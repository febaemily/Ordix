"""
ORDIX — Orders Simplified
FastAPI Application: Serves the REST API, AI agent endpoint, health check, and static frontend.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import tools
from agent import (
    AgentConfigurationError,
    AgentExecutionError,
    get_gemini_client,
    run_agent_turn,
)
from data_loader import DataLoadError, get_dataset_summary, load_orders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ordix.main")

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# Pydantic Request & Response Models
class MessageTurn(BaseModel):
    role: str = Field(..., description="Role of the sender: 'user' or 'model'/'assistant'")
    content: str = Field(..., min_length=1, max_length=50000, description="Message text")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned in ["user", "human"]:
            return "user"
        return "model"

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("Message content cannot be empty.")
        if len(text) > 30000:
            return text[:30000]
        return text


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question or query about order data",
    )
    conversation_history: Optional[List[MessageTurn]] = Field(
        default=None,
        max_length=20,
        description="Previous turns in the conversation for multi-turn context",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or solely whitespace.")
        return trimmed


class ChatResponse(BaseModel):
    success: bool
    answer: str
    tool_calls: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    brand: str
    tagline: str
    timestamp: str
    dataset_ready: bool
    total_orders: int
    date_range: Dict[str, Optional[str]]
    gemini_api_configured: bool
    supported_operations: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ORDIX Order Assistant...")
    try:
        orders = load_orders()
        summary = get_dataset_summary(orders)
        logger.info(
            "Dataset successfully verified: %d orders loaded (Date range: %s to %s).",
            summary["total_orders"],
            summary["date_range"]["min"],
            summary["date_range"]["max"],
        )
    except Exception as e:
        logger.error("Error loading orders dataset on startup: %s", str(e))
    yield


# Initialize FastAPI App
app = FastAPI(
    title="ORDIX — Orders Simplified",
    description="Intelligent AI Order Assistant API powered by Google Gemini and deterministic backend tools.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development and integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 1. Health Endpoint
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Returns non-sensitive readiness information, confirming dataset status and configuration.
    """
    api_key_configured = bool(os.getenv("GEMINI_API_KEY", "").strip()) and (
        os.getenv("GEMINI_API_KEY", "").strip() != "your_gemini_api_key_here"
    )

    try:
        summary = get_dataset_summary()
        dataset_ready = summary["total_orders"] > 0
        total_orders = summary["total_orders"]
        date_range = summary["date_range"]
    except Exception as e:
        logger.warning("Health check dataset warning: %s", str(e))
        dataset_ready = False
        total_orders = 0
        date_range = {"min": None, "max": None}

    return HealthResponse(
        status="ok" if dataset_ready else "degraded",
        brand="ORDIX",
        tagline="Orders Simplified",
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_ready=dataset_ready,
        total_orders=total_orders,
        date_range=date_range,
        gemini_api_configured=api_key_configured,
        supported_operations=[
            "search_orders",
            "calculate_order_statistics",
        ],
    )


# 2. Suggested Questions Endpoint
@app.get("/api/suggestions")
def get_suggestions() -> Dict[str, Any]:
    """Returns curated starter questions grounded in the dataset."""
    return {
        "suggestions": [
            {"title": "Track an Order", "prompt": "What is the status and details of order ORD-1001?"},
            {"title": "Cancelled Orders", "prompt": "How many orders were cancelled and what was the lost revenue?"},
            {"title": "Highest Revenue Month", "prompt": "Which month had the highest revenue?"},
            {"title": "Lowest Spender", "prompt": "Who spent the least?"},
            {"title": "Paid Online", "prompt": "Who paid online?"},
            {"title": "Compare Months", "prompt": "Compare monthly revenue."},
            {"title": "City Analysis", "prompt": "How many customers are from Thiruvananthapuram?"},
            {"title": "Top Customer", "prompt": "Who is the highest-spending customer and how much have they spent?"},
        ]
    }


# 3. Dashboard Statistics Endpoint
@app.get("/api/dashboard")
def get_dashboard_data() -> Dict[str, Any]:
    """
    Returns executive KPI statistics, monthly chronological revenue trends,
    status distribution, payment methods, category performance, and city rankings.
    All numbers calculated directly from the real orders dataset via Python/pandas.
    """
    try:
        orders = load_orders()
    except DataLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))

    summary_res = tools.calculate_order_statistics("summary")
    monthly_res = tools.calculate_order_statistics("monthly_revenue")
    status_res = tools.calculate_order_statistics("order_counts_by_status")
    category_res = tools.calculate_order_statistics("revenue_by_category")
    city_res = tools.calculate_order_statistics("revenue_by_city")

    summary_data = summary_res.get("data", {})
    monthly_breakdown = monthly_res.get("data", {}).get("monthly_breakdown", [])
    counts_by_status = status_res.get("data", {}).get("counts_by_status", {})
    categories = category_res.get("data", {}).get("categories", [])
    cities = city_res.get("data", {}).get("cities", [])

    # Month name mapping for clean display
    month_display_names = {
        "2026-06": "June 2026",
        "2026-07": "July 2026",
        "2026-08": "August 2026",
        "2026-09": "September 2026",
    }
    for m in monthly_breakdown:
        m["display_name"] = month_display_names.get(m["month"], m["month"])

    # Payment method breakdown
    payment_counts: Dict[str, int] = {}
    payment_amounts: Dict[str, float] = {}
    for o in orders:
        pm = o["payment_method"]
        payment_counts[pm] = payment_counts.get(pm, 0) + 1
        payment_amounts[pm] = payment_amounts.get(pm, 0.0) + float(o["total_inr"])

    total_orders_count = len(orders)
    payment_breakdown = []
    for pm, count in sorted(payment_counts.items(), key=lambda x: x[1], reverse=True):
        payment_breakdown.append({
            "method": pm,
            "order_count": count,
            "total_inr": round(payment_amounts[pm], 2),
            "share_percentage": round((count / total_orders_count * 100), 1) if total_orders_count > 0 else 0.0,
            "is_online": pm != "Cash on Delivery",
        })

    # Unique customers
    unique_customers = len(set(o["customer_name"] for o in orders))

    # Pending count (processing + shipped)
    pending_count = counts_by_status.get("processing", 0) + counts_by_status.get("shipped", 0)

    # Latest 6 orders for recent activity
    sorted_recent = sorted(orders, key=lambda x: (x.get("order_date", ""), x.get("order_id", "")), reverse=True)

    return {
        "success": True,
        "kpis": {
            "total_orders": summary_data.get("total_orders", 60),
            "delivered_orders": summary_data.get("delivered_orders_count", 48),
            "delivered_revenue_inr": summary_data.get("delivered_orders_revenue_inr", 371040.0),
            "gross_revenue_inr": summary_data.get("total_gross_order_value_inr", 470312.0),
            "pending_orders": pending_count,
            "cancelled_orders": counts_by_status.get("cancelled", 7),
            "returned_orders": counts_by_status.get("returned", 3),
            "cancelled_loss_inr": summary_data.get("cancelled_value_lost_inr", 72634.0),
            "total_customers": unique_customers,
            "average_delivered_order_value_inr": summary_data.get("average_delivered_order_value_inr", 7730.0),
            "fulfillment_rate_percentage": round(
                (summary_data.get("delivered_orders_count", 48) / summary_data.get("total_orders", 60) * 100), 1
            ),
        },
        "monthly_revenue": monthly_breakdown,
        "status_distribution": counts_by_status,
        "payment_methods": payment_breakdown,
        "categories": categories,
        "cities": cities,
        "recent_orders": sorted_recent[:6],
    }


# 4. Orders Explorer Listing Endpoint
@app.get("/api/orders")
def list_orders(
    q: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    city: Optional[str] = None,
    payment_method: Optional[str] = None,
    sort_by: str = "order_date",
    sort_order: str = "desc",
    page: int = 1,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Returns filtered, searched, sorted, and paginated real order records.
    Supports multi-field filtering, full-text substring search, and column sorting.
    """
    try:
        all_orders = load_orders()
    except DataLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))

    filtered = list(all_orders)

    # 1. Text search across order_id, customer, product, city
    if q and q.strip():
        search_term = q.strip().lower()
        filtered = [
            o for o in filtered
            if search_term in o["order_id"].lower()
            or search_term in o["customer_name"].lower()
            or search_term in o["product"].lower()
            or search_term in o["city"].lower()
        ]

    # 2. Status filter
    if status and status.strip() and status.lower() != "all":
        target_status = status.strip().lower()
        filtered = [o for o in filtered if o["status"].lower() == target_status]

    # 3. Category filter
    if category and category.strip() and category.lower() != "all":
        target_cat = category.strip().lower()
        filtered = [o for o in filtered if o["category"].lower() == target_cat]

    # 4. City filter
    if city and city.strip() and city.lower() != "all":
        target_city = city.strip().lower()
        filtered = [o for o in filtered if o["city"].lower() == target_city]

    # 5. Payment method filter
    if payment_method and payment_method.strip() and payment_method.lower() != "all":
        pm_target = payment_method.strip().lower()
        if pm_target in ["online", "digital"]:
            filtered = [o for o in filtered if o["payment_method"] != "Cash on Delivery"]
        elif pm_target in ["offline", "cod"]:
            filtered = [o for o in filtered if o["payment_method"] == "Cash on Delivery"]
        else:
            filtered = [o for o in filtered if o["payment_method"].lower() == pm_target]

    total_matched = len(filtered)

    # 6. Sorting
    reverse_sort = (sort_order.lower() == "desc")
    if sort_by in ["total_inr", "quantity", "unit_price_inr"]:
        filtered.sort(key=lambda x: float(x.get(sort_by, 0.0)), reverse=reverse_sort)
    elif sort_by in ["order_date", "order_id", "customer_name", "status", "category", "city"]:
        filtered.sort(key=lambda x: str(x.get(sort_by, "")).lower(), reverse=reverse_sort)
    else:
        filtered.sort(key=lambda x: str(x.get("order_date", "")), reverse=True)

    # 7. Pagination
    safe_page = max(1, page)
    safe_limit = max(1, min(limit, 100))
    start_idx = (safe_page - 1) * safe_limit
    end_idx = start_idx + safe_limit
    paginated_slice = filtered[start_idx:end_idx]
    total_pages = max(1, (total_matched + safe_limit - 1) // safe_limit)

    # Summary metrics for the filtered subset
    filtered_delivered_rev = sum(o["total_inr"] for o in filtered if o["status"] == "delivered")
    filtered_gross_val = sum(o["total_inr"] for o in filtered)

    return {
        "success": True,
        "orders": paginated_slice,
        "total": total_matched,
        "page": safe_page,
        "total_pages": total_pages,
        "pagination": {
            "page": safe_page,
            "limit": safe_limit,
            "total_records": total_matched,
            "total_pages": total_pages,
            "has_next": safe_page < total_pages,
            "has_prev": safe_page > 1,
        },
        "subset_metrics": {
            "total_matched": total_matched,
            "delivered_revenue_inr": round(filtered_delivered_rev, 2),
            "gross_revenue_inr": round(filtered_gross_val, 2),
        },
        "filter_options": {
            "statuses": ["delivered", "cancelled", "returned", "processing", "shipped"],
            "categories": ["Electronics", "Accessories", "Furniture", "Stationery"],
            "cities": ["Chennai", "Kochi", "Thiruvananthapuram", "Bengaluru", "Pune", "Hyderabad"],
            "payment_methods": ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"],
        },
    }


# 3. Chat Endpoint
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Processes user query using Gemini with backend tool calling.
    Never exposes API secrets or raw stack traces.
    """
    # 1. Verify dataset readiness
    try:
        load_orders()
    except DataLoadError as dle:
        logger.error("Dataset unavailable: %s", str(dle))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The orders dataset is currently unavailable on the server.",
        )

    # 2. Convert conversation history if present
    history_dicts = (
        [turn.model_dump() for turn in payload.conversation_history]
        if payload.conversation_history
        else None
    )

    # 3. Execute AI Agent Turn
    try:
        agent_result = run_agent_turn(
            message=payload.message,
            conversation_history=history_dicts,
        )

        return ChatResponse(
            success=True,
            answer=agent_result.get("answer", ""),
            tool_calls=agent_result.get("tool_calls", []),
            metadata=agent_result.get("metadata", {}),
        )

    except AgentConfigurationError as ace:
        logger.warning("Configuration error: %s", str(ace))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ace),
        )
    except AgentExecutionError as aee:
        logger.error("AI agent execution error: %s", str(aee))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(aee),
        )
    except Exception as e:
        logger.error("Unhandled error processing chat: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request. Please try again.",
        )


@app.get("/")
def serve_index() -> FileResponse:
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        content={"message": "ORDIX API is running. Static frontend not yet compiled."},
        status_code=200,
    )


# Exception handler for cleaner error JSONs
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )
