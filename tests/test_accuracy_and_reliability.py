"""
Tests for ORDIX Accuracy and Reliability Fixes
Validates:
1. Revenue calculations for different months return distinct and accurate values.
2. Monthly revenue breakdowns always include both fulfilled and overall gross revenue totals.
3. 'Who paid online?' returns actual matching customer and order records directly.
4. 'How many customers are from Thiruvananthapuram?' returns exact unique count and handles multi-turn conversation history without 2000-char validation error.
5. 'Which customer has spent the least?' returns verified lowest spending customer (Nikhil Gupta) without failing or triggering Gemini quota warning.
6. Spending rankings handle ties and exclude cancelled/refunded orders from delivered spend.
7. Graceful fallback on rate-limits, quota exhaustion, and server busy (503) returns calculated Python answers.
8. Small controlled synthetic dataset tests verifying all corner cases.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from google.genai.errors import APIError

import tools
import router
import query_engine
import query_planner
import plan_executor
import agent
from main import app

client = TestClient(app)


# ==============================================================================
# 1. Monthly Revenue Accuracy & Distinctness
# ==============================================================================

def test_august_and_september_revenue_are_distinct():
    """Verify August and September return distinct, accurate revenue calculations."""
    aug_res = query_engine.process_natural_query("August revenue")
    sep_res = query_engine.process_natural_query("September revenue")

    assert aug_res is not None
    assert sep_res is not None

    # August 2026: 20 orders, ₹1,39,059.00 delivered, ₹1,51,301.00 gross
    assert "₹139,059.00" in aug_res["answer"]
    assert "₹151,301.00" in aug_res["answer"]
    assert "20 total order(s)" in aug_res["answer"]

    # September 2026: 8 orders, ₹41,292.00 delivered, ₹47,436.00 gross
    assert "₹41,292.00" in sep_res["answer"]
    assert "₹47,436.00" in sep_res["answer"]
    assert "8 total order(s)" in sep_res["answer"]

    # Ensure answers are completely distinct
    assert aug_res["answer"] != sep_res["answer"]


def test_monthly_revenue_table_includes_gross_and_delivered_totals():
    """Verify month-by-month table includes individual months and the overall gross & delivered totals."""
    # Test via query_engine
    q_engine_res = query_engine.process_natural_query("What was the total revenue for each month from June to September 2026?")
    assert q_engine_res is not None
    assert "Overall Total" in q_engine_res["answer"]
    assert "₹371,040.00" in q_engine_res["answer"]  # Total delivered
    assert "₹470,312.00" in q_engine_res["answer"]  # Total gross
    assert "60" in q_engine_res["answer"]  # Total orders

    # Test via router
    router_res = router.route_monthly_revenue()
    assert router_res is not None
    assert "| **Total** |" in router_res["answer"]
    assert "₹371,040.00" in router_res["answer"]
    assert "₹470,312.00" in router_res["answer"]
    assert "—" not in router_res["answer"]  # No missing dashes in total row!


# ==============================================================================
# 2. Online Payments Query
# ==============================================================================

def test_who_paid_online_returns_actual_matching_orders():
    """Verify 'Who paid online?' returns matching orders and customer names based on dataset payment methods."""
    res = query_engine.process_natural_query("Who paid online?")
    assert res is not None
    assert res["metadata"]["intent"] == "who_paid_online"

    # In orders.csv, exactly 50 of 60 orders were paid online (UPI: 15, Debit Card: 12, Credit Card: 12, Net Banking: 11)
    tool_call = res["tool_calls"][0]
    matched_count = tool_call["result"]["total_matched"]
    assert matched_count == 50

    # Returned orders should only contain online payment methods
    orders = tool_call["result"]["orders"]
    for o in orders:
        assert o["payment_method"] in ["UPI", "Credit Card", "Debit Card", "Net Banking"]
        assert o["payment_method"] != "Cash on Delivery"

    # Ensure customer names are present
    assert "Rahul Sharma" in res["answer"]
    assert "Sneha Pillai" in res["answer"]


def test_tools_search_orders_supports_online_payment_filter():
    """Verify tools.search_orders correctly filters online payment methods."""
    res = tools.search_orders(payment_method="online", limit=60)
    assert res["success"] is True
    assert res["total_matched"] == 50
    for o in res["orders"]:
        assert o["payment_method"] in ["UPI", "Credit Card", "Debit Card", "Net Banking"]


# ==============================================================================
# 3. Thiruvananthapuram and 2000-Char Validation Error Fix
# ==============================================================================

def test_thiruvananthapuram_customers_count_and_details():
    """Verify city questions return the exact count and records from the CSV."""
    res = query_engine.process_natural_query("How many customers are from Thiruvananthapuram?")
    assert res is not None
    assert "4 unique customers" in res["answer"]
    assert "Divya Joseph" in res["answer"]
    assert "Priya Menon" in res["answer"]
    assert "Rohan Das" in res["answer"]
    assert "Suresh Babu" in res["answer"]


def test_2000_char_validation_error_fixed_in_api():
    """
    Verify regression where a short question followed by a conversation history
    turn > 2,000 characters no longer triggers HTTP 422 'String should have at most 2000 characters'.
    """
    long_prior_answer = (
        "### Customers Who Paid Online (50 Orders / 15 Customers)\n\n"
        + ("| ORD-1001 | Rahul Sharma | Debit Card | Wireless Mouse | ₹1,598.00 | Delivered |\n" * 40)
    )
    assert len(long_prior_answer) > 2500

    payload = {
        "message": "How many customers are from Thiruvananthapuram?",
        "conversation_history": [
            {"role": "user", "content": "Who paid online?"},
            {"role": "model", "content": long_prior_answer},
        ],
    }

    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["success"] is True
    assert "4 unique customer" in data["answer"]
    assert "Thiruvananthapuram" in data["answer"]


# ==============================================================================
# 4. Lowest Spending Customer & Quota Fallback
# ==============================================================================

def test_which_customer_has_spent_the_least():
    """Verify 'Which customer has spent the least?' returns Nikhil Gupta directly with verified metrics."""
    res = query_engine.process_natural_query("Which customer has spent the least?")
    assert res is not None
    assert "Nikhil Gupta" in res["answer"]
    assert "₹2,146.00" in res["answer"]
    assert "3 fulfilled order(s)" in res["answer"]


def test_who_spent_the_least_variations():
    """Verify phrasing variations for lowest spending customer resolve correctly."""
    variations = [
        "Who spent the least?",
        "Which customer spent the least?",
        "Lowest spending customer",
        "Least spending customer",
        "who is the lowest spender?",
    ]
    for q in variations:
        res = query_engine.process_natural_query(q)
        assert res is not None, f"Failed for query: {q}"
        assert "Nikhil Gupta" in res["answer"]
        assert "₹2,146.00" in res["answer"]


def test_agent_turn_answers_least_spender_without_gemini():
    """Verify run_agent_turn answers 'Which customer has spent the least?' with 0 API calls."""
    res = agent.run_agent_turn("Which customer has spent the least?", client=None, use_fast_path=True)
    assert res is not None
    assert "Nikhil Gupta" in res["answer"]
    assert "₹2,146.00" in res["answer"]


def test_gemini_quota_exhaustion_does_not_replace_answer_with_warning():
    """
    Verify when Gemini returns 429 quota exhaustion or busy 503, the assistant
    safely returns the actual calculated Python answer instead of a generic summary.
    """
    mock_client = MagicMock()
    mock_error = APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED: quota exceeded"}})
    mock_client.models.generate_content.side_effect = mock_error

    # Test with use_fast_path=False to simulate falling through to Gemini and catching 429
    res = agent.run_agent_turn(
        message="Which customer has spent the least?",
        client=mock_client,
        use_fast_path=False,
        fallback_on_quota=True,
    )

    assert res is not None
    assert "Nikhil Gupta" in res["answer"]
    assert "₹2,146.00" in res["answer"]


def test_gemini_busy_503_fallback_delivers_answer():
    """Verify when Gemini returns 503 Service Unavailable, fallback returns calculated answer."""
    mock_client = MagicMock()
    mock_error = APIError(503, {"error": {"message": "The service is currently unavailable or high demand"}})
    mock_client.models.generate_content.side_effect = mock_error

    res = agent.run_agent_turn(
        message="August revenue",
        client=mock_client,
        use_fast_path=False,
        fallback_on_quota=True,
    )

    assert res is not None
    assert "₹139,059.00" in res["answer"]
    assert "August 2026" in res["answer"]


# ==============================================================================
# 5. Controlled Synthetic Dataset with Ties & Cancelled Orders
# ==============================================================================

def test_controlled_synthetic_dataset_with_ties_and_cancellations():
    """
    Validates calculations on a small controlled synthetic dataset with known values,
    ties in spending, and cancelled/refunded orders.
    """
    synthetic_csv = (
        "order_id,order_date,customer_name,city,product,category,quantity,unit_price_inr,total_inr,payment_method,status\n"
        "ORD-001,2026-08-01,Customer Alpha,Thiruvananthapuram,Mouse,Accessories,1,100,100,UPI,delivered\n"
        "ORD-002,2026-08-02,Customer Beta,Kochi,Mouse,Accessories,1,100,100,Credit Card,delivered\n"
        "ORD-003,2026-09-01,Customer Gamma,Chennai,Chair,Furniture,1,500,500,Cash on Delivery,delivered\n"
        "ORD-004,2026-09-02,Customer Alpha,Thiruvananthapuram,Desk,Furniture,1,1000,1000,Debit Card,cancelled\n"
        "ORD-005,2026-09-03,Customer Delta,Bengaluru,Laptop,Electronics,1,2000,2000,UPI,cancelled\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(synthetic_csv)
        temp_csv_path = f.name

    try:
        # 1. Total statistics
        stats = tools.calculate_order_statistics("summary", csv_path=temp_csv_path)
        data = stats["data"]
        assert data["total_orders"] == 5
        assert data["delivered_orders_count"] == 3
        assert data["delivered_orders_revenue_inr"] == 700.0  # 100 + 100 + 500
        assert data["total_gross_order_value_inr"] == 3700.0  # 100 + 100 + 500 + 1000 + 2000
        assert data["cancelled_value_lost_inr"] == 3000.0   # 1000 + 2000

        # 2. Monthly revenue distinctness on synthetic data
        aug_res = query_engine.execute_single_month_revenue(8, "August", query_engine.get_dataframe(temp_csv_path), year=2026)
        sep_res = query_engine.execute_single_month_revenue(9, "September", query_engine.get_dataframe(temp_csv_path), year=2026)
        assert "₹200.00" in aug_res["answer"]  # August delivered
        assert "₹500.00" in sep_res["answer"]  # September delivered

        # 3. Online payments on synthetic data
        online_res = tools.search_orders(payment_method="online", csv_path=temp_csv_path)
        assert online_res["total_matched"] == 4  # ORD-001, ORD-002, ORD-004, ORD-005

        # 4. Lowest spending customer with tie handling
        # Customer Alpha delivered = 100, Customer Beta delivered = 100 (Tie!)
        # Customer Delta delivered = 0 (only has cancelled order)
        least_res = query_engine.execute_customer_frequency_and_spending(
            query_engine.get_dataframe(temp_csv_path), ranking_by="least_spending"
        )
        assert "Customer Delta" in least_res["answer"]  # 0 delivered spend
        assert "₹0.00" in least_res["answer"]

        # Top spending customer
        top_res = query_engine.execute_customer_frequency_and_spending(
            query_engine.get_dataframe(temp_csv_path), ranking_by="spending"
        )
        assert "Customer Gamma" in top_res["answer"]
        assert "₹500.00" in top_res["answer"]

    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
