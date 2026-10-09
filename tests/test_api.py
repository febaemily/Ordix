"""
Tests for API Endpoints and AI Agent Tool Execution
Uses FastAPI TestClient and mocks external Gemini network calls.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from main import app
from agent import AgentConfigurationError, AgentExecutionError

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns readiness details and dataset confirmation."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["brand"] == "ORDIX"
    assert data["tagline"] == "Orders Simplified"
    assert data["dataset_ready"] is True
    assert data["total_orders"] == 60
    assert data["date_range"]["min"] == "2026-06-01"
    assert data["date_range"]["max"] == "2026-09-28"
    assert "search_orders" in data["supported_operations"]
    assert "calculate_order_statistics" in data["supported_operations"]


def test_suggestions_endpoint():
    """Verify /api/suggestions returns starter questions."""
    response = client.get("/api/suggestions")
    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) >= 4
    titles = [s["title"] for s in data["suggestions"]]
    assert "Track an Order" in titles
    assert "Cancelled Orders" in titles


def test_chat_empty_query_rejected():
    """Verify empty query or whitespace is rejected with validation error."""
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_excessively_long_query_rejected():
    """Verify questions exceeding character limit are rejected."""
    long_msg = "A" * 1005
    response = client.post("/api/chat", json={"message": long_msg})
    assert response.status_code == 422


def test_chat_missing_api_key_returns_503(monkeypatch):
    """Verify clean 503 error when GEMINI_API_KEY is not configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/chat", json={"message": "What is the status of ORD-1001?"})
    assert response.status_code == 503
    data = response.json()
    assert data["success"] is False
    assert "gemini api key is not configured" in data["error"].lower()


@patch("main.run_agent_turn")
def test_chat_successful_mocked_turn(mock_run_agent, monkeypatch):
    """Verify successful AI answer with tool execution metadata."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key")
    mock_run_agent.return_value = {
        "answer": "Order ORD-1001 was placed by Rahul Sharma on 2026-06-01 for 2 Wireless Mice totaling ₹1,598, and its status is delivered.",
        "tool_calls": [
            {
                "tool": "search_orders",
                "args": {"order_id": "ORD-1001"},
                "result": {
                    "success": True,
                    "orders": [{"order_id": "ORD-1001", "status": "delivered"}],
                },
            }
        ],
        "metadata": {
            "model": "gemini-3.8-flash",
            "iterations": 1,
            "execution_time_ms": 120.5,
        },
    }

    response = client.post(
        "/api/chat",
        json={"message": "What is the status of order ORD-1001?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "ORD-1001" in data["answer"]
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool"] == "search_orders"
    assert data["metadata"]["model"] == "gemini-3.8-flash"


@patch("main.run_agent_turn")
def test_chat_provider_failure_returns_502(mock_run_agent, monkeypatch):
    """Verify AI provider error returns 502 with user-friendly message."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key")
    mock_run_agent.side_effect = AgentExecutionError("Gemini AI provider returned an error: Quota exceeded")

    response = client.post(
        "/api/chat",
        json={"message": "Show monthly revenue"},
    )
    assert response.status_code == 502
    data = response.json()
    assert data["success"] is False
    assert "gemini ai provider returned an error" in data["error"].lower()


def test_serve_index_html():
    """Verify root GET / serves the frontend interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ORDIX — Orders Simplified" in response.text
    assert "Welcome to ORDIX" in response.text


def test_regression_short_question_with_long_conversation_history():
    """
    Regression test: 'How many customers from Thiruvananthapuram?'
    Previously triggered 'String should have at most 2000 characters' when conversation_history
    contained a previous detailed response (e.g., from 'who paid online') exceeding 2000 chars.
    Verifies HTTP 200, no 422 validation failure, and accurate answers calculated directly from orders.csv.
    """
    long_prior_response = (
        "### Customers Who Paid Online (50 Orders / 15 Customers)\n\n"
        + "| Order ID | Customer | Payment Method | Product | Amount | Date | Status |\n"
        + "| :--- | :--- | :--- | :--- | :--- | :--- | :---: |\n"
        + "\n".join([f"| **ORD-{1000+i}** | Customer {i} | UPI | Mouse | ₹1,598.00 | 2026-06-01 | Delivered |" for i in range(50)])
    )
    assert len(long_prior_response) > 2000

    response = client.post(
        "/api/chat",
        json={
            "message": "How many customers from Thiruvananthapuram?",
            "conversation_history": [
                {"role": "user", "content": "who paid online"},
                {"role": "model", "content": long_prior_response},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "4" in data["answer"]
    assert "Thiruvananthapuram" in data["answer"]
    assert "Rohan Das" in data["answer"]
    assert "Priya Menon" in data["answer"]
    assert "Divya Joseph" in data["answer"]
    assert "Suresh Babu" in data["answer"]


def test_dashboard_endpoint():
    """Verify /api/dashboard returns correct precalculated metrics from orders.csv."""
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    kpis = data["kpis"]
    assert kpis["total_orders"] == 60
    assert kpis["delivered_orders"] == 48
    assert kpis["delivered_revenue_inr"] == 371040.0
    assert kpis["gross_revenue_inr"] == 470312.0
    assert kpis["cancelled_orders"] == 7
    assert kpis["pending_orders"] == 2
    assert kpis["total_customers"] == 15
    assert len(data["monthly_revenue"]) == 4
    assert len(data["payment_methods"]) >= 5


def test_orders_list_endpoint():
    """Verify /api/orders returns filterable, searchable, and paginated records."""
    response = client.get("/api/orders?page=1&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 60
    assert len(data["orders"]) == 10
    assert data["page"] == 1
    assert data["total_pages"] == 6

    # Test search
    res_search = client.get("/api/orders?q=Thiruvananthapuram")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total"] == 16
    assert all("Thiruvananthapuram" in o["city"] for o in search_data["orders"])

    # Test status filter
    res_status = client.get("/api/orders?status=cancelled")
    assert res_status.status_code == 200
    assert res_status.json()["total"] == 7

