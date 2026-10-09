"""
Tests for Query Router Module
Validates deterministic fast-path query routing, response formatting, and quota fallback.
"""

from unittest.mock import MagicMock
import pytest
from google.genai.errors import APIError

import router
from agent import run_agent_turn


def test_router_order_id_lookup():
    """Verify ORD-1001 exact match and details formatting."""
    queries = [
        "What is the status and details of order ORD-1001?",
        "ORD-1001",
        "details of order ORD-1001",
        "track order ORD-1001",
        "Status of ord-1001",
    ]
    for q in queries:
        res = router.try_fast_path_answer(q)
        assert res is not None, f"Query '{q}' should have matched fast path"
        assert res["metadata"]["routing"] == "fast_path"
        assert res["metadata"]["pattern"] == "order_id_lookup"
        assert len(res["tool_calls"]) == 1
        assert res["tool_calls"][0]["tool"] == "search_orders"
        assert "ORD-1001" in res["answer"]
        assert "Rahul Sharma" in res["answer"]
        assert "₹1,598.00" in res["answer"]
        assert "Delivered" in res["answer"]


def test_router_order_id_not_found():
    """Verify nonexistent order ID returns clear notice without crashing."""
    res = router.try_fast_path_answer("Check status of ORD-9999")
    assert res is not None
    assert "was not found in the dataset" in res["answer"]
    assert res["tool_calls"][0]["result"]["total_matched"] == 0


def test_router_cancelled_and_lost_revenue():
    """Verify cancelled orders and lost revenue query matches calculate_order_statistics."""
    q = "How many orders were cancelled and what was the lost revenue?"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "cancelled_and_pending"
    assert res["tool_calls"][0]["tool"] == "calculate_order_statistics"
    assert "7" in res["answer"]  # 7 cancelled orders
    assert "₹72,634.00" in res["answer"]  # lost revenue


def test_router_pending_or_shipped_orders():
    """Verify pending and shipped order search."""
    q = "Show all processing or shipped orders."
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "pending_or_shipped_orders"
    assert "ORD-1059" in res["answer"]
    assert "ORD-1060" in res["answer"]
    assert len(res["tool_calls"][0]["result"]["orders"]) == 2


def test_router_monthly_revenue():
    """Verify monthly revenue table generation."""
    q = "What was the total revenue for each month from June to September 2026?"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "monthly_revenue"
    assert "June 2026" in res["answer"]
    assert "August 2026" in res["answer"]
    assert "₹139,059.00" in res["answer"]


def test_router_total_revenue():
    """Verify overall revenue summary."""
    q = "What is the total revenue?"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "revenue_total"
    assert "₹371,040.00" in res["answer"]  # delivered revenue
    assert "₹470,312.00" in res["answer"]  # gross revenue


def test_router_top_customers():
    """Verify highest-spending customer report."""
    q = "Who is the highest-spending customer and how much have they spent?"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "top_customers"
    assert "Rohan Das" in res["answer"]
    assert "Leaderboard" in res["answer"]


def test_router_revenue_by_category():
    """Verify product category performance breakdown."""
    q = "Which product category generated the highest revenue?"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "revenue_by_category"
    assert "Electronics" in res["answer"]
    assert "Furniture" in res["answer"]


def test_router_revenue_by_city():
    """Verify city breakdown."""
    q = "Show revenue by city"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "revenue_by_city"
    assert "Chennai" in res["answer"]
    assert "Thiruvananthapuram" in res["answer"]


def test_router_summary():
    """Verify overall order summary."""
    q = "Give me an overview of orders"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "order_summary"
    assert "60" in res["answer"]


def test_router_customer_lookup():
    """Verify customer specific lookup."""
    q = "What orders did Sneha Pillai place?"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "customer_lookup"
    assert "Sneha Pillai" in res["answer"]


def test_router_product_lookup():
    """Verify product specific lookup."""
    q = "Show orders for Wireless Mouse"
    res = router.try_fast_path_answer(q)
    assert res is not None
    assert res["metadata"]["pattern"] == "product_lookup"
    assert "Wireless Mouse" in res["answer"]


def test_complex_query_routes_to_gemini():
    """Verify complex or non-deterministic questions return None so Gemini handles them."""
    complex_questions = [
        "What marketing campaign should we run to boost stationery sales next quarter?",
        "Compare customer satisfaction in Pune vs Bengaluru and suggest improvements.",
        "Tell me a fun story about online shopping.",
    ]
    for q in complex_questions:
        res = router.try_fast_path_answer(q)
        assert res is None, f"Query '{q}' should NOT match fast-path, but route to Gemini"


def test_run_agent_turn_fast_path(monkeypatch):
    """Verify run_agent_turn uses fast-path router when API key is set."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_valid_looking_key")

    res = run_agent_turn("What is the status of order ORD-1001?")
    assert res["metadata"]["routing"] == "fast_path"
    assert "ORD-1001" in res["answer"]
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["tool"] == "search_orders"


def test_run_agent_turn_quota_fallback(monkeypatch):
    """Verify quota exhaustion activates graceful deterministic fallback."""
    # When use_fast_path=False (e.g. complex query), and Gemini returns a 429 quota error:
    mock_client = MagicMock()
    quota_error = APIError(429, {"error": {"message": "Resource exhausted: quota exceeded"}})
    mock_client.models.generate_content.side_effect = quota_error

    # Give mock client a real client class name so it triggers the client branch
    type(mock_client).__name__ = "Client"

    res = run_agent_turn(
        message="Complex analysis question",
        client=mock_client,
        use_fast_path=False,
        fallback_on_quota=True,
    )

    assert res["metadata"]["routing"] == "fallback"
    assert "quota is currently exhausted" in res["answer"].lower()
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["tool"] == "calculate_order_statistics"
