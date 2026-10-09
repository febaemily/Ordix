"""
Tests for Backend Tools Module
Validates deterministic search and statistical calculations against orders.csv.
"""

import pytest
from tools import calculate_order_statistics, search_orders, validate_date


# ==============================================================================
# Tests for search_orders Tool
# ==============================================================================

def test_search_orders_exact_id():
    """Verify exact order lookup returns the real record."""
    result = search_orders(order_id="ORD-1001")
    assert result["success"] is True
    assert result["total_matched"] == 1
    assert result["returned_count"] == 1
    order = result["orders"][0]
    assert order["order_id"] == "ORD-1001"
    assert order["customer_name"] == "Rahul Sharma"
    assert order["city"] == "Hyderabad"
    assert order["product"] == "Wireless Mouse"
    assert order["total_inr"] == 1598.0
    assert order["status"] == "delivered"


def test_search_orders_nonexistent_id():
    """Verify lookup for a nonexistent order returns 0 matches gracefully without inventing records."""
    result = search_orders(order_id="ORD-999999")
    assert result["success"] is True
    assert result["total_matched"] == 0
    assert result["returned_count"] == 0
    assert len(result["orders"]) == 0
    assert "no orders found" in result["message"].lower()


def test_search_orders_filter_by_status():
    """Verify filtering by order status (e.g. cancelled)."""
    result = search_orders(status="cancelled", limit=50)
    assert result["success"] is True
    assert result["total_matched"] == 7
    for o in result["orders"]:
        assert o["status"] == "cancelled"


def test_search_orders_filter_by_customer_partial():
    """Verify partial customer name matching."""
    result = search_orders(customer_name="Pillai")
    assert result["success"] is True
    assert result["total_matched"] >= 1
    for o in result["orders"]:
        assert "pillai" in o["customer_name"].lower()


def test_search_orders_filter_by_category():
    """Verify filtering by product category."""
    result = search_orders(category="Furniture")
    assert result["success"] is True
    assert result["total_matched"] > 0
    for o in result["orders"]:
        assert o["category"].lower() == "furniture"


def test_search_orders_filter_by_city():
    """Verify filtering by city."""
    result = search_orders(city="Chennai")
    assert result["success"] is True
    assert result["total_matched"] > 0
    for o in result["orders"]:
        assert o["city"].lower() == "chennai"


def test_search_orders_filter_by_payment_method():
    """Verify filtering by payment method."""
    result = search_orders(payment_method="UPI")
    assert result["success"] is True
    assert result["total_matched"] > 0
    for o in result["orders"]:
        assert "upi" in o["payment_method"].lower()


def test_search_orders_date_filtering():
    """Verify filtering within a specific date range."""
    result = search_orders(start_date="2026-08-01", end_date="2026-08-31", limit=50)
    assert result["success"] is True
    assert result["total_matched"] == 20
    for o in result["orders"]:
        assert "2026-08-01" <= o["order_date"] <= "2026-08-31"


def test_search_orders_invalid_date_format():
    """Verify invalid date format returns an error message without crashing."""
    result = search_orders(start_date="01-08-2026")
    assert result["success"] is False
    assert "invalid date format" in result["error"].lower()


def test_search_orders_limit_applied():
    """Verify limit capping."""
    result = search_orders(limit=3)
    assert result["success"] is True
    assert result["returned_count"] == 3
    assert result["total_matched"] == 60


# ==============================================================================
# Tests for calculate_order_statistics Tool
# ==============================================================================

def test_calculate_summary():
    """Verify overall order summary metrics."""
    result = calculate_order_statistics(operation="summary")
    assert result["success"] is True
    assert result["operation"] == "summary"
    data = result["data"]
    assert data["total_orders"] == 60
    assert data["delivered_orders_count"] == 48
    assert data["delivered_orders_revenue_inr"] > 0
    assert data["cancelled_value_lost_inr"] > 0
    assert data["average_delivered_order_value_inr"] > 0
    assert data["status_breakdown"]["delivered"] == 48
    assert data["status_breakdown"]["cancelled"] == 7


def test_calculate_order_counts_by_status():
    """Verify exact count for each order status."""
    result = calculate_order_statistics(operation="order_counts_by_status")
    assert result["success"] is True
    counts = result["data"]["counts_by_status"]
    assert counts["delivered"] == 48
    assert counts["cancelled"] == 7
    assert counts["returned"] == 3
    assert counts["processing"] == 1
    assert counts["shipped"] == 1
    assert sum(counts.values()) == 60


def test_calculate_cancelled_and_pending():
    """Verify cancelled, returned, and processing/shipped counts and values."""
    result = calculate_order_statistics(operation="cancelled_and_pending")
    assert result["success"] is True
    data = result["data"]
    assert data["cancelled_count"] == 7
    assert data["cancelled_value_inr"] > 0
    assert data["returned_count"] == 3
    assert data["processing_count"] == 1
    assert data["shipped_count"] == 1


def test_calculate_revenue_total():
    """Verify delivered revenue calculation vs total gross order value."""
    result = calculate_order_statistics(operation="revenue_total")
    assert result["success"] is True
    data = result["data"]
    assert data["delivered_order_count"] == 48
    assert data["delivered_revenue_inr"] == 371040.0
    assert data["net_revenue_excluding_cancelled_inr"] == 397678.0
    assert data["total_gross_value_including_cancelled_inr"] == 470312.0
    assert data["cancelled_order_value_inr"] == 72634.0


def test_calculate_monthly_revenue():
    """Verify monthly revenue calculations from June to September 2026."""
    result = calculate_order_statistics(operation="monthly_revenue")
    assert result["success"] is True
    breakdown = result["data"]["monthly_breakdown"]
    assert len(breakdown) == 4

    months = {item["month"]: item for item in breakdown}
    assert "2026-06" in months
    assert "2026-07" in months
    assert "2026-08" in months
    assert "2026-09" in months

    assert months["2026-06"]["order_count"] == 18
    assert months["2026-07"]["order_count"] == 14
    assert months["2026-08"]["order_count"] == 20
    assert months["2026-09"]["order_count"] == 8

    assert months["2026-08"]["delivered_revenue_inr"] == 139059.0


def test_calculate_revenue_by_category():
    """Verify revenue and quantity breakdowns by product category."""
    result = calculate_order_statistics(operation="revenue_by_category")
    assert result["success"] is True
    categories = result["data"]["categories"]
    assert len(categories) == 4

    cat_names = [c["category"] for c in categories]
    assert "Electronics" in cat_names
    assert "Furniture" in cat_names
    assert "Accessories" in cat_names
    assert "Stationery" in cat_names

    top_cat = result["data"]["top_category_by_delivered_revenue"]
    assert top_cat in ["Electronics", "Furniture"]


def test_calculate_top_customers():
    """Verify highest spending customer calculation."""
    result = calculate_order_statistics(operation="top_customers")
    assert result["success"] is True
    top_customer = result["data"]["highest_spending_customer"]
    assert top_customer is not None
    assert top_customer["customer_name"] == "Rohan Das"
    assert top_customer["delivered_spend_inr"] > 0
    assert top_customer["order_count"] >= 5


def test_calculate_revenue_by_city():
    """Verify city-level aggregated revenues."""
    result = calculate_order_statistics(operation="revenue_by_city")
    assert result["success"] is True
    cities = result["data"]["cities"]
    assert len(cities) == 6
    city_names = [c["city"] for c in cities]
    assert "Chennai" in city_names
    assert "Kochi" in city_names
    assert "Thiruvananthapuram" in city_names


def test_calculate_unsupported_operation():
    """Verify error returned on unknown operation."""
    result = calculate_order_statistics(operation="predict_future_stock")
    assert result["success"] is False
    assert "unsupported operation" in result["error"].lower()


def test_calculate_with_filters():
    """Verify statistical calculation restricted to a specific date window."""
    result = calculate_order_statistics(
        operation="revenue_total",
        start_date="2026-08-01",
        end_date="2026-08-31",
    )
    assert result["success"] is True
    assert result["total_orders_in_scope"] == 20
    assert result["data"]["delivered_revenue_inr"] == 139059.0


def test_calculate_city_customers():
    """Verify city_customers operation returns unique customers and stats."""
    result = calculate_order_statistics(operation="city_customers", city="Thiruvananthapuram")
    assert result["success"] is True
    cities = result["data"]["cities"]
    assert len(cities) == 1
    th = cities[0]
    assert th["city"] == "Thiruvananthapuram"
    assert th["unique_customers_count"] == 4
    assert sorted(th["customers"]) == ["Divya Joseph", "Priya Menon", "Rohan Das", "Suresh Babu"]
    assert th["total_orders"] == 16
