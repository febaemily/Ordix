"""
Tests for Local Pandas Data-Query Engine (query_engine.py)
Validates verified arithmetic calculations, multi-criteria filtering,
distinct monthly revenues, online payments, follow-ups, and fallback handling.
"""

import os
import tempfile
import pytest
import pandas as pd

import query_engine
from agent import run_agent_turn


# ==============================================================================
# Tests with Real orders.csv
# ==============================================================================

def test_august_and_september_revenue_are_distinct_and_accurate():
    """Verify August and September return distinct, mathematically accurate numbers."""
    aug_res = query_engine.process_natural_query("August revenue")
    sep_res = query_engine.process_natural_query("September revenue")

    assert aug_res is not None
    assert sep_res is not None

    # August checks: 18 delivered, ₹139,059.00 delivered revenue, 2 cancelled (₹12,242.00)
    assert "₹139,059.00" in aug_res["answer"]
    assert "18" in aug_res["answer"]
    assert "₹151,301.00" in aug_res["answer"]  # Gross
    assert "₹12,242.00" in aug_res["answer"]   # Cancelled loss

    # September checks: 5 delivered, ₹41,292.00 delivered revenue, 1 cancelled (₹3,798.00)
    assert "₹41,292.00" in sep_res["answer"]
    assert "5" in sep_res["answer"]
    assert "₹47,436.00" in sep_res["answer"]   # Gross
    assert "₹3,798.00" in sep_res["answer"]    # Cancelled loss

    # Crucial assertion: August and September answers must NEVER be identical
    assert aug_res["answer"] != sep_res["answer"]
    assert "₹139,059.00" not in sep_res["answer"]
    assert "₹41,292.00" not in aug_res["answer"]


def test_monthly_breakdown_includes_overall_total():
    """Verify grouped monthly summary includes the overall total row."""
    res = query_engine.process_natural_query("What was the total revenue for each month from June to September 2026?")
    assert res is not None
    assert "Monthly Revenue Breakdown" in res["answer"]
    assert "June 2026" in res["answer"]
    assert "July 2026" in res["answer"]
    assert "August 2026" in res["answer"]
    assert "September 2026" in res["answer"]

    # Verify overall total is present in table
    assert "Overall Total" in res["answer"]
    assert "₹371,040.00" in res["answer"]  # Overall delivered revenue
    assert "₹470,312.00" in res["answer"]  # Overall gross revenue
    assert "60" in res["answer"]           # Overall orders count


def test_who_paid_online_returns_actual_matching_orders():
    """Verify online payment inquiry returns matching customers, breakdown, and rows."""
    res = query_engine.process_natural_query("Who paid online?")
    assert res is not None
    assert "Customers Who Paid Online" in res["answer"]
    assert "50 Orders" in res["answer"]
    assert "UPI" in res["answer"]
    assert "Credit Card" in res["answer"]
    assert "Debit Card" in res["answer"]
    assert "Net Banking" in res["answer"]

    # Must contain actual matching rows table
    assert "| Order ID | Customer | Payment Method | Product | Amount | Date | Status |" in res["answer"]
    assert "ORD-1001" in res["answer"]
    assert "Rahul Sharma" in res["answer"]


def test_payment_methods_breakdown():
    """Verify payment methods breakdown includes overall total and shares."""
    res = query_engine.process_natural_query("Show payment method breakdown")
    assert res is not None
    assert "Payment Methods Breakdown" in res["answer"]
    assert "UPI" in res["answer"]
    assert "Cash on Delivery" in res["answer"]
    assert "Overall Total" in res["answer"]
    assert "100.0%" in res["answer"]


def test_which_month_had_highest_revenue():
    """Verify peak month calculation accurately identifies July for revenue and August for volume."""
    res = query_engine.process_natural_query("Which month had the highest revenue?")
    assert res is not None
    assert "Peak Performance Months" in res["answer"]
    assert "July 2026" in res["answer"]      # Highest delivered revenue: ₹155,771.00
    assert "₹155,771.00" in res["answer"]
    assert "August 2026" in res["answer"]    # Highest order volume: 20 orders


def test_month_comparison():
    """Verify comparative analysis between August and September."""
    res = query_engine.process_natural_query("Compare August and September")
    assert res is not None
    assert "Revenue Comparison: August vs September 2026" in res["answer"]
    assert "Delivered Revenue" in res["answer"]
    assert "₹139,059.00" in res["answer"]
    assert "₹41,292.00" in res["answer"]
    assert "Difference" in res["answer"]


def test_most_frequent_customers():
    """Verify customer frequency ranking."""
    res = query_engine.process_natural_query("Who are the most frequent customers?")
    assert res is not None
    assert "Most Frequent Customers" in res["answer"]
    assert "Vikram Reddy" in res["answer"]  # 9 orders
    assert "Rohan Das" in res["answer"]     # 7 orders


def test_cancelled_orders_details():
    """Verify cancelled orders count, loss, and table."""
    res = query_engine.process_natural_query("How many orders were cancelled and what was the lost revenue?")
    assert res is not None
    assert "Cancelled Orders Overview" in res["answer"]
    assert "7" in res["answer"]
    assert "₹72,634.00" in res["answer"]
    assert "ORD-1041" in res["answer"]
    assert "ORD-1045" in res["answer"]


def test_contextual_follow_up():
    """Verify conversational follow-up 'What about September?' inherits revenue operation."""
    history = [
        {"role": "user", "content": "What was the revenue in August?"},
        {"role": "model", "content": "August 2026 delivered revenue was ₹139,059.00."},
    ]
    res = query_engine.process_natural_query("What about September?", conversation_history=history)
    assert res is not None
    assert "September 2026" in res["answer"]
    assert "₹41,292.00" in res["answer"]
    assert res["metadata"]["intent"] == "follow_up_month_revenue"


def test_combined_filters():
    """Verify multi-criteria entity filtering (customer + status)."""
    res = query_engine.process_natural_query("Show orders for Sneha Pillai")
    assert res is not None
    assert "Sneha Pillai" in res["answer"]
    assert "Filtered Orders" in res["answer"]
    assert "4" in res["answer"]  # Sneha Pillai placed 4 orders


def test_run_agent_turn_local_resolution_zero_gemini_calls(monkeypatch):
    """Verify run_agent_turn answers analytical questions locally without calling Gemini API."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key_only")

    res = run_agent_turn("August revenue")
    assert res["metadata"]["routing"] == "fast_path"
    assert "₹139,059.00" in res["answer"]
    assert len(res["tool_calls"]) == 1

    res_sep = run_agent_turn("September revenue")
    assert res_sep["metadata"]["routing"] == "fast_path"
    assert "₹41,292.00" in res_sep["answer"]
    assert len(res_sep["tool_calls"]) == 1


# ==============================================================================
# Tests with Controlled Test Dataset
# ==============================================================================

def test_controlled_dataset():
    """Verify calculations on a small controlled synthetic CSV dataset."""
    sample_csv_content = (
        "order_id,order_date,customer_name,city,product,category,quantity,unit_price_inr,total_inr,payment_method,status\n"
        "ORD-9001,2026-08-01,Alice Smith,Chennai,Mouse,Electronics,2,500,1000,UPI,delivered\n"
        "ORD-9002,2026-08-15,Bob Jones,Pune,Desk,Furniture,1,5000,5000,Cash on Delivery,cancelled\n"
        "ORD-9003,2026-09-01,Alice Smith,Chennai,Keyboard,Electronics,1,2000,2000,Credit Card,delivered\n"
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as tmp:
        tmp.write(sample_csv_content)
        tmp_path = tmp.name

    try:
        # August query on synthetic data
        res_aug = query_engine.execute_single_month_revenue(8, "August", query_engine.get_dataframe(tmp_path))
        assert "₹1,000.00" in res_aug["answer"]  # Delivered revenue
        assert "₹6,000.00" in res_aug["answer"]  # Gross revenue
        assert "₹5,000.00" in res_aug["answer"]  # Cancelled loss

        # September query on synthetic data
        res_sep = query_engine.execute_single_month_revenue(9, "September", query_engine.get_dataframe(tmp_path))
        assert "₹2,000.00" in res_sep["answer"]  # Delivered revenue
        assert "₹5,000.00" not in res_sep["answer"]

        # Online payments on synthetic data
        res_online = query_engine.execute_who_paid_online(query_engine.get_dataframe(tmp_path))
        assert "2 Orders" in res_online["answer"]  # 2 online orders (ORD-9001, ORD-9003)
        assert "Alice Smith" in res_online["answer"]
        assert "Bob Jones" not in res_online["answer"]  # Bob paid COD
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_city_customers_thiruvananthapuram():
    """Verify 'How many customers from Thiruvananthapuram?' returns verified 4 unique customers and stats."""
    res = query_engine.process_natural_query("How many customers from Thiruvananthapuram?")
    assert res is not None
    assert res["metadata"]["intent"] == "city_customers"
    assert "4" in res["answer"]
    assert "Rohan Das" in res["answer"]
    assert "Priya Menon" in res["answer"]
    assert "Divya Joseph" in res["answer"]
    assert "Suresh Babu" in res["answer"]
    assert "16 order(s)" in res["answer"]
    assert "175,065.00" in res["answer"]

    # Tool call verification
    assert len(res["tool_calls"]) == 1
    tool_res = res["tool_calls"][0]["result"]
    assert tool_res["unique_customers_count"] == 4
    assert sorted(tool_res["customers"]) == ["Divya Joseph", "Priya Menon", "Rohan Das", "Suresh Babu"]
    assert tool_res["total_orders"] == 16


def test_total_customers_inquiry():
    """Verify 'How many customers in total?' returns verified 15 unique customers."""
    res = query_engine.process_natural_query("How many customers in total?")
    assert res is not None
    assert res["metadata"]["intent"] == "total_customers"
    assert "15" in res["answer"]
    assert "60 orders" in res["answer"]
