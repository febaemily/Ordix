"""
ORDIX — Orders Simplified
Tests for General-Purpose Natural-Language Data Analyst Engine.
Tests data understanding, query planning, plan execution, combinations of filters,
rankings, comparisons, distinct values, follow-ups, and Gemini-unavailable operation.
"""

import os
import tempfile
from unittest.mock import MagicMock
import pytest
from google.genai.errors import APIError

import schema_inspector
from query_plan import QueryPlan, FilterSpec, AggregationMetric, validate_plan
import plan_executor
import query_planner
import query_engine
from agent import run_agent_turn


# ==============================================================================
# 1. Data Understanding & Schema Introspection Tests
# ==============================================================================

def test_dynamic_schema_inspection():
    """Verify automatic introspection of columns, types, date bounds, and distinct values from orders.csv."""
    schema = schema_inspector.inspect_schema()
    
    assert "order_id" in schema["columns"]
    assert "total_inr" in schema["columns"]
    assert "city" in schema["columns"]
    assert "product" in schema["columns"]
    assert "category" in schema["columns"]
    assert "status" in schema["columns"]
    assert "payment_method" in schema["columns"]

    # Inferred types
    assert schema["column_types"]["total_inr"] == "numeric"
    assert schema["column_types"]["quantity"] == "numeric"
    assert schema["column_types"]["order_date"] == "datetime"
    assert schema["column_types"]["category"] == "categorical"

    # Distinct categories from actual CSV
    assert "Electronics" in schema["distinct_values"]["category"]
    assert "Furniture" in schema["distinct_values"]["category"]
    assert "Chennai" in schema["distinct_values"]["city"]
    assert "Thiruvananthapuram" in schema["distinct_values"]["city"]

    # Date range bounds
    assert schema["date_range"]["min"] == "2026-06-01"
    assert schema["date_range"]["max"] == "2026-09-28"


def test_schema_summary_for_llm_is_compact():
    """Verify that the LLM schema summary is concise and contains no raw data rows."""
    summary = schema_inspector.get_schema_summary_for_llm()
    assert "columns" in summary
    assert "numeric_metrics" in summary
    assert "categories" in summary
    assert "orders" not in summary  # no raw rows leaked


# ==============================================================================
# 2. Query Operations: Sums, Counts, Averages, Min/Max, Distinct
# ==============================================================================

def test_aggregation_metrics_sum_count_mean():
    """Verify calculations for sum, count, and average over filtered dataset."""
    plan = QueryPlan(
        intent="aggregate",
        filters=[FilterSpec(column="category", operator="==", value="Electronics")],
        metrics=[
            AggregationMetric(column="total_inr", agg="sum"),
            AggregationMetric(column="total_inr", agg="mean"),
        ],
    )
    res = plan_executor.execute_plan(plan)
    assert res["tool_calls"][0]["result"]["success"] is True
    assert res["tool_calls"][0]["result"]["total_orders"] > 0
    assert "Delivered Orders Revenue" in res["answer"]
    assert "Average Delivered Order Value" in res["answer"]


def test_distinct_values_products():
    """Verify listing distinct products dynamically."""
    plan = query_planner.parse_query_locally("What are the distinct products sold?")
    assert plan is not None
    assert plan.intent == "distinct_values"
    res = plan_executor.execute_plan(plan)
    assert "Wireless Mouse" in res["answer"]
    assert "Mechanical Keyboard" in res["answer"]
    assert "Standing Desk" in res["answer"]


def test_distinct_values_cities():
    """Verify listing distinct cities."""
    plan = query_planner.parse_query_locally("What cities do we have orders from?")
    assert plan is not None
    assert plan.intent == "distinct_values"
    res = plan_executor.execute_plan(plan)
    assert "Chennai" in res["answer"]
    assert "Thiruvananthapuram" in res["answer"]
    assert "Kochi" in res["answer"]


# ==============================================================================
# 3. Rankings, Top-K, Bottom-K, and Customer Spending
# ==============================================================================

def test_top_spending_customers_ranking():
    """Verify ranking top spending customers with explicit cancelled orders documentation."""
    plan = query_planner.parse_query_locally("Who are the top 5 spending customers?")
    assert plan is not None
    assert plan.intent == "ranking"
    assert plan.sort_ascending is False
    res = plan_executor.execute_plan(plan)
    assert "Rohan Das" in res["answer"]
    assert "Delivered revenue strictly reflects fulfilled orders" in res["answer"]
    assert len(res["tool_calls"][0]["result"]["data"]) == 5


def test_bottom_or_least_frequent_ranking():
    """Verify ranking entities in ascending order (bottom-K / least)."""
    plan = QueryPlan(
        intent="ranking",
        group_by=["customer_name"],
        sort_by="order_count",
        sort_ascending=True,
        limit=3,
    )
    res = plan_executor.execute_plan(plan)
    data = res["tool_calls"][0]["result"]["data"]
    assert len(data) == 3
    # First should have <= orders than last
    assert data[0]["order_count"] <= data[-1]["order_count"]


# ==============================================================================
# 4. Comparisons & Side-by-Side Variances
# ==============================================================================

def test_month_comparison_variance():
    """Verify comparing two months produces verified differences."""
    plan = query_planner.parse_query_locally("Compare August and September revenue")
    assert plan is not None
    assert plan.intent == "comparison"
    res = plan_executor.execute_plan(plan)
    assert "August" in res["answer"]
    assert "September" in res["answer"]
    assert "Key Variance" in res["answer"]
    assert "₹139,059.00" in res["answer"]
    assert "₹41,292.00" in res["answer"]


def test_payment_method_comparison():
    """Verify comparing payment methods."""
    plan = QueryPlan(
        intent="comparison",
        group_by=["payment_method"],
        filters=[FilterSpec(column="payment_method", operator="in", value=["UPI", "Cash on Delivery"])],
    )
    res = plan_executor.execute_plan(plan)
    assert "UPI" in res["answer"]
    assert "Cash on Delivery" in res["answer"]
    assert len(res["tool_calls"][0]["result"]["comparison"]) == 2


# ==============================================================================
# 5. Grouped Summaries with Grand Totals & Percentages
# ==============================================================================

def test_grouped_by_payment_method_with_grand_total_and_percentages():
    """Verify payment method breakdown includes grand total row and percentages."""
    plan = query_planner.parse_query_locally("Show breakdown by payment method")
    assert plan is not None
    assert plan.intent == "group_by"
    assert plan.include_grand_total is True
    res = plan_executor.execute_plan(plan)
    assert "Grand Total" in res["answer"]
    assert "100.0%" in res["answer"]
    assert "UPI" in res["answer"]
    assert "Credit Card" in res["answer"]
    assert "Cash on Delivery" in res["answer"]


def test_grouped_by_city_with_grand_total():
    """Verify regional city breakdown includes grand total row."""
    plan = query_planner.parse_query_locally("Breakdown of orders by city")
    assert plan is not None
    assert plan.intent == "group_by"
    res = plan_executor.execute_plan(plan)
    assert "Grand Total" in res["answer"]
    assert "60" in res["answer"]  # total orders


# ==============================================================================
# 6. Combined Multi-Condition Filtering
# ==============================================================================

def test_combined_filters_city_and_category_and_payment():
    """Verify combining multiple filters: category=Electronics, city=Chennai, payment=UPI."""
    query = "Show Electronics in Chennai paid with UPI"
    res = query_engine.process_natural_query(query)
    assert res is not None
    assert "Chennai" in res["answer"]
    assert "Electronics" in res["answer"]
    assert len(res["tool_calls"]) >= 1


def test_date_range_between_filter():
    """Verify date filtering using between operator."""
    query = "Show orders between 2026-06-01 and 2026-06-15"
    res = query_engine.process_natural_query(query)
    assert res is not None
    assert len(res["tool_calls"]) >= 1
    # Check that orders in June 1-15 are matched
    assert "ORD-" in res["answer"]


# ==============================================================================
# 7. Follow-Up Questions (Context Handling Without Filter Leakage)
# ==============================================================================

def test_contextual_follow_up_without_filter_leakage():
    """Verify follow-up query 'What about September?' resolves with September metrics without filter contamination."""
    history = [
        {"role": "user", "content": "What is the revenue for August?"},
        {"role": "model", "content": "August delivered revenue was ₹139,059.00."},
    ]
    plan = query_planner.parse_query_locally("What about September?", conversation_history=history)
    assert plan is not None
    assert plan.intent == "aggregate"
    # Ensure filter is specifically September
    assert plan.filters[0].value == "September"
    res = plan_executor.execute_plan(plan)
    assert "September" in res["answer"]
    assert "₹41,292.00" in res["answer"]
    # Ensure August revenue is NOT reported as September revenue
    assert "₹139,059.00" not in res["answer"]


def test_contextual_follow_up_city():
    """Verify follow-up query 'What about Kochi?'."""
    history = [
        {"role": "user", "content": "How many orders in Chennai?"},
        {"role": "model", "content": "There were 21 orders in Chennai."},
    ]
    plan = query_planner.parse_query_locally("What about Kochi?", conversation_history=history)
    assert plan is not None
    assert plan.filters[0].column == "city"
    assert plan.filters[0].value == "Kochi"
    res = plan_executor.execute_plan(plan)
    assert "Kochi" in res["answer"]


# ==============================================================================
# 8. Ambiguous Questions & Clarifications
# ==============================================================================

def test_ambiguous_question_returns_clarification():
    """Verify ambiguous or non-specific questions ask for clarification instead of guessing."""
    plan = query_planner.parse_query_locally("is it good?")
    assert plan is not None
    assert plan.intent == "clarification"
    res = plan_executor.execute_plan(plan)
    assert "Please specify what you would like to analyze" in res["answer"]
    assert len(res["tool_calls"]) == 0


def test_greeting_returns_friendly_clarification():
    """Verify greetings return a welcoming guidance message."""
    plan = query_planner.parse_query_locally("hello")
    assert plan is not None
    assert plan.intent == "clarification"
    res = plan_executor.execute_plan(plan)
    assert "Hello! I am ORDIX" in res["answer"]


# ==============================================================================
# 9. Local Query Engine when Gemini is Unavailable
# ==============================================================================

def test_local_engine_works_when_gemini_unavailable():
    """Verify queries execute 100% locally when Gemini API is unavailable (no key or mock error)."""
    res = query_engine.process_natural_query(
        message="What is the gross revenue across all months?",
        client=None,
        allow_gemini_planner=False,
    )
    assert res is not None
    assert "470,312.00" in res["answer"]  # Total gross order value
    assert len(res["tool_calls"]) >= 1


def test_quota_fallback_with_local_engine(monkeypatch):
    """Verify that when Gemini raises 429 quota exhaustion, local engine handles the query."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = APIError(429, {"error": {"message": "Quota exceeded"}})
    type(mock_client).__name__ = "Client"

    res = run_agent_turn(
        message="August revenue",
        client=mock_client,
        use_fast_path=False,  # force past fast path to trigger quota handler
        fallback_on_quota=True,
    )
    assert "August" in res["answer"]
    assert "139,059.00" in res["answer"]
    assert res["metadata"]["routing"] == "fallback"


# ==============================================================================
# 10. Controlled Synthetic Dataset Tests
# ==============================================================================

def test_controlled_synthetic_dataset_edge_cases():
    """Verify edge cases (ties, empty results, missing filters) on a controlled synthetic CSV."""
    sample_csv = (
        "order_id,order_date,customer_name,city,product,category,quantity,unit_price_inr,total_inr,payment_method,status\n"
        "ORD-7001,2026-07-01,User Alpha,Goa,Headphones,Electronics,2,1000,2000,UPI,delivered\n"
        "ORD-7002,2026-07-05,User Beta,Goa,Mouse,Electronics,1,1000,1000,UPI,delivered\n"
        "ORD-7003,2026-07-10,User Gamma,Delhi,Chair,Furniture,1,5000,5000,Cash on Delivery,cancelled\n"
        "ORD-7004,2026-07-15,User Alpha,Goa,Desk,Furniture,1,8000,8000,Credit Card,delivered\n"
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as tmp:
        tmp.write(sample_csv)
        tmp_path = tmp.name

    try:
        # 1. Total delivered revenue in Goa on synthetic data: 2000 + 1000 + 8000 = 11000
        plan_goa = QueryPlan(
            intent="aggregate",
            filters=[FilterSpec(column="city", operator="==", value="Goa")],
        )
        res_goa = plan_executor.execute_plan(plan_goa, csv_path=tmp_path)
        assert "₹11,000.00" in res_goa["answer"]
        assert "₹5,000.00" not in res_goa["answer"]  # Delhi cancelled order excluded

        # 2. Cancelled order in Delhi: 1 cancelled, ₹5000 loss
        plan_delhi = QueryPlan(
            intent="aggregate",
            filters=[FilterSpec(column="city", operator="==", value="Delhi")],
        )
        res_delhi = plan_executor.execute_plan(plan_delhi, csv_path=tmp_path)
        assert "₹5,000.00" in res_delhi["answer"]  # Cancelled loss
        assert "₹0.00" in res_delhi["answer"]      # 0 delivered revenue

        # 3. Nonexistent filter returns clean empty result
        plan_empty = QueryPlan(
            intent="search",
            filters=[FilterSpec(column="city", operator="==", value="Mumbai")],
        )
        res_empty = plan_executor.execute_plan(plan_empty, csv_path=tmp_path)
        assert "No orders matched" in res_empty["answer"]

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
