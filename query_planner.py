"""
ORDIX — Orders Simplified
Query Planner Module: Translates natural-language inquiries into validated QueryPlans.
Combines high-speed deterministic local parsing with LLM structured planning.
Zero code execution by LLM; strictly validated schema operations.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError

from query_plan import QueryPlan, FilterSpec, AggregationMetric, validate_plan
from schema_inspector import inspect_schema, get_schema_summary_for_llm, DEFAULT_CSV_PATH

logger = logging.getLogger("ordix.planner")

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_NUMBER_TO_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

ONLINE_PAYMENTS = ["UPI", "Credit Card", "Debit Card", "Net Banking"]


def extract_dates_and_months(text: str) -> Dict[str, Any]:
    """Extracts explicit date ranges, single dates, months, and years from text."""
    lower = text.lower()
    res: Dict[str, Any] = {
        "months": [],
        "year": None,
        "date_between": None,
        "date_single": None,
    }

    # Month names
    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", lower):
            full_name = MONTH_NUMBER_TO_NAME[num]
            if (num, full_name) not in res["months"]:
                res["months"].append((num, full_name))

    # YYYY-MM
    ym_matches = re.findall(r"\b2026[-/](0?[1-9]|1[0-2])\b", text)
    for m in ym_matches:
        num = int(m)
        full_name = MONTH_NUMBER_TO_NAME[num]
        if (num, full_name) not in res["months"]:
            res["months"].append((num, full_name))

    # Year
    yr_match = re.search(r"\b(202[4-9])\b", text)
    if yr_match:
        res["year"] = int(yr_match.group(1))

    # Date range: between YYYY-MM-DD and YYYY-MM-DD
    range_match = re.search(r"(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if range_match:
        res["date_between"] = (range_match.group(1), range_match.group(2))

    # Single date: YYYY-MM-DD
    if not res["date_between"]:
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if date_match:
            res["date_single"] = date_match.group(1)

    return res


def extract_entities_from_schema(text: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extracts values belonging to known categorical/text columns based on live schema."""
    lower = text.lower()
    matched_entities: Dict[str, List[str]] = {}

    for col, lookup in schema.get("value_lookup", {}).items():
        matched_for_col = []
        for val_lower, canonical_val in lookup.items():
            pattern = rf"\b{re.escape(val_lower)}\b"
            if re.search(pattern, lower):
                if canonical_val not in matched_for_col:
                    matched_for_col.append(canonical_val)
        if matched_for_col:
            matched_entities[col] = matched_for_col

    return matched_entities


def resolve_follow_up_context(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]],
    schema: Dict[str, Any],
) -> Optional[QueryPlan]:
    """
    Interprets follow-up questions (e.g. 'What about September?', 'What about Kochi?', 'Which was highest?')
    by leveraging prior context without contaminating unrelated filters.
    """
    if not conversation_history:
        return None

    clean_msg = message.strip()
    lower_msg = clean_msg.lower()

    # Get last user turn
    last_user_turns = [t.get("content", "").strip() for t in conversation_history if t.get("role") == "user"]
    if not last_user_turns:
        return None
    prev_user_query = last_user_turns[-1].lower()

    # Pattern A: "What about [Month]?" or "How about [Month]?"
    elliptical_month = re.search(r"^(?:what\s+about|how\s+about|and|what\s+of)\s+([a-z]+)\??$", lower_msg)
    if elliptical_month:
        target_name = elliptical_month.group(1)
        if target_name in MONTH_MAP:
            m_num = MONTH_MAP[target_name]
            m_name = MONTH_NUMBER_TO_NAME[m_num]
            # Inherit previous intent (e.g. revenue, orders)
            plan = QueryPlan(
                intent="aggregate",
                filters=[FilterSpec(column="month_name", operator="==", value=m_name)],
                metrics=[AggregationMetric(column="total_inr", agg="sum", alias="delivered_revenue")],
                title=f"Revenue & Orders Summary: {m_name} 2026",
            )
            return plan

    # Pattern B: "What about [City]?" or "In [City]?"
    elliptical_city = re.search(r"^(?:what\s+about|how\s+about|and|in)\s+([a-z\s]+)\??$", lower_msg)
    if elliptical_city:
        cand_city = elliptical_city.group(1).strip()
        if "city" in schema.get("value_lookup", {}):
            lookup = schema["value_lookup"]["city"]
            if cand_city in lookup:
                canonical_city = lookup[cand_city]
                plan = QueryPlan(
                    intent="aggregate",
                    filters=[FilterSpec(column="city", operator="==", value=canonical_city)],
                    metrics=[AggregationMetric(column="total_inr", agg="sum", alias="delivered_revenue")],
                    title=f"Orders & Revenue in {canonical_city}",
                )
                return plan

    # Pattern C: "Which month had the highest revenue?" or "Which was highest?"
    if any(p in lower_msg for p in ["which month had the highest", "which month was highest", "which was highest", "peak month"]):
        if "month" in prev_user_query or "revenue" in prev_user_query:
            return QueryPlan(
                intent="ranking",
                group_by=["month_name"],
                sort_by="delivered_revenue",
                sort_ascending=False,
                limit=1,
                title="Highest Revenue Month",
            )

    return None


def parse_query_locally(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    csv_path: Optional[str] = None,
) -> Optional[QueryPlan]:
    """
    Deterministic rule/NLP parser that translates natural-language queries into
    strictly validated QueryPlan specifications without calling Gemini.
    """
    schema = inspect_schema(csv_path)
    clean_msg = message.strip()
    lower_msg = clean_msg.lower()

    # 1. Ambiguity Detection (short non-specific questions)
    ambiguous_prompts = [
        "is it good", "how is it", "what do you think", "tell me something",
        "help me", "analyze", "explain data", "what's up", "hello", "hi", "hey"
    ]
    if lower_msg.rstrip("?!. ") in ambiguous_prompts or len(clean_msg.split()) == 1 and lower_msg not in ["summary", "overview", "revenue", "orders"]:
        if lower_msg in ["hi", "hello", "hey"]:
            prompt = (
                "Hello! I am ORDIX, your dataset analyst. You can ask me any question about your orders, "
                "such as monthly revenue, customer spending rankings, city performance, payment methods, "
                "or filtered order lists."
            )
        else:
            prompt = (
                "Your inquiry is quite general. Please specify what you would like to analyze — for example: "
                "`monthly revenue`, `who paid online`, `top spending customers`, `orders by city`, or "
                "`cancelled orders`."
            )
        return QueryPlan(intent="clarification", clarification_prompt=prompt)

    # 2. Contextual follow-up resolution
    follow_up_plan = resolve_follow_up_context(message, conversation_history, schema)
    if follow_up_plan:
        return validate_plan(follow_up_plan, schema)

    date_info = extract_dates_and_months(clean_msg)
    entities = extract_entities_from_schema(clean_msg, schema)

    # Order ID specific lookup
    id_match = re.search(r"\bORD[-_\s]?(\d{4})\b", clean_msg, re.IGNORECASE)
    if not id_match:
        id_match = re.search(r"\border\s*(?:#|id\s*)?(\d{4})\b", clean_msg, re.IGNORECASE)
    if id_match:
        full_id = f"ORD-{id_match.group(1)}"
        plan = QueryPlan(
            intent="search",
            filters=[FilterSpec(column="order_id", operator="==", value=full_id)],
            limit=1,
            title=f"Order Details: {full_id}",
        )
        return validate_plan(plan, schema)

    # 3. Peak Month / Highest Month Query
    if any(p in lower_msg for p in ["which month had the highest", "which month was the highest", "highest revenue month", "peak month", "best month"]):
        plan = QueryPlan(
            intent="ranking",
            group_by=["month_name"],
            sort_by="delivered_revenue",
            sort_ascending=False,
            limit=1,
            title="Highest Revenue Month",
        )
        return validate_plan(plan, schema)

    # 4. Month Comparisons (e.g., "Compare August and September" or "August vs September")
    if len(date_info["months"]) >= 2 and any(k in lower_msg for k in ["vs", "compare", "difference", "comparison"]):
        month_names = [m[1] for m in date_info["months"]]
        plan = QueryPlan(
            intent="comparison",
            group_by=["month_name"],
            filters=[FilterSpec(column="month_name", operator="in", value=month_names)],
            title=f"Comparison: {' vs '.join(month_names)}",
        )
        return validate_plan(plan, schema)

    # 5. Distinct Values (e.g. "what products are sold", "which cities", "distinct categories")
    if any(k in lower_msg for k in ["what products", "which products", "list of products", "distinct products", "all products"]) and not any(k in lower_msg for k in ["revenue", "sales", "highest", "top"]):
        return validate_plan(QueryPlan(intent="distinct_values", group_by=["product"], title="Distinct Products in Dataset"), schema)
    if any(k in lower_msg for k in ["what cities", "which cities", "list of cities", "distinct cities", "all cities"]) and not any(k in lower_msg for k in ["revenue", "sales", "highest", "top"]):
        return validate_plan(QueryPlan(intent="distinct_values", group_by=["city"], title="Distinct Cities in Dataset"), schema)
    if any(k in lower_msg for k in ["what categories", "which categories", "distinct categories"]):
        return validate_plan(QueryPlan(intent="distinct_values", group_by=["category"], title="Distinct Product Categories"), schema)
    if any(k in lower_msg for k in ["what payment methods", "which payment methods", "distinct payment"]):
        return validate_plan(QueryPlan(intent="distinct_values", group_by=["payment_method"], title="Payment Methods Accepted"), schema)

    # 6. Customer Specific Queries
    is_cust_inquiry = any(kw in lower_msg for kw in ["customer", "customers", "client", "who bought", "who ordered", "who made orders"])
    
    # "How many customers from [City]?" or "customers in [City]"
    if "city" in entities and is_cust_inquiry:
        target_cities = entities["city"]
        plan = QueryPlan(
            intent="ranking",
            group_by=["customer_name"],
            filters=[FilterSpec(column="city", operator="in", value=target_cities)],
            sort_by="delivered_revenue",
            sort_ascending=False,
            limit=20,
            title=f"Customers in {', '.join(target_cities)}",
        )
        return validate_plan(plan, schema)

    # "Top spending customers" or "Who spent the most"
    if (re.search(r"\b(?:top|highest|best)\s+(?:\d+\s+)?(?:spending\s+)?customers?\b", lower_msg) or any(k in lower_msg for k in ["top customer", "top customers", "highest spending", "spent the most", "best customer", "highest spender", "who spent the most"])) and not any(k in lower_msg for k in ["least", "lowest"]):
        num_match = re.search(r"\b(?:top|best)\s+(\d+)\b", lower_msg)
        k_limit = int(num_match.group(1)) if num_match else 5
        plan = QueryPlan(
            intent="ranking",
            group_by=["customer_name"],
            sort_by="delivered_revenue",
            sort_ascending=False,
            limit=k_limit,
            title=f"Top {k_limit} Spending Customers (By Delivered Spend)",
        )
        return validate_plan(plan, schema)

    # "Lowest/least spending customers" or "Who spent the least" / "Which customer has spent the least?"
    if re.search(r"\b(?:least|lowest|minimum|bottom)\s+(?:\d+\s+)?(?:spending\s+)?customers?\b", lower_msg) or any(
        k in lower_msg for k in [
            "spent the least",
            "spent least",
            "least spending",
            "lowest spending",
            "lowest spender",
            "least spender",
            "which customer has spent the least",
            "which customer spent the least",
            "who spent the least",
            "who has spent the least",
        ]
    ):
        num_match = re.search(r"\b(?:least|lowest|bottom)\s+(\d+)\b", lower_msg)
        k_limit = int(num_match.group(1)) if num_match else 5
        if any(k in lower_msg for k in ["which customer", "who spent", "lowest customer", "least customer"]):
            k_limit = 1
        plan = QueryPlan(
            intent="ranking",
            group_by=["customer_name"],
            sort_by="delivered_revenue",
            sort_ascending=True,
            limit=k_limit,
            title="Lowest Spending Customers (By Delivered Spend)",
        )
        return validate_plan(plan, schema)

    # "Most frequent customers" or "Who ordered the most"
    if any(k in lower_msg for k in ["frequent customer", "frequent customers", "ordered the most", "most orders", "frequency"]):
        plan = QueryPlan(
            intent="ranking",
            group_by=["customer_name"],
            sort_by="order_count",
            sort_ascending=False,
            limit=5,
            title="Most Frequent Customers (By Order Volume)",
        )
        return validate_plan(plan, schema)

    # 7. Online Payments
    if any(k in lower_msg for k in ["who paid online", "paid online", "online payment", "online payments", "digital payment"]):
        plan = QueryPlan(
            intent="search",
            filters=[FilterSpec(column="payment_method", operator="in", value=ONLINE_PAYMENTS)],
            limit=12,
            title="Orders Paid via Online Payment Methods",
        )
        return validate_plan(plan, schema)

    # 8. Cancelled & Returned Orders
    if "cancel" in lower_msg and any(k in lower_msg for k in ["order", "orders", "revenue", "loss", "how many", "count", "value"]):
        plan = QueryPlan(
            intent="search",
            filters=[FilterSpec(column="status", operator="==", value="cancelled")],
            limit=10,
            title="Cancelled Orders & Lost Value",
        )
        return validate_plan(plan, schema)

    if "return" in lower_msg and any(k in lower_msg for k in ["order", "orders", "revenue", "how many", "count"]):
        plan = QueryPlan(
            intent="search",
            filters=[FilterSpec(column="status", operator="==", value="returned")],
            limit=10,
            title="Returned Orders Breakdown",
        )
        return validate_plan(plan, schema)

    # 9. Single Specific Month Revenue
    if len(date_info["months"]) == 1 and not date_info["date_between"] and not date_info["date_single"] and not any(k in lower_msg for k in ["vs", "compare"]):
        m_num, m_name = date_info["months"][0]
        # Month revenue / orders
        plan = QueryPlan(
            intent="aggregate",
            filters=[FilterSpec(column="month_name", operator="==", value=m_name)],
            metrics=[AggregationMetric(column="total_inr", agg="sum", alias="delivered_revenue")],
            title=f"Revenue & Orders: {m_name} 2026",
        )
        return validate_plan(plan, schema)

    # 10. Grouped Summaries / Breakdowns across Categories, Cities, Payment Methods, Months
    if any(k in lower_msg for k in ["monthly revenue", "revenue by month", "all months", "every month", "each month", "monthly breakdown", "orders by month"]):
        plan = QueryPlan(
            intent="group_by",
            group_by=["month_name"],
            calculate_percentage=False,
            include_grand_total=True,
            title="Monthly Revenue & Orders Breakdown",
        )
        return validate_plan(plan, schema)

    if any(k in lower_msg for k in ["by payment", "payment method", "payment methods", "payment breakdown", "payment distribution"]):
        plan = QueryPlan(
            intent="group_by",
            group_by=["payment_method"],
            calculate_percentage=True,
            include_grand_total=True,
            title="Orders & Revenue by Payment Method",
        )
        return validate_plan(plan, schema)

    if any(k in lower_msg for k in ["by category", "category breakdown", "revenue by category", "category performance"]):
        plan = QueryPlan(
            intent="group_by",
            group_by=["category"],
            calculate_percentage=False,
            include_grand_total=True,
            title="Product Category Performance Breakdown",
        )
        return validate_plan(plan, schema)

    if any(k in lower_msg for k in ["by city", "city breakdown", "revenue by city", "city performance", "regional breakdown"]):
        plan = QueryPlan(
            intent="group_by",
            group_by=["city"],
            calculate_percentage=True,
            include_grand_total=True,
            title="Regional City Performance Breakdown",
        )
        return validate_plan(plan, schema)

    if any(k in lower_msg for k in ["by status", "status breakdown", "counts by status", "distribution of status"]):
        plan = QueryPlan(
            intent="group_by",
            group_by=["status"],
            calculate_percentage=True,
            include_grand_total=True,
            title="Order Status Distribution",
        )
        return validate_plan(plan, schema)

    # 11. Multi-Condition Filtering (Combinations of City, Category, Product, Customer, Payment, Status, Dates)
    filters: List[FilterSpec] = []
    for col, vals in entities.items():
        if len(vals) == 1:
            filters.append(FilterSpec(column=col, operator="==", value=vals[0]))
        else:
            filters.append(FilterSpec(column=col, operator="in", value=vals))

    if date_info["date_between"]:
        filters.append(FilterSpec(column="order_date", operator="between", value=list(date_info["date_between"])))
    elif date_info["date_single"]:
        filters.append(FilterSpec(column="order_date", operator="==", value=date_info["date_single"]))

    if filters:
        is_list_query = any(k in lower_msg for k in ["show", "list", "which", "find", "see", "records", "orders between"])
        plan_intent = "search" if is_list_query else "aggregate"
        plan = QueryPlan(
            intent=plan_intent,
            filters=filters,
            title="Filtered Orders Query",
        )
        return validate_plan(plan, schema)

    # 12. General Dataset Summary / Total Revenue
    if any(k in lower_msg for k in ["total revenue", "overall revenue", "gross revenue", "total sales", "dataset summary", "summary", "overview"]):
        plan = QueryPlan(
            intent="aggregate",
            title="Overall Dataset Revenue & Orders Summary",
        )
        return validate_plan(plan, schema)

    return None


def parse_query_with_gemini(
    message: str,
    client: genai.Client,
    model_name: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    csv_path: Optional[str] = None,
) -> Optional[QueryPlan]:
    """
    Leverages Gemini strictly as a structured query planner for complex, unfamiliar, or colloquial queries.
    Never passes raw CSV records or allows code generation. Returns a validated QueryPlan.
    """
    schema = inspect_schema(csv_path)
    schema_summary = get_schema_summary_for_llm(csv_path)

    system_prompt = (
        "You are ORDIX's Structured Query Planner. "
        "Your task is to convert a user's natural language question about an orders dataset into a JSON QueryPlan. "
        "NEVER calculate numbers or execute code. ONLY return valid JSON adhering to the specified schema.\n\n"
        f"DATASET SCHEMA:\n{json.dumps(schema_summary, indent=2)}\n\n"
        "ALLOWED INTENTS: 'aggregate', 'group_by', 'search', 'ranking', 'comparison', 'distinct_values', 'clarification'\n"
        "ALLOWED OPERATORS: '==', '!=', 'in', 'not in', '>', '>=', '<', '<=', 'between', 'contains'\n"
        "ALLOWED AGGREGATIONS: 'sum', 'count', 'mean', 'min', 'max', 'nunique'\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        '  "intent": "group_by | aggregate | search | ranking | comparison | distinct_values | clarification",\n'
        '  "filters": [{"column": "city", "operator": "==", "value": "Chennai"}],\n'
        '  "group_by": ["category"],\n'
        '  "metrics": [{"column": "total_inr", "agg": "sum", "alias": "revenue"}],\n'
        '  "sort_by": "total_inr",\n'
        '  "sort_ascending": false,\n'
        '  "limit": 10,\n'
        '  "include_grand_total": true,\n'
        '  "calculate_percentage": false,\n'
        '  "revenue_definition": "delivered",\n'
        '  "clarification_prompt": null,\n'
        '  "title": "Optional Display Title"\n'
        "}"
    )

    history_context = ""
    if conversation_history:
        recent = conversation_history[-2:]
        history_context = "\nRecent Conversation:\n" + "\n".join([f"{t.get('role')}: {t.get('content')[:200]}" for t in recent])

    prompt = f"User Question: {message}{history_context}\n\nReturn ONLY the JSON QueryPlan object."

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        resp_text = response.text or "{}"
        plan_dict = json.loads(resp_text)
        raw_plan = QueryPlan(**plan_dict)
        return validate_plan(raw_plan, schema)

    except Exception as e:
        logger.warning("Gemini query planner encountered error: %s. Falling back to local planner.", str(e))
        return None
