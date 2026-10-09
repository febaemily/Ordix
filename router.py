"""
ORDIX — Orders Simplified
Deterministic Fast-Path Query Router
Routes common order queries directly to Python backend tools (tools.py)
without consuming Gemini GenAI API quota or tokens.
"""

import re
import time
from typing import Any, Dict, List, Optional
import tools


def format_inr(amount: float) -> str:
    """Formats a numeric amount into an Indian Rupee currency string."""
    return f"₹{amount:,.2f}"


def route_order_id_lookup(order_id: str, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Handles single order ID lookup (e.g. ORD-1001)."""
    clean_id = order_id.upper().strip()
    result = tools.search_orders(order_id=clean_id, csv_path=csv_path)

    if result.get("success") and result.get("orders"):
        ord_data = result["orders"][0]
        st = ord_data.get("status", "unknown").capitalize()
        answer = (
            f"### Order Details: **{ord_data['order_id']}**\n\n"
            f"- **Status**: **{st}**\n"
            f"- **Customer**: {ord_data['customer_name']}\n"
            f"- **Order Date**: {ord_data['order_date']}\n"
            f"- **Product**: {ord_data['product']} ({ord_data['category']})\n"
            f"- **Quantity**: {ord_data['quantity']} unit(s)\n"
            f"- **Unit Price**: {format_inr(ord_data['unit_price_inr'])}\n"
            f"- **Total Amount**: **{format_inr(ord_data['total_inr'])}**\n"
            f"- **Payment Method**: {ord_data['payment_method']}\n"
            f"- **City**: {ord_data['city']}"
        )
    else:
        answer = (
            f"Order **{clean_id}** was not found in the dataset. "
            "Please verify the order ID (valid dataset order IDs range from **ORD-1001** to **ORD-1060**)."
        )

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "search_orders",
                "args": {"order_id": clean_id},
                "result": result,
            }
        ],
    }


def route_cancelled_and_pending(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Handles queries regarding cancelled, returned, and pending/processing orders and lost revenue."""
    result = tools.calculate_order_statistics("cancelled_and_pending", csv_path=csv_path)
    data = result.get("data", {})

    cancelled_count = data.get("cancelled_count", 0)
    cancelled_val = data.get("cancelled_value_inr", 0.0)
    returned_count = data.get("returned_count", 0)
    returned_val = data.get("returned_value_inr", 0.0)
    processing_count = data.get("processing_count", 0)
    processing_val = data.get("processing_value_inr", 0.0)
    shipped_count = data.get("shipped_count", 0)
    shipped_val = data.get("shipped_value_inr", 0.0)

    answer = (
        "### Cancelled & Non-Delivered Orders Summary\n\n"
        f"- **Cancelled Orders**: **{cancelled_count}** order(s) with **{format_inr(cancelled_val)}** in lost revenue.\n"
        f"- **Returned Orders**: **{returned_count}** order(s) totaling **{format_inr(returned_val)}**.\n"
        f"- **Processing Orders**: **{processing_count}** order(s) totaling **{format_inr(processing_val)}**.\n"
        f"- **Shipped Orders**: **{shipped_count}** order(s) totaling **{format_inr(shipped_val)}**.\n\n"
        f"In total, **{cancelled_count}** orders were cancelled, resulting in a direct gross revenue loss of **{format_inr(cancelled_val)}**."
    )

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "cancelled_and_pending"},
                "result": result,
            }
        ],
    }


def route_pending_orders_count(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculates exact pending order count directly from orders data."""
    cnt = tools.count_pending_orders(csv_path=csv_path)
    proc_res = tools.search_orders(status="processing", limit=50, csv_path=csv_path)
    ship_res = tools.search_orders(status="shipped", limit=50, csv_path=csv_path)
    p_cnt = len(proc_res.get("orders", []))
    s_cnt = len(ship_res.get("orders", []))
    active_cnt = p_cnt + s_cnt

    if cnt == 0 and active_cnt > 0:
        answer = (
            f"There are **0** orders with the literal status `pending` in the dataset.\n\n"
            f"However, there are **{active_cnt} active orders** currently pending completion/fulfillment:\n"
            f"- **Processing**: {p_cnt} order\n"
            f"- **Shipped**: {s_cnt} order\n\n"
            f"Total active orders pending delivery: **{active_cnt} orders**."
        )
    else:
        answer = f"There are **{cnt}** pending orders in the dataset."

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "count_pending_orders",
                "args": {},
                "result": {"pending_count": cnt, "active_pending_count": active_cnt, "success": True},
            }
        ],
    }


def route_pending_or_shipped_orders(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Lists orders that are currently processing or shipped."""
    proc_res = tools.search_orders(status="processing", limit=50, csv_path=csv_path)
    ship_res = tools.search_orders(status="shipped", limit=50, csv_path=csv_path)

    combined_orders = proc_res.get("orders", []) + ship_res.get("orders", [])

    lines = [
        f"### Pending & In-Transit Orders ({len(combined_orders)} total)\n",
        f"There are **{len(combined_orders)}** orders currently pending fulfillment (**{len(proc_res.get('orders', []))} processing** and **{len(ship_res.get('orders', []))} shipped**):\n",
    ]

    for ord_item in combined_orders:
        lines.append(
            f"- **{ord_item['order_id']}** ({ord_item['status'].upper()}): "
            f"{ord_item['product']} by **{ord_item['customer_name']}** in {ord_item['city']} "
            f"— {format_inr(ord_item['total_inr'])} (Placed on {ord_item['order_date']})"
        )

    answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "search_orders",
                "args": {"status": "processing,shipped"},
                "result": {
                    "success": True,
                    "total_matched": len(combined_orders),
                    "returned_count": len(combined_orders),
                    "orders": combined_orders,
                },
            }
        ],
    }


def route_monthly_revenue(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates monthly revenue breakdown table."""
    result = tools.calculate_order_statistics("monthly_revenue", csv_path=csv_path)
    data = result.get("data", {})
    breakdown = data.get("monthly_breakdown", [])

    month_names = {
        "2026-06": "June 2026",
        "2026-07": "July 2026",
        "2026-08": "August 2026",
        "2026-09": "September 2026",
    }

    table_rows = [
        "| Month | Delivered Revenue | Gross Revenue | Orders |",
        "| :--- | :--- | :--- | :--- |",
    ]

    total_deliv = 0.0
    total_gross = 0.0
    total_orders = 0

    for item in breakdown:
        m_name = month_names.get(item["month"], item["month"])
        deliv = item["delivered_revenue_inr"]
        gross = item["gross_revenue_inr"]
        cnt = item["order_count"]
        total_deliv += deliv
        total_gross += gross
        total_orders += cnt
        table_rows.append(
            f"| **{m_name}** | {format_inr(deliv)} | {format_inr(gross)} | {cnt} |"
        )

    table_rows.append(
        f"| **Total** | **{format_inr(total_deliv)}** | **{format_inr(total_gross)}** | **{total_orders}** |"
    )

    best_item = max(breakdown, key=lambda x: x["delivered_revenue_inr"]) if breakdown else None
    if best_item:
        best_name = month_names.get(best_item["month"], best_item["month"])
        note = f"\n\n*Note: {best_name} was the strongest month by volume and revenue with {best_item['order_count']} orders totaling {format_inr(best_item['delivered_revenue_inr'])} in delivered sales.*"
    else:
        note = ""

    answer = (
        "### Monthly Revenue Breakdown (June – September 2026)\n\n"
        + "\n".join(table_rows)
        + note
    )

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "monthly_revenue"},
                "result": result,
            }
        ],
    }


def route_revenue_total(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates total revenue summary."""
    result = tools.calculate_order_statistics("revenue_total", csv_path=csv_path)
    data = result.get("data", {})

    deliv_rev = data.get("delivered_revenue_inr", 0.0)
    net_rev = data.get("net_revenue_excluding_cancelled_inr", 0.0)
    gross_val = data.get("total_gross_value_including_cancelled_inr", 0.0)
    cancelled_val = data.get("cancelled_order_value_inr", 0.0)
    deliv_cnt = data.get("delivered_order_count", 0)

    answer = (
        "### Revenue Summary\n\n"
        f"- **Delivered Orders Revenue**: **{format_inr(deliv_rev)}** across {deliv_cnt} fulfilled orders.\n"
        f"- **Net Revenue (excluding cancelled)**: **{format_inr(net_rev)}**.\n"
        f"- **Total Gross Order Value (all orders)**: **{format_inr(gross_val)}**.\n"
        f"- **Cancelled Order Loss**: **{format_inr(cancelled_val)}** in forfeited order value.\n\n"
        "*Delivered revenue reflects finalized orders received by customers.*"
    )

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "revenue_total"},
                "result": result,
            }
        ],
    }


def route_top_customers(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates top spending customer report."""
    result = tools.calculate_order_statistics("top_customers", csv_path=csv_path)
    data = result.get("data", {})
    top_cust = data.get("highest_spending_customer", {})
    ranked = data.get("top_ranked_customers", [])

    lines = [
        "### Top Customer & Spending Analysis\n",
        f"The highest-spending customer is **{top_cust.get('customer_name', 'Unknown')}**:\n"
        f"- **Delivered Spend**: **{format_inr(top_cust.get('delivered_spend_inr', 0.0))}**\n"
        f"- **Total Orders Placed**: {top_cust.get('order_count', 0)}\n"
        f"- **City**: {', '.join(top_cust.get('cities', []))}\n",
        "#### Top Customers Leaderboard\n",
        "| Rank | Customer Name | Delivered Spend | Total Spend | Orders |",
        "| :---: | :--- | :--- | :--- | :---: |",
    ]

    for idx, c in enumerate(ranked[:5], start=1):
        lines.append(
            f"| {idx} | **{c['customer_name']}** | {format_inr(c['delivered_spend_inr'])} | "
            f"{format_inr(c['total_spend_inr'])} | {c['order_count']} |"
        )

    answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "top_customers"},
                "result": result,
            }
        ],
    }


def route_lowest_spending_customer(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates lowest-spending customer report with verified figures and tie handling."""
    result = tools.calculate_order_statistics("lowest_spending_customer", csv_path=csv_path)
    data = result.get("data", {})
    lowest_cust = data.get("lowest_spending_customer") or data.get("primary_customer")
    ranked = data.get("ranked_customers", [])

    lines = [
        "### Lowest-Spending Customer Ranking\n",
    ]

    if lowest_cust:
        lines.append(
            f"The customer with the lowest delivered spend is **{lowest_cust['customer_name']}** "
            f"with **{format_inr(lowest_cust['delivered_spend_inr'])}** in delivered spend across "
            f"{lowest_cust['order_count']} order(s) (Total Gross Spend: {format_inr(lowest_cust['total_spend_inr'])}).\n"
        )

    lines.extend([
        "| Rank | Customer Name | Delivered Spend | Total Spend | Orders |",
        "| :---: | :--- | :--- | :--- | :---: |",
    ])

    for idx, c in enumerate(ranked[:5], start=1):
        lines.append(
            f"| {idx} | **{c['customer_name']}** | {format_inr(c['delivered_spend_inr'])} | "
            f"{format_inr(c['total_spend_inr'])} | {c['order_count']} |"
        )

    lines.append(
        "\n*Note on Revenue Treatment: Rankings reflect delivered spend on fulfilled orders. "
        "Cancelled and refunded orders are excluded from delivered spend.*"
    )

    return {
        "answer": "\n".join(lines),
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "lowest_spending_customer"},
                "result": result,
            }
        ],
    }


def route_who_paid_online(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Returns actual orders and customers who paid using online payment methods."""
    result = tools.search_orders(payment_method="online", limit=12, csv_path=csv_path)
    matched = result.get("total_matched", 0)
    orders_list = result.get("orders", [])

    custs = sorted(list({o["customer_name"] for o in orders_list}))

    lines = [
        f"### Orders & Customers Paid Online ({matched} Total Orders Found)\n",
        f"A total of **{matched} orders** were paid using online payment methods (UPI, Credit Card, Debit Card, Net Banking).\n",
        f"#### Customers Who Paid Online ({len(custs)} unique customers in sample):",
        ", ".join([f"**{c}**" for c in custs]) + "\n",
        "#### Matching Online Orders:",
        "| Order ID | Customer | Payment Method | Product | Amount | Status | City |",
        "| :--- | :--- | :--- | :--- | :--- | :---: | :--- |",
    ]

    for o in orders_list:
        lines.append(
            f"| **{o['order_id']}** | {o['customer_name']} | {o['payment_method']} | "
            f"{o['product']} | {format_inr(o['total_inr'])} | {o['status'].capitalize()} | {o['city']} |"
        )

    return {
        "answer": "\n".join(lines),
        "tool_calls": [
            {
                "tool": "search_orders",
                "args": {"payment_method": "online", "limit": 12},
                "result": result,
            }
        ],
    }


def route_revenue_by_category(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates category performance report."""
    result = tools.calculate_order_statistics("revenue_by_category", csv_path=csv_path)
    data = result.get("data", {})
    top_cat = data.get("top_category_by_delivered_revenue", "Unknown")
    categories = data.get("categories", [])

    lines = [
        f"### Product Category Performance\n",
        f"The product category generating the highest delivered revenue is **{top_cat}**.\n",
        "| Category | Delivered Revenue | Gross Revenue | Units Sold | Orders |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ]

    for cat in categories:
        lines.append(
            f"| **{cat['category']}** | {format_inr(cat['delivered_revenue_inr'])} | "
            f"{format_inr(cat['gross_revenue_inr'])} | {cat['total_quantity_sold']} | {cat['order_count']} |"
        )

    answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "revenue_by_category"},
                "result": result,
            }
        ],
    }


def route_revenue_by_city(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates city / regional performance breakdown."""
    result = tools.calculate_order_statistics("revenue_by_city", csv_path=csv_path)
    data = result.get("data", {})
    cities = data.get("cities", [])

    lines = [
        "### Regional & City Performance\n",
        "| City | Delivered Revenue | Gross Revenue | Orders |",
        "| :--- | :--- | :--- | :---: |",
    ]

    for ct in cities:
        lines.append(
            f"| **{ct['city']}** | {format_inr(ct['delivered_revenue_inr'])} | "
            f"{format_inr(ct['gross_revenue_inr'])} | {ct['order_count']} |"
        )

    answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "revenue_by_city"},
                "result": result,
            }
        ],
    }


def route_order_summary(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates general dataset overview and status breakdown."""
    result = tools.calculate_order_statistics("summary", csv_path=csv_path)
    data = result.get("data", {})
    breakdown = data.get("status_breakdown", {})

    total_orders = data.get("total_orders", 0)
    deliv_rev = data.get("delivered_orders_revenue_inr", 0.0)
    aov = data.get("average_delivered_order_value_inr", 0.0)
    cancelled_loss = data.get("cancelled_value_lost_inr", 0.0)

    answer = (
        "### Orders Performance Summary\n\n"
        f"- **Total Dataset Orders**: **{total_orders}**\n"
        f"- **Delivered Orders**: **{breakdown.get('delivered', 0)}** ({format_inr(deliv_rev)})\n"
        f"- **Average Order Value (Delivered)**: **{format_inr(aov)}**\n"
        f"- **Cancelled Orders Lost Value**: **{format_inr(cancelled_loss)}**\n\n"
        "#### Orders by Status:\n"
        f"- Delivered: **{breakdown.get('delivered', 0)}**\n"
        f"- Cancelled: **{breakdown.get('cancelled', 0)}**\n"
        f"- Returned: **{breakdown.get('returned', 0)}**\n"
        f"- Processing: **{breakdown.get('processing', 0)}**\n"
        f"- Shipped: **{breakdown.get('shipped', 0)}**"
    )

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "summary"},
                "result": result,
            }
        ],
    }


def route_status_breakdown(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates order counts by status."""
    result = tools.calculate_order_statistics("order_counts_by_status", csv_path=csv_path)
    counts = result.get("data", {}).get("counts_by_status", {})
    total = result.get("data", {}).get("total_orders", 0)

    lines = [
        f"### Order Counts by Status ({total} Total Orders)\n",
    ]
    for st, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- **{st.capitalize()}**: **{count}** order(s)")

    answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "order_counts_by_status"},
                "result": result,
            }
        ],
    }


def route_customer_lookup(customer_name: str, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Looks up orders placed by a specific customer."""
    result = tools.search_orders(customer_name=customer_name, limit=20, csv_path=csv_path)
    orders = result.get("orders", [])

    if not orders:
        answer = f"No orders found for customer **{customer_name}**."
    else:
        total_spent = sum(o["total_inr"] for o in orders if o["status"] == "delivered")
        lines = [
            f"### Orders for **{customer_name}** ({len(orders)} order(s))\n",
            f"Total delivered spend: **{format_inr(total_spent)}**\n",
        ]
        for o in orders:
            lines.append(
                f"- **{o['order_id']}** ({o['status'].capitalize()}): {o['product']} — "
                f"{format_inr(o['total_inr'])} on {o['order_date']}"
            )
        answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "search_orders",
                "args": {"customer_name": customer_name},
                "result": result,
            }
        ],
    }


def route_product_lookup(product_name: str, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Looks up orders for a specific product."""
    result = tools.search_orders(product=product_name, limit=20, csv_path=csv_path)
    orders = result.get("orders", [])

    if not orders:
        answer = f"No orders found for product **{product_name}**."
    else:
        total_units = sum(o["quantity"] for o in orders)
        delivered_units = sum(o["quantity"] for o in orders if o["status"] == "delivered")
        lines = [
            f"### Orders for **{product_name}** ({len(orders)} order(s))\n",
            f"Total units ordered: **{total_units}** ({delivered_units} delivered)\n",
        ]
        for o in orders[:8]:
            lines.append(
                f"- **{o['order_id']}** by {o['customer_name']}: {o['quantity']} unit(s) — "
                f"{format_inr(o['total_inr'])} ({o['status'].capitalize()})"
            )
        if len(orders) > 8:
            lines.append(f"\n*...and {len(orders) - 8} additional orders.*")
        answer = "\n".join(lines)

    return {
        "answer": answer,
        "tool_calls": [
            {
                "tool": "search_orders",
                "args": {"product": product_name},
                "result": result,
            }
        ],
    }


# Known entities in orders.csv for accurate entity matching
KNOWN_CUSTOMERS = [
    "Rahul Sharma", "Aditya Kumar", "Sneha Pillai", "Fathima Rahman",
    "Vikram Reddy", "Ananya Iyer", "Arjun Nair", "Rohan Das",
    "Meera Thomas", "Lakshmi Varma", "Priya Menon", "Nikhil Gupta",
    "Divya Joseph", "Karthik Rao", "Suresh Babu",
]

KNOWN_PRODUCTS = [
    "Wireless Mouse", "Backpack", "Gel Pen Set", "Desk Lamp",
    "Laptop Sleeve", "Water Bottle", "Standing Desk", "Notebook Pack",
    "Mechanical Keyboard", "USB-C Hub", "Noise Cancelling Headphones",
    "27-inch Monitor", "Office Chair",
]


def try_fast_path_answer(message: str, csv_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Attempts to answer common order inquiries deterministically.
    Returns structured response dict if matched, or None if the query
    should be delegated to Gemini.
    """
    start_time = time.time()
    clean_msg = message.strip()
    lower_msg = clean_msg.lower()

    # 1. Specific Order ID Lookup (e.g. ORD-1001, ord 1005, order #1001)
    order_id_match = re.search(r"\bORD[-_\s]?(\d{4})\b", clean_msg, re.IGNORECASE)
    if not order_id_match:
        # Check for phrases like "order 1001" or "order #1001"
        order_id_match = re.search(r"\border\s*(?:#|id\s*)?(\d{4})\b", clean_msg, re.IGNORECASE)

    if order_id_match:
        ord_num = order_id_match.group(1)
        full_id = f"ORD-{ord_num}"
        res = route_order_id_lookup(full_id, csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "order_id_lookup",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 2. Cancelled orders & lost revenue
    if "cancel" in lower_msg and any(kw in lower_msg for kw in ["revenue", "lost", "loss", "how many", "count", "value"]):
        res = route_cancelled_and_pending(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "cancelled_and_pending",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 2b. Processing / shipped / pending completion inquiries
    is_proc_ship = bool(re.search(r"\b(?:process\w*|procssee\w*|shipp?\w*)\b", lower_msg))
    if is_proc_ship and any(
        kw in lower_msg for kw in ["show", "list", "all", "orders", "what are", "how many", "count", "number of", "pending"]
    ):
        res = route_pending_or_shipped_orders(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "pending_or_shipped_orders",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 2c. Pure pending orders count inquiry
    if "pending" in lower_msg and any(kw in lower_msg for kw in ["how many", "count", "number of", "are pending", "total pending", "orders are pending"]):
        res = route_pending_orders_count(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "count_pending_orders",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 4. Monthly revenue & monthly breakdown
    if ("month" in lower_msg or "monthly" in lower_msg or "june to september" in lower_msg) and any(
        kw in lower_msg for kw in ["revenue", "sales", "breakdown", "total", "orders", "trend"]
    ):
        res = route_monthly_revenue(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "monthly_revenue",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 5. Total revenue / gross revenue
    if any(
        phrase in lower_msg
        for phrase in [
            "total revenue",
            "overall revenue",
            "gross revenue",
            "net revenue",
            "total sales",
            "how much revenue",
            "revenue total",
            "what is the revenue",
        ]
    ):
        res = route_revenue_total(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "revenue_total",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 6. Highest-spending customer / top customer
    if any(
        phrase in lower_msg
        for phrase in [
            "highest-spending",
            "highest spending",
            "top customer",
            "top customers",
            "spent the most",
            "best customer",
            "most valuable customer",
            "who spent the most",
        ]
    ) and not any(kw in lower_msg for kw in ["least", "lowest"]):
        res = route_top_customers(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "top_customers",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 6b. Lowest-spending customer / least spending customer
    if any(
        phrase in lower_msg
        for phrase in [
            "lowest-spending",
            "lowest spending",
            "least spending",
            "spent the least",
            "spent least",
            "lowest spender",
            "least spender",
            "which customer has spent the least",
            "which customer spent the least",
            "who spent the least",
            "who has spent the least",
        ]
    ) or (re.search(r"\b(?:least|lowest|minimum|bottom)\s+(?:spending\s+)?customers?\b", lower_msg)):
        res = route_lowest_spending_customer(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "lowest_spending_customer",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 6c. Who paid online
    if any(
        phrase in lower_msg
        for phrase in [
            "who paid online",
            "paid online",
            "online payment",
            "online payments",
            "digital payment",
            "pay online",
        ]
    ):
        res = route_who_paid_online(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "who_paid_online",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 7. Category performance / highest revenue category
    if "category" in lower_msg and any(
        kw in lower_msg for kw in ["revenue", "highest", "top", "sales", "performance", "breakdown", "generated"]
    ):
        res = route_revenue_by_category(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "revenue_by_category",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 8. Regional / city revenue & breakdown
    if ("city" in lower_msg or "cities" in lower_msg or "regional" in lower_msg) and any(
        kw in lower_msg for kw in ["revenue", "sales", "breakdown", "orders", "performance", "most"]
    ):
        res = route_revenue_by_city(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "revenue_by_city",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 9. Status breakdown / counts by status
    if any(phrase in lower_msg for phrase in ["status breakdown", "counts by status", "orders by status", "breakdown of status"]):
        res = route_status_breakdown(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "status_breakdown",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 10. General overview / summary
    if any(phrase in lower_msg for phrase in ["order summary", "overview of orders", "performance summary", "dataset summary", "summary of orders"]) or (
        lower_msg in ["summary", "overview", "total orders", "how many orders"]
    ):
        res = route_order_summary(csv_path=csv_path)
        res["metadata"] = {
            "model": "deterministic-router",
            "routing": "fast_path",
            "pattern": "order_summary",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        return res

    # 11. Customer search
    for cust in KNOWN_CUSTOMERS:
        if cust.lower() in lower_msg:
            res = route_customer_lookup(cust, csv_path=csv_path)
            res["metadata"] = {
                "model": "deterministic-router",
                "routing": "fast_path",
                "pattern": "customer_lookup",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            }
            return res

    # 12. Product search
    for prod in KNOWN_PRODUCTS:
        if prod.lower() in lower_msg and any(kw in lower_msg for kw in ["order", "orders", "show", "how many", "who", "bought"]):
            res = route_product_lookup(prod, csv_path=csv_path)
            res["metadata"] = {
                "model": "deterministic-router",
                "routing": "fast_path",
                "pattern": "product_lookup",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            }
            return res

    # No fast-path match: route to Gemini for complex/conversational reasoning
    return None
