"""
ORDIX — Orders Simplified
Tools Module: Backend tools for order lookup and statistical calculations.
Deterministic, robust, and grounded strictly in the real orders dataset.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from data_loader import load_orders, DataLoadError


def validate_date(date_str: Optional[str], field_name: str) -> Optional[str]:
    """Validates YYYY-MM-DD date format if provided."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise ValueError(f"Invalid date format for '{field_name}': '{date_str}'. Expected format is YYYY-MM-DD.")


def search_orders(
    order_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    status: Optional[str] = None,
    product: Optional[str] = None,
    category: Optional[str] = None,
    city: Optional[str] = None,
    payment_method: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search actual order records matching the provided filters.

    Args:
        order_id: Exact or partial Order ID (e.g., 'ORD-1001' or '1001')
        customer_name: Customer name or partial substring (e.g., 'Sneha' or 'Rahul Sharma')
        status: Order status ('delivered', 'cancelled', 'returned', 'processing', 'shipped')
        product: Product name or partial substring (e.g., 'Wireless Mouse', 'Keyboard')
        category: Category name ('Electronics', 'Accessories', 'Stationery', 'Furniture')
        city: City name (e.g., 'Chennai', 'Kochi', 'Bengaluru', 'Pune', 'Hyderabad', 'Thiruvananthapuram')
        payment_method: Payment method (e.g., 'UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash on Delivery')
        start_date: Filter orders placed on or after this date (YYYY-MM-DD)
        end_date: Filter orders placed on or before this date (YYYY-MM-DD)
        limit: Maximum number of matching records to return (1-50, default 10)
        csv_path: Optional custom path to dataset (used in testing)

    Returns:
        Structured dictionary containing matched orders count, criteria, and record list.
    """
    try:
        orders = load_orders(csv_path)
    except DataLoadError as e:
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "orders": [],
        }

    # Validate limits
    try:
        limit = max(1, min(int(limit), 50))
    except (ValueError, TypeError):
        limit = 10

    # Validate dates
    try:
        validated_start = validate_date(start_date, "start_date")
        validated_end = validate_date(end_date, "end_date")
    except ValueError as ve:
        return {
            "success": False,
            "error": str(ve),
            "count": 0,
            "orders": [],
        }

    # Clean query parameters
    q_order_id = order_id.strip().upper() if order_id else None
    q_customer = customer_name.strip().lower() if customer_name else None
    q_status = status.strip().lower() if status else None
    q_product = product.strip().lower() if product else None
    q_category = category.strip().lower() if category else None
    q_city = city.strip().lower() if city else None
    q_payment = payment_method.strip().lower() if payment_method else None

    matches: List[Dict[str, Any]] = []

    for order in orders:
        # Match order_id: allow exact match or substring
        if q_order_id:
            ord_id_val = order["order_id"].upper()
            if q_order_id != ord_id_val and q_order_id not in ord_id_val:
                continue

        # Match customer_name substring
        if q_customer and q_customer not in order["customer_name"].lower():
            continue

        # Match status
        if q_status and order["status"] != q_status:
            continue

        # Match product substring
        if q_product and q_product not in order["product"].lower():
            continue

        # Match category
        if q_category and q_category != order["category"].lower() and q_category not in order["category"].lower():
            continue

        # Match city
        if q_city and q_city != order["city"].lower() and q_city not in order["city"].lower():
            continue

        # Match payment method (supports specific method, 'online', 'offline', or comma-separated lists)
        if q_payment:
            ord_pm = order["payment_method"].lower()
            if q_payment in ["online", "online payment", "online payments", "digital", "digital payment"]:
                if ord_pm not in ["upi", "credit card", "debit card", "net banking"]:
                    continue
            elif q_payment in ["offline", "cod", "cash", "cash on delivery"]:
                if ord_pm != "cash on delivery":
                    continue
            elif "," in q_payment:
                allowed_methods = [m.strip().lower() for m in q_payment.split(",")]
                if ord_pm not in allowed_methods:
                    continue
            elif q_payment != ord_pm and q_payment not in ord_pm:
                continue

        # Match date range
        order_dt = order.get("order_date", "")
        if validated_start and order_dt < validated_start:
            continue
        if validated_end and order_dt > validated_end:
            continue

        matches.append(order)

    total_found = len(matches)
    truncated = matches[:limit]

    return {
        "success": True,
        "total_matched": total_found,
        "returned_count": len(truncated),
        "limit_applied": limit,
        "orders": truncated,
        "message": (
            f"Found {total_found} matching order(s)."
            if total_found > 0
            else "No orders found matching the specified criteria."
        ),
    }


def count_pending_orders(csv_path: Optional[str] = None) -> int:
    """
    Calculate the pending count directly from the loaded order data using the actual status column.
    Normalizes status with strip() and lower(), then counts rows where status equals 'pending'.
    Returns the exact integer count.
    """
    orders = load_orders(csv_path)
    return sum(1 for o in orders if str(o.get("status", "")).strip().lower() == "pending")


def calculate_order_statistics(
    operation: str,
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    city: Optional[str] = None,
    csv_path: Optional[str] = None,
    month: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate deterministic analytics and metrics directly from the order dataset.

    Args:
        operation: The aggregation metric to calculate.
            Supported operations:
            - 'summary': General overview of total orders, revenue, status breakdown, AOV
            - 'order_counts_by_status': Distribution of orders across all statuses
            - 'cancelled_and_pending': Count and value of cancelled, returned, and pending/processing orders
            - 'revenue_total': Total revenue breakdown (all orders vs delivered vs non-cancelled)
            - 'monthly_revenue': Revenue and order count by calendar month (e.g. 2026-06 to 2026-09)
            - 'revenue_by_category': Total sales and items sold per product category
            - 'top_customers': Customers ranked by total spending and order frequency
            - 'revenue_by_city': Regional breakdown of revenue and orders by city
            - 'orders_by_month': Order volume grouped by month
        status_filter: Optional filter to restrict calculations to a specific status (e.g., 'delivered')
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        category: Optional category filter
        city: Optional city filter
        csv_path: Optional custom path to dataset (used in testing)
        month: Optional month filter (e.g., 'August', 'September', '2026-08', '2026-09')

    Returns:
        Structured statistical dictionary with computed values.
    """
    try:
        orders = load_orders(csv_path)
    except DataLoadError as e:
        return {
            "success": False,
            "error": str(e),
            "operation": operation,
        }

    # Automatically resolve month parameter into start_date and end_date if not set
    if month and not (start_date and end_date):
        m_str = str(month).strip().lower()
        month_to_dates = {
            "june": ("2026-06-01", "2026-06-30"),
            "jun": ("2026-06-01", "2026-06-30"),
            "6": ("2026-06-01", "2026-06-30"),
            "06": ("2026-06-01", "2026-06-30"),
            "2026-06": ("2026-06-01", "2026-06-30"),
            "july": ("2026-07-01", "2026-07-31"),
            "jul": ("2026-07-01", "2026-07-31"),
            "7": ("2026-07-01", "2026-07-31"),
            "07": ("2026-07-01", "2026-07-31"),
            "2026-07": ("2026-07-01", "2026-07-31"),
            "august": ("2026-08-01", "2026-08-31"),
            "aug": ("2026-08-01", "2026-08-31"),
            "8": ("2026-08-01", "2026-08-31"),
            "08": ("2026-08-01", "2026-08-31"),
            "2026-08": ("2026-08-01", "2026-08-31"),
            "september": ("2026-09-01", "2026-09-30"),
            "sep": ("2026-09-01", "2026-09-30"),
            "sept": ("2026-09-01", "2026-09-30"),
            "9": ("2026-09-01", "2026-09-30"),
            "09": ("2026-09-01", "2026-09-30"),
            "2026-09": ("2026-09-01", "2026-09-30"),
        }
        if m_str in month_to_dates:
            start_date, end_date = month_to_dates[m_str]

    # Validate dates
    try:
        validated_start = validate_date(start_date, "start_date")
        validated_end = validate_date(end_date, "end_date")
    except ValueError as ve:
        return {
            "success": False,
            "error": str(ve),
            "operation": operation,
        }

    op = str(operation).strip().lower()

    # Pre-filter dataset if filters are provided
    filtered: List[Dict[str, Any]] = []
    q_status = status_filter.strip().lower() if status_filter else None
    q_category = category.strip().lower() if category else None
    q_city = city.strip().lower() if city else None

    for order in orders:
        if q_status and order["status"] != q_status:
            continue
        if q_category and q_category != order["category"].lower() and q_category not in order["category"].lower():
            continue
        if q_city and q_city != order["city"].lower() and q_city not in order["city"].lower():
            continue
        order_dt = order.get("order_date", "")
        if validated_start and order_dt < validated_start:
            continue
        if validated_end and order_dt > validated_end:
            continue
        filtered.append(order)

    if not filtered:
        return {
            "success": True,
            "operation": op,
            "total_orders_in_scope": 0,
            "message": "No orders matched the specified filters for this statistical calculation.",
            "data": {},
        }

    # 1. Summary
    if op in ["summary", "overall", "performance"]:
        status_counts = defaultdict(int)
        delivered_revenue = 0.0
        cancelled_revenue = 0.0
        total_gross_value = 0.0

        for o in filtered:
            st = o["status"]
            val = o["total_inr"]
            status_counts[st] += 1
            total_gross_value += val
            if st == "delivered":
                delivered_revenue += val
            elif st == "cancelled":
                cancelled_revenue += val

        delivered_count = status_counts.get("delivered", 0)
        aov = round(delivered_revenue / delivered_count, 2) if delivered_count > 0 else 0.0

        return {
            "success": True,
            "operation": "summary",
            "total_orders_in_scope": len(filtered),
            "data": {
                "total_orders": len(filtered),
                "delivered_orders_revenue_inr": round(delivered_revenue, 2),
                "total_gross_order_value_inr": round(total_gross_value, 2),
                "cancelled_value_lost_inr": round(cancelled_revenue, 2),
                "delivered_orders_count": delivered_count,
                "average_delivered_order_value_inr": aov,
                "status_breakdown": dict(status_counts),
            },
        }

    # 2. Order counts by status
    elif op in ["order_counts_by_status", "status_counts", "status_breakdown"]:
        counts = defaultdict(int)
        for o in filtered:
            counts[o["status"]] += 1
        return {
            "success": True,
            "operation": "order_counts_by_status",
            "total_orders_in_scope": len(filtered),
            "data": {
                "counts_by_status": dict(counts),
                "total_orders": len(filtered),
            },
        }

    # 3. Cancelled and pending
    elif op in ["cancelled_and_pending", "cancelled_orders", "pending_orders"]:
        cancelled = [o for o in filtered if o["status"] == "cancelled"]
        returned = [o for o in filtered if o["status"] == "returned"]
        processing = [o for o in filtered if o["status"] == "processing"]
        shipped = [o for o in filtered if o["status"] == "shipped"]

        return {
            "success": True,
            "operation": "cancelled_and_pending",
            "total_orders_in_scope": len(filtered),
            "data": {
                "cancelled_count": len(cancelled),
                "cancelled_value_inr": round(sum(o["total_inr"] for o in cancelled), 2),
                "returned_count": len(returned),
                "returned_value_inr": round(sum(o["total_inr"] for o in returned), 2),
                "processing_count": len(processing),
                "processing_value_inr": round(sum(o["total_inr"] for o in processing), 2),
                "shipped_count": len(shipped),
                "shipped_value_inr": round(sum(o["total_inr"] for o in shipped), 2),
            },
        }

    # 3b. Count pending orders
    elif op in ["count_pending_orders", "pending_count", "pending_orders_count"]:
        pending_cnt = sum(1 for o in filtered if str(o.get("status", "")).strip().lower() == "pending")
        return {
            "success": True,
            "operation": "count_pending_orders",
            "total_orders_in_scope": len(filtered),
            "pending_count": pending_cnt,
            "data": {
                "pending_count": pending_cnt,
                "total_orders": len(filtered),
            },
        }

    # 4. Total revenue
    elif op in ["revenue_total", "total_revenue", "revenue"]:
        delivered_revenue = sum(o["total_inr"] for o in filtered if o["status"] == "delivered")
        non_cancelled_revenue = sum(o["total_inr"] for o in filtered if o["status"] != "cancelled")
        total_all_orders_value = sum(o["total_inr"] for o in filtered)
        cancelled_revenue = sum(o["total_inr"] for o in filtered if o["status"] == "cancelled")

        return {
            "success": True,
            "operation": "revenue_total",
            "total_orders_in_scope": len(filtered),
            "data": {
                "delivered_revenue_inr": round(delivered_revenue, 2),
                "net_revenue_excluding_cancelled_inr": round(non_cancelled_revenue, 2),
                "total_gross_value_including_cancelled_inr": round(total_all_orders_value, 2),
                "cancelled_order_value_inr": round(cancelled_revenue, 2),
                "delivered_order_count": sum(1 for o in filtered if o["status"] == "delivered"),
            },
        }

    # 5. Monthly revenue
    elif op in ["monthly_revenue", "revenue_by_month"]:
        months_delivered = defaultdict(float)
        months_gross = defaultdict(float)
        months_count = defaultdict(int)

        for o in filtered:
            dt = o.get("order_date", "")
            month_key = dt[:7] if len(dt) >= 7 else "Unknown"
            months_count[month_key] += 1
            months_gross[month_key] += o["total_inr"]
            if o["status"] == "delivered":
                months_delivered[month_key] += o["total_inr"]

        monthly_breakdown = []
        for m in sorted(months_count.keys()):
            monthly_breakdown.append({
                "month": m,
                "order_count": months_count[m],
                "delivered_revenue_inr": round(months_delivered[m], 2),
                "gross_revenue_inr": round(months_gross[m], 2),
            })

        return {
            "success": True,
            "operation": "monthly_revenue",
            "total_orders_in_scope": len(filtered),
            "data": {
                "monthly_breakdown": monthly_breakdown,
            },
        }

    # 6. Orders by month
    elif op in ["orders_by_month", "monthly_orders"]:
        months_count = defaultdict(int)
        for o in filtered:
            dt = o.get("order_date", "")
            month_key = dt[:7] if len(dt) >= 7 else "Unknown"
            months_count[month_key] += 1

        breakdown = [{"month": m, "order_count": months_count[m]} for m in sorted(months_count.keys())]
        return {
            "success": True,
            "operation": "orders_by_month",
            "total_orders_in_scope": len(filtered),
            "data": {
                "monthly_order_counts": breakdown,
            },
        }

    # 7. Revenue by product category
    elif op in ["revenue_by_category", "category_revenue", "category_breakdown"]:
        cat_revenue_delivered = defaultdict(float)
        cat_revenue_gross = defaultdict(float)
        cat_quantity = defaultdict(int)
        cat_orders = defaultdict(int)

        for o in filtered:
            c = o["category"]
            cat_orders[c] += 1
            cat_quantity[c] += o["quantity"]
            cat_revenue_gross[c] += o["total_inr"]
            if o["status"] == "delivered":
                cat_revenue_delivered[c] += o["total_inr"]

        categories_list = []
        for c in sorted(cat_orders.keys(), key=lambda x: cat_revenue_delivered[x], reverse=True):
            categories_list.append({
                "category": c,
                "delivered_revenue_inr": round(cat_revenue_delivered[c], 2),
                "gross_revenue_inr": round(cat_revenue_gross[c], 2),
                "total_quantity_sold": cat_quantity[c],
                "order_count": cat_orders[c],
            })

        top_cat = categories_list[0]["category"] if categories_list else None

        return {
            "success": True,
            "operation": "revenue_by_category",
            "total_orders_in_scope": len(filtered),
            "data": {
                "top_category_by_delivered_revenue": top_cat,
                "categories": categories_list,
            },
        }

    # 8. Top & lowest spending customers
    elif op in ["top_customers", "highest_spending_customer", "lowest_spending_customer", "least_spending_customer", "customer_analysis"]:
        cust_delivered_spend = defaultdict(float)
        cust_total_spend = defaultdict(float)
        cust_orders = defaultdict(int)
        cust_cities = defaultdict(set)

        for o in filtered:
            cust = o["customer_name"]
            cust_orders[cust] += 1
            cust_total_spend[cust] += o["total_inr"]
            cust_cities[cust].add(o["city"])
            if o["status"] == "delivered":
                cust_delivered_spend[cust] += o["total_inr"]

        is_least = op in ["lowest_spending_customer", "least_spending_customer"]

        # Sort by delivered spend primarily, fallback to total spend
        ranked = sorted(
            cust_orders.keys(),
            key=lambda c: (cust_delivered_spend[c], cust_total_spend[c]),
            reverse=not is_least,
        )

        customer_list = []
        for c in ranked:
            customer_list.append({
                "customer_name": c,
                "delivered_spend_inr": round(cust_delivered_spend[c], 2),
                "total_spend_inr": round(cust_total_spend[c], 2),
                "order_count": cust_orders[c],
                "cities": sorted(list(cust_cities[c])),
            })

        target_customer = customer_list[0] if customer_list else None

        return {
            "success": True,
            "operation": op,
            "total_orders_in_scope": len(filtered),
            "data": {
                "primary_customer": target_customer,
                "highest_spending_customer": customer_list[0] if not is_least and customer_list else None,
                "lowest_spending_customer": customer_list[0] if is_least and customer_list else None,
                "ranked_customers": customer_list[:10],
                "top_ranked_customers": customer_list[:10],
            },
        }

    # 9. Revenue by city
    elif op in ["revenue_by_city", "city_statistics", "regional_breakdown"]:
        city_delivered = defaultdict(float)
        city_gross = defaultdict(float)
        city_orders = defaultdict(int)
        city_custs = defaultdict(set)

        for o in filtered:
            ct = o["city"]
            city_orders[ct] += 1
            city_gross[ct] += o["total_inr"]
            city_custs[ct].add(o["customer_name"])
            if o["status"] == "delivered":
                city_delivered[ct] += o["total_inr"]

        city_list = []
        for ct in sorted(city_orders.keys(), key=lambda x: city_delivered[x], reverse=True):
            city_list.append({
                "city": ct,
                "delivered_revenue_inr": round(city_delivered[ct], 2),
                "gross_revenue_inr": round(city_gross[ct], 2),
                "order_count": city_orders[ct],
                "unique_customers_count": len(city_custs[ct]),
                "customers": sorted(list(city_custs[ct])),
            })

        return {
            "success": True,
            "operation": "revenue_by_city",
            "total_orders_in_scope": len(filtered),
            "data": {
                "cities": city_list,
            },
        }

    # 10. City Customers
    elif op in ["city_customers", "customers_by_city"]:
        city_custs = defaultdict(set)
        city_cust_orders = defaultdict(lambda: defaultdict(list))
        for o in filtered:
            ct = o["city"]
            cust = o["customer_name"]
            city_custs[ct].add(cust)
            city_cust_orders[ct][cust].append(o)

        breakdown = []
        for ct in sorted(city_custs.keys()):
            cust_details = []
            for cust in sorted(city_custs[ct]):
                ords = city_cust_orders[ct][cust]
                deliv_rev = sum(x["total_inr"] for x in ords if x["status"] == "delivered")
                gross_val = sum(x["total_inr"] for x in ords)
                cust_details.append({
                    "customer_name": cust,
                    "order_count": len(ords),
                    "delivered_spend_inr": round(deliv_rev, 2),
                    "gross_spend_inr": round(gross_val, 2),
                })
            breakdown.append({
                "city": ct,
                "unique_customers_count": len(city_custs[ct]),
                "customers": sorted(list(city_custs[ct])),
                "total_orders": sum(len(ords) for ords in city_cust_orders[ct].values()),
                "customer_details": cust_details,
            })

        return {
            "success": True,
            "operation": "city_customers",
            "total_orders_in_scope": len(filtered),
            "data": {
                "cities": breakdown,
            },
        }

    else:
        return {
            "success": False,
            "operation": op,
            "error": (
                f"Unsupported operation '{operation}'. Supported operations: "
                "'summary', 'order_counts_by_status', 'cancelled_and_pending', "
                "'revenue_total', 'monthly_revenue', 'orders_by_month', "
                "'revenue_by_category', 'top_customers', 'lowest_spending_customer', 'revenue_by_city', 'city_customers'."
            ),
        }
