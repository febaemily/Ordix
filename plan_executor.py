"""
ORDIX — Orders Simplified
Plan Executor Module: Deterministic Python & pandas execution of structured query plans.
100% verified calculations strictly grounded in orders.csv. Zero hallucination.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from query_plan import QueryPlan, FilterSpec, AggregationMetric
from schema_inspector import inspect_schema, DEFAULT_CSV_PATH


def format_inr(amount: float) -> str:
    """Formats numeric amount as Indian Rupee string."""
    return f"₹{float(amount):,.2f}"


def to_native_types(obj: Any) -> Any:
    """Recursively converts numpy/pandas scalar types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native_types(item) for item in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)[:10]
    return obj


def load_dataframe_for_execution(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads DataFrame with enriched temporal and boolean fields for query execution.
    """
    target_path = csv_path if csv_path is not None else DEFAULT_CSV_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Orders CSV dataset not found at: {target_path}")

    df = pd.read_csv(target_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Clean string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Numeric conversions
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    if "unit_price_inr" in df.columns:
        df["unit_price_inr"] = pd.to_numeric(df["unit_price_inr"], errors="coerce").fillna(0.0)
    if "total_inr" in df.columns:
        df["total_inr"] = pd.to_numeric(df["total_inr"], errors="coerce").fillna(0.0)

    # Date handling
    date_col = "order_date" if "order_date" in df.columns else None
    if date_col:
        df["order_date_str"] = df[date_col].astype(str).str.strip()
        df["order_date_dt"] = pd.to_datetime(df["order_date_str"], format="%Y-%m-%d", errors="coerce")
        df["year"] = df["order_date_dt"].dt.year
        df["month"] = df["order_date_dt"].dt.month
        df["month_name"] = df["order_date_dt"].dt.strftime("%B")
        df["year_month"] = df["order_date_dt"].dt.strftime("%Y-%m")

    # Status lowercasing and boolean flags
    if "status" in df.columns:
        df["status_lower"] = df["status"].str.lower()
        df["is_delivered"] = df["status_lower"] == "delivered"
        df["is_cancelled"] = df["status_lower"] == "cancelled"
        df["is_returned"] = df["status_lower"] == "returned"
        df["is_processing"] = df["status_lower"] == "processing"
        df["is_shipped"] = df["status_lower"] == "shipped"
    else:
        df["is_delivered"] = True
        df["is_cancelled"] = False
        df["is_returned"] = False

    return df


def apply_filters(df: pd.DataFrame, filters: List[FilterSpec]) -> pd.DataFrame:
    """
    Applies validated FilterSpec list safely to the DataFrame.
    """
    filtered = df.copy()
    for f in filters:
        col = f.column
        op = f.operator
        val = f.value

        if col not in filtered.columns:
            continue

        if op == "==":
            if isinstance(val, str) and filtered[col].dtype == object:
                filtered = filtered[filtered[col].str.lower() == val.lower()]
            else:
                filtered = filtered[filtered[col] == val]
        elif op == "!=":
            if isinstance(val, str) and filtered[col].dtype == object:
                filtered = filtered[filtered[col].str.lower() != val.lower()]
            else:
                filtered = filtered[filtered[col] != val]
        elif op == "in":
            if isinstance(val, list):
                if filtered[col].dtype == object:
                    vals_lower = [str(v).lower() for v in val]
                    filtered = filtered[filtered[col].str.lower().isin(vals_lower)]
                else:
                    filtered = filtered[filtered[col].isin(val)]
        elif op == "not in":
            if isinstance(val, list):
                if filtered[col].dtype == object:
                    vals_lower = [str(v).lower() for v in val]
                    filtered = filtered[~filtered[col].str.lower().isin(vals_lower)]
                else:
                    filtered = filtered[~filtered[col].isin(val)]
        elif op == ">":
            filtered = filtered[filtered[col] > val]
        elif op == ">=":
            filtered = filtered[filtered[col] >= val]
        elif op == "<":
            filtered = filtered[filtered[col] < val]
        elif op == "<=":
            filtered = filtered[filtered[col] <= val]
        elif op == "between":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                v_min, v_max = val
                filtered = filtered[(filtered[col] >= v_min) & (filtered[col] <= v_max)]
        elif op == "contains":
            if filtered[col].dtype == object:
                filtered = filtered[filtered[col].str.lower().str.contains(str(val).lower(), regex=False, na=False)]

    return filtered


def execute_plan(plan: QueryPlan, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for executing a validated QueryPlan.
    Returns structured answer, tool_calls trace, and metadata.
    """
    start_time = time.time()
    
    # 1. Ambiguous query / clarification requested
    if plan.intent == "clarification":
        clarification_msg = plan.clarification_prompt or (
            "Could you please specify which columns or filters you would like to analyze? "
            "For example, you can ask about revenue by month, customer spending, product sales, "
            "order status breakdown, or orders in a specific city."
        )
        return {
            "answer": clarification_msg,
            "tool_calls": [],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "clarification",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    df = load_dataframe_for_execution(csv_path)
    f_df = apply_filters(df, plan.filters)

    # 2. Distinct Values Query
    if plan.intent == "distinct_values":
        target_col = plan.group_by[0] if (plan.group_by and len(plan.group_by) > 0) else (plan.sort_by or "product")
        if target_col in f_df.columns:
            unique_vals = sorted(f_df[target_col].dropna().unique().tolist())
        else:
            unique_vals = []

        title = plan.title or f"Distinct Values for {target_col.replace('_', ' ').title()}"
        lines = [
            f"### {title} ({len(unique_vals)} Found)\n",
            f"There are **{len(unique_vals)} unique values** recorded in **{target_col}**:\n",
        ]
        for val in unique_vals[:30]:
            lines.append(f"- **{val}**")
        if len(unique_vals) > 30:
            lines.append(f"\n*...plus {len(unique_vals) - 30} additional values.*")

        return {
            "answer": "\n".join(lines),
            "tool_calls": [
                {
                    "tool": "calculate_order_statistics",
                    "args": {"operation": "distinct_values", "column": target_col},
                    "result": {"success": True, "column": target_col, "count": len(unique_vals), "values": unique_vals},
                }
            ],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "distinct_values",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    # 3. Search / Filtered Records Listing
    if plan.intent == "search":
        matched_count = len(f_df)
        if matched_count == 0:
            return {
                "answer": "No orders matched your specified criteria in the dataset.",
                "tool_calls": [
                    {
                        "tool": "search_orders",
                        "args": {"filters": [f.model_dump() for f in plan.filters]},
                        "result": {"success": True, "count": 0, "orders": []},
                    }
                ],
                "metadata": {
                    "engine": "general-plan-executor",
                    "intent": "search",
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            }

        limit = plan.limit or 10
        display_df = f_df.head(limit)
        
        deliv_rev = f_df[f_df["is_delivered"]]["total_inr"].sum()
        gross_val = f_df["total_inr"].sum()
        deliv_cnt = f_df["is_delivered"].sum()

        title = plan.title or f"Matching Orders ({matched_count} Found)"
        lines = [
            f"### {title}\n",
            f"- **Matching Orders**: **{matched_count}** order(s)",
            f"- **Delivered Orders Revenue**: **{format_inr(deliv_rev)}** ({deliv_cnt} fulfilled order(s))",
            f"- **Total Gross Value**: **{format_inr(gross_val)}**",
            f"- **Total Units**: **{f_df['quantity'].sum()}**\n",
            "| Order ID | Customer | Product | Amount | Status | City | Date |",
            "| :--- | :--- | :--- | :--- | :---: | :--- | :--- |",
        ]
        for _, r in display_df.iterrows():
            lines.append(
                f"| **{r['order_id']}** | {r['customer_name']} | {r['product']} | "
                f"{format_inr(r['total_inr'])} | {r['status'].capitalize()} | {r['city']} | {r['order_date_str']} |"
            )

        if matched_count > limit:
            lines.append(f"\n*...plus {matched_count - limit} additional matching orders.*")

        sample_orders = display_df.to_dict(orient="records")
        return {
            "answer": "\n".join(lines),
            "tool_calls": [
                {
                    "tool": "search_orders",
                    "args": {"limit": limit, "filters": [f.model_dump() for f in plan.filters]},
                    "result": {
                        "success": True,
                        "total_matched": matched_count,
                        "returned_count": len(sample_orders),
                        "orders": to_native_types(sample_orders),
                    },
                }
            ],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "search",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    # 4. Rankings (Top-N / Bottom-N Entities)
    if plan.intent == "ranking":
        rank_col = plan.group_by[0] if (plan.group_by and len(plan.group_by) > 0) else "customer_name"
        limit = plan.limit or 5

        # Aggregate by entity
        grouped = f_df.groupby(rank_col).agg(
            order_count=("order_id", "count"),
            delivered_revenue=("total_inr", lambda s: s[f_df.loc[s.index, "is_delivered"]].sum()),
            gross_revenue=("total_inr", "sum"),
            units_sold=("quantity", "sum"),
        ).reset_index()

        # Sort metric
        sort_col = plan.sort_by if plan.sort_by in grouped.columns else "delivered_revenue"
        grouped = grouped.sort_values(
            by=[sort_col, "order_count", rank_col],
            ascending=[plan.sort_ascending, plan.sort_ascending, True],
        )

        # Dense / minimum rank for accurate tie handling
        grouped["rank"] = grouped[sort_col].rank(method="min", ascending=plan.sort_ascending).astype(int)

        # If limit is 1 but there are ties for first/lowest place, include all tied entities
        if limit == 1 and (grouped["rank"] == 1).sum() > 1:
            top_slice = grouped[grouped["rank"] == 1]
        else:
            top_slice = grouped.head(limit)

        direction_str = "Lowest" if plan.sort_ascending else "Top"
        metric_label = "Delivered Revenue" if sort_col == "delivered_revenue" else sort_col.replace("_", " ").title()
        title = plan.title or f"{direction_str} {limit} {rank_col.replace('_', ' ').title()}s by {metric_label}"

        lines = [
            f"### {title}\n",
        ]

        # Provide direct lead answer for single entity inquiries (e.g. "Which customer has spent the least?")
        rank1_slice = grouped[grouped["rank"] == 1]
        if limit == 1 or len(top_slice) == 1:
            direction_word = "least" if plan.sort_ascending else "highest"
            if len(rank1_slice) == 1:
                lead_row = rank1_slice.iloc[0]
                lines.append(
                    f"The {rank_col.replace('_', ' ')} with the {direction_word} {metric_label.lower()} is **{lead_row[rank_col]}** "
                    f"with **{format_inr(lead_row[sort_col])}** across {lead_row['order_count']} order(s) "
                    f"(Total Gross Value: {format_inr(lead_row['gross_revenue'])}).\n"
                )
            else:
                names = ", ".join([f"**{r[rank_col]}**" for _, r in rank1_slice.iterrows()])
                lines.append(
                    f"There is a tie for the {direction_word} {metric_label.lower()} among {names}, "
                    f"each with **{format_inr(rank1_slice.iloc[0][sort_col])}**.\n"
                )

        lines.extend([
            f"| Rank | {rank_col.replace('_', ' ').title()} | Orders | Delivered Revenue | Gross Value | Units |",
            "| :-: | :--- | :---: | :--- | :--- | :---: |",
        ])

        for _, row in top_slice.iterrows():
            lines.append(
                f"| {row['rank']} | **{row[rank_col]}** | {row['order_count']} | "
                f"{format_inr(row['delivered_revenue'])} | {format_inr(row['gross_revenue'])} | {row['units_sold']} |"
            )

        lines.append("\n*Note on Revenue Treatment: Delivered revenue strictly reflects fulfilled orders received by customers. Cancelled and refunded orders are excluded from delivered revenue figures.*")
        if grouped["rank"].nunique() < len(grouped):
            lines.append("*Tie Handling: In case of equal metric values, entities share the same rank number.*")

        return {
            "answer": "\n".join(lines),
            "tool_calls": [
                {
                    "tool": "calculate_order_statistics",
                    "args": {
                        "operation": "ranking",
                        "entity": rank_col,
                        "sort_by": sort_col,
                        "ascending": plan.sort_ascending,
                        "limit": limit,
                    },
                    "result": {
                        "success": True,
                        "data": to_native_types(top_slice.to_dict(orient="records")),
                    },
                }
            ],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "ranking",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    # 5. Group-By Aggregations with Grand Totals & Percentages
    if plan.intent == "group_by":
        group_cols = plan.group_by or ["category"]
        primary_col = group_cols[0]

        grouped = f_df.groupby(group_cols).agg(
            order_count=("order_id", "count"),
            delivered_revenue=("total_inr", lambda s: s[f_df.loc[s.index, "is_delivered"]].sum()),
            gross_revenue=("total_inr", "sum"),
            quantity=("quantity", "sum"),
        ).reset_index()

        # Sort
        sort_col = plan.sort_by if (plan.sort_by and plan.sort_by in grouped.columns) else "delivered_revenue"
        grouped = grouped.sort_values(by=sort_col, ascending=plan.sort_ascending)

        total_orders = len(f_df)
        total_delivered_rev = f_df[f_df["is_delivered"]]["total_inr"].sum()
        total_gross_val = f_df["total_inr"].sum()
        total_quantity = f_df["quantity"].sum()

        title = plan.title or f"Breakdown by {', '.join([c.replace('_', ' ').title() for c in group_cols])}"
        
        has_pct = plan.calculate_percentage
        if has_pct:
            lines = [
                f"### {title}\n",
                f"| {primary_col.replace('_', ' ').title()} | Orders | Share | Delivered Revenue | Gross Value | Units |",
                "| :--- | :---: | :---: | :--- | :--- | :---: |",
            ]
            for _, row in grouped.iterrows():
                share = (row["order_count"] / total_orders * 100) if total_orders > 0 else 0.0
                lines.append(
                    f"| **{row[primary_col]}** | {row['order_count']} | {share:.1f}% | "
                    f"{format_inr(row['delivered_revenue'])} | {format_inr(row['gross_revenue'])} | {row['quantity']} |"
                )
        else:
            lines = [
                f"### {title}\n",
                f"| {primary_col.replace('_', ' ').title()} | Orders | Delivered Revenue | Gross Value | Units |",
                "| :--- | :---: | :--- | :--- | :---: |",
            ]
            for _, row in grouped.iterrows():
                lines.append(
                    f"| **{row[primary_col]}** | {row['order_count']} | "
                    f"{format_inr(row['delivered_revenue'])} | {format_inr(row['gross_revenue'])} | {row['quantity']} |"
                )

        # Append Grand Total Row if requested or standard breakdown
        if plan.include_grand_total or len(grouped) > 1:
            if has_pct:
                lines.append(
                    f"| **Grand Total** | **{total_orders}** | **100.0%** | "
                    f"**{format_inr(total_delivered_rev)}** | **{format_inr(total_gross_val)}** | **{total_quantity}** |"
                )
            else:
                lines.append(
                    f"| **Grand Total** | **{total_orders}** | "
                    f"**{format_inr(total_delivered_rev)}** | **{format_inr(total_gross_val)}** | **{total_quantity}** |"
                )

        lines.append("\n*Revenue Definition: Delivered revenue strictly excludes cancelled and returned orders. Gross value covers all orders placed.*")

        return {
            "answer": "\n".join(lines),
            "tool_calls": [
                {
                    "tool": "calculate_order_statistics",
                    "args": {"operation": f"group_by_{primary_col}", "columns": group_cols},
                    "result": {
                        "success": True,
                        "data": to_native_types(grouped.to_dict(orient="records")),
                        "grand_total": {
                            "order_count": total_orders,
                            "delivered_revenue_inr": round(float(total_delivered_rev), 2),
                            "gross_revenue_inr": round(float(total_gross_val), 2),
                            "quantity": int(total_quantity),
                        },
                    },
                }
            ],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "group_by",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    # 6. Comparisons (e.g. Month A vs Month B, UPI vs Credit Card)
    if plan.intent == "comparison":
        comp_col = plan.group_by[0] if (plan.group_by and len(plan.group_by) > 0) else "month_name"
        grouped = f_df.groupby(comp_col).agg(
            order_count=("order_id", "count"),
            delivered_revenue=("total_inr", lambda s: s[f_df.loc[s.index, "is_delivered"]].sum()),
            gross_revenue=("total_inr", "sum"),
            quantity=("quantity", "sum"),
        ).reset_index()

        title = plan.title or f"Comparison by {comp_col.replace('_', ' ').title()}"
        lines = [
            f"### {title}\n",
            f"| {comp_col.replace('_', ' ').title()} | Orders | Delivered Revenue | Gross Value | Units |",
            "| :--- | :---: | :--- | :--- | :---: |",
        ]
        for _, row in grouped.iterrows():
            lines.append(
                f"| **{row[comp_col]}** | {row['order_count']} | "
                f"{format_inr(row['delivered_revenue'])} | {format_inr(row['gross_revenue'])} | {row['quantity']} |"
            )

        if len(grouped) == 2:
            r1, r2 = grouped.iloc[0], grouped.iloc[1]
            diff_rev = r1["delivered_revenue"] - r2["delivered_revenue"]
            diff_orders = r1["order_count"] - r2["order_count"]
            higher_rev = r1[comp_col] if diff_rev > 0 else r2[comp_col]
            lines.append(
                f"\n**Key Variance**: **{higher_rev}** outperformed with **{format_inr(abs(diff_rev))}** higher delivered revenue "
                f"and **{abs(diff_orders)}** difference in order volume."
            )

        return {
            "answer": "\n".join(lines),
            "tool_calls": [
                {
                    "tool": "calculate_order_statistics",
                    "args": {"operation": "comparison", "column": comp_col},
                    "result": {"success": True, "comparison": to_native_types(grouped.to_dict(orient="records"))},
                }
            ],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "comparison",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    # 7. Aggregate Metrics (Sum, Count, Avg, Min, Max over Filtered Scope)
    tot_orders = len(f_df)
    if tot_orders == 0:
        return {
            "answer": "No orders matched your specified criteria in the dataset.",
            "tool_calls": [
                {
                    "tool": "calculate_order_statistics",
                    "args": {"operation": "aggregate", "filters": [f.model_dump() for f in plan.filters]},
                    "result": {"success": True, "count": 0},
                }
            ],
            "metadata": {
                "engine": "general-plan-executor",
                "intent": "aggregate",
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        }

    deliv_df = f_df[f_df["is_delivered"]]
    deliv_rev = deliv_df["total_inr"].sum()
    gross_val = f_df["total_inr"].sum()
    canc_val = f_df[f_df["is_cancelled"]]["total_inr"].sum()
    deliv_cnt = len(deliv_df)
    aov = (deliv_rev / deliv_cnt) if deliv_cnt > 0 else 0.0

    title = plan.title or "Summary & Key Metrics"
    lines = [
        f"### {title}\n",
        f"- **Total In-Scope Orders**: **{tot_orders}** order(s)",
        f"- **Delivered Orders Revenue**: **{format_inr(deliv_rev)}** ({deliv_cnt} fulfilled order(s))",
        f"- **Total Gross Order Value**: **{format_inr(gross_val)}**",
        f"- **Average Delivered Order Value (AOV)**: **{format_inr(aov)}**",
        f"- **Cancelled Orders Lost**: **{format_inr(canc_val)}** ({f_df['is_cancelled'].sum()} order(s))",
        f"- **Total Items/Quantity**: **{f_df['quantity'].sum()}** unit(s)",
    ]

    # Specific custom metric requests (e.g. min/max/mean)
    for m in plan.metrics:
        col = m.column
        agg = m.agg
        if col in f_df.columns:
            if agg == "mean":
                val = f_df[col].mean()
                lines.append(f"- **Average {col.replace('_', ' ').title()}**: {val:,.2f}")
            elif agg == "max":
                val = f_df[col].max()
                lines.append(f"- **Maximum {col.replace('_', ' ').title()}**: {val:,.2f}")
            elif agg == "min":
                val = f_df[col].min()
                lines.append(f"- **Minimum {col.replace('_', ' ').title()}**: {val:,.2f}")

    lines.append("\n*Revenue Definition: Delivered revenue strictly reflects fulfilled orders received by customers. Cancelled and refunded orders are excluded from delivered revenue figures.*")

    return {
        "answer": "\n".join(lines),
        "tool_calls": [
            {
                "tool": "calculate_order_statistics",
                "args": {"operation": "summary", "filters": [f.model_dump() for f in plan.filters]},
                "result": {
                    "success": True,
                    "total_orders": tot_orders,
                    "delivered_orders": deliv_cnt,
                    "delivered_revenue_inr": round(float(deliv_rev), 2),
                    "gross_revenue_inr": round(float(gross_val), 2),
                    "cancelled_revenue_inr": round(float(canc_val), 2),
                    "aov_inr": round(float(aov), 2),
                },
            }
        ],
        "metadata": {
            "engine": "general-plan-executor",
            "intent": "aggregate",
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
        },
    }
