"""
ORDIX — Orders Simplified
Schema Inspector Module: Dynamic introspection and caching of orders.csv metadata.
Extracts column names, types, date formats, distinct values, and synonyms without hardcoding.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.csv")

# In-memory schema cache: csv_path -> (mtime, schema_dict)
_SCHEMA_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def inspect_schema(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Dynamically inspects the actual CSV file to determine columns, dtypes,
    distinct categories, numeric fields, date bounds, and value mappings.
    Caches metadata based on file modification timestamp.
    """
    target_path = csv_path if csv_path is not None else DEFAULT_CSV_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Orders CSV dataset not found at: {target_path}")

    mtime = os.path.getmtime(target_path)
    if target_path in _SCHEMA_CACHE:
        cached_mtime, cached_schema = _SCHEMA_CACHE[target_path]
        if cached_mtime == mtime:
            return cached_schema

    df = pd.read_csv(target_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    columns: List[str] = list(df.columns)
    column_types: Dict[str, str] = {}
    distinct_values: Dict[str, List[str]] = {}
    value_to_column: Dict[str, str] = {}  # lowercase value -> column name
    value_lookup: Dict[str, Dict[str, str]] = {}  # column -> {lowercase_val: canonical_val}
    numeric_columns: List[str] = []
    text_columns: List[str] = []
    date_columns: List[str] = []
    missing_counts: Dict[str, int] = df.isnull().sum().to_dict()

    for col in columns:
        sample_series = df[col].dropna()
        # Check if numeric
        is_num = pd.to_numeric(sample_series, errors="coerce").notnull().all() and len(sample_series) > 0
        
        # Check if date
        is_date = False
        if not is_num and col.lower() in ["order_date", "date", "created_at", "timestamp"]:
            try:
                parsed_dates = pd.to_datetime(sample_series, errors="coerce")
                if parsed_dates.notnull().mean() > 0.8:
                    is_date = True
            except Exception:
                pass

        if is_date:
            column_types[col] = "datetime"
            date_columns.append(col)
        elif is_num and col not in ["order_id"]:
            column_types[col] = "numeric"
            numeric_columns.append(col)
        else:
            column_types[col] = "categorical" if df[col].nunique() <= 30 else "text"
            text_columns.append(col)
            # Record distinct values for categorical / lookup fields
            unique_vals = [str(v).strip() for v in sample_series.unique() if str(v).strip()]
            distinct_values[col] = sorted(unique_vals)
            lookup = {v.lower(): v for v in unique_vals}
            value_lookup[col] = lookup
            for v in unique_vals:
                v_lower = v.lower()
                # Map value to column if unambiguous
                if v_lower not in value_to_column:
                    value_to_column[v_lower] = col

    # Date range analysis
    date_col = date_columns[0] if date_columns else ("order_date" if "order_date" in columns else None)
    date_range: Dict[str, Optional[str]] = {"min": None, "max": None}
    if date_col and date_col in df.columns:
        dt_series = pd.to_datetime(df[date_col].dropna(), errors="coerce")
        if dt_series.notnull().any():
            date_range["min"] = dt_series.min().strftime("%Y-%m-%d")
            date_range["max"] = dt_series.max().strftime("%Y-%m-%d")

    # Column synonyms and semantic keywords mapping
    synonyms: Dict[str, List[str]] = {
        "revenue": ["total_inr", "total", "amount", "sales", "gross", "net", "turnover", "spend", "value"],
        "price": ["unit_price_inr", "unit_price", "rate", "cost"],
        "quantity": ["quantity", "qty", "volume", "units", "items", "count of items", "number of items"],
        "customer": ["customer_name", "customer", "client", "buyer", "user", "who"],
        "city": ["city", "location", "place", "town", "region", "area"],
        "product": ["product", "item", "article", "goods", "merchandise"],
        "category": ["category", "department", "type of product", "product category"],
        "payment": ["payment_method", "payment", "paid with", "payment mode", "mode", "payment type"],
        "status": ["status", "order status", "state", "fulfillment status"],
        "order": ["order_id", "order", "id", "order number", "order id"],
        "date": ["order_date", "date", "day", "month", "year", "when", "time"],
    }

    # Reverse mapping: lowercase keyword -> canonical column
    keyword_to_column: Dict[str, str] = {}
    for col in columns:
        keyword_to_column[col.lower()] = col
        col_no_under = col.lower().replace("_", " ")
        keyword_to_column[col_no_under] = col

    for category, syn_list in synonyms.items():
        matched_target_col = None
        for syn in syn_list:
            for actual_col in columns:
                if syn == actual_col.lower() or syn in actual_col.lower():
                    matched_target_col = actual_col
                    break
            if matched_target_col:
                break

        if matched_target_col:
            for syn in syn_list:
                keyword_to_column[syn] = matched_target_col

    schema: Dict[str, Any] = {
        "csv_path": target_path,
        "mtime": mtime,
        "row_count": len(df),
        "columns": columns,
        "column_types": column_types,
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "date_columns": date_columns,
        "date_column": date_col,
        "date_range": date_range,
        "distinct_values": distinct_values,
        "value_lookup": value_lookup,
        "value_to_column": value_to_column,
        "keyword_to_column": keyword_to_column,
        "missing_counts": missing_counts,
    }

    _SCHEMA_CACHE[target_path] = (mtime, schema)
    return schema


def get_schema_summary_for_llm(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns a compact schema summary designed specifically for LLM query planning.
    Includes columns, data types, sample distinct categorical values, and date ranges.
    Consumes minimal tokens (<250 tokens).
    """
    s = inspect_schema(csv_path)
    summary: Dict[str, Any] = {
        "columns": s["columns"],
        "numeric_metrics": s["numeric_columns"],
        "date_column": s["date_column"],
        "date_range": s["date_range"],
        "categories": {},
    }
    for col, vals in s["distinct_values"].items():
        summary["categories"][col] = vals[:15]  # provide sample values
    return summary
