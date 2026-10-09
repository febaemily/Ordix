"""
ORDIX — Orders Simplified
Data Loader Module: Robust loading, validation, and caching of order records from CSV.
"""

import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.csv")

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_name",
    "city",
    "product",
    "category",
    "quantity",
    "unit_price_inr",
    "total_inr",
    "payment_method",
    "status",
]


class DataLoadError(Exception):
    """Raised when orders CSV cannot be found, read, or validated."""
    pass


def load_orders(csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads and validates orders from the CSV file.
    Resolves file paths reliably relative to the module root.
    """
    target_path = csv_path if csv_path is not None else DEFAULT_CSV_PATH

    if not os.path.exists(target_path):
        raise DataLoadError(f"Orders dataset not found at path: {target_path}")

    orders: List[Dict[str, Any]] = []

    try:
        with open(target_path, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise DataLoadError("Orders CSV is empty or missing headers.")

            # Check header schema
            cleaned_fieldnames = [col.strip() for col in reader.fieldnames if col]
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in cleaned_fieldnames]
            if missing_cols:
                raise DataLoadError(f"Orders CSV is missing required columns: {missing_cols}")

            for row_idx, raw_row in enumerate(reader, start=2):
                # Clean row values
                row = {k.strip(): (v.strip() if v is not None else "") for k, v in raw_row.items() if k}

                order_id = row.get("order_id", "")
                if not order_id:
                    continue  # skip blank lines

                # Parse & validate numeric fields
                try:
                    quantity = int(row.get("quantity", 0))
                except (ValueError, TypeError):
                    quantity = 0

                try:
                    unit_price = float(row.get("unit_price_inr", 0.0))
                except (ValueError, TypeError):
                    unit_price = 0.0

                try:
                    total_inr = float(row.get("total_inr", 0.0))
                except (ValueError, TypeError):
                    total_inr = round(quantity * unit_price, 2)

                # Validate date format (YYYY-MM-DD)
                raw_date = row.get("order_date", "")
                validated_date = raw_date
                try:
                    # check parsing
                    datetime.strptime(raw_date, "%Y-%m-%d")
                except ValueError:
                    # fallback if non-empty
                    validated_date = raw_date

                cleaned_order = {
                    "order_id": order_id,
                    "order_date": validated_date,
                    "customer_name": row.get("customer_name", ""),
                    "city": row.get("city", ""),
                    "product": row.get("product", ""),
                    "category": row.get("category", ""),
                    "quantity": quantity,
                    "unit_price_inr": unit_price,
                    "total_inr": total_inr,
                    "payment_method": row.get("payment_method", ""),
                    "status": row.get("status", "").lower(),
                }
                orders.append(cleaned_order)

    except (OSError, UnicodeDecodeError) as e:
        raise DataLoadError(f"Failed to read orders file: {str(e)}") from e

    return orders


def get_dataset_summary(orders: Optional[List[Dict[str, Any]]] = None, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes quick schema & readiness statistics for health checks and system prompts.
    """
    if orders is None:
        orders = load_orders(csv_path)

    total_orders = len(orders)
    if total_orders == 0:
        return {
            "total_orders": 0,
            "date_range": {"min": None, "max": None},
            "statuses": [],
            "categories": [],
            "cities": [],
            "total_revenue_inr": 0.0,
        }

    valid_dates = [o["order_date"] for o in orders if o.get("order_date")]
    statuses: Set[str] = set(o["status"] for o in orders if o.get("status"))
    categories: Set[str] = set(o["category"] for o in orders if o.get("category"))
    cities: Set[str] = set(o["city"] for o in orders if o.get("city"))
    total_revenue = sum(o["total_inr"] for o in orders if o.get("status") != "cancelled")

    return {
        "total_orders": total_orders,
        "date_range": {
            "min": min(valid_dates) if valid_dates else None,
            "max": max(valid_dates) if valid_dates else None,
        },
        "statuses": sorted(list(statuses)),
        "categories": sorted(list(categories)),
        "cities": sorted(list(cities)),
        "total_revenue_inr": round(total_revenue, 2),
    }
