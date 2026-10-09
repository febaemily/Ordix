"""
Tests for Data Loader Module
Validates CSV loading, schema checking, edge cases, and summaries.
"""

import os
import tempfile
import pytest

from data_loader import (
    DataLoadError,
    get_dataset_summary,
    load_orders,
    REQUIRED_COLUMNS,
)


def test_load_real_orders_csv():
    """Verify the real orders.csv loads successfully with 60 records."""
    orders = load_orders()
    assert len(orders) == 60
    assert isinstance(orders, list)

    first_order = orders[0]
    for col in REQUIRED_COLUMNS:
        assert col in first_order, f"Column {col} missing from loaded order"

    assert first_order["order_id"] == "ORD-1001"
    assert first_order["customer_name"] == "Rahul Sharma"
    assert first_order["city"] == "Hyderabad"
    assert first_order["product"] == "Wireless Mouse"
    assert first_order["category"] == "Electronics"
    assert first_order["quantity"] == 2
    assert first_order["unit_price_inr"] == 799.0
    assert first_order["total_inr"] == 1598.0
    assert first_order["payment_method"] == "Debit Card"
    assert first_order["status"] == "delivered"


def test_dataset_summary():
    """Verify summary metadata calculation for the dataset."""
    summary = get_dataset_summary()
    assert summary["total_orders"] == 60
    assert summary["date_range"]["min"] == "2026-06-01"
    assert summary["date_range"]["max"] == "2026-09-28"
    assert "delivered" in summary["statuses"]
    assert "cancelled" in summary["statuses"]
    assert "Electronics" in summary["categories"]
    assert "Hyderabad" in summary["cities"]
    assert summary["total_revenue_inr"] > 0


def test_missing_csv_file_raises_error():
    """Verify loading from nonexistent path raises DataLoadError."""
    with pytest.raises(DataLoadError) as exc_info:
        load_orders("nonexistent_orders_file_12345.csv")
    assert "not found" in str(exc_info.value).lower()


def test_corrupted_csv_headers_raises_error():
    """Verify loading a CSV with missing required columns raises DataLoadError."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write("order_id,customer_name\nORD-999,Test User\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(DataLoadError) as exc_info:
            load_orders(tmp_path)
        assert "missing required columns" in str(exc_info.value).lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_empty_csv_raises_error():
    """Verify loading an empty CSV raises DataLoadError."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write("")
        tmp_path = tmp.name

    try:
        with pytest.raises(DataLoadError) as exc_info:
            load_orders(tmp_path)
        assert "empty or missing headers" in str(exc_info.value).lower()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
