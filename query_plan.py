"""
ORDIX — Orders Simplified
Query Plan Module: Safe, structured query specifications and schema validation.
Never executes arbitrary code or unvalidated user/model operations.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


ALLOWED_OPERATORS = {
    "==", "!=", "in", "not in", ">", ">=", "<", "<=", "between", "contains"
}

ALLOWED_AGGREGATIONS = {
    "sum", "count", "mean", "avg", "min", "max", "nunique"
}

ALLOWED_INTENTS = {
    "aggregate", "group_by", "search", "ranking", "comparison", "distinct_values", "clarification"
}


class FilterSpec(BaseModel):
    """Declarative filter condition applied to DataFrame rows."""
    column: str = Field(..., description="Target column name")
    operator: str = Field(..., description="Comparison operator: ==, !=, in, not in, >, >=, <, <=, between, contains")
    value: Any = Field(..., description="Target value, list of values, or range")

    @field_validator("operator")
    @classmethod
    def validate_op(cls, v: str) -> str:
        op = str(v).strip().lower()
        if op == "=":
            op = "=="
        if op not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported filter operator: '{v}'. Must be one of {ALLOWED_OPERATORS}")
        return op


class AggregationMetric(BaseModel):
    """Specification of an aggregate metric to compute."""
    column: str = Field(..., description="Column on which aggregation is performed")
    agg: str = Field(..., description="Aggregation function: sum, count, mean, min, max, nunique")
    alias: Optional[str] = Field(None, description="Human-readable label for output tables")

    @field_validator("agg")
    @classmethod
    def validate_agg(cls, v: str) -> str:
        a = str(v).strip().lower()
        if a == "avg":
            a = "mean"
        if a not in ALLOWED_AGGREGATIONS:
            raise ValueError(f"Unsupported aggregation: '{v}'. Must be one of {ALLOWED_AGGREGATIONS}")
        return a


class QueryPlan(BaseModel):
    """
    Validated, deterministic query plan ready for Python/pandas execution.
    Contains zero arbitrary code or unverified instructions.
    """
    intent: str = Field(..., description="Primary query intent: aggregate, group_by, search, ranking, comparison, distinct_values, clarification")
    filters: List[FilterSpec] = Field(default_factory=list, description="List of filter conditions")
    group_by: Optional[List[str]] = Field(None, description="Columns to group by for breakdown")
    metrics: List[AggregationMetric] = Field(default_factory=list, description="Metrics to compute")
    sort_by: Optional[str] = Field(None, description="Column or metric to sort results by")
    sort_ascending: bool = Field(False, description="Ascending sort order if True, descending if False")
    limit: Optional[int] = Field(None, description="Maximum number of rows to return")
    include_grand_total: bool = Field(False, description="Include overall dataset total row in group_by results")
    calculate_percentage: bool = Field(False, description="Calculate percentage share of total for group rows")
    revenue_definition: str = Field("delivered", description="Revenue definition: 'delivered' (default), 'gross' (all orders), or 'net' (excl cancelled)")
    clarification_prompt: Optional[str] = Field(None, description="Clarification message if query is ambiguous")
    title: Optional[str] = Field(None, description="Optional presentation title for the generated answer")

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        i = str(v).strip().lower()
        if i not in ALLOWED_INTENTS:
            raise ValueError(f"Unknown query plan intent: '{v}'. Must be one of {ALLOWED_INTENTS}")
        return i


def validate_plan(plan: QueryPlan, schema: Dict[str, Any]) -> QueryPlan:
    """
    Validates every column, filter value, and metric against the live CSV schema.
    Normalizes casing and raises ValueError on invalid fields.
    """
    valid_cols = set(schema["columns"])
    # Allow virtual time dimensions
    valid_cols.update(["month", "year", "year_month", "month_name", "order_date_str"])

    # Validate & normalize filters
    for f in plan.filters:
        col = f.column.strip()
        # Check column synonym if not direct match
        if col not in valid_cols:
            col_lower = col.lower()
            if col_lower in schema["keyword_to_column"]:
                f.column = schema["keyword_to_column"][col_lower]
                col = f.column
            else:
                raise ValueError(f"Invalid column '{f.column}' in filter. Valid columns are: {sorted(list(valid_cols))}")

        # Value normalization for categorical columns
        if col in schema["value_lookup"]:
            lookup = schema["value_lookup"][col]
            if isinstance(f.value, str):
                v_lower = f.value.strip().lower()
                if v_lower in lookup:
                    f.value = lookup[v_lower]
            elif isinstance(f.value, list):
                norm_list = []
                for item in f.value:
                    item_str = str(item).strip().lower()
                    norm_list.append(lookup.get(item_str, item))
                f.value = norm_list

    # Validate group_by
    if plan.group_by:
        norm_groups = []
        for g in plan.group_by:
            g_clean = g.strip()
            if g_clean in valid_cols:
                norm_groups.append(g_clean)
            elif g_clean.lower() in schema["keyword_to_column"]:
                norm_groups.append(schema["keyword_to_column"][g_clean.lower()])
            elif g_clean.lower() in ["month", "year"]:
                norm_groups.append(g_clean.lower())
            else:
                raise ValueError(f"Invalid group_by column '{g}'. Valid columns: {sorted(list(valid_cols))}")
        plan.group_by = norm_groups

    # Validate metrics
    for m in plan.metrics:
        col = m.column.strip()
        if col not in valid_cols and col not in ["*", "order_id"]:
            col_lower = col.lower()
            if col_lower in schema["keyword_to_column"]:
                m.column = schema["keyword_to_column"][col_lower]
            else:
                raise ValueError(f"Invalid column '{m.column}' in metric. Valid columns: {sorted(list(valid_cols))}")

    # Validate sort_by
    if plan.sort_by:
        s_col = plan.sort_by.strip()
        metric_aliases = {m.alias for m in plan.metrics if m.alias}
        metric_cols = {m.column for m in plan.metrics}
        if s_col not in valid_cols and s_col not in metric_aliases and s_col not in metric_cols:
            s_lower = s_col.lower()
            if s_lower in schema["keyword_to_column"]:
                plan.sort_by = schema["keyword_to_column"][s_lower]

    return plan
