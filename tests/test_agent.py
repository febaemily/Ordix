"""
Tests for AI Agent Module
Tests tool loop orchestration, argument handling, configuration checks, and safety limits.
"""

from unittest.mock import MagicMock
import pytest

from agent import (
    AgentConfigurationError,
    AgentExecutionError,
    execute_backend_tool,
    get_gemini_client,
    run_agent_turn,
)


def test_missing_api_key_raises_configuration_error(monkeypatch):
    """Verify that omitting GEMINI_API_KEY raises AgentConfigurationError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AgentConfigurationError) as exc_info:
        get_gemini_client()
    assert "not configured" in str(exc_info.value).lower()


def test_placeholder_api_key_raises_configuration_error(monkeypatch):
    """Verify that leaving the placeholder key raises AgentConfigurationError."""
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(AgentConfigurationError) as exc_info:
        get_gemini_client()
    assert "not configured" in str(exc_info.value).lower()


def test_execute_backend_tool_search_orders():
    """Verify execute_backend_tool correctly routes to search_orders."""
    res = execute_backend_tool("search_orders", {"order_id": "ORD-1001"})
    assert res["success"] is True
    assert res["total_matched"] == 1
    assert res["orders"][0]["order_id"] == "ORD-1001"


def test_execute_backend_tool_calculate_statistics():
    """Verify execute_backend_tool correctly routes to calculate_order_statistics."""
    res = execute_backend_tool("calculate_order_statistics", {"operation": "summary"})
    assert res["success"] is True
    assert res["data"]["total_orders"] == 60


def test_execute_backend_tool_unknown_name():
    """Verify execute_backend_tool safely handles unknown tool names."""
    res = execute_backend_tool("arbitrary_execution_tool", {})
    assert res["success"] is False
    assert "unknown tool name" in res["error"].lower()


def test_run_agent_turn_with_mocked_gemini_workflow():
    """
    Verify complete tool-calling loop:
    Turn 1: Model requests calculate_order_statistics('order_counts_by_status')
    Turn 2: Tool executes, returns counts, Model produces grounded answer.
    """
    mock_client = MagicMock()

    # Create mock tool call
    mock_func_call = MagicMock()
    mock_func_call.name = "calculate_order_statistics"
    mock_func_call.args = {"operation": "order_counts_by_status"}

    mock_resp_1 = MagicMock()
    mock_resp_1.function_calls = [mock_func_call]
    mock_resp_1.candidates = [MagicMock(content=MagicMock())]
    mock_resp_1.text = None

    # Response 2 provides the final natural language summary
    mock_resp_2 = MagicMock()
    mock_resp_2.function_calls = None
    mock_resp_2.text = "There are 48 delivered orders and 7 cancelled orders in your dataset."

    mock_client.models.generate_content.side_effect = [mock_resp_1, mock_resp_2]

    result = run_agent_turn(
        message="How many orders were cancelled?",
        client=mock_client,
        model_name="gemini-3.8-flash",
    )

    assert result["answer"] == "There are 48 delivered orders and 7 cancelled orders in your dataset."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "calculate_order_statistics"
    assert result["tool_calls"][0]["result"]["success"] is True
    assert result["metadata"]["iterations"] == 2


def test_run_agent_turn_iteration_limit():
    """Verify tool iteration limit prevents infinite loops."""
    mock_client = MagicMock()

    mock_func_call = MagicMock()
    mock_func_call.name = "calculate_order_statistics"
    mock_func_call.args = {"operation": "summary"}

    # Repeatedly return tool requests
    infinite_resp = MagicMock()
    infinite_resp.function_calls = [mock_func_call]
    infinite_resp.candidates = [MagicMock(content=MagicMock())]
    infinite_resp.text = None

    summary_resp = MagicMock()
    summary_resp.text = "Summary reached after iteration limit."

    # First 5 calls return tool request, 6th call returns summary
    mock_client.models.generate_content.side_effect = [
        infinite_resp,
        infinite_resp,
        infinite_resp,
        infinite_resp,
        infinite_resp,
        summary_resp,
    ]

    result = run_agent_turn(
        message="Keep analyzing orders",
        client=mock_client,
    )

    assert result["metadata"]["iterations"] == 5
    assert "notice" in result["metadata"]
    assert len(result["tool_calls"]) == 5
