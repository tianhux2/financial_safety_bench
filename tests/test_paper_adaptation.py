import time
import pytest
from finance_agent.prompt import get_system_prompt, SYSTEM_PROMPT_TEMPLATE
from finance_agent.langchain_agent import BudgetTracker, LangChainFinanceAgent
from finance_agent.tools import (
    Calculator,
    EDGARSearch,
    ParseHtmlPage,
    PriceHistory,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    set_max_end_date,
    MAX_END_DATE,
)


def test_system_prompt_adaptation():
    """Verify system prompt matches paper specifications."""
    prompt = get_system_prompt(date="2024-05-15", max_tool_calls=200, max_tool_call_time=300)
    assert "current date is 2024-05-15" in prompt
    assert "The maximum number of tool calls allowed (None for no limit): 200" in prompt
    assert "The maximum wall-clock seconds allowed for tool calling (None for no limit): 300" in prompt
    assert "IMPORTANT: If any tool response returns with instructions indicating that tool call limit" in prompt
    assert "submit_final_result" in prompt


def test_date_clamping():
    """Verify tool date cutoff can be dynamically adjusted."""
    set_max_end_date("2024-01-01")
    from finance_agent import tools
    assert tools.MAX_END_DATE == "2024-01-01"
    set_max_end_date("2026-03-01")


def test_budget_tracker_limits():
    """Verify tool call limits and enforcement."""
    tracker = BudgetTracker(max_tool_calls=3, max_tool_call_time=10.0)

    # First 3 calls should pass
    ok1, msg1 = tracker.check_and_record_tool_call("web_search")
    assert ok1 is True
    assert msg1 is None

    ok2, msg2 = tracker.check_and_record_tool_call("calculator")
    assert ok2 is True

    ok3, msg3 = tracker.check_and_record_tool_call("price_history")
    assert ok3 is True

    # 4th call exceeds limit of 3
    ok4, msg4 = tracker.check_and_record_tool_call("web_search")
    assert ok4 is False
    assert "Tool call limit or time limit has reached" in msg4
    assert "submit_final_result" in msg4

    # submit_final_result should still be allowed after limit is reached
    ok_sub, msg_sub = tracker.check_and_record_tool_call("submit_final_result")
    assert ok_sub is True


def test_budget_tracker_time_limit():
    """Verify time limit enforcement."""
    tracker = BudgetTracker(max_tool_calls=100, max_tool_call_time=0.01)
    time.sleep(0.05)
    assert tracker.is_over_budget() is True

    ok, msg = tracker.check_and_record_tool_call("web_search")
    assert ok is False
    assert "Tool call limit or time limit has reached" in msg


def test_agent_tool_building():
    """Verify LangChain agent wraps all 6 specialized tools + submit_final_result."""
    class DummyModel:
        def bind_tools(self, tools):
            return self

    agent = LangChainFinanceAgent(
        model=DummyModel(),
        as_of_date="2024-05-15",
        max_tool_calls=200,
        max_tool_call_time=300.0,
    )

    tool_names = [t.name for t in agent.langchain_tools]
    assert "calculator" in tool_names
    assert "edgar_search" in tool_names
    assert "price_history" in tool_names
    assert "web_search" in tool_names
    assert "parse_html_page" in tool_names
    assert "retrieve_information" in tool_names
    assert "submit_final_result" in tool_names
    assert len(tool_names) == 7


if __name__ == "__main__":
    test_system_prompt_adaptation()
    print("test_system_prompt_adaptation passed!")
    test_date_clamping()
    print("test_date_clamping passed!")
    test_budget_tracker_limits()
    print("test_budget_tracker_limits passed!")
    test_budget_tracker_time_limit()
    print("test_budget_tracker_time_limit passed!")
    test_agent_tool_building()
    print("test_agent_tool_building passed!")
    print("\nALL 5 TESTS PASSED SUCCESSFULLY!")

