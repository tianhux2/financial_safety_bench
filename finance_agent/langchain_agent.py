"""LangChain-based Agent Orchestration for Finance Agent v2.

Adapted strictly according to the paper:
'Measuring Frontier Intelligence of Finance Agents' (arXiv: 2608.11683).

Key adaptations:
1. LangChain-based orchestration loop supporting any modern LLM backend.
2. Dynamic temporal anchor ({date} / query_date) in prompt and tools.
3. Finite budget enforcement: 200 tool calls and 300 seconds wall-clock time limit.
4. Runtime limit interception with mandatory 'submit_final_result' fallback.
5. End-to-end compatibility with FrontierFinance dataset (frontier_finance_public.jsonl)
   and grader (system_summaries.json).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .prompt import QUESTION_PROMPT, get_system_prompt
from .tools import (
    Calculator,
    EDGARSearch,
    ParseHtmlPage,
    PriceHistory,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    set_max_end_date,
)

logger = logging.getLogger("finance_agent.langchain")


class BudgetTracker:
    """Tracks tool calls and wall-clock time, enforcing paper-specified limits."""

    def __init__(
        self,
        max_tool_calls: int = 200,
        max_tool_call_time: float = 300.0,
    ):
        self.max_tool_calls = max_tool_calls
        self.max_tool_call_time = max_tool_call_time
        self.tool_call_count = 0
        self.start_time = time.time()
        self.limit_breached = False
        self.interception_issued = False
        self.current_turn = 0
        self.interception_turn: Optional[int] = None
        self.final_result: Optional[str] = None
        self.run_failed: bool = False

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def is_over_budget(self) -> bool:
        if self.tool_call_count >= self.max_tool_calls:
            return True
        if self.elapsed_seconds() >= self.max_tool_call_time:
            return True
        return False

    def check_and_record_tool_call(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """Checks if tool call is permitted.

        Returns (allowed, optional_interception_message).
        """
        if self.is_over_budget():
            self.limit_breached = True

        if self.limit_breached:
            if tool_name == "submit_final_result":
                return True, None
            if not self.interception_issued:
                self.interception_issued = True
                self.interception_turn = self.current_turn
                msg = (
                    "Tool call limit or time limit has reached. Your VERY NEXT action MUST "
                    "be to call 'submit_final_result' with the best answer you have gathered so far. "
                    "Do not make any other tool calls except 'submit_final_result'. "
                    "Failure to do so will cause the entire run to fail with no answer recorded."
                )
                return False, msg
            elif self.interception_turn is not None and self.current_turn == self.interception_turn:
                # Same turn/batch where interception was first issued: suppress without failing
                msg = (
                    "Tool call suppressed: Tool or time budget limit reached. You MUST call "
                    "'submit_final_result' in your immediate next response turn."
                )
                return False, msg
            else:
                self.run_failed = True
                msg = (
                    "RUN FAILED: The agent attempted an unauthorized tool call after the limit was "
                    "reached instead of calling 'submit_final_result'. The entire run is aborted."
                )
                return False, msg

        self.tool_call_count += 1
        return True, None


class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate, e.g. '(5000 - 3200) * 0.21'")


class SubmitFinalResultInput(BaseModel):
    final_result: str = Field(description="The final comprehensive answer to submit, including step-by-step reasoning, calculations, and dictionary of cited sources.")


class TavilyWebSearchInput(BaseModel):
    search_query: str = Field(description="Search query string to search on the public internet")
    start_date: Optional[str] = Field(default=None, description="Optional start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="Optional end date in YYYY-MM-DD format")


class EDGARSearchInput(BaseModel):
    search_query: str = Field(description="The search terms or keyword query to search SEC filings")
    start_date: Optional[str] = Field(default="1900-01-01", description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format")
    form_types: Optional[List[str]] = Field(default=None, description="Filters results to specified form types, e.g. ['10-K', '10-Q']")
    ciks: Optional[List[str]] = Field(default=None, description="Filters results to filings from specified CIKs, e.g. ['0000320193']")
    top_n_results: Optional[int] = Field(default=10, description="Number of results to return")


class ParseHtmlPageInput(BaseModel):
    url: str = Field(description="The URL of the HTML page to parse")
    key: str = Field(description="The data storage key to save the parsed text under, e.g. 'company_10k'")


class RetrieveInformationInput(BaseModel):
    prompt: str = Field(description="The prompt containing at least one storage key in {{key_name}} format to query the stored document")
    input_character_ranges: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional list of character range dicts")


class PriceHistoryInput(BaseModel):
    ticker: str = Field(description="Ticker symbol, e.g. 'AAPL', 'MSFT', 'btcusd'")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    asset_class: str = Field(default="equity", description="Asset class: 'equity', 'etf', 'crypto', 'fx'")


TOOL_SCHEMAS: Dict[str, Any] = {
    "calculator": CalculatorInput,
    "submit_final_result": SubmitFinalResultInput,
    "web_search": TavilyWebSearchInput,
    "edgar_search": EDGARSearchInput,
    "parse_html_page": ParseHtmlPageInput,
    "retrieve_information": RetrieveInformationInput,
    "price_history": PriceHistoryInput,
}


class LangChainFinanceAgent:
    """Finance Agent v2 using LangChain orchestration matching FrontierFinance paper."""

    def __init__(
        self,
        model: Any,
        as_of_date: str = "March 1, 2026",
        max_tool_calls: int = 200,
        max_tool_call_time: float = 300.0,
        log_level: int = logging.INFO,
        environment: Any = None,
        system_suffix: Optional[str] = None,
    ):
        self.model = model
        self.as_of_date = as_of_date
        self.max_tool_calls = max_tool_calls
        self.max_tool_call_time = max_tool_call_time
        # Optional adversary.ToolEnvironment; None keeps default (non-adversarial) behavior.
        self._environment = environment
        # Optional text appended to the system prompt. Used ONLY to build a deliberately
        # weakened agent for judge calibration; None keeps the paper-faithful prompt.
        self._system_suffix = system_suffix

        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("LangChainFinanceAgent")

        # Set tool date clamping
        set_max_end_date(as_of_date)

        # Initialize internal storage state
        self.state: Dict[str, str] = {}
        self.budget = BudgetTracker(
            max_tool_calls=max_tool_calls,
            max_tool_call_time=max_tool_call_time,
        )

        # Build tools
        self._raw_tools = {
            "calculator": Calculator(),
            "submit_final_result": SubmitFinalResult(),
            "web_search": TavilyWebSearch(),
            "edgar_search": EDGARSearch(),
            "parse_html_page": ParseHtmlPage(),
            "price_history": PriceHistory(),
            "retrieve_information": RetrieveInformation(llm=self._retrieve_llm_adapter),
        }
        self.langchain_tools = self._build_langchain_tools()

    async def _retrieve_llm_adapter(self, prompt: str) -> str:
        """Adapter allowing RetrieveInformation to query the model directly."""
        try:
            if hasattr(self.model, "ainvoke"):
                res = await self.model.ainvoke(prompt)
                return res.content if hasattr(res, "content") else str(res)
            elif callable(self.model):
                res = self.model(prompt)
                if asyncio.iscoroutine(res):
                    res = await res
                return getattr(res, "content", str(res))
            return str(self.model)
        except Exception as e:
            self.logger.warning(f"Error querying model for retrieval: {e}")
            return f"Error retrieving: {e}"

    def _build_langchain_tools(self) -> List[StructuredTool]:
        """Wrap the 6 specialized tools + submit_final_result into LangChain tools."""
        lc_tools = []

        for name, tool_obj in self._raw_tools.items():
            # Build wrapper closure for each tool
            def make_executor(t_name: str, raw_t: Any):
                async def _tool_func(**kwargs: Any) -> str:
                    allowed, intercept_msg = self.budget.check_and_record_tool_call(t_name)
                    if not allowed:
                        return intercept_msg or "Tool call not allowed."

                    try:
                        if self._environment is not None:
                            output = await self._environment.intercept(
                                t_name,
                                kwargs,
                                self.state,
                                self.logger,
                                run_raw=lambda: raw_t.execute(kwargs, self.state, self.logger),
                            )
                        else:
                            output_obj = await raw_t.execute(kwargs, self.state, self.logger)
                            output = output_obj.output
                        if t_name == "submit_final_result":
                            self.budget.final_result = kwargs.get("final_result", "")
                        return output
                    except Exception as e:
                        err = f"Execution error in {t_name}: {str(e)}"
                        self.logger.warning(err)
                        return err

                return _tool_func

            # Schema schemas for pydantic args
            tool_fn = make_executor(name, tool_obj)
            schema = TOOL_SCHEMAS.get(name)
            lc_tool = StructuredTool.from_function(
                coroutine=tool_fn,
                name=tool_obj.name,
                description=tool_obj.description,
                args_schema=schema,
            )
            lc_tools.append(lc_tool)

        return lc_tools

    async def run(
        self,
        question: str,
        query_id: Optional[str] = None,
        save_trajectory: bool = True,
        runs_dir: str | Path = "runs",
    ) -> str:
        """Executes the agent loop on a single question with budget monitoring and full trajectory recording."""
        start_ts = time.time()
        self.budget = BudgetTracker(
            max_tool_calls=self.max_tool_calls,
            max_tool_call_time=self.max_tool_call_time,
        )
        self.state.clear()
        if self._environment is not None:
            self._environment.reset()
        steps: List[Dict[str, Any]] = []

        system_prompt = get_system_prompt(
            date=self.as_of_date,
            max_tool_calls=self.max_tool_calls,
            max_tool_call_time=int(self.max_tool_call_time),
        )
        if self._system_suffix:
            system_prompt = f"{system_prompt}\n\n{self._system_suffix}"
        user_prompt = QUESTION_PROMPT.format(question=question)

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Bind tools to chat model
        if hasattr(self.model, "bind_tools"):
            model_with_tools = self.model.bind_tools(self.langchain_tools)
        else:
            model_with_tools = self.model

        tools_by_name = {t.name: t for t in self.langchain_tools}

        self.logger.info(f"Starting agent run for question (as of {self.as_of_date})...")

        final_text = ""
        while True:
            # Check abort condition
            if self.budget.run_failed:
                self.logger.error("Run aborted due to budget violation.")
                break

            if self.budget.final_result is not None:
                self.logger.info("Agent submitted final result successfully.")
                break

            # Invoke model
            self.budget.current_turn += 1
            try:
                ai_message: AIMessage = await model_with_tools.ainvoke(messages)
            except Exception as e:
                self.logger.error(f"Model invocation failed: {e}")
                final_text = f"Error: {e}"
                break

            messages.append(ai_message)
            if isinstance(ai_message.content, str):
                content_text = ai_message.content
            elif isinstance(ai_message.content, list):
                parts = []
                for item in ai_message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                content_text = "\n".join(parts).strip()
            else:
                content_text = str(ai_message.content)

            # Check if model requested tool calls
            tool_calls = getattr(ai_message, "tool_calls", [])
            if not tool_calls:
                final_text = content_text
                steps.append({
                    "step_index": len(steps) + 1,
                    "thought": content_text,
                    "tool_name": "direct_response",
                    "args": {},
                    "output": "Completed without further tool calls.",
                    "elapsed_ms": 0,
                })
                break

            # Execute tool calls
            for tc in tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                t_id = tc.get("id", f"call_{int(time.time()*1000)}")

                self.logger.info(f"Tool call [{self.budget.tool_call_count+1}]: {t_name}")

                t_start = time.time()
                if t_name in tools_by_name:
                    tool_instance = tools_by_name[t_name]
                    tool_output = await tool_instance.ainvoke(t_args)
                else:
                    tool_output = f"Unknown tool: {t_name}"
                t_elapsed_ms = round((time.time() - t_start) * 1000, 1)

                messages.append(ToolMessage(content=str(tool_output), tool_call_id=t_id))
                steps.append({
                    "step_index": len(steps) + 1,
                    "thought": content_text if content_text else "",
                    "tool_name": t_name,
                    "args": t_args,
                    "output": str(tool_output),
                    "elapsed_ms": t_elapsed_ms,
                })

                if self.budget.final_result is not None:
                    break
                if self.budget.run_failed:
                    break

        final_answer = self.budget.final_result or final_text or ""

        # Extract cited sources JSON if present
        sources: List[Dict[str, str]] = []
        source_match = re.search(r'\{\s*"sources"\s*:\s*\[.*?\]\s*\}', final_answer, re.DOTALL)
        if source_match:
            try:
                parsed = json.loads(source_match.group(0))
                sources = parsed.get("sources", [])
            except Exception:
                pass

        model_name = getattr(self.model, "model_name", getattr(self.model, "model", str(self.model)))
        trajectory_id = query_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        duration = round(time.time() - start_ts, 2)

        trajectory = {
            "id": trajectory_id,
            "question": question,
            "as_of_date": self.as_of_date,
            "model": model_name,
            "max_tool_calls": self.max_tool_calls,
            "max_tool_call_time": self.max_tool_call_time,
            "start_time": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "tool_call_count": self.budget.tool_call_count,
            "status": "completed" if self.budget.final_result is not None else ("failed" if self.budget.run_failed else "finished"),
            "final_answer": final_answer,
            "sources": sources,
            "steps": steps,
            "env_log": self._environment.log.to_list() if self._environment is not None else [],
        }
        self.last_trajectory = trajectory

        if save_trajectory:
            traj_path = save_trajectory_record(trajectory, runs_dir=runs_dir)
            self.logger.info(f"Trajectory saved to {traj_path}")

        return final_answer


def save_trajectory_record(trajectory: Dict[str, Any], runs_dir: str | Path = "runs") -> Path:
    """Saves complete trajectory to runs/trajectories/<id>.json and updates runs/index.json."""
    runs_path = Path(runs_dir)
    traj_dir = runs_path / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    traj_file = traj_dir / f"{trajectory['id']}.json"
    with open(traj_file, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)

    index_file = runs_path / "index.json"
    index_data: List[Dict[str, Any]] = []
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception:
            index_data = []

    summary_item = {
        "id": trajectory["id"],
        "question": trajectory["question"],
        "as_of_date": trajectory["as_of_date"],
        "model": trajectory["model"],
        "duration_seconds": trajectory["duration_seconds"],
        "tool_call_count": trajectory["tool_call_count"],
        "status": trajectory["status"],
        "timestamp": trajectory["start_time"],
        "file": str(traj_file.name),
    }

    index_data = [item for item in index_data if item.get("id") != trajectory["id"]]
    index_data.insert(0, summary_item)

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    return traj_file


async def run_frontier_finance_batch(
    dataset_path: str | Path,
    output_path: str | Path,
    model: Any,
    max_tool_calls: int = 200,
    max_tool_call_time: float = 300.0,
    max_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Runs Finance Agent v2 on FrontierFinance dataset and outputs system_summaries.json."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    queries: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            queries.append(item)

    if max_queries:
        queries = queries[:max_queries]

    print(f"Loaded {len(queries)} queries from {dataset_path}")

    summaries: List[Dict[str, str]] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                summaries = json.load(f)
            print(f"Resuming: found {len(summaries)} existing summaries in {output_path}")
        except Exception:
            summaries = []

    completed_ids = {s["query_id"] for s in summaries}

    for idx, q in enumerate(queries, 1):
        qid = q["query_id"]
        if qid in completed_ids:
            continue

        query_text = q["query"]
        query_date = q.get("query_date", "March 1, 2026")

        print(f"\n[{idx}/{len(queries)}] Processing {qid} (as of {query_date}): {query_text[:70]}...")

        agent = LangChainFinanceAgent(
            model=model,
            as_of_date=query_date,
            max_tool_calls=max_tool_calls,
            max_tool_call_time=max_tool_call_time,
        )

        try:
            answer = await agent.run(query_text, query_id=qid, save_trajectory=True)
        except Exception as e:
            print(f"Query {qid} failed with error: {e}")
            answer = ""

        summaries.append({
            "query_id": qid,
            "system_summary": answer,
        })

        # Save incrementally
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nBatch completed! All {len(summaries)} responses saved to {output_path}")
    return summaries
