"""CLI entrypoint for running Finance Agent v2 (FrontierFinance paper adapted)."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from .langchain_agent import LangChainFinanceAgent, run_frontier_finance_batch


def get_model(provider: str, model_name: str, reasoning_effort: str | None = "none"):
    """Instantiate a chat model via LangChain."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {}
        if reasoning_effort and reasoning_effort != "default":
            kwargs["reasoning_effort"] = reasoning_effort
        return ChatOpenAI(model=model_name, **kwargs)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name)
    else:
        # Fallback to langchain.chat_models
        from langchain.chat_models import init_chat_model
        return init_chat_model(model_name, model_provider=provider)


async def main_async():
    parser = argparse.ArgumentParser(
        description="Run Finance Agent v2 adapted per FrontierFinance paper"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="A single question to research and answer",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="March 1, 2026",
        help="Temporal anchor date (YYYY-MM-DD or 'March 1, 2026')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to FrontierFinance public dataset (e.g., frontier_finance_public.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="system_summaries.json",
        help="Output JSON path for generated summaries (for grading)",
    )
    default_provider = "openai" if os.getenv("OPENAI_API_KEY") else ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai")
    default_model = os.getenv("DEFAULT_MODEL", "gpt-6-luna" if default_provider == "openai" else "claude-3-7-sonnet-20250219")

    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=["anthropic", "openai", "google"],
        help=f"Model provider (default: {default_provider})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model name (default: {default_model})",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=200,
        help="Max tool calls allowed (paper default: 200)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Max wall-clock seconds allowed (paper default: 300)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries from dataset to run",
    )

    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high", "default"],
        help="OpenAI reasoning effort level (none disables internal encrypted tokens so thoughts output in visible text)",
    )

    args = parser.parse_args()

    model = get_model(args.provider, args.model, reasoning_effort=args.reasoning_effort)

    if args.dataset:
        await run_frontier_finance_batch(
            dataset_path=args.dataset,
            output_path=args.output,
            model=model,
            max_tool_calls=args.max_tool_calls,
            max_tool_call_time=args.max_time,
            max_queries=args.limit,
        )
    elif args.question:
        agent = LangChainFinanceAgent(
            model=model,
            as_of_date=args.date,
            max_tool_calls=args.max_tool_calls,
            max_tool_call_time=args.max_time,
        )
        answer = await agent.run(args.question, save_trajectory=True)
        safe_print("\n" + "=" * 50)
        safe_print("FINAL ANSWER:")
        safe_print("=" * 50)
        safe_print(answer)
        if agent.last_trajectory:
            safe_print(f"\n[Full Trajectory Log Saved]: runs/trajectories/{agent.last_trajectory['id']}.json")
    else:
        parser.print_help()


def safe_print(text: str):
    try:
        print(text)
    except Exception:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
