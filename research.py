"""CLI entry point: python research.py "<topic>" [--researchers N] [--max-turns N] [--model NAME]"""
import argparse
import os
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from agent import build_graph

REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY")


def require_env_keys(env=os.environ):
    missing = [k for k in REQUIRED_KEYS if not env.get(k)]
    if missing:
        raise SystemExit(
            f"Missing required API keys: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parallel multi-researcher report agent (LangGraph)")
    parser.add_argument("topic", help="The topic to research")
    parser.add_argument("--researchers", type=int, default=3,
                        help="Number of researcher personas (default: 3)")
    parser.add_argument("--max-turns", type=int, default=2,
                        help="Q&A rounds per researcher (default: 2)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    return parser.parse_args(argv)


def print_team(payload):
    print("\nProposed researcher team:")
    for r in payload["researchers"]:
        print(f"  - {r['name']} ({r['role']}): {r['focus']}")


def main(argv=None):
    args = parse_args(argv)
    load_dotenv()
    require_env_keys()

    graph = build_graph(model=args.model)
    config = {"recursion_limit": 100,
              "configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"Researching: {args.topic}")
    print(f"  researchers={args.researchers}  max_turns={args.max_turns}  model={args.model}")
    result = graph.invoke({"topic": args.topic,
                           "num_researchers": args.researchers,
                           "max_turns": args.max_turns}, config)

    while "__interrupt__" in result:
        print_team(result["__interrupt__"][0].value)
        feedback = input("\nPress Enter to approve, or type feedback to regenerate: ").strip()
        if feedback:
            print("Regenerating the team with your feedback...")
        else:
            print("Approved. Researchers are working (this can take a minute)...")
        result = graph.invoke(Command(resume=feedback), config)

    print("\n" + "=" * 72 + "\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
