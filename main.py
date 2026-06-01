"""
main.py

Entry point for the FreshMart data agent.

Before the first run:
    1. Make sure Ollama is running locally
    2. Pull the model once:                  ollama pull qwen2.5
    3. Generate sample data:                 uv run generate_data.py
    4. Start the agent:                      uv run main.py

Usage:
  uv run main.py                      # interactive REPL
  uv run main.py "your question here" # one-shot question
  uv run main.py --verbose "..."      # show debug info (SQL, snapshot used)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.path.exists("schema_registry.json"):
    print("Error: schema_registry.json not found.")
    print("  Run first:  uv run generate_data.py")
    sys.exit(1)

from agent import run_agent  # noqa: E402 (import after env check)
from opensearch_agent import run_agent_opensearch  # noqa: E402

EXAMPLE_QUESTIONS = [
    "How do almond milk prices compare between branch_a and branch_b?",
    "What products are out of stock in branch_c?",
    "What are your latest announcements?",
    "What is the total sales amount for branch_a?",
    "How do you do a backflip?",  # off-topic demo
]


def main():
    argv = sys.argv[1:]
    verbose = "--verbose" in argv

    backend = "sql"
    if "--backend" in argv:
        idx = argv.index("--backend")
        if idx + 1 >= len(argv):
            print("Error: --backend requires a value (sql or opensearch)")
            sys.exit(1)
        backend = argv[idx + 1].strip().lower()
        if backend not in {"sql", "opensearch"}:
            print("Error: --backend must be either 'sql' or 'opensearch'")
            sys.exit(1)

    filtered = []
    skip_next = False
    for i, a in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if a == "--verbose":
            continue
        if a == "--backend":
            skip_next = True
            continue
        filtered.append(a)
    args = filtered

    runner = run_agent if backend == "sql" else run_agent_opensearch

    if args:
        # One-shot mode
        query = " ".join(args)
        print(f"\nQ: {query}")
        print(f"A: {runner(query, verbose=verbose)}\n")
    else:
        # Interactive REPL
        print(
            "FreshMart Agent"
            f" [{backend}]  —  type 'quit' to exit, 'examples' for sample questions\n"
        )
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() == "examples":
                for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
                    print(f"  {i}. {q}")
                print()
                continue
            print(f"Agent: {runner(query, verbose=verbose)}\n")


if __name__ == "__main__":
    main()