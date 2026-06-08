"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point for the Autonomous Customer Profiling & Financial Clustering Agent.

Run:
    python main.py

Environment variables required in .env:
    OPENAI_API_KEY=sk-...

What happens when you run this script
──────────────────────────────────────
1. The .env file is loaded so that OPENAI_API_KEY is available.
2. The ReAct agent is instantiated (LLM + tools + persona prompt).
3. A complex natural-language query is submitted:
   a) Preprocess the German Credit dataset.
   b) Train a mixed-data clustering model.
   c) Analyse a new applicant and produce a risk opinion.
4. The agent prints its full reasoning trace in real-time.
5. The final answer is printed in a formatted block.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# ── Load environment ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

if not os.getenv("OPENAI_API_KEY"):
    sys.exit(
        "❌  OPENAI_API_KEY is not set.\n"
        "    Create a .env file in the project root with:\n"
        "    OPENAI_API_KEY=sk-...\n"
    )

# ── Import agent (after env is loaded) ──────────────────────────────────────
from agents.react_agent import create_credit_agent  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────────────────
AGENT_QUERY = """
Load the dataset at 'data/german_credit_data.csv' and preprocess it.
Then, segment the customers using a mixed-data clustering algorithm
(Gower Distance + K-Medoids) with 4 clusters to discover hidden borrower
profiles.

Finally, a new customer just applied for credit:
{
  "Age": 24,
  "Sex": "male",
  "Job": 2,
  "Housing": "rent",
  "Saving accounts": "little",
  "Checking account": "moderate",
  "Credit amount": 4500,
  "Duration": 36,
  "Purpose": "business"
}

Analyse this customer:
  1. Which cluster do they belong to?
  2. What does that cluster tell us about their borrower profile?
  3. Should the bank be cautious about approving this application?

Provide a full, professional risk assessment.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _banner(title: str, width: int = 70) -> str:
    border = "═" * width
    pad = (width - len(title) - 2) // 2
    return f"\n╔{border}╗\n║{' ' * pad}{title}{' ' * (width - pad - len(title))}║\n╚{border}╝\n"


def _print_intermediate_steps(steps: list) -> None:
    """Pretty-print the (action, observation) pairs from the agent trace."""
    print(_banner("AGENT REASONING TRACE", width=68))
    for i, (action, observation) in enumerate(steps, start=1):
        print(f"  ── Step {i} ────────────────────────────────────────────────")
        print(f"  Action     : {action.tool}")
        # Truncate long action inputs for readability
        action_input_str = str(action.tool_input)
        if len(action_input_str) > 300:
            action_input_str = action_input_str[:300] + " … [truncated]"
        print(f"  Input      : {action_input_str}")
        # Truncate long observations
        obs_str = str(observation)
        if len(obs_str) > 800:
            obs_str = obs_str[:800] + "\n  … [observation truncated for display]"
        print(f"  Observation:\n{textwrap.indent(obs_str, '    ')}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(_banner("AUTONOMOUS CUSTOMER PROFILING & FINANCIAL CLUSTERING AGENT"))

    print("  Initialising ReAct agent…")
    agent_executor = create_credit_agent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        verbose=True,      # prints live Thought/Action/Obs during execution
        max_iterations=15,
    )

    print(_banner("SUBMITTING QUERY TO AGENT"))
    print(textwrap.indent(AGENT_QUERY, "  "))
    print()

    # ── Invoke ───────────────────────────────────────────────────────────────
    result = agent_executor.invoke({"input": AGENT_QUERY})

    # ── Intermediate steps ───────────────────────────────────────────────────
    _print_intermediate_steps(result.get("intermediate_steps", []))

    # ── Final answer ─────────────────────────────────────────────────────────
    print(_banner("FINAL AGENT RESPONSE"))
    final = result.get("output", "No output produced.")
    print(textwrap.fill(final, width=80, initial_indent="  ", subsequent_indent="  "))
    print()

    print("  ✅  Pipeline artefacts written to: models/")
    print("  ✅  Run complete.\n")


if __name__ == "__main__":
    main()
