"""
agents/react_agent.py
─────────────────────────────────────────────────────────────────────────────
Builds and returns a fully configured LangChain ReAct AgentExecutor.

The agent persona is a Senior Financial Analyst who:
  • Orchestrates the full ML pipeline (preprocessing → clustering → inference)
  • Communicates findings in a structured, professional financial tone
  • Is transparent about its reasoning at every step (Thought → Action → Obs)

Architecture
────────────
  LLM            : ChatOpenAI  (gpt-4o-mini, temperature=0)
  Tools          : preprocess_german_credit_data
                   train_mixed_data_clustering
                   analyze_new_customer
  Prompt         : Hub-pulled ReAct prompt with a custom system persona
  Agent          : create_react_agent  (standard ReAct loop)
  Executor       : AgentExecutor  (verbose=True for full thought trace)
"""

from __future__ import annotations

from typing import Any

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from tools.clustering_tools import (
    analyze_new_customer,
    preprocess_german_credit_data,
    train_mixed_data_clustering,
)

# ─────────────────────────────────────────────────────────────────────────────
# System persona injected into the ReAct prompt
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PERSONA = """You are an Autonomous Senior Financial Analyst Agent at a
leading European retail bank. Your mission is to apply advanced unsupervised
machine learning techniques to segment banking customers into meaningful credit
profiles and to evaluate the risk of new loan applicants against those profiles.

You operate with the following principles:
1. SEQUENTIAL WORKFLOW – always preprocess the data before training, and
   always train before analysing a new customer.
2. TRANSPARENCY – explain every decision in your Thought steps so the
   compliance team can audit your reasoning.
3. PRECISION – use exact numbers from tool outputs; never round or estimate.
4. RISK AWARENESS – when in doubt, flag caution; the cost of a bad loan
   far exceeds the cost of a declined application.
5. PROFESSIONAL TONE – communicate findings as you would in an executive
   credit committee report.

You have access to three specialised data-science tools. Use them in order."""

# ─────────────────────────────────────────────────────────────────────────────
# Build the ReAct prompt
# ─────────────────────────────────────────────────────────────────────────────
def _build_react_prompt() -> PromptTemplate:
    """
    Construct the ReAct prompt template.

    We pull the canonical hwchase17/react template from LangChain Hub and
    prepend our persona string to the 'instructions' / system section.
    The template expects: {tools}, {tool_names}, {input}, {agent_scratchpad}.
    """
    # Standard ReAct template (Answer: prefix)
    template = _SYSTEM_PERSONA + """

You have access to the following tools:
{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    return PromptTemplate.from_template(template)


# ─────────────────────────────────────────────────────────────────────────────
# Public factory function
# ─────────────────────────────────────────────────────────────────────────────
def create_credit_agent(
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    verbose: bool = True,
    max_iterations: int = 15,
    handle_parsing_errors: bool = True,
) -> AgentExecutor:
    """
    Instantiate and return a configured AgentExecutor for credit profiling.

    Parameters
    ──────────
    model_name : str
        OpenAI model identifier (default: 'gpt-4o-mini').
    temperature : float
        Sampling temperature for the LLM (default: 0 for deterministic output).
    verbose : bool
        When True the full ReAct trace (Thought/Action/Observation) is printed.
    max_iterations : int
        Hard cap on ReAct cycles to prevent infinite loops.
    handle_parsing_errors : bool
        Pass malformed LLM output back to the model with a corrective prompt.

    Returns
    ───────
    AgentExecutor
        Ready-to-invoke agent executor.
    """
    # ── LLM ─────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    # ── Tools ────────────────────────────────────────────────────────────────
    tools = [
        preprocess_german_credit_data,
        train_mixed_data_clustering,
        analyze_new_customer,
    ]

    # ── Prompt ───────────────────────────────────────────────────────────────
    prompt = _build_react_prompt()

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # ── Executor ─────────────────────────────────────────────────────────────
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=handle_parsing_errors,
        return_intermediate_steps=True,
    )

    return executor
