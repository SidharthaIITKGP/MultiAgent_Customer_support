"""
agents/action_agent.py

THE CORE AGENTIC PIECE.

Implements an explicit ReAct (Reason + Act) loop using LangChain's tool-calling:
  - Tools are @tool-decorated functions that call mock_backend.py
  - LLM bound via model.bind_tools() — it decides which tool to call and with what args
  - Each iteration: Thought (from AI message) → Action (tool call) → Observation (tool result)
  - Every step is stored in react_trace for the demo
  - Max 5 iterations; on tool failure, the next Thought explicitly reasons about the failure
  - After the loop, the LLM evaluates whether knowledge context was sufficient (sets insufficient_context)

Run in isolation to test:
    python -m agents.action_agent
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm_utils import build_llm, extract_text
from agents.state import ReActStep, SupportTicketState, ToolCallRecord
from logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
MAX_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Tools — @tool decorated, call mock_backend.py via HTTP
# ---------------------------------------------------------------------------

def _call_backend(method: str, path: str, **kwargs) -> dict:
    """Shared HTTP helper. Returns dict with success/error info."""
    url = f"{BACKEND_URL}{path}"
    try:
        response = httpx.request(method, url, timeout=10, **kwargs)
        if response.status_code < 300:
            return response.json()
        else:
            # Return structured error so the LLM can reason about it
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            return {
                "success": False,
                "error": True,
                "status_code": response.status_code,
                "detail": detail,
            }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": True,
            "detail": f"Cannot connect to backend at {BACKEND_URL}. Is mock_backend.py running?",
        }
    except Exception as e:
        return {"success": False, "error": True, "detail": str(e)}


@tool
def check_order_status(order_id: str) -> dict:
    """
    Check the status, delivery date, refund eligibility, and amount for an order.
    Use this BEFORE processing a refund to verify the order exists and is eligible.
    Returns: order_id, user_id, status, amount, delivery_date, refund_eligible, refunded
    """
    return _call_backend("GET", f"/order/{order_id}")


@tool
def process_refund(order_id: str, user_id: str, amount: float, reason: str = "customer_request") -> dict:
    """
    Process a refund for an order. The account must not be locked, the order must
    be delivered and eligible, and must not have been refunded already.
    Returns: success, refund_id, eta_days, or error detail explaining why it failed.
    """
    return _call_backend(
        "POST",
        "/refund",
        json={"order_id": order_id, "user_id": user_id, "amount": amount, "reason": reason},
    )


@tool
def reset_password(user_id: str, method: str = "email") -> dict:
    """
    Send a password reset link to the user via email or sms.
    Works even when the account is locked.
    Returns: success, sent_to, expires_in_minutes, or error detail.
    """
    return _call_backend(
        "POST",
        "/reset-password",
        json={"user_id": user_id, "method": method},
    )


@tool
def check_account_status(user_id: str) -> dict:
    """
    Retrieve account details including status (active/locked), plan, name, and email.
    Use this to diagnose why an action failed (e.g., account locked prevents refund).
    Returns: user_id, name, email, status, plan, or error detail.
    """
    return _call_backend("GET", f"/account/{user_id}")


@tool
def update_account(user_id: str, field: str, value: str) -> dict:
    """
    Update an account field. Supported fields: email, plan, name.
    Fails if the account is locked. 
    Returns: success, user_id, updated field/value, or error detail.
    """
    return _call_backend(
        "PATCH",
        f"/account/{user_id}",
        json={"field": field, "value": value},
    )


ALL_TOOLS = [
    check_order_status,
    process_refund,
    reset_password,
    check_account_status,
    update_account,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
# Resolve model-supplied names at runtime while keeping one tool list for model
# binding and execution.

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ACTION_SYSTEM_PROMPT = """You are the Action Agent in a multi-agent customer support system.
Your role is to take concrete actions on behalf of the customer using the available tools.

You have access to these tools:
- check_order_status: Look up order details and refund eligibility
- process_refund: Issue a refund for an eligible order
- reset_password: Send a password reset email/SMS
- check_account_status: Look up account status (active/locked), plan, email
- update_account: Update account fields (email, plan, name)

Context from previous agents:
- Classification: {classification} | Urgency: {urgency} | Sentiment: {sentiment}
- Intake reasoning: {intake_reasoning}
- Retrieved policy context: {retrieved_context}
- Knowledge agent reasoning: {knowledge_reasoning}

Customer ticket: {ticket_text}
Customer user_id: {user_id}

CRITICAL INSTRUCTIONS:
1. Use your tools to resolve the customer's issue. Do not just explain — ACT.
2. If a tool call FAILS, your next message must explicitly reason about WHY it failed
   and what different action you should try (e.g., if refund fails with "account locked",
   check the account status first, then decide next steps).
3. After completing your actions, evaluate: did the retrieved policy context give you
   everything you needed? If you needed information that wasn't in the context
   (e.g., specific eligibility rules, policy details), explicitly state this.
4. Be concrete. Include order IDs, amounts, and results in your final summary.
5. Each round-trip is expensive — when you're confident an action should succeed (the
   ticket clearly describes an eligible request), call multiple independent tools in
   the SAME response rather than checking status, waiting a full turn, then acting
   in a second turn. The backend enforces eligibility either way, so batching costs
   nothing when you're right and just becomes your next failure to reason about when
   you're not. Only go step-by-step when a later call genuinely depends on seeing an
   earlier result first (e.g., diagnosing why something failed).

Think step by step. Use tools. Reason about failures explicitly."""


# ---------------------------------------------------------------------------
# ReAct loop implementation
# ---------------------------------------------------------------------------

def _extract_thought(message: AIMessage, tool_calls: list | None = None) -> str:
    """
    Extract the text portion of an AIMessage as the 'Thought'.
    Gemini often returns empty text when making tool calls (unlike Claude).
    In that case, we synthesize a thought from the tool call intent.
    """
    text = extract_text(message.content)
    if text:
        return text

    # Gemini gave no text — synthesize from tool call names for a readable trace
    tc_list = tool_calls or (message.tool_calls if hasattr(message, 'tool_calls') else [])
    if tc_list:
        tool_names = ", ".join(tc["name"] for tc in tc_list)
        args_preview = ", ".join(
            f"{k}={repr(v)}" for tc in tc_list for k, v in list(tc.get("args", {}).items())[:2]
        )
        return f"Calling {tool_names}({args_preview}) to gather information needed to resolve this issue."
    return "[reasoning implicit — proceeding to action]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_action_agent(state: SupportTicketState, config=None) -> dict:
    """
    LangGraph node function.
    Runs the ReAct loop and returns state updates.

    `config` intentionally has no type annotation: LangGraph only injects the
    runtime RunnableConfig into a node if this param is unannotated or typed
    exactly RunnableConfig/Optional[RunnableConfig] (checked by literal string
    match, since this module uses `from __future__ import annotations`).
    """
    llm = build_llm(
        config,
        model=MODEL_NAME,
        temperature=0,
        max_output_tokens=4096,
    ).bind_tools(ALL_TOOLS)

    # Build the system prompt with context from previous agents
    system_prompt = ACTION_SYSTEM_PROMPT.format(
        classification=state.get("classification", "unknown"),
        urgency=state.get("urgency", "medium"),
        sentiment=state.get("sentiment", "neutral"),
        intake_reasoning=state.get("intake_reasoning", "No intake reasoning available."),
        retrieved_context=state.get("retrieved_context", "No context retrieved."),
        knowledge_reasoning=state.get("knowledge_reasoning", "No knowledge reasoning available."),
        ticket_text=state.get("ticket_text", ""),
        user_id=state.get("user_id", "unknown"),
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please resolve this customer's issue: {state.get('ticket_text', '')}"),
    ]

    # Accumulated state for this node
    new_trace_steps: list[ReActStep] = []
    tool_calls_made: list[ToolCallRecord] = []
    tool_retry_count = 0
    action_result = "No action taken."
    action_success = False
    insufficient_context = False
    loop_completed_cleanly = False  # True only if the model gave a final answer
                                     # (no tool calls) rather than hitting MAX_ITERATIONS

    logger.info("action_agent start model=%s max_iterations=%d", MODEL_NAME, MAX_ITERATIONS)

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.debug("action iteration %d/%d", iteration, MAX_ITERATIONS)

        # Get LLM response
        response: AIMessage = llm.invoke(messages)
        messages.append(response)

        thought = _extract_thought(response, response.tool_calls)
        logger.debug("THOUGHT: %s", thought)

        # Check if LLM wants to call tools
        if not response.tool_calls:
            # No tool calls → LLM is done reasoning, this is the final answer
            action = "Final answer / no further tool calls"
            observation = thought  # The thought IS the final observation

            step = ReActStep(
                agent="action",
                iteration=iteration,
                thought=thought,
                action=action,
                observation=observation,
                timestamp=_now(),
            )
            new_trace_steps.append(step)
            logger.debug("ACTION: %s", action)
            logger.debug("OBSERVATION: %s...", observation[:200])
            action_result = thought
            action_success = True  # Completed without error
            loop_completed_cleanly = True
            break

        # Process tool calls
        tool_results_for_next_message = []

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            action_str = f"{tool_name}({json.dumps(tool_args)})"
            logger.debug("ACTION: %s", action_str)

            # Execute the tool
            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                raw_result = {"error": True, "detail": f"Unknown tool: {tool_name}"}
                success = False
            else:
                try:
                    raw_result = tool_fn.invoke(tool_args)
                    success = not raw_result.get("error", False)
                except Exception as e:
                    raw_result = {"error": True, "detail": str(e)}
                    success = False

            if not success:
                tool_retry_count += 1

            observation_str = json.dumps(raw_result, indent=2)
            logger.debug("OBSERVATION: %s", observation_str[:300])

            # Record the tool call
            tool_calls_made.append(ToolCallRecord(
                tool_name=tool_name,
                arguments=tool_args,
                result=raw_result,
                success=success,
                error_message=raw_result.get("detail") if not success else None,
            ))

            # Record the ReAct step
            step = ReActStep(
                agent="action",
                iteration=iteration,
                thought=thought,
                action=action_str,
                observation=observation_str,
                timestamp=_now(),
            )
            new_trace_steps.append(step)

            # Prepare tool result message for the LLM
            tool_results_for_next_message.append(
                ToolMessage(
                    content=observation_str,
                    tool_call_id=tc["id"],
                )
            )

            # Update action_result and action_success based on last tool call
            if success:
                action_success = True
                action_result = f"Tool '{tool_name}' succeeded: {json.dumps(raw_result)}"
            else:
                action_success = False
                action_result = f"Tool '{tool_name}' failed: {raw_result.get('detail', 'unknown error')}"

        messages.extend(tool_results_for_next_message)

    else:
        # Max iterations reached without breaking
        logger.warning("max iterations (%d) reached", MAX_ITERATIONS)
        action_result = f"Max iterations reached. Last status: {action_result}"

    # ---------------------------------------------------------------------------
    # Post-loop: evaluate whether retrieved context was sufficient
    # ---------------------------------------------------------------------------
    # This costs one extra sequential LLM round-trip, so skip it in the common
    # case where nothing indicates a problem: the loop ended with a clean final
    # answer (not cut off by MAX_ITERATIONS) and every tool call that was made
    # succeeded (vacuously true if none were made — e.g. a ticket that needed
    # no tools at all). If the task actually got done cleanly, there's no
    # realistic signal that policy context was missing; if anything failed or
    # the loop ran out of iterations, we still ask the LLM to reflect on why.
    all_tools_succeeded = all(tc.success for tc in tool_calls_made)

    if loop_completed_cleanly and all_tools_succeeded:
        insufficient_context = False
        context_eval_reason = "Skipped: task completed cleanly with no tool failures."
    else:
        # Ask the LLM to reflect on whether the knowledge context covered what it needed
        context_eval_prompt = f"""Reflect on the actions you just took.

The knowledge_agent retrieved this context:
---
{state.get('retrieved_context', 'No context retrieved.')}
---

Knowledge reasoning: {state.get('knowledge_reasoning', 'N/A')}

Question: Was this retrieved context sufficient for you to complete the task?
- Did you need policy information (e.g., refund eligibility rules, eligibility windows,
  account policy) that was NOT present in the retrieved context?
- Or did you have everything you needed?

Respond with EXACTLY this JSON format:
{{
  "insufficient_context": true/false,
  "reason": "brief explanation of what was missing (if insufficient) or confirmation (if sufficient)"
}}"""

        eval_response = build_llm(config, model=MODEL_NAME, temperature=0, max_output_tokens=512).invoke([
            SystemMessage(content="You are evaluating whether retrieved context was sufficient."),
            HumanMessage(content=context_eval_prompt),
        ])

        eval_text = extract_text(eval_response.content)
        insufficient_context = False
        context_eval_reason = "Context evaluation not parseable."

        try:
            # Strip markdown code fences (Gemini wraps JSON in ```json ... ```)
            cleaned = re.sub(r'```(?:json)?\s*', '', eval_text).strip().rstrip('`').strip()
            json_match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
            if json_match:
                eval_json = json.loads(json_match.group())
                insufficient_context = bool(eval_json.get("insufficient_context", False))
                context_eval_reason = eval_json.get("reason", "")
        except Exception as e:
            logger.warning("context sufficiency eval unparseable: %s", e)

    # Add a final reflection step to the trace
    new_trace_steps.append(ReActStep(
        agent="action",
        iteration=MAX_ITERATIONS + 1,  # Denotes post-loop reflection
        thought=f"Context sufficiency check: insufficient_context={insufficient_context}. {context_eval_reason}",
        action="evaluate_context_sufficiency()",
        observation=f"insufficient_context={insufficient_context} | reason: {context_eval_reason}",
        timestamp=_now(),
    ))

    logger.info(
        "action_agent done success=%s tool_calls=%d insufficient_context=%s reason=%s",
        action_success, len(tool_calls_made), insufficient_context, context_eval_reason,
    )

    # Merge with existing trace
    existing_trace = state.get("react_trace", [])

    return {
        "tool_calls_made": tool_calls_made,
        "action_result": action_result,
        "action_success": action_success,
        "tool_retry_count": tool_retry_count,
        "insufficient_context": insufficient_context,
        "react_trace": existing_trace + new_trace_steps,
    }


# ---------------------------------------------------------------------------
# Isolation test — run directly to see the ReAct trace
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ACTION AGENT — ISOLATION TEST")
    print("Make sure mock_backend.py is running: uvicorn mock_backend:app --port 8000")
    print("="*60)

    # Test 1: Happy path — eligible refund
    print("\n[TEST 1] Happy path refund")
    test_state_1: SupportTicketState = {
        "ticket_id": "test-001",
        "user_id": "u001",
        "ticket_text": "I need a refund for my order 1001. The item arrived damaged.",
        "classification": "refund",
        "urgency": "medium",
        "sentiment": "negative",
        "classification_confidence": 0.92,
        "intake_reasoning": "Customer explicitly requested refund for order 1001. Mentions damaged item.",
        "retrieved_context": "Refund Policy: Physical goods eligible for refund within 30 days of delivery. Defective items eligible within 90 days. Locked accounts cannot process refunds.",
        "knowledge_reasoning": "Retrieved refund policy directly relevant to this refund request.",
        "react_trace": [],
        "tool_calls_made": [],
        "knowledge_retry_count": 0,
    }

    result_1 = run_action_agent(test_state_1)
    print("\n✓ Test 1 complete.")
    print(f"  action_success: {result_1['action_success']}")
    print(f"  action_result: {result_1['action_result']}")
    print(f"  tool_calls: {len(result_1['tool_calls_made'])}")
    print(f"  insufficient_context: {result_1['insufficient_context']}")

    print("\n" + "-"*60)

    # Test 2: Tool failure + intra-loop retry (account locked)
    print("\n[TEST 2] Account locked — refund fails, agent must adapt")
    test_state_2: SupportTicketState = {
        "ticket_id": "test-002",
        "user_id": "u002",   # u002 has status=locked in mock_backend
        "ticket_text": "Please refund my order 1003. I've been waiting too long.",
        "classification": "refund",
        "urgency": "high",
        "sentiment": "negative",
        "classification_confidence": 0.88,
        "intake_reasoning": "Customer requesting refund for order 1003. Frustrated tone.",
        "retrieved_context": "Refund Policy: Locked accounts cannot process refunds until account is restored. Password reset is available even for locked accounts.",
        "knowledge_reasoning": "Retrieved refund policy and locked account rules — directly relevant.",
        "react_trace": [],
        "tool_calls_made": [],
        "knowledge_retry_count": 0,
    }

    result_2 = run_action_agent(test_state_2)
    print("\n✓ Test 2 complete.")
    print(f"  action_success: {result_2['action_success']}")
    print(f"  action_result: {result_2['action_result']}")
    print(f"  tool_retry_count (failures): {result_2['tool_retry_count']}")
    print(f"  tool_calls: {len(result_2['tool_calls_made'])}")
    for tc in result_2['tool_calls_made']:
        status = "✓" if tc.success else "✗"
        print(f"    {status} {tc.tool_name}({tc.arguments}) → {tc.error_message or 'ok'}")

    print(f"\n  Full ReAct trace ({len(result_2['react_trace'])} steps):")
    for step in result_2['react_trace']:
        print(f"\n  [Iter {step.iteration}] THOUGHT: {step.thought[:150]}")
        print(f"            ACTION: {step.action}")
        print(f"            OBSERVATION: {step.observation[:150]}")
