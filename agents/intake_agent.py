"""
agents/intake_agent.py

Classifies an incoming support ticket using a ReAct-style loop:
  Thought → (is the ticket ambiguous? does memory reveal context?) → Action → Observation

Key behaviors:
- Retrieves similar past tickets from memory BEFORE classifying
  (retrieval-augmented reasoning, not just RAG over docs)
- If classification is ambiguous (confidence < 0.7), the agent's Thought explicitly
  reasons about the ambiguity before proceeding or generating a clarifying question
- Produces: classification, urgency, sentiment, classification_confidence, intake_reasoning
  — all passed downstream so action_agent can reference WHY the ticket was classified this way
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm_utils import build_llm, extract_text
from agents.state import ReActStep, SupportTicketState
from logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

INTAKE_SYSTEM_PROMPT = """You are the Intake Agent for a customer support system.
Your job is to analyze an incoming support ticket and classify it.

You will be given:
1. The ticket text
2. The customer's user_id
3. Similar past tickets from this user's history (if any)
4. Conversation context for the CURRENT chat, if this is a follow-up message (if any)

Conversation context vs. past tickets — these are different things:
- "Conversation context" is THIS live chat's own back-and-forth (what was just said,
  moments ago, in this same conversation).
- "Past tickets" is history from OTHER, separate tickets this user filed previously.

If conversation context shows the assistant just asked the customer a clarifying
question, treat the new message as the customer ANSWERING that question — combine
it with the original issue rather than classifying it as a brand-new, standalone
ticket. Say so explicitly in intake_reasoning (e.g. "customer is answering the prior
request for an order ID"). If the new message clearly raises something unrelated to
what was asked, classify it as a new issue instead and say so in intake_reasoning.

Produce a JSON response with this exact structure:
{{
  "classification": "<billing|technical|refund|account|general>",
  "urgency": "<low|medium|high|critical>",
  "sentiment": "<positive|neutral|negative|angry>",
  "classification_confidence": <0.0-1.0>,
  "intake_reasoning": "<2-3 sentences explaining WHY you classified this way, what signals you used, and how past ticket history influenced your decision. This is passed to the next agent and must be informative.>",
  "needs_clarification": <true|false>,
  "clarifying_question": "<question to ask customer if needs_clarification is true, else null>"
}}

Classification guidelines:
- billing: charges, invoices, payment failures, subscription billing questions
- technical: app errors, crashes, bugs, error codes, performance issues
- refund: explicit refund requests, return requests
- account: login, password, locked account, profile changes, 2FA
- general: shipping, tracking, product questions, anything else

Urgency guidelines:
- critical: service down, account compromise, anger + no resolution after multiple attempts
- high: angry customer, blocking issue, business impact
- medium: standard issue, some frustration
- low: informational, positive tone, minor request

IMPORTANT: If the user has 3 or more past unresolved tickets about the same issue,
increase urgency to at least "high" and note this in intake_reasoning.
This is a strong escalation signal that must be captured explicitly."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_conversation_context(history: list[dict], previous_turn_requested_info: bool) -> str:
    """
    Format THIS chat's own recent turn history for the intake prompt.
    Distinct from memory_context (past tickets, a different mechanism) — see the
    system prompt's explicit note on the difference.
    """
    if not history:
        return "\n\nNo prior messages in this conversation (this is the first message)."

    lines = ["\n\nConversation context (this chat, most recent last):"]
    for turn in history:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        label = "Customer" if role == "user" else "Assistant"
        lines.append(f"  [{label}] {content}")

    if previous_turn_requested_info:
        lines.append(
            "\nNOTE: The assistant's last message asked the customer a clarifying "
            "question. The newest customer message below is likely an ANSWER to that "
            "question, not a fresh, unrelated ticket."
        )
    return "\n".join(lines)


def run_intake_agent(state: SupportTicketState, config=None) -> dict:
    """LangGraph node function for the intake agent.

    `config` intentionally has no type annotation: LangGraph only injects the
    runtime RunnableConfig into a node if this param is unannotated or typed
    exactly RunnableConfig/Optional[RunnableConfig] (checked by literal string
    match, since this module uses `from __future__ import annotations`).
    """

    llm = build_llm(config, model=MODEL_NAME, temperature=0, max_output_tokens=1536)

    ticket_text = state.get("ticket_text", "")
    user_id = state.get("user_id", "unknown")
    # This is best-effort enrichment: a memory outage must not prevent a new
    # ticket from being classified.
    new_trace_steps: list[ReActStep] = []

    # ---- Retrieve similar past tickets from memory ----
    similar_past_tickets: list[dict] = []
    try:
        # Import here to avoid circular dependency at module load time
        from memory.ticket_memory import retrieve_similar_tickets
        similar_past_tickets = retrieve_similar_tickets(user_id, ticket_text, k=5)
    except Exception as e:
        # Memory is not critical — proceed without it
        logger.warning("memory retrieval skipped: %s", e)

    # Unresolved tickets about the same topic
    unresolved_similar = [t for t in similar_past_tickets if not t.get("resolved", True)]

    # ---- Iteration 1: Initial classification attempt ----
    memory_context = ""
    if similar_past_tickets:
        memory_context = "\n\nPast tickets from this user:\n"
        for i, pt in enumerate(similar_past_tickets[:5], 1):
            resolved_str = "resolved" if pt.get("resolved") else "UNRESOLVED"
            memory_context += f"  {i}. [{resolved_str}] {pt.get('text', '')} → {pt.get('resolution', 'no resolution')}\n"
        if unresolved_similar:
            memory_context += f"\nNOTE: {len(unresolved_similar)} unresolved similar ticket(s) found for this user."
    else:
        memory_context = "\n\nNo past tickets found for this user."

    conversation_context = _format_conversation_context(
        state.get("conversation_history", []),
        state.get("previous_turn_requested_info", False),
    )

    iter1_prompt = f"""Classify this support ticket:

Ticket: {ticket_text}
User ID: {user_id}
{conversation_context}
{memory_context}

Respond with the JSON structure described in your instructions."""

    thought_1 = (
        f"Analyzing ticket for user {user_id}. "
        f"Found {len(similar_past_tickets)} past ticket(s), "
        f"{len(unresolved_similar)} unresolved similar. "
        f"Proceeding with classification."
    )
    logger.debug("THOUGHT: %s", thought_1)

    response_1 = llm.invoke([
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(content=iter1_prompt),
    ])

    response_text = extract_text(response_1.content)
    logger.debug("ACTION: classify_ticket()")
    logger.debug("OBSERVATION: %s", response_text[:400])

    step_1 = ReActStep(
        agent="intake",
        iteration=1,
        thought=thought_1,
        action="classify_ticket(ticket_text, user_id, past_tickets)",
        observation=response_text[:800],
        timestamp=_now(),
    )
    new_trace_steps.append(step_1)

    # Parse the JSON response
    classification_data = _parse_classification_json(response_text)

    # ---- Iteration 2 (conditional): Ambiguity check ----
    confidence = classification_data.get("classification_confidence", 0.0)

    if confidence < 0.70:
        thought_2 = (
            f"Classification confidence is {confidence:.2f} — below 0.70 threshold. "
            f"This ticket is ambiguous. Re-analyzing with focused prompting."
        )
        logger.debug("THOUGHT: %s", thought_2)

        clarify_prompt = f"""The initial classification had low confidence ({confidence:.2f}).

Original ticket: {ticket_text}

Re-examine this carefully. Consider:
- What is the customer's PRIMARY need? (refund vs billing vs account vs technical vs general)
- Are there multiple issues? If so, what is the DOMINANT one?
- Is there enough information to classify confidently, or should we ask the customer?

Revise your JSON classification. If still uncertain, set needs_clarification=true and provide a specific clarifying question."""

        response_2 = build_llm(config, model=MODEL_NAME, temperature=0, max_output_tokens=1536).invoke([
            SystemMessage(content=INTAKE_SYSTEM_PROMPT),
            HumanMessage(content=iter1_prompt),
            response_1,
            HumanMessage(content=clarify_prompt),
        ])

        response_text_2 = extract_text(response_2.content)
        logger.debug("ACTION: re_classify_with_ambiguity_check()")
        logger.debug("OBSERVATION: %s", response_text_2[:400])

        step_2 = ReActStep(
            agent="intake",
            iteration=2,
            thought=thought_2,
            action="re_classify_with_ambiguity_check()",
            observation=response_text_2[:800],
            timestamp=_now(),
        )
        new_trace_steps.append(step_2)

        classification_data = _parse_classification_json(response_text_2)

    # ---- Build the intake_reasoning string (passed to downstream agents) ----
    reasoning = classification_data.get("intake_reasoning", "No reasoning provided.")
    if unresolved_similar:
        reasoning = (
            f"[Memory signal: {len(unresolved_similar)} prior unresolved ticket(s) about similar issue] "
            + reasoning
        )

    existing_trace = state.get("react_trace", [])

    logger.info(
        "classification=%s urgency=%s sentiment=%s confidence=%.2f",
        classification_data.get("classification"),
        classification_data.get("urgency"),
        classification_data.get("sentiment"),
        classification_data.get("classification_confidence"),
    )

    return {
        "classification": classification_data.get("classification", "general"),
        "urgency": classification_data.get("urgency", "medium"),
        "sentiment": classification_data.get("sentiment", "neutral"),
        "classification_confidence": float(classification_data.get("classification_confidence", 0.5)),
        "intake_reasoning": reasoning,
        "similar_past_tickets": similar_past_tickets,
        "react_trace": existing_trace + new_trace_steps,
        "knowledge_retry_count": 0,
        "tool_calls_made": [],
        "insufficient_context": False,
    }


def _parse_classification_json(text: str) -> dict:
    """Extract and parse the JSON classification from the LLM response."""
    # Strip markdown code fences (Gemini wraps JSON in ```json ... ```)
    cleaned = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
    json_match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback defaults
    return {
        "classification": "general",
        "urgency": "medium",
        "sentiment": "neutral",
        "classification_confidence": 0.5,
        "intake_reasoning": "Could not parse classification response — defaulting to general/medium.",
        "needs_clarification": False,
        "clarifying_question": None,
    }
