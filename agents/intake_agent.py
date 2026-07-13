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
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.state import ReActStep, SupportTicketState

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

INTAKE_SYSTEM_PROMPT = """You are the Intake Agent for a customer support system.
Your job is to analyze an incoming support ticket and classify it.

You will be given:
1. The ticket text
2. The customer's user_id
3. Similar past tickets from this user's history (if any)

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


def run_intake_agent(state: SupportTicketState) -> dict:
    """LangGraph node function for the intake agent."""

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_output_tokens=1024)

    ticket_text = state.get("ticket_text", "")
    user_id = state.get("user_id", "unknown")
    new_trace_steps: list[ReActStep] = []

    # ---- Retrieve similar past tickets from memory ----
    similar_past_tickets: list[dict] = []
    try:
        # Import here to avoid circular dependency at module load time
        from memory.ticket_memory import retrieve_similar_tickets
        similar_past_tickets = retrieve_similar_tickets(user_id, ticket_text, k=5)
    except Exception as e:
        # Memory is not critical — proceed without it
        print(f"[intake_agent] Memory retrieval skipped: {e}")

    # Unresolved tickets about the same topic
    unresolved_similar = [t for t in similar_past_tickets if not t.get("resolved", True)]

    # ---- Iteration 1: Initial classification attempt ----
    memory_context = ""
    if similar_past_tickets:
        memory_context = f"\n\nPast tickets from this user:\n"
        for i, pt in enumerate(similar_past_tickets[:5], 1):
            resolved_str = "resolved" if pt.get("resolved") else "UNRESOLVED"
            memory_context += f"  {i}. [{resolved_str}] {pt.get('text', '')} → {pt.get('resolution', 'no resolution')}\n"
        if unresolved_similar:
            memory_context += f"\nNOTE: {len(unresolved_similar)} unresolved similar ticket(s) found for this user."
    else:
        memory_context = "\n\nNo past tickets found for this user."

    iter1_prompt = f"""Classify this support ticket:

Ticket: {ticket_text}
User ID: {user_id}
{memory_context}

Respond with the JSON structure described in your instructions."""

    thought_1 = (
        f"Analyzing ticket for user {user_id}. "
        f"Found {len(similar_past_tickets)} past ticket(s), "
        f"{len(unresolved_similar)} unresolved similar. "
        f"Proceeding with classification."
    )
    print(f"\n[intake_agent] THOUGHT: {thought_1}")

    response_1 = llm.invoke([
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(content=iter1_prompt),
    ])

    response_text = response_1.content if isinstance(response_1.content, str) else str(response_1.content)
    print(f"[intake_agent] ACTION: classify_ticket()")
    print(f"[intake_agent] OBSERVATION: {response_text[:400]}")

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
        print(f"\n[intake_agent] THOUGHT: {thought_2}")

        clarify_prompt = f"""The initial classification had low confidence ({confidence:.2f}).

Original ticket: {ticket_text}

Re-examine this carefully. Consider:
- What is the customer's PRIMARY need? (refund vs billing vs account vs technical vs general)
- Are there multiple issues? If so, what is the DOMINANT one?
- Is there enough information to classify confidently, or should we ask the customer?

Revise your JSON classification. If still uncertain, set needs_clarification=true and provide a specific clarifying question."""

        response_2 = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_output_tokens=1024).invoke([
            SystemMessage(content=INTAKE_SYSTEM_PROMPT),
            HumanMessage(content=iter1_prompt),
            response_1,
            HumanMessage(content=clarify_prompt),
        ])

        response_text_2 = response_2.content if isinstance(response_2.content, str) else str(response_2.content)
        print(f"[intake_agent] ACTION: re_classify_with_ambiguity_check()")
        print(f"[intake_agent] OBSERVATION: {response_text_2[:400]}")

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

    print(f"\n[intake_agent] Classification: {classification_data.get('classification')} | "
          f"Urgency: {classification_data.get('urgency')} | "
          f"Sentiment: {classification_data.get('sentiment')} | "
          f"Confidence: {classification_data.get('classification_confidence'):.2f}")

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
