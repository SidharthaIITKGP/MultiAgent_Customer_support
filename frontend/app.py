"""
frontend/app.py — Streamlit demo frontend

Features:
  - Submit a ticket with user_id
  - Live streaming of each agent's Thought → Action → Observation as the graph runs
  - Color-coded by agent (blue=intake, purple=knowledge, orange=action, green=escalation)
  - Final resolution card with confidence breakdown
  - Full trace viewer (expandable)

Run: streamlit run frontend/app.py
(Start mock_backend.py and api.py first)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Support System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Agent color scheme
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "intake":     {"bg": "#1a3a5c", "accent": "#4a9edd", "label": "🔵 Intake Agent"},
    "knowledge":  {"bg": "#2d1b5c", "accent": "#9b7de8", "label": "🟣 Knowledge Agent"},
    "action":     {"bg": "#5c2a0e", "accent": "#e8703a", "label": "🟠 Action Agent"},
    "escalation": {"bg": "#0e3d1a", "accent": "#2ecc71", "label": "🟢 Escalation Agent"},
}

DECISION_COLORS = {
    "auto_resolve": "#2ecc71",
    "escalate":     "#e74c3c",
    "request_info": "#f39c12",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("Preset Tickets")
    preset_options = {
        "Select a preset...": ("", ""),
        "TC-01: Simple Refund (happy path)":          ("u001", "I need a refund for my order 1001. The item arrived damaged."),
        "TC-06: Locked Account Refund (tool failure)": ("u002", "I need a refund for order 1003. I've been waiting too long."),
        "TC-04: Technical Bug E-4023":                 ("u003", "The app crashes every time I try to log in. I keep getting error E-4023."),
        "TC-03: Angry Customer":                       ("u004", "I AM ABSOLUTELY FURIOUS. Nobody has helped me for THREE DAYS. I want a refund NOW."),
        "TC-05: Multi-tool (check then refund)":       ("u001", "Can you check if my order 1001 has been delivered? If it has, I'd like a refund."),
        "TC-14: Multi-intent (password + email)":      ("u003", "Two things: I need to reset my password AND update my email to carol_new@example.com."),
    }

    selected_preset = st.selectbox("Load preset", list(preset_options.keys()))
    preset_uid, preset_text = preset_options[selected_preset]

    st.markdown("---")
    st.markdown("""
**Architecture:**
- LangGraph StateGraph
- Claude via Anthropic API  
- ChromaDB (local RAG)
- ReAct: Thought → Action → Observation

**Agents:**
1. 🔵 Intake (classify + memory)
2. 🟣 Knowledge (RAG, max 3 iters)
3. 🟠 Action (tool-calling, max 5 iters)
4. 🟢 Escalation (trace reasoning)
""")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🤖 Multi-Agent Customer Support System")
st.markdown("*Real-time ReAct reasoning trace across 4 specialized agents*")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    ticket_text = st.text_area(
        "Customer Ticket",
        value=preset_text,
        height=120,
        placeholder="Describe your issue here...",
    )

with col2:
    user_id = st.text_input(
        "User ID",
        value=preset_uid or "u001",
        help="User IDs in mock backend: u001 (active), u002 (locked), u003 (active), u004 (active)",
    )
    st.markdown("**Available users:**")
    st.markdown("- `u001` — Alice (active, premium)\n- `u002` — Bob (🔒 locked)\n- `u003` — Carol (active)\n- `u004` — Dan (active)")

submit = st.button("▶ Submit Ticket", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Graph streaming logic
# ---------------------------------------------------------------------------

def run_ticket_sync(ticket_text: str, user_id: str):
    """Run the graph synchronously and yield trace steps for display."""
    from agents.graph import get_compiled_app
    from agents.state import SupportTicketState

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    initial_state: SupportTicketState = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "ticket_text": ticket_text,
        "react_trace": [],
        "tool_calls_made": [],
        "knowledge_retry_count": 0,
        "tool_retry_count": 0,
        "insufficient_context": False,
    }

    app = get_compiled_app()
    results = {}
    for chunk in app.stream(initial_state, config={"recursion_limit": 15}):
        for node_name, node_output in chunk.items():
            results[node_name] = node_output
            yield node_name, node_output

    return results


def render_react_step(step: dict, container):
    """Render a single Thought/Action/Observation step in a styled box."""
    agent = step.get("agent", "?")
    colors = AGENT_COLORS.get(agent, {"bg": "#333", "accent": "#ccc", "label": agent})
    iteration = step.get("iteration", "?")

    with container:
        st.markdown(f"""
<div style="
    background: {colors['bg']};
    border-left: 4px solid {colors['accent']};
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
">
<div style="color: {colors['accent']}; font-weight: bold; margin-bottom: 8px;">
    {colors['label']} | Iteration {iteration}
</div>
<div style="color: #e0e0e0; margin-bottom: 6px;">
    <span style="color: #7ec8e3;">💭 THOUGHT:</span> {step.get('thought', '')[:400]}
</div>
<div style="color: #e0e0e0; margin-bottom: 6px;">
    <span style="color: #f0a500;">⚡ ACTION:</span> <code>{step.get('action', '')[:200]}</code>
</div>
<div style="color: #e0e0e0;">
    <span style="color: #7bcf8e;">👁 OBSERVATION:</span> {step.get('observation', '')[:400]}
</div>
</div>
""", unsafe_allow_html=True)


def render_confidence_bar(label: str, value: float, weight: float):
    """Render a single confidence component as a progress bar."""
    col_a, col_b, col_c = st.columns([3, 5, 2])
    with col_a:
        st.markdown(f"**{label}**")
    with col_b:
        st.progress(value)
    with col_c:
        st.markdown(f"`{value:.3f}` (w={weight:.0%})")


# ---------------------------------------------------------------------------
# Display results after submission
# ---------------------------------------------------------------------------

if submit and ticket_text.strip():

    st.markdown("---")
    st.markdown("## 🔄 Live Agent Reasoning Trace")

    trace_container = st.container()
    status_placeholder = st.empty()
    final_placeholder = st.empty()

    start_time = time.time()
    all_trace_steps = []
    final_state_snapshot = {}

    with st.spinner("Running multi-agent pipeline..."):
        for node_name, node_output in run_ticket_sync(ticket_text, user_id):
            # Show new trace steps as they arrive
            new_steps = node_output.get("react_trace", [])
            # Only show steps added by THIS node
            truly_new = new_steps[len(all_trace_steps):]
            for step in truly_new:
                if isinstance(step, dict):
                    render_react_step(step, trace_container)
                    all_trace_steps.append(step)

            # Update status
            agent_label = AGENT_COLORS.get(node_name, {}).get("label", node_name)
            status_placeholder.info(f"⚙️ {agent_label} completed")

            # Snapshot the latest state for final display
            final_state_snapshot.update(node_output)

    elapsed = time.time() - start_time
    status_placeholder.success(f"✅ Pipeline complete in {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Final resolution card
    # ---------------------------------------------------------------------------

    st.markdown("---")
    decision = final_state_snapshot.get("escalation_decision", "unknown")
    decision_color = DECISION_COLORS.get(decision, "#888")

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        st.markdown(f"""
<div style="
    background: #1a1a2e;
    border: 2px solid {decision_color};
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
">
<h3 style="color: {decision_color}; margin: 0 0 12px 0;">
    Resolution: {decision.upper().replace('_', ' ')}
</h3>
<p style="color: #e0e0e0; font-size: 15px; line-height: 1.5;">
    {final_state_snapshot.get('final_response', 'No response generated.')}
</p>
<hr style="border-color: #333; margin: 12px 0;">
<p style="color: #aaa; font-size: 12px;">
    <strong>Justification:</strong> {final_state_snapshot.get('escalation_justification', 'N/A')}
</p>
</div>
""", unsafe_allow_html=True)

    with col_r2:
        st.markdown("### Confidence Breakdown")
        cb = final_state_snapshot.get("confidence_breakdown", {})
        if isinstance(cb, dict):
            weights = cb.get("weights_used", {})
            render_confidence_bar("Retrieval Relevance", cb.get("retrieval_relevance", 0), weights.get("retrieval_relevance", 0.2))
            render_confidence_bar("Tool Call Success",   cb.get("tool_call_success", 0),   weights.get("tool_call_success", 0.4))
            render_confidence_bar("Classification Conf", cb.get("classification_confidence", 0), weights.get("classification_confidence", 0.2))
            render_confidence_bar("Sentiment Score",     cb.get("sentiment_score", 0),      weights.get("sentiment_score", 0.2))
            st.markdown(f"**Overall: `{cb.get('overall', 0):.3f}`**")
        else:
            st.markdown("*Confidence breakdown not available*")

    # ---------------------------------------------------------------------------
    # Additional metadata
    # ---------------------------------------------------------------------------

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Classification", final_state_snapshot.get("classification", "?").capitalize())
    col_m2.metric("Urgency",       final_state_snapshot.get("urgency", "?").capitalize())
    col_m3.metric("Sentiment",     final_state_snapshot.get("sentiment", "?").capitalize())
    col_m4.metric("Tool Calls",    len(final_state_snapshot.get("tool_calls_made", [])))

    # ---------------------------------------------------------------------------
    # Full trace viewer (collapsed)
    # ---------------------------------------------------------------------------

    with st.expander("📋 Full Reasoning Trace (all agents)"):
        for step in all_trace_steps:
            render_react_step(step, st)

    with st.expander("🔧 Tool Call Log"):
        tool_calls = final_state_snapshot.get("tool_calls_made", [])
        if tool_calls:
            for tc in tool_calls:
                success = tc.get("success", False)
                icon = "✓" if success else "✗"
                color = "#2ecc71" if success else "#e74c3c"
                st.markdown(f"""
<span style="color:{color}">**{icon} {tc.get('tool_name', '?')}**</span>
`args: {json.dumps(tc.get('arguments', {}))}`
`result: {json.dumps(tc.get('result', {}))[:200]}`
""")
        else:
            st.markdown("*No tool calls made*")

    with st.expander("📜 Raw State JSON"):
        displayable = {k: v for k, v in final_state_snapshot.items() if k != "react_trace"}
        st.json(displayable)

elif submit and not ticket_text.strip():
    st.warning("Please enter a ticket description.")
