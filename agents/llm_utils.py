"""
agents/llm_utils.py

Central factory for constructing the Gemini chat model used by every agent node.

Why this exists: to let a caller (a chat visitor) bring their own Gemini API key
instead of always using the server's GOOGLE_API_KEY, without ever putting that key
into SupportTicketState — state gets persisted verbatim into TicketRun.state_json,
and a secret must never go through that path.

The key travels through LangGraph's `config` (RunnableConfig), which every node
function may optionally accept as a second parameter and which is threaded through
graph.invoke()/astream() calls via config={"configurable": {...}}. This is the same
mechanism agents/graph.py already uses for recursion_limit — additive, not new.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

RunnableConfig = dict[str, Any]


def resolve_api_key(config: Optional[RunnableConfig]) -> Optional[str]:
    """Extract the caller-supplied Gemini API key from a LangGraph config, if any."""
    if not config:
        return None
    return config.get("configurable", {}).get("gemini_api_key")


def build_llm(config: Optional[RunnableConfig], **kwargs: Any) -> ChatGoogleGenerativeAI:
    """
    Construct a ChatGoogleGenerativeAI instance, using the caller's API key from
    `config` if one was supplied, otherwise falling back to the GOOGLE_API_KEY
    environment variable (ChatGoogleGenerativeAI's own default behavior).

    Defaults `thinking_level="minimal"` for Gemini 3+ models. Without this, a
    "thinking" model (e.g. gemini-3.5-flash) can spend most of max_output_tokens
    on internal reasoning tokens before writing any visible output, truncating
    the JSON every agent here depends on parsing. This system already elicits
    explicit, externalized reasoning via prompts (react_trace, intake_reasoning,
    etc.) — the model's own hidden chain-of-thought isn't part of that design,
    so keeping it minimal is correct here, not just a workaround. Measured on
    the intake classification prompt: "minimal" produced valid JSON with 0
    reasoning tokens (vs 109 at "low") and was faster end-to-end, with no
    quality loss on that prompt — per-call latency is dominated by network +
    base generation time, not reasoning depth, so this is a modest but real
    win that compounds across the ~5-6 sequential calls a full ticket makes.
    Callers may override by passing thinking_level explicitly.
    """
    api_key = resolve_api_key(config)
    if api_key:
        kwargs["google_api_key"] = api_key
    kwargs.setdefault("thinking_level", "minimal")
    return ChatGoogleGenerativeAI(**kwargs)


def extract_text(content: Any) -> str:
    """
    Extract human-readable text from a LangChain AIMessage.content value.

    Newer Gemini responses (e.g. gemini-3.x) return `content` as a list of typed
    blocks — text plus non-text metadata like thought signatures — rather than a
    plain string. Naively doing `str(content)` on that list embeds a Python dict
    repr (single-quoted, with a giant opaque signature blob) around the actual
    text, which silently corrupts any JSON the model wrote. This pulls out just
    the text blocks, so callers that regex/json.loads the result keep working
    regardless of which content shape the model returned.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()
