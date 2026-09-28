"""CMN-C1-101 Cross-Session Memory & Interaction-History Q&A — State.

Flat TypedDict per the platform's node and state-safety contracts:
every field is an Optional primitive or a JSON-serialized string (msgpack
round-trips cleanly). No Pydantic, no dataclass, no arbitrary Python objects,
and never JWT / API keys / credentials. **No other user's data** ever enters
State — retrieval is user-scoped and OutputValidate blocks cross-user leaks.

Pipeline (docs/02_design.md §Architecture Overview):
    InputParse (S-1) → SessionHistoryFetch → MemoryRetrieve → ContextMerge
    → ResponseGenerate → OutputValidate (S-3 self-reference + cross-user block + S-4 audit)
"""

from __future__ import annotations

from typing import Optional

from framework.schemas.agent_state import AgentState


class CrossSessionMemoryState(AgentState):
    """State for the cross-session memory Q&A pipeline.

    Shared fields (user_input, status, session_id, node_history, error_log, …) are
    inherited from AgentState. Only domain fields are declared here; LangGraph drops
    keys not declared in the schema, so every field the pipeline writes MUST appear below.
    """

    # ─── Input (caller supplies via user_input + input_context) ───
    user_id: Optional[str]  # scope key — authorizes which user's memory is read
    query: Optional[str]  # NL cross-session history question
    session_window: Optional[str]  # optional retrieval scope (JSON range or session-id allowlist)

    # ─── InputParseNode (pre_process, S-1 input boundary) ───
    validated_query: Optional[str]  # sanitized / normalized query

    # ─── SessionHistoryFetchNode (user-scoped) ───
    history_log: Optional[str]  # JSON: [{session_id, ts, role, text}] (this user only)
    history_count: Optional[int]

    # ─── MemoryRetrieveNode (user-scoped) ───
    memory_chunks: Optional[str]  # JSON: [{session_id, ts, text, score}] (this user only)
    memory_hit_count: Optional[int]

    # ─── ContextMergeNode (deterministic recency-weighted) ───
    merged_context: Optional[str]  # JSON: [{session_id, ts, text, weight}] (token-budget capped)
    context_size_estimate: Optional[int]

    # ─── ResponseGenerateNode (LLM) ───
    answer: Optional[str]  # NL answer grounded in the merged context
    cited_session_ids: Optional[str]  # JSON: [session_id, ...] the answer is grounded in

    # ─── OutputValidateNode (S-3 self-reference + cross-user block + PII sanitize + S-4) ───
    validation_status: Optional[str]  # "passed" | "redacted" | "rejected" | "cross_user_blocked"
    pii_redaction_count: Optional[int]  # PII values redacted out of the answer
    cross_user_blocked: Optional[bool]  # True if a cited session was outside this user's scope
    audit_logged: Optional[bool]  # True once the S-4 audit event is emitted

    # ─── Error propagation (any node; downstream nodes self-skip) ───
    error_code: Optional[str]
    error_message: Optional[str]


# Backward-compat alias: the scaffold (graph.py / server.py) references `State`.
State = CrossSessionMemoryState
