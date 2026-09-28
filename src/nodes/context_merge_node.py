"""ContextMergeNode — main slot sub-node.

Deterministic recency-weighted, token-budget merge of session history + memory chunks
(no LLM). Each candidate's weight = recency_weight(ts) [× relevance score for memory
chunks]; candidates are sorted by weight and accumulated until the token budget is hit.
Determinism keeps *which sessions were selected* auditable — the LLM only writes the
answer, never picks the context.
"""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import recency_weight, estimate_tokens, TOKEN_BUDGET
from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel


class ContextMergeNode(FunctionNode):
    """Recency-weighted token-budget merge of history + memory."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        try:
            history = json.loads(state.get("history_log") or "[]")
        except (json.JSONDecodeError, TypeError):
            history = []
        try:
            memory = json.loads(state.get("memory_chunks") or "[]")
        except (json.JSONDecodeError, TypeError):
            memory = []

        candidates: list[dict[str, Any]] = []
        for r in history:
            candidates.append(
                {
                    "session_id": r.get("session_id"),
                    "ts": int(r.get("ts", 0)),
                    "text": str(r.get("text", "")),
                    "source": "history",
                    "relevance": 1.0,
                }
            )
        for c in memory:
            candidates.append(
                {
                    "session_id": c.get("session_id"),
                    "ts": int(c.get("ts", 0)),
                    "text": str(c.get("text", "")),
                    "source": "memory",
                    "relevance": float(c.get("score", 0.5)),
                }
            )

        if not candidates:
            return {
                "merged_context": json.dumps([], ensure_ascii=False),
                "context_size_estimate": 0,
                "status": AgentStatus.SUCCESS.value,
            }

        now = max(c["ts"] for c in candidates)
        for c in candidates:
            c["weight"] = round(recency_weight(c["ts"], now) * c["relevance"], 4)
        candidates.sort(key=lambda c: c["weight"], reverse=True)

        merged: list[dict[str, Any]] = []
        tokens = 0
        for c in candidates:
            t = estimate_tokens(c["text"])
            if tokens + t > TOKEN_BUDGET and merged:
                break
            merged.append({"session_id": c["session_id"], "ts": c["ts"], "text": c["text"], "weight": c["weight"]})
            tokens += t

        emit_trace_event("context_merged", {"merged_count": len(merged), "token_estimate": tokens}, state)
        return {
            "merged_context": json.dumps(merged, ensure_ascii=False),
            "context_size_estimate": tokens,
            "status": AgentStatus.SUCCESS.value,
        }
