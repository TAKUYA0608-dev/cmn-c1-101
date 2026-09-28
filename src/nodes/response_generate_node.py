"""ResponseGenerateNode — main slot sub-node.

LLM call over the deterministically-merged context → a natural-language answer plus the
set of `cited_session_ids` the answer is grounded in. The cited session list is built
**from the merged context** (not free-form from the LLM), so every cited session is one
that was actually retrieved for this user — OutputValidate then enforces that invariant.

A zero-context result yields a safe "no prior context found" answer (no fabrication).
With context but NO LLM bound, the node degrades with `LLM_NOT_CONFIGURED` (SUCCESS +
error_code, no answer) — it never substitutes a stub reply for a synthesized one.
"""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import LLMClient
from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel

_NO_CONTEXT = (
    "I don't find any prior session context for you on this topic, so I can't answer from "
    "memory. You may not have discussed this before, or it falls outside the retrieval window."
)


class ResponseGenerateNode(FunctionNode):
    """Generate the NL answer + cited_session_ids from the merged context."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__()
        # None = no LLM bound. Resolved by the graph (explicit kw > config["llm"]);
        # a bare node stays unbound and degrades rather than answering from a stub.
        self._llm: LLMClient | None = llm_client

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        try:
            merged = json.loads(state.get("merged_context") or "[]")
        except (json.JSONDecodeError, TypeError):
            merged = []

        if not merged:
            return {
                "answer": _NO_CONTEXT,
                "cited_session_ids": json.dumps([], ensure_ascii=False),
                "status": AgentStatus.SUCCESS.value,
            }

        cited = sorted({m.get("session_id") for m in merged if m.get("session_id")})
        if self._llm is None:
            # Context was retrieved but nothing can synthesize an answer from it.
            # Say so (named code) instead of echoing the context as if it were one.
            emit_trace_event("llm_not_configured", {"cited_session_count": len(cited)}, state)
            return {
                "error_code": "LLM_NOT_CONFIGURED",
                "error_message": (
                    "ResponseGenerateNode: no LLM bound (llm_client / config['llm']); "
                    "context was retrieved but no answer was synthesized"
                ),
                "cited_session_ids": json.dumps(cited, ensure_ascii=False),
                "status": AgentStatus.SUCCESS.value,
            }
        query = state.get("validated_query") or state.get("query") or ""
        context_text = " | ".join(str(m.get("text", "")) for m in merged)
        prompt = (
            "Answer the user's question using ONLY their prior session context below. "
            "Do not invent facts.\n\n"
            f"Question: {query}\n\nContext: {context_text}"
        )
        answer = (self._llm.generate(prompt) or "").strip() or _NO_CONTEXT
        # Provenance footer makes the grounding explicit (and gives the S-3 gate a marker).
        answer = f"{answer}\n\n_(grounded in sessions: {', '.join(cited)})_"

        emit_trace_event("response_generated", {"cited_session_count": len(cited)}, state)
        return {
            "answer": answer,
            "cited_session_ids": json.dumps(cited, ensure_ascii=False),
            "status": AgentStatus.SUCCESS.value,
        }
