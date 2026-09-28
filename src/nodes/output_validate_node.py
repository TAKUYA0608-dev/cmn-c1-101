"""OutputValidateNode — post_process slot. **S-3 gate + S-4 audit.**

The terminal node. Three responsibilities (docs/02 Step 6):
  1. **Self-reference gate** — a substantive answer must cite ≥1 session it is grounded
     in (`cited_session_ids` non-empty). A "no prior context" answer is exempt.
  2. **Cross-user-leak block** (the #1 risk) — every cited session MUST belong to this
     user's retrieved set (`history_log` ∪ `memory_chunks`, both user-scoped). If any
     cited session is outside that set, the answer is rejected and `cross_user_blocked`.
  3. **PII sanitization** — emails / phone numbers / マイナンバー are redacted out of the
     answer before it leaves the agent.
Then S-4: emit one redacted per-invocation audit event. Always fires (even on the error
path) — silent failure is prohibited.

Domain validation in `execute()` — distinct from the framework `@final` credential gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b(?:0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}|\+\d{8,15})\b")
_MYNUMBER = re.compile(r"\b(?:\d[ \-]?){12}\b")
_REDACTED = "[REDACTED:pii]"

_REJECTED = (
    "I can't return this answer because it referenced session context that isn't grounded in "
    "your own history. Withholding to prevent any cross-user information leak."
)

# chars: below this length an answer is treated as a short refusal / notice
# (e.g. the "no prior context" path), not a grounded claim — so it is exempt from the
# self-reference / cross-user gates.
_MIN_SUBSTANTIVE = 40


def _sanitize_pii(text: str) -> tuple[str, int]:
    count = 0

    def _sub(_m: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        return _REDACTED

    text = _EMAIL.sub(_sub, text)
    text = _MYNUMBER.sub(_sub, text)
    text = _PHONE.sub(_sub, text)
    return text, count


def _allowed_sessions(state: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for field in ("history_log", "memory_chunks"):
        try:
            for r in json.loads(state.get(field) or "[]"):
                sid = r.get("session_id")
                if sid:
                    allowed.add(sid)
        except (json.JSONDecodeError, TypeError):
            continue
    return allowed


class OutputValidateNode(FunctionNode):
    """S-3 self-reference + cross-user-leak block + PII sanitize + S-4 audit (terminal)."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        answer = state.get("answer") or ""
        try:
            cited = json.loads(state.get("cited_session_ids") or "[]")
        except (json.JSONDecodeError, TypeError):
            cited = []

        cross_user_blocked = False
        validation_status = "passed"

        substantive = bool(answer) and len(answer) >= _MIN_SUBSTANTIVE and not state.get("error_code")

        if substantive and cited:
            # ── cross-user-leak block: every cited session must be in this user's scope ──
            allowed = _allowed_sessions(state)
            foreign = [s for s in cited if s not in allowed]
            if foreign:
                emit_trace_event("cross_user_leak_blocked", {"foreign_session_count": len(foreign)}, state)
                answer = _REJECTED
                cross_user_blocked = True
                validation_status = "cross_user_blocked"
        elif substantive and not cited:
            # ── self-reference gate: a grounded answer must cite a session ──
            # Exempt only the explicit "no prior context" path (short / context-less).
            if "prior session context" not in answer and "grounded in sessions" not in answer:
                answer = _REJECTED
                validation_status = "rejected"

        # ── PII sanitization ──
        answer, pii_redaction_count = _sanitize_pii(answer)
        if pii_redaction_count and validation_status == "passed":
            validation_status = "redacted"

        # ── S-4 audit (always fires) ──
        payload: dict[str, Any] = {
            "history_count": state.get("history_count") or 0,
            "memory_hit_count": state.get("memory_hit_count") or 0,
            "cited_session_count": len(cited),
            "answer_length": len(answer),
            "validation_status": validation_status,
            "pii_redaction_count": pii_redaction_count,
            "cross_user_blocked": cross_user_blocked,
        }
        if state.get("error_code"):
            payload["error_code"] = state["error_code"]
        emit_trace_event("agent_invoke_complete", payload, state)

        return {
            "answer": answer,
            "validation_status": validation_status,
            "pii_redaction_count": pii_redaction_count,
            "cross_user_blocked": cross_user_blocked,
            "audit_logged": True,
            "status": AgentStatus.SUCCESS.value,
        }
