"""InputParseNode — pre_process slot / S-1 input boundary.

Validate + normalize the cross-session memory request: require `user_id` (the scope
key), validate `query`, NFKC normalize, reject injection, size-cap. Parse only — no
retrieval (that is the main-slot sub-nodes' responsibility).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.utils.audit import emit_trace_event

_MAX_LEN = 2000

_INJECTION = re.compile(
    r"(?i)(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"
    r"|disregard\s+(?:the\s+)?(?:system|previous)\s+(?:prompt|instructions)"
    r"|reveal\s+(?:your\s+)?system\s+prompt"
    r"|<\s*script\b|</\s*script\s*>)"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputParseNode(FunctionNode):
    """S-1: validate user_id + query (+ optional session_window); reject injection."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        ic = state.get("input_context") or {}
        # Identity resolution, widest-to-narrowest. `caller_id` is last but is
        # the one that matters in production: the Marketplace runner builds
        # `InvocationContext(caller_id=env["USER_ID"], ...)` and passes an
        # `input_context` of its own making — it never carries a `user_id` key.
        # Reading only the first two sources meant every Marketplace run failed
        # with USER_ID_MISSING (verified against the shipped runner), while the
        # HTTP path kept working because callers set it explicitly there.
        user_id = state.get("user_id") or ic.get("user_id") or state.get("caller_id")
        raw = state.get("query") or state.get("user_input") or ""

        if not user_id or not str(user_id).strip():
            return {
                "error_code": "USER_ID_MISSING",
                "error_message": "InputParseNode: user_id is required (memory is user-scoped)",
                "status": AgentStatus.SUCCESS.value,
            }
        if not raw or not str(raw).strip():
            return {
                "error_code": "INPUT_EMPTY",
                "error_message": "InputParseNode: query is empty or missing",
                "status": AgentStatus.SUCCESS.value,
            }
        if len(str(raw)) > _MAX_LEN:
            return {
                "error_code": "INPUT_TOO_LONG",
                "error_message": f"InputParseNode: query exceeds {_MAX_LEN} chars",
                "status": AgentStatus.SUCCESS.value,
            }
        if _INJECTION.search(str(raw)):
            return {
                "error_code": "INJECTION_DETECTED",
                "error_message": "InputParseNode: prompt-injection pattern rejected",
                "status": AgentStatus.SUCCESS.value,
            }

        validated = unicodedata.normalize("NFKC", str(raw))
        validated = _CONTROL.sub("", validated).strip()
        validated = re.sub(r"\s+", " ", validated)

        out: dict[str, Any] = {
            "user_id": str(user_id),
            "query": validated,
            "validated_query": validated,
            "status": AgentStatus.SUCCESS.value,
        }
        window = state.get("session_window") or ic.get("session_window")
        if window:
            out["session_window"] = str(window)
        # S-4 (gate-audit-trace-check): record that this
        # boundary node completed. Field NAMES only — never values.
        emit_trace_event("input_parse_completed", {}, state)
        return out
