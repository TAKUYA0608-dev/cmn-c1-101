"""SessionHistoryFetchNode — main slot sub-node.

Fetch the `user_id`-scoped interaction history from the session-log store (DI). The
store NEVER returns other users' rows — user-scoping is the first defense against
cross-user memory leak. A zero-history result is not an error.
"""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import SessionLogStore, InMemorySessionLogStore, seed_stores
from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel

_LIMIT = 20


class SessionHistoryFetchNode(FunctionNode):
    """Fetch user-scoped session history."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, session_store: SessionLogStore | None = None) -> None:
        super().__init__()
        # No store injected → an EMPTY process-local store, never the seeded demo
        # data: illustrative rows must not be served to a caller as their history.
        self._store: SessionLogStore = session_store if session_store is not None else InMemorySessionLogStore()

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        user_id = state.get("user_id") or ""
        window = state.get("session_window")
        rows = self._store.fetch(user_id, window=window, limit=_LIMIT) or []
        emit_trace_event("history_fetched", {"history_count": len(rows)}, state)

        return {
            "history_log": json.dumps(rows, ensure_ascii=False),
            "history_count": len(rows),
            "status": AgentStatus.SUCCESS.value,
        }


__all__ = ["SessionHistoryFetchNode", "InMemorySessionLogStore", "seed_stores"]
