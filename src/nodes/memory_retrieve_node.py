"""MemoryRetrieveNode — main slot sub-node.

Retrieve top-k `user_id`-scoped memory chunks from the external memory store (mem0 /
Letta / pgvector, DI). User-scoped query — never another user's chunks. A zero-hit
result is not an error.
"""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import MemoryStore, InMemoryMemoryStore, seed_stores
from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel

_TOP_K = 6


class MemoryRetrieveNode(FunctionNode):
    """Retrieve user-scoped semantic memory chunks."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        super().__init__()
        # No store injected → an EMPTY process-local store, never the seeded demo
        # data: illustrative chunks must not be served to a caller as their memory.
        self._store: MemoryStore = memory_store if memory_store is not None else InMemoryMemoryStore()

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        user_id = state.get("user_id") or ""
        query = state.get("validated_query") or state.get("query") or ""
        chunks = self._store.search(user_id, query, top_k=_TOP_K) or []
        emit_trace_event("memory_retrieved", {"memory_hit_count": len(chunks)}, state)

        return {
            "memory_chunks": json.dumps(chunks, ensure_ascii=False),
            "memory_hit_count": len(chunks),
            "status": AgentStatus.SUCCESS.value,
        }


__all__ = ["MemoryRetrieveNode", "InMemoryMemoryStore", "seed_stores"]
