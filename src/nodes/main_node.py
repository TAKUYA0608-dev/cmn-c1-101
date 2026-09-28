"""MemoryQAMainNode (main slot) — composes the 4 core reasoning steps.

    SessionHistoryFetch → MemoryRetrieve → ContextMerge → ResponseGenerate

Sub-node composition: sub-nodes are instantiated in
`__init__()` (`self._seq`) and called via `sub_node.execute(state, config)` **directly —
not `__call__()`**; the composite owns the single S-2/S-3/S-4 boundary. Each sub-node
returns only its changed fields and self-skips on `error_code`. Both stores + the LLM are
dependency-injected into the sub-nodes that need them.
"""

from __future__ import annotations

from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import SessionLogStore, MemoryStore, LLMClient
from src.nodes.session_history_fetch_node import SessionHistoryFetchNode
from src.nodes.memory_retrieve_node import MemoryRetrieveNode
from src.nodes.context_merge_node import ContextMergeNode
from src.nodes.response_generate_node import ResponseGenerateNode
from framework.schemas.trust_level import TrustLevel
from src.utils.audit import emit_trace_event


class MemoryQAMainNode(FunctionNode):
    """main slot — SessionHistoryFetch → MemoryRetrieve → ContextMerge → ResponseGenerate."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        session_store: SessionLogStore | None = None,
        memory_store: MemoryStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__()
        self._seq: list[FunctionNode] = [
            SessionHistoryFetchNode(session_store=session_store),
            MemoryRetrieveNode(memory_store=memory_store),
            ContextMergeNode(),
            ResponseGenerateNode(llm_client=llm_client),
        ]

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            # Upstream pre_process rejected the input: skip the composite
            # sub-pipeline. The framework router still advances to post_process,
            # whose terminal S-4 audit always runs.
            return {"status": AgentStatus.SUCCESS.value}
        working = dict(state)
        deltas: dict[str, Any] = {}
        try:
            for node in self._seq:
                updates = node.execute(working, config) or {}
                working.update(updates)
                deltas.update(updates)
        except Exception as e:
            # Degrade instead of crashing the node: with missing upstream state
            # (e.g. the PB-6 synthetic state) or unavailable dependencies the
            # sub-pipeline cannot run. post_process still runs the terminal
            # S-4 audit and surfaces the error_code.
            return {
                "error_code": "MAIN_PIPELINE_ERROR",
                "error_message": f"{type(e).__name__}: {str(e)[:160]}",
                "status": AgentStatus.SUCCESS.value,
            }
        # Intentional: composite reports SUCCESS so the framework router always advances to
        # post_process (OutputValidate), where the S-4 audit + cross-user gate always run.
        # A sub-node failure is carried via the `error_code` delta. Design ref: docs/02
        # "Composition Pattern"; same pattern as its sibling templates.
        deltas["status"] = AgentStatus.SUCCESS.value
        # S-4 (gate-audit-trace-check): record that this
        # boundary node completed. Field NAMES only — never values.
        emit_trace_event("memory_q_a_main_completed", {}, state)
        return deltas
