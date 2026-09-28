"""CMN-C1-101 CrossSessionMemoryQAAgent — graph composition.

L1-direct inheritance from AgentBaseGraph (Cat 1). The 6-step SoT workflow is composed
into the framework's three writable slots (START → initialize → pre_process → main →
post_process → finalize → END, RETRY back to pre_process).

    pre_process  = InputParseNode        (S-1 input boundary; user_id scope key)
    main         = MemoryQAMainNode       (SessionHistoryFetch → MemoryRetrieve
                                          → ContextMerge → ResponseGenerate)
    post_process = OutputValidateNode     (S-3 self-reference + cross-user-leak block
                                          + PII sanitize + S-4 audit)

Cat 1 shape: `main` is a composite FunctionNode (single graph slot) — NOT a Cat 2
GraphNode-in-main. The two stores (session-log + memory) and the LLM are dependency-
injected so the agent is offline-testable. All retrieval is user-scoped.

Default binding (option (a), decided 2026-09-03): when no store is injected the graph
binds EMPTY process-local in-memory stores and states that scope in every envelope
(`memory_scope` + one sentence in the output text). The LLM resolves as explicit
`llm_client` > `config["llm"]` > none (→ `LLM_NOT_CONFIGURED`, no invented answer).
"""

from __future__ import annotations
from typing import Any

from framework.graph.agent_base_graph import AgentBaseGraph

from src.schemas.state import CrossSessionMemoryState
from src.services.service import (
    MEMORY_SCOPE_NOTICE,
    InMemoryMemoryStore,
    InMemorySessionLogStore,
    LLMClient,
    MemoryStore,
    SessionLogStore,
    memory_scope_for,
    resolve_llm_client,
)
from src.nodes.input_parse_node import InputParseNode
from src.nodes.main_node import MemoryQAMainNode
from src.nodes.output_validate_node import OutputValidateNode


# Error codes raised in response to a hostile input. The output for these must
# not name the code or echo the input: doing so tells a prober which filter
# fired and that it fired at all. Kept as data so the policy is one edit, not a
# condition scattered through the node code.
_ADVERSARIAL_CODES = frozenset({"INJECTION_DETECTED"})


class CrossSessionMemoryQAAgent(AgentBaseGraph):
    """Cross-session memory & interaction-history Q&A agent (Cat 1).

    Construction contract
    ---------------------
    The Marketplace runner constructs the agent as ``agent_cls(config=<config.yaml>)``
    — no keyword stores, no ``llm_client`` — so whatever this initializer binds by
    default IS the Marketplace deployment.

    * ``session_store`` / ``memory_store`` not injected → EMPTY ``InMemorySessionLogStore``
      / ``InMemoryMemoryStore`` (``src/services/service.py``). They are process-local:
      on the Marketplace every execution runs in a fresh Pod, so memory lasts ONE
      execution and nothing is recalled across sessions. This is the deliberate
      option (a) — the agent is loaded with in-memory stores and says so plainly
      (``memory_scope="this_execution_only"`` in the envelope + one sentence in the
      output text) rather than implying cross-session recall. Persistence requires
      injecting real stores at the registry boundary; then the scope reads
      ``"persistent"``. The scope is derived from the bound objects, never from a flag.
    * ``llm_client`` explicit kw > ``config["llm"]`` (an object answering ``invoke``
      / ``complete``, adapted to ``LLMClient``) > none. With none, the answer step
      degrades with ``LLM_NOT_CONFIGURED`` (SUCCESS + error_code + non-empty notice);
      it never invents an answer.
    * Memory-store WRITES are out of scope by design (a separate Update Agent).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        session_store: SessionLogStore | None = None,
        memory_store: MemoryStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._session_store: SessionLogStore = session_store if session_store is not None else InMemorySessionLogStore()
        self._memory_store: MemoryStore = memory_store if memory_store is not None else InMemoryMemoryStore()
        self._memory_scope = memory_scope_for(self._session_store, self._memory_store)
        self._llm_client = resolve_llm_client(llm_client, config)
        super().__init__(config)

    @property
    def memory_scope(self) -> str:
        """`"this_execution_only"` (in-memory default) or `"persistent"` (real stores bound)."""
        return self._memory_scope

    @property
    def name(self) -> str:
        return "CrossSessionMemoryQAAgent"

    @property
    def state_schema(self) -> type:
        """Domain State so per-node fields survive node merges (LangGraph drops
        keys not declared in the schema)."""
        return CrossSessionMemoryState

    def register_nodes(self) -> None:
        super().register_nodes()  # framework injects InitializeNode + FinalizeNode
        self._nodes["pre_process"] = InputParseNode()
        self._nodes["main"] = MemoryQAMainNode(
            session_store=self._session_store,
            memory_store=self._memory_store,
            llm_client=self._llm_client,
        )
        self._nodes["post_process"] = OutputValidateNode()

    def get_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Surface the memory-Q&A payload (this agent writes answer/cited_session_ids,
        not the framework-default output)."""
        # A degraded run (SUCCESS + error_code, no answer) would otherwise
        # surface an empty output, and the Marketplace runner rejects a
        # successful invocation whose output is missing — verified on the Pod
        # with a sibling template. Surface the degradation itself instead: this states
        # what happened, it does not invent an answer.
        #
        # Only on SUCCESS. A refused request (framework S-2 refusal, status
        # ERROR) must keep publishing nothing — that is the security property
        # `assert_framework_refused` pins, and the runner treats a non-success
        # invocation as a failure regardless of output, so there is nothing to
        # rescue there.
        answer = state.get("answer")
        if not answer and str(state.get("status", "")).lower().endswith("success"):
            code = state.get("error_code") or "NO_CONTENT"
            if code == "LLM_NOT_CONFIGURED":
                # Context may have been retrieved; nothing synthesized it. Name the
                # cause and the counts — a notice, not an answer.
                answer = (
                    "This request could not be completed (error_code=LLM_NOT_CONFIGURED). "
                    "No language model is bound to this deployment, so no answer was "
                    "synthesized from your session context "
                    f"(history entries: {state.get('history_count') or 0}, "
                    f"memory hits: {state.get('memory_hit_count') or 0})."
                )
            elif code in _ADVERSARIAL_CODES:
                # Injection policy: a caller who probed the input filter learns
                # only that the request was not completed. Naming the code
                # would confirm the probe worked and hand back which filter
                # fired. The framework refuses high-confidence markers outright
                # (status ERROR, no output); this is the template-side branch,
                # where the run stays SUCCESS and therefore must still carry an
                # output for the runner — but a deliberately uninformative one.
                answer = "This request could not be completed."
            else:
                answer = (
                    "This request could not be completed "
                    f"(error_code={code}). No answer content was produced; "
                    "see error_code and error_log for the degradation cause."
                )
        # Every published text carries the memory-scope sentence (option (a)): the
        # reader must never be led to believe cross-session recall happened when the
        # stores are process-local. A refused request (no answer) publishes nothing.
        if answer:
            answer = f"{answer}\n\n{MEMORY_SCOPE_NOTICE[self._memory_scope]}"
        return {
            "output": answer,
            "memory_scope": self._memory_scope,
            "answer": state.get("answer"),
            "cited_session_ids": state.get("cited_session_ids"),
            "history_count": state.get("history_count"),
            "memory_hit_count": state.get("memory_hit_count"),
            "context_size_estimate": state.get("context_size_estimate"),
            "validation_status": state.get("validation_status"),
            "pii_redaction_count": state.get("pii_redaction_count"),
            "cross_user_blocked": state.get("cross_user_blocked"),
            "audit_logged": state.get("audit_logged"),
            "status": state.get("status"),
            "error_code": state.get("error_code"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
            "error_log": state.get("error_log", []),
        }


# Backward-compat alias — the scaffold (api/server.py) imports `Graph`.
Graph = CrossSessionMemoryQAAgent
