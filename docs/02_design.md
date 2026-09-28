# Template Design Specification — CMN-C1-101

**Template ID:** CMN-C1-101
**Agent Class:** CrossSessionMemoryQAAgent
**Category:** Cat 1 (single technical capability — cross-session memory & interaction-history Q&A)
**Industry:** CMN (cross-industry)
**SoT:** scaffold/#626, SoT note #936771 (JP) / #936770 (EN)

> **Clarifications carried from evaluation (SoT §1):**
> - L1 base is `AgentBaseGraph` (L1-direct). The architect spec's `ChatAgent`
>   notation reflects the pre-2026-05-18 policy; the ChatAgent-style pipe-and-filter shape
>   (input → fetch → retrieve → merge → generate → validate) is a *conceptual reference only*.
> - **Out of scope** (SoT §4): memory-store *writes* (a separate Update Agent) and
>   summarization/compression. This template is **read + Q&A** over existing memory.
> - **Security-critical property:** all retrieval is **user-scoped**; cross-user memory leak is
>   the primary risk, enforced by OutputValidate (every cited session must belong to `user_id`).

## Position in AgentCore Architecture

- **L1 Base: `AgentBaseGraph`** (Cat 1, L1-direct per the graph contract; `agents/base/*` not used)
- **Three-Layer Separation:**
  - State: flat TypedDict composition (`CrossSessionMemoryState(AgentState)`) — no Pydantic
  - Node: L1 inheritance (`FunctionNode`, `execute(self, state, config=None) -> dict` override only)
  - Graph: composition (`register_nodes()` fills the 3 writable slots)

## Architecture Overview

### Node Configuration

The SoT 6-step workflow is composed into the framework's three writable slots. The domain
nodes map as:

| Slot | Node | Responsibility | Inherits |
|------|------|---------------|----------|
| initialize | InitializeNode | schema_version, session_id, trust_level | default (framework) |
| **pre_process** | **InputParseNode** | S-1 input boundary: validate user_id + query + optional session_window; NFKC normalize, injection reject, size gate. **parse only** — no retrieval. | FunctionNode |
| **main** | **MemoryQAMainNode** composing → | composite of the 4 sub-nodes below | FunctionNode |
| ↳ | SessionHistoryFetchNode | fetch the `user_id`-scoped interaction history from the session-log store (DI; **default = empty process-local `InMemorySessionLogStore`**, ADR-7); never reads other users' logs | FunctionNode |
| ↳ | MemoryRetrieveNode | retrieve top-k `user_id`-scoped chunks from the external memory store (mem0 / Letta / pgvector, DI; **default = empty process-local `InMemoryMemoryStore`**, ADR-7); user-scoped query | FunctionNode |
| ↳ | ContextMergeNode | **deterministic recency-weighted, token-budget merge** of history + memory chunks (temporal decay × relevance); no LLM | FunctionNode |
| ↳ | ResponseGenerateNode | LLM call over the merged context → natural-language answer + the set of `cited_session_ids` it drew from. LLM = explicit `llm_client` > `config["llm"]` (adapted) > none; **with none and non-empty context → `LLM_NOT_CONFIGURED`** (SUCCESS + error_code, no answer) | FunctionNode |
| **post_process** | **OutputValidateNode** | **S-3 self-reference gate** (a substantive answer must cite ≥1 session it is grounded in) + **cross-user-leak block** (every cited session ∈ this user's scope, else reject) + **S-2 PII sanitization** of the output; **S-4 audit** (always fires) | FunctionNode |
| finalize | FinalizeNode | response_metadata, total_time_ms | default (framework) |

### Data Flow

```
START → initialize → pre_process(InputParse) → main(MemoryQAMain) → {route}
        → post_process(OutputValidate) → finalize → END
                                            ↓ (retry, max 3)
                                          pre_process
```

Error propagation: any node sets `error_code`; downstream sub-nodes self-skip (`return {}`).
A zero-history / zero-memory result flows to a safe "no prior context found" answer (no
fabricated history). OutputValidate's S-4 audit fires on every path, including errors.

Every published envelope carries `memory_scope` (`"this_execution_only"` | `"persistent"`,
derived from the bound stores — ADR-7) and the output text ends with the matching
one-sentence notice (`service.MEMORY_SCOPE_NOTICE`). A refused request (status `error`)
publishes nothing, so no scope text either.

### Error Codes (degraded runs: `status=success` + `error_code` + non-empty `output`)

| error_code | Set by | Meaning | Output text |
|-----------|--------|---------|-------------|
| `USER_ID_MISSING` | InputParse | no `user_id` / `input_context.user_id` / `caller_id` | generic notice naming the code |
| `INPUT_EMPTY` / `INPUT_TOO_LONG` | InputParse | query empty / > 2000 chars | generic notice naming the code |
| `INJECTION_DETECTED` | InputParse | template-side injection pattern | deliberately uninformative (code not named, probe not echoed) |
| `MAIN_PIPELINE_ERROR` | MemoryQAMain | a sub-node raised (missing upstream state / store failure) | generic notice naming the code |
| `LLM_NOT_CONFIGURED` | ResponseGenerate | context retrieved but no LLM bound (`llm_client` / `config["llm"]`) | notice naming the code + history/memory counts; **never a stub answer** |

### State Definition (`src/schemas/state.py`)

| Field | Type | Purpose | Written by |
|-------|------|---------|-----------|
| user_id / query / session_window | Optional[str] | caller input (user_id = scope key; **session_window** = optional retrieval scope as a JSON string — `{"from_ts","to_ts"}` ISO-8601 range **or** a session-id allowlist; `None` = full user scope) | caller |
| validated_query | Optional[str] | sanitized / normalized query | InputParse |
| history_log / history_count | Optional[str] / Optional[int] | user-scoped history JSON [{session_id, ts, role, text}] + count | SessionHistoryFetch |
| memory_chunks / memory_hit_count | Optional[str] / Optional[int] | user-scoped memory chunk JSON [{session_id, ts, text, score}] + count | MemoryRetrieve |
| merged_context / context_token_estimate | Optional[str] / Optional[int] | recency-weighted merged context JSON + token estimate | ContextMerge |
| answer / cited_session_ids | Optional[str] | NL answer + JSON list of session_ids it cites | ResponseGenerate |
| validation_status / pii_redaction_count / cross_user_blocked / audit_logged | Optional[str/int/bool] | S-3/S-2/S-4 outcome | OutputValidate |
| error_code / error_message | Optional[str] | error propagation | any node |

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serialized strings); no Pydantic / dataclass (msgpack)
- No JWT, API keys, credentials in State; **no other user's data** ever enters State (user-scoped retrieval)
- InvocationContext (incl. caller identity used to authorize `user_id` scope) via `config["configurable"]` only
- PII in the answer is sanitized by OutputValidate before it leaves the agent

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (caller_trust_level, session_id) — via `config["configurable"]`
- [ ] ConnectionPolicy (retry/timeout) — N/A at the agent layer; store retry/timeout is the DI backend's responsibility (`SessionLogStore` / `MemoryStore`)
- [x] SecurityViolationError — the framework S-1/S-2 `@final` gates raise it; domain checks (injection / cross-user leak) propagate via `error_code`
- [x] S-2: framework `@final` `_security_gate_input()` runs automatically (FunctionNode). No domain `_extra_security_gate_input()` needed — InputParse does NFKC + injection reject + size cap in `execute()`; **output PII sanitization** is the domain S-2-style check, placed in OutputValidate.
- [x] S-3: framework `@final` `_security_gate_output()` (credential scan) runs automatically; **domain S-3 logic** (self-reference / cited-session gate + cross-user-leak block) lives in OutputValidateNode.`execute()`.
- [x] S-4: `emit_trace_event()` (`src/utils/audit.py` wrapper → `shared.utils.audit_logger`) called inside `execute()` of SessionHistoryFetch, MemoryRetrieve, ContextMerge, ResponseGenerate (incl. `llm_not_configured`), OutputValidate. `node_start`/`node_complete`/`node_error` are NOT emitted by templates.

### Composition Pattern
- **Pattern:** Standalone (Cat 1) — single-agent, no GraphNode/RemoteAgentNode subgraph
- **Sub-node invocation:** `MemoryQAMainNode` instantiates its 4 sub-nodes in `__init__()` (`self._seq = [...]`) and calls each via `sub_node.execute(state, config)` **directly — not `__call__()`**; the composite owns the single S-2/S-3/S-4 boundary. Same composite pattern as its sibling templates.
- **Error propagation strategy:** propagate via `error_code`; terminal audit always fires

## Import Isolation Confirmation
- [x] Template does not import `agenticstar` (Level 0) — PB-4 AST scan
- [x] Import targets: `framework/` (FunctionNode, AgentBaseGraph, AgentState, AgentStatus, InvocationContext) + `shared/` (audit_logger via wrapper) only

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed 6-step Q&A pipeline, no autonomous loop (Cat 1) |
| InputParse scope | parse + retrieve | parse / validate only | **parse only** | retrieval is the main-slot sub-nodes' responsibility; keep S-1 boundary thin |
| Stores | one combined store | **two DI Protocols** (SessionLogStore + MemoryStore) | **two DI Protocols** | history (recent, structured) and semantic memory are distinct backends (mem0/Letta/pgvector); both user-scoped, both offline-testable |
| Context merge | LLM-summarized | deterministic recency-weighted token-budget | **deterministic** | auditability + reproducibility of which sessions were selected; LLM only generates the answer |
| Output gate | best-effort | self-reference + cross-user-leak block + PII sanitize | **strict** | cross-user memory leak is the #1 risk; every cited session must be in `user_id` scope or the answer is rejected |
| Backends | committed clients | DI Protocol + in-memory stubs | **DI Protocol** | offline-testable; prod binds the real session-log + memory stores |
| **ADR-7** Default store binding on the Marketplace (2026-09-03) | (b) require injected persistent stores (fail at construction / stay off the Marketplace until a platform store exists) | (a) bind empty in-memory stores by default and state the scope explicitly | **(a)** | see below |

### ADR-7 — Option (a): in-memory default binding + explicit memory scope

- **Context.** The Marketplace runner constructs the agent as `agent_cls(config=<config.yaml>)`
  — no keyword stores, no `llm_client`. Before this ADR the bare construction silently bound the
  *seeded demo* stores (`seed_stores()`) and a `StubLLMClient`, so a caller whose id happened to
  be `u1` was served illustrative rows as their own history, and every other caller got a
  "no prior context" answer with no indication that the deployment cannot remember anyone.
  The platform offers no persistent session-log / memory store yet.
- **Decision (TAKUYA, 2026-09-03).** Load CMN-C1-101 on the Marketplace with **empty, process-local
  in-memory stores bound by default** and **say plainly that memory lasts one execution only**:
  the envelope carries `memory_scope: "this_execution_only"` and the output text ends with
  `service.MEMORY_SCOPE_NOTICE[...]`. The scope is derived from the bound objects
  (`service.memory_scope_for`): `"persistent"` only when both stores are injected and neither is
  an in-memory class — it cannot be claimed by a flag. Precedent: the MemoryAgent pattern is
  registered with a process-local backend.
- **LLM.** `llm_client` kw > `config["llm"]` (an object answering `invoke(prompt)` /
  `complete(prompt)`, adapted by `service.ConfigLLMAdapter` to `generate(prompt)`) > none. With
  none and non-empty context the answer step degrades with `LLM_NOT_CONFIGURED`; the stub LLM is
  never bound by default (an echoed context is not an answer). An unusable `config["llm"]` object
  raises at construction rather than degrading quietly.
- **Consequences.** No cross-session recall on the Marketplace until a real
  `SessionLogStore` / `MemoryStore` is injected at the registry boundary (docs/07 §2); reads within
  one execution work; writes remain out of scope (separate Update Agent). Node-by-node tests could
  not see any of this, so `tests/integration/test_full_path_invoke.py` pins the bare-construction
  path on the real SDK.


## Cat 1 Genericity & Parameterization

The capability — **user-scoped cross-session memory retrieval + grounded Q&A** — embeds no
use case. `SessionLogStore` / `MemoryStore` / `LLMClient` are injected Protocols (swap mem0
for Letta/pgvector with no node change); recency-decay half-life, `top_k`, and token budget
are module constants; the State holds generic memory artifacts (history_log / memory_chunks /
cited_session_ids), so a Cat 2 industry sibling (e.g. a CX-specific memory agent) can inherit
this as its `main`-slot building block.
