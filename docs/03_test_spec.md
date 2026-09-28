# Test Specification — CMN-C1-101 CrossSessionMemoryQAAgent

## Test Strategy
- Coverage target: **≥ 80%** (achieved **89%** across `src/`, excl. FastAPI entrypoint)
- Test types: Unit (per-node + agent) / Integration (3-slot pipeline) / Proof-of-Boundary
- Backends are dependency-injected: tests bind `seed_stores()` (in-memory session-log + memory) + `StubLLMClient` — no network, deterministic, user-scoped.

> **Note:** `_extra_security_gate_input()` / `_extra_security_gate_output()` are **not used** —
> domain input checks live in `InputParseNode.execute()`, and domain S-3 (self-reference gate /
> cross-user-leak block / PII sanitize) lives in `OutputValidateNode.execute()`; the framework
> `@final` gates run automatically. TC-09/TC-10 (`_extra` hooks non-trivial) are therefore N/A.

## Framework Compliance Tests (Mandatory)

| TC-ID | Test | Result |
|-------|------|--------|
| TC-01 | State contract: flat TypedDict (`CrossSessionMemoryState`) | ✅ PB-2 |
| TC-02 | Input rejection (missing user_id / empty / oversize / injection) | ✅ |
| TC-03 | No JWT/Credential in `src/` (CI gate-credential-scan) | ✅ CI |
| TC-04 | InvocationContext via `config["configurable"]` only | ✅ |
| TC-05 | S-4: no duplicate lifecycle events in `execute()` | ✅ |
| TC-06/07 | S-2/S-3 `@final` gates not overridden | ✅ |
| TC-08 | `required_trust_level` = `VERIFIED_EXTERNAL` (valid enum) | ✅ |
| TC-11 | S-4: ≥1 domain `emit_trace_event()` per side-effect node | ✅ (5 nodes) |

## Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Result |
|-------|----------|--------|
| PB-2 | State serialization (primitives + JSON strings; no credential-named fields) | ✅ |
| PB-4 | Import isolation (no Level 0 `agenticstar`) | ✅ |
| PB-1/5/6 | Audit / checkpoint / invoke order via agent e2e | ✅ |

## Business Logic Tests (domain)

| Area | Coverage |
|------|----------|
| InputParse | **user_id required**, injection/empty rejection, normalization |
| SessionHistoryFetch / MemoryRetrieve | **user-scoped** (other user gets 0), hit counts |
| ContextMerge | **deterministic recency-weight decay**, recency-sorted merge, token-budget cap, empty |
| ResponseGenerate | cites merged sessions, no-context safe answer |
| OutputValidate | self-reference gate, **cross-user-leak block** (cited session ∉ user scope → blocked), PII redaction (email/phone/マイナンバー), audit on error path |
| main composite | 4-step run, node contract |
| agent e2e | invoke: grounded answer, missing user_id, injection, no-history user, node order |
| ResponseGenerate (no LLM) | context retrieved + no LLM bound → `LLM_NOT_CONFIGURED`, no stub answer |
| full-path invoke (`tests/integration/test_full_path_invoke.py`, real SDK) | bare `Graph(config=<config.yaml>)` exactly as the Marketplace runner constructs it: SUCCESS + non-empty output + `memory_scope=this_execution_only` + scope sentence; default stores start empty (demo user id `u1` gets no seeded rows); `caller_id` → scope key; `config["llm"]` reaches the answer node (prompt seen, answer derives from its reply), explicit `llm_client` wins; no LLM → `LLM_NOT_CONFIGURED`; injected seeded store shows in the answer; non in-memory stores → `memory_scope=persistent`; refusal publishes no scope text |

## Test Execution Summary
- Total tests: **56** (real SDK `agenticstar-agentcore` 1.0.x) — Pass: **54** / Fail: 0 / Skip: 2 (framework injection-policy probes skipped where the gate module is absent)
- Coverage: **89%** (`--cov=src`)
