"""CMN-C1-101 — service layer: session-log + memory stores + LLM (DI).

Two external stores (the user-scoped session-log store, and the semantic memory store)
plus the LLM are injected as Protocols so the template is offline-testable and
deployment-agnostic. A persistent deployment binds the real session-log DB + memory
store (mem0 / Letta / pgvector); when nothing is injected the graph binds the EMPTY
in-memory stores defined here (option (a): process-local, one execution only) and
reports that scope in every envelope — see `memory_scope_for`. The LLM comes from an
explicit `llm_client`, else `config["llm"]` (adapted by `resolve_llm_client`), else
none — in which case the answer step degrades with `LLM_NOT_CONFIGURED` instead of
inventing an answer.

**Both stores are user-scoped**: every read takes `user_id` and returns ONLY that user's
records — the first line of defense against cross-user memory leak.

No `agenticstar` (Level 0) imports; no `framework.*` dependency — pure domain logic.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

# Recency-decay half-life (in the same integer "ts" unit the stores use). Older records
# decay toward zero weight; this makes ContextMerge selection deterministic + auditable.
RECENCY_HALFLIFE = 10
TOKEN_BUDGET = 1200  # approx token budget for the merged context
_TOKENS_PER_CHAR = 0.25  # crude char→token estimate for offline budgeting

_STOPWORDS: frozenset[str] = frozenset(
    "a an the is are was were be been to of for and or in on at by with from as that "
    "this it its do does did what which who how when where why can could should would "
    "will not no our your their we i you my about last week".split()
)


def _terms(text: str) -> set[str]:
    return {t for t in text.lower().split() if t and t not in _STOPWORDS and len(t) > 2}


def recency_weight(ts: int, now: int) -> float:
    """Exponential recency decay: weight = 0.5 ** (age / halflife), clamped to (0, 1]."""
    age = max(0, now - int(ts))
    return round(cast(float, 0.5 ** (age / RECENCY_HALFLIFE)), 4)


@runtime_checkable
class SessionLogStore(Protocol):
    """User-scoped interaction-history store.

    `fetch` returns this user's recent history entries, each shaped:
        {"session_id": str, "ts": int, "role": str, "text": str}
    `window` is an OPTIONAL retrieval-scope spec passed through as a JSON string —
    either an ISO-8601 range `{"from_ts","to_ts"}` or a session-id allowlist `["s-101", ...]`;
    `None` = full user scope. The backend parses/applies it; the agent only forwards it.
    """

    def fetch(self, user_id: str, window: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...


@runtime_checkable
class MemoryStore(Protocol):
    """User-scoped semantic memory store.

    `search` returns this user's top-k relevant memory chunks, each shaped:
        {"session_id": str, "ts": int, "text": str, "score": float}
    """

    def search(self, user_id: str, query: str, top_k: int = 6) -> list[dict[str, Any]]: ...


@runtime_checkable
class LLMClient(Protocol):
    """LLM boundary — `generate(prompt) -> str`."""

    def generate(self, prompt: str) -> str: ...


class InMemorySessionLogStore:
    """In-memory SessionLogStore (user-scoped). Starts EMPTY; bound by default when no
    store is injected (process-local — lost when the process ends)."""

    def __init__(self) -> None:
        self._rows: dict[str, list[dict[str, Any]]] = {}

    def add(self, user_id: str, rows: list[dict[str, Any]]) -> None:
        self._rows.setdefault(user_id, []).extend(rows)

    def fetch(self, user_id: str, window: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        rows = list(self._rows.get(user_id, []))
        rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
        return rows[:limit]


class InMemoryMemoryStore:
    """In-memory MemoryStore (user-scoped, lexical ranking). Starts EMPTY; bound by
    default when no store is injected (process-local — lost when the process ends)."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[dict[str, Any]]] = {}

    def add(self, user_id: str, chunks: list[dict[str, Any]]) -> None:
        self._chunks.setdefault(user_id, []).extend(chunks)

    def search(self, user_id: str, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        q = _terms(query)
        scored = []
        for c in self._chunks.get(user_id, []):
            words = _terms(str(c.get("text", "")))
            score = (len(q & words) / len(q)) if q and words else 0.0
            if score > 0.0:
                scored.append({**c, "score": round(score, 4)})
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_k]


class StubLLMClient:
    """Deterministic stub LLM for tests / local runs. Never bound by default: with no
    LLM the answer step degrades with `LLM_NOT_CONFIGURED` (see `resolve_llm_client`)."""

    def __init__(self, canned: str | None = None) -> None:
        self._canned = canned

    def generate(self, prompt: str) -> str:
        if self._canned is not None:
            return self._canned
        tail = prompt.strip().splitlines()[-1][:200] if prompt.strip() else ""
        return f"Based on your prior sessions: {tail}"


# ── Memory scope (option (a)) ─────────────────────────────────────────────────
# The graph binds these when no store is injected. They are process-local: on the
# Marketplace every execution is a fresh Pod, so nothing survives to the next
# session. Rather than hide that, every envelope carries a machine-readable scope
# plus one user-facing sentence, so a reader never mistakes a cold start for
# "no memory of you exists". Kept here (one place) so the wording cannot drift.
MEMORY_SCOPE_THIS_EXECUTION = "this_execution_only"
MEMORY_SCOPE_PERSISTENT = "persistent"

MEMORY_SCOPE_NOTICE: dict[str, str] = {
    MEMORY_SCOPE_THIS_EXECUTION: (
        "Memory scope: this execution only. This deployment keeps session memory "
        "in-process and starts empty on every run, so nothing from your earlier "
        "sessions was recalled here."
    ),
    MEMORY_SCOPE_PERSISTENT: (
        "Memory scope: persistent. This deployment reads a persistent memory store, "
        "so the answer may draw on your earlier sessions."
    ),
}

_IN_MEMORY_STORE_TYPES: tuple[type, ...] = (InMemorySessionLogStore, InMemoryMemoryStore)


def memory_scope_for(session_store: Any, memory_store: Any) -> str:
    """Derive the scope statement from what is actually bound.

    "persistent" only when BOTH stores are bound and NEITHER is one of the in-memory
    classes above; anything else (a missing store, or an in-memory one — even when it
    was injected explicitly) is "this_execution_only". Decided from the bound objects,
    not from a flag, so the claim cannot be set without the backing store.
    """
    if session_store is None or memory_store is None:
        return MEMORY_SCOPE_THIS_EXECUTION
    if isinstance(session_store, _IN_MEMORY_STORE_TYPES) or isinstance(memory_store, _IN_MEMORY_STORE_TYPES):
        return MEMORY_SCOPE_THIS_EXECUTION
    return MEMORY_SCOPE_PERSISTENT


# ── LLM seam: config["llm"] → LLMClient ───────────────────────────────────────
def _as_text(result: Any) -> str:
    """Coerce a client reply to text: str as-is, message-like objects via `.content`."""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return "" if result is None else str(result)


class ConfigLLMAdapter:
    """Adapt a `config["llm"]` object to this template's `LLMClient` Protocol.

    The fleet entry point places a lazily-resolved chat client under `config["llm"]`
    that answers `invoke(prompt) -> str` (and `complete(prompt, **kw) -> str`); this
    template's nodes speak `generate(prompt) -> str`. The adapter forwards to `invoke`
    first, then `complete`, and never swallows the client's exceptions — a configured
    but failing LLM must surface, not silently degrade.
    """

    def __init__(self, client: Any) -> None:
        call = getattr(client, "invoke", None)
        if not callable(call):
            call = getattr(client, "complete", None)
        if not callable(call):
            raise TypeError(
                "config['llm'] must expose generate(prompt), invoke(prompt) or "
                f"complete(prompt); got {type(client).__name__}"
            )
        self._client = client
        self._call = call

    def generate(self, prompt: str) -> str:
        return _as_text(self._call(prompt))


def resolve_llm_client(explicit: LLMClient | None, config: Any) -> LLMClient | None:
    """Precedence: explicit `llm_client` kw > `config["llm"]` > None.

    None means "no LLM bound": ResponseGenerate then degrades with `LLM_NOT_CONFIGURED`
    (SUCCESS + error_code + non-empty notice) rather than producing an answer. An
    object under `config["llm"]` that answers none of generate/invoke/complete raises
    at construction (misconfiguration is not a reason to degrade quietly).
    """
    if explicit is not None:
        return explicit
    candidate = config.get("llm") if isinstance(config, dict) else None
    if candidate is None:
        return None
    if isinstance(candidate, LLMClient):
        return candidate
    return ConfigLLMAdapter(candidate)


def estimate_tokens(text: str) -> int:
    return int(len(text) * _TOKENS_PER_CHAR)


def seed_stores(user_id: str = "u1") -> tuple[InMemorySessionLogStore, InMemoryMemoryStore]:
    """Return (session-log, memory) stores seeded for `user_id` — illustrative offline data.

    Tests / demos only. The graph never binds seeded stores by default: illustrative
    rows must not be served to a real caller as their own history."""
    logs = InMemorySessionLogStore()
    logs.add(
        user_id,
        [
            {
                "session_id": "s-101",
                "ts": 100,
                "role": "user",
                "text": "we decided to use postgres for the billing service",
            },
            {
                "session_id": "s-101",
                "ts": 101,
                "role": "assistant",
                "text": "agreed, postgres for billing; revisit sharding later",
            },
            {
                "session_id": "s-104",
                "ts": 104,
                "role": "user",
                "text": "my preference is dark mode and concise answers",
            },
        ],
    )
    mem = InMemoryMemoryStore()
    mem.add(
        user_id,
        [
            {"session_id": "s-101", "ts": 100, "text": "decision: postgres chosen for the billing service backend"},
            {"session_id": "s-104", "ts": 104, "text": "user preference: dark mode, concise responses"},
            {"session_id": "s-090", "ts": 90, "text": "older context: evaluated mysql vs postgres for billing"},
        ],
    )
    return logs, mem
