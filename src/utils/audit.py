"""S-4 structured audit logging.

Binds the platform ``emit_trace_event()`` from ``shared.utils.audit_logger`` when
importable; otherwise falls back to a stderr JSON-Lines sink (same shape), so audit
events are never silently dropped. Field NAMES + counts only — never raw memory text,
PII, or another user's data.
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:  # pragma: no cover
    from shared.utils.audit_logger import emit_trace_event as _platform_emit
except Exception:
    _platform_emit = None


def emit_trace_event(
    event_type: str, payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None
) -> None:
    """Emit one domain audit event (platform sink when available, else stderr)."""
    if _platform_emit is not None:  # pragma: no cover
        _platform_emit(event_type, payload or {}, state or {})
        return
    event: dict[str, Any] = {"template_id": "CMN-C1-101", "event_type": event_type}
    if payload:
        event.update(payload)
    if state is not None:
        tid = state.get("trace_id")
        if tid:
            event["trace_id"] = tid
        sid = state.get("session_id")
        if sid:
            event["session_id"] = sid
    print(json.dumps(event, ensure_ascii=False), file=sys.stderr)
