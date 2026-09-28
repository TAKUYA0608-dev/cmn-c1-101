# CMN-C1-101 — Integration: full pre_process → main → post_process pipeline

from framework.schemas.agent_status import AgentStatus

from src.nodes.input_parse_node import InputParseNode
from src.nodes.main_node import MemoryQAMainNode
from src.nodes.output_validate_node import OutputValidateNode
from src.services.service import seed_stores, StubLLMClient


def _run(initial: dict, canned: str = "You chose postgres for billing.") -> dict:
    logs, mem = seed_stores("u1")
    pre = InputParseNode()
    main = MemoryQAMainNode(session_store=logs, memory_store=mem, llm_client=StubLLMClient(canned=canned))
    post = OutputValidateNode()
    state = dict(initial)
    for node in (pre, main, post):
        state.update(node.execute(state) or {})
    return state


class TestAgentE2E:
    def test_happy_path(self):
        import json
        s = _run({"user_id": "u1", "query": "what did we decide about the billing database?"})
        assert s["status"] == AgentStatus.SUCCESS.value
        assert s["history_count"] >= 1
        assert json.loads(s["cited_session_ids"])
        assert s["validation_status"] in ("passed", "redacted")
        assert s["cross_user_blocked"] is False
        assert s["audit_logged"] is True

    def test_missing_user_id_short_circuits_but_audit_fires(self):
        s = _run({"query": "anything?"})
        assert s["error_code"] == "USER_ID_MISSING"
        assert s["audit_logged"] is True

    def test_no_history_user(self):
        import json
        s = _run({"user_id": "u_nobody", "query": "what did we discuss?"})
        assert s["history_count"] == 0
        assert json.loads(s["cited_session_ids"]) == []
        assert s["audit_logged"] is True
