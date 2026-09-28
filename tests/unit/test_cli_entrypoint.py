"""Unit tests — Marketplace entrypoint wiring and the construction-time LLM decision.

Two silent regressions are pinned here. (1) An entry point that still starts
and quietly ignores this template's configuration: the config assertions pin
the loaded VALUES, not the type — `isinstance(config, dict)` passes against a
hard-coded `config={}`. (2) An entry point that builds the agent bare on the
Marketplace so `config["llm"]` is never set and every run silently takes the
deterministic fallback while the registration carries the Azure keys: the LLM
decision is asserted from a provider with and without the keys, and on the
object the runner actually receives.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cli
from src.services.azure_llm import SecretsAzureLLM, secrets_available

REPO_ROOT = Path(cli.__file__).resolve().parent

_FULL = {
    "AZURE_OPENAI_API_KEY": "unit-test-placeholder",
    "AZURE_OPENAI_ENDPOINT": "https://unit.test.example",
    "AZURE_OPENAI_DEPLOYMENT": "unit-deployment",
}


class _Provider:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def _capture(monkeypatch, *, egress=None, provider=None):
    """Run cli.main() with the runner, the provider builder and the egress helper stood in.

    The helper is always stood in — left real, it would try to reach a sidecar
    that does not exist here and block every test for its full 15s timeout.
    """
    seen = {"order": []}
    provider = _Provider({}) if provider is None else provider

    async def _fake_wait():
        seen["order"].append("egress")
        return True

    platform = types.ModuleType("agenticstar_platform")
    if egress != "absent":
        platform.wait_for_egress = _fake_wait if egress is None else egress
    monkeypatch.setitem(sys.modules, "agenticstar_platform", platform)

    def _fake_run(agent_cls, **kwargs):
        from framework.secrets.context import current_secrets

        seen["order"].append("runner")
        seen["agent_cls"] = agent_cls
        seen["bound"] = current_secrets()
        seen.update(kwargs)
        return None

    def _fake_builder(*, agent_name, namespace):
        seen["builder_identity"] = (agent_name, namespace)
        return provider

    module = types.ModuleType("shared.bootstrap.marketplace_app")
    module.run_agent_marketplace = _fake_run
    module.build_marketplace_secret_provider = _fake_builder
    monkeypatch.setitem(sys.modules, "shared.bootstrap.marketplace_app", module)
    cli.main()
    return seen


class TestEntrypointWiring:
    def test_the_loaded_config_yaml_reaches_the_runner(self, monkeypatch):
        expected = cli.load_this_agents_config()
        assert (
            expected
        ), "config/config.yaml must load non-empty, or this test proves nothing"
        config = _capture(monkeypatch)["config"]
        assert {k: v for k, v in config.items() if k != "llm"} == expected

    def test_the_configured_values_arrive(self, monkeypatch):
        # Pinned independently of the loader; these are this repository's own
        # committed values.
        config = _capture(monkeypatch)["config"]
        assert config["max_retry"] == 3
        assert config["timeout_seconds"] == 30

    def test_identity_and_namespace(self, monkeypatch):
        seen = _capture(monkeypatch)
        assert seen["agent_name"] == "cmn-c1-101"
        # Same value src/api/server.py passes to secrets_factory.
        assert seen["namespace"] == "cmn-c1-101"
        assert seen["builder_identity"] == ("cmn-c1-101", "cmn-c1-101")

    def test_the_agent_class_is_the_one_the_registry_declares(self, monkeypatch):
        from src.graph.graph import CrossSessionMemoryQAAgent

        assert _capture(monkeypatch)["agent_cls"] is CrossSessionMemoryQAAgent

    def test_the_runner_reuses_the_bound_provider_instance(self, monkeypatch):
        # One provider object for the LLM decision and for the runner; asserting
        # kwargs alone stays green with the binding deleted (found in review).
        provider = _Provider({})
        assert _capture(monkeypatch, provider=provider)["bound"] is provider


class TestConstructionTimeLlmDecision:
    """No keys -> no config["llm"] -> the template's deliberate deterministic
    fallback. Keys present -> the lazy client rides in on the object the runner
    receives. This decision preserves the reviewed `llm is None` semantics on
    the Marketplace path.
    """

    def test_a_provider_without_keys_decides_no_llm(self, monkeypatch):
        config = _capture(monkeypatch, provider=_Provider({}))["config"]
        # config.yaml may carry an `llm:` SETTINGS mapping; what must be absent is a client.
        assert not callable(getattr(config.get("llm"), "invoke", None))

    def test_partial_keys_mean_not_configured(self, monkeypatch):
        partial = dict(_FULL)
        partial.pop("AZURE_OPENAI_DEPLOYMENT")
        assert not secrets_available(_Provider(partial))
        assert not callable(
            getattr(
                _capture(monkeypatch, provider=_Provider(partial))["config"].get("llm"),
                "invoke",
                None,
            )
        )

    def test_full_keys_put_the_lazy_client_on_the_runner_config(self, monkeypatch):
        config = _capture(monkeypatch, provider=_Provider(_FULL))["config"]
        assert isinstance(config["llm"], SecretsAzureLLM)
        # The file config still rides alongside the client.
        assert {
            k: v for k, v in config.items() if k != "llm"
        } == cli.load_this_agents_config()


class TestTheLazyClient:
    def test_construction_touches_no_client(self):
        # Constructed before secrets are provisioned: nothing may
        # be resolved until the first call.
        llm = SecretsAzureLLM(_Provider({}))
        assert llm._client is None

    def test_both_call_shapes_return_the_model_text(self, monkeypatch):
        calls = []

        class _FakeAzure:
            def __init__(self, cfg):
                calls.append(("init", cfg["azure_deployment"]))

            def complete(self, messages):
                calls.append(("complete", messages[0]["content"]))
                return {"content": "model-text"}

        module = types.ModuleType("shared.services.llm.azure_openai_client")
        module.AzureOpenAIClient = _FakeAzure
        monkeypatch.setitem(
            sys.modules, "shared.services.llm.azure_openai_client", module
        )
        llm = SecretsAzureLLM(_Provider(_FULL))
        assert llm.invoke("p1") == "model-text"
        assert llm.complete("p2", api_key="ignored-on-purpose") == "model-text"
        assert calls == [
            ("init", "unit-deployment"),
            ("complete", "p1"),
            ("complete", "p2"),
        ]

    def test_a_missing_secret_raises_instead_of_fabricating(self):
        import pytest

        with pytest.raises(RuntimeError):
            SecretsAzureLLM(_Provider({})).invoke("p")


class TestEgressWait:
    def test_the_wait_runs_before_the_runner(self, monkeypatch):
        assert _capture(monkeypatch)["order"] == ["egress", "runner"]

    def test_the_run_continues_when_the_helper_is_absent(self, monkeypatch):
        assert _capture(monkeypatch, egress="absent")["order"] == ["runner"]

    def test_the_run_continues_when_the_wait_fails(self, monkeypatch):
        async def _boom():
            raise RuntimeError("sidecar probe failed")

        assert _capture(monkeypatch, egress=_boom)["order"] == ["runner"]
