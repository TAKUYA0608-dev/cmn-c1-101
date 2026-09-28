"""Marketplace Pod entrypoint — one-shot process, referenced by the Dockerfile CMD.

Template-owned (the scaffold ships a blueprint with placeholders): it targets
this template's own Graph and makes the construction-time LLM decision this
template's reviewed design depends on.

`config["llm"]` decides behaviour, so it is resolved before the agent is built:

  - keys present  -> `config["llm"] = SecretsAzureLLM(provider)`; a call that
    later fails is a withheld item (this template's reviewed per-item design);
  - keys absent   -> no `config["llm"]`, this template's deliberate
    deterministic-only fallback — identical to a keyless HTTP deployment.

One provider, one identity: the chain is built once here with the same
`namespace` `src/api/server.py` hands to `secrets_factory`, the LLM decision
is made from that provider, and it is bound so the runner reuses the same
instance rather than building its own.

Which wheel the image carries is a deploy-time choice, and they differ
(measured against both): `load_agent_config` and
`build_marketplace_secret_provider` exist only in agentcore 1.0.2, so each has
its same-behaviour path on the vendored 1.0.1 written out below;
`wait_for_egress` is awaited by 1.0.2's `marketplace_app` and by neither 1.0.1
wheel, so the wait is owned here. The bootstrap imports stay inside functions:
they ship in the wheel only with the `[marketplace]` extra.
"""

from pathlib import Path

from framework.utils.config_loader import load_config

from src.graph.graph import CrossSessionMemoryQAAgent as Agent

AGENT_NAME = "cmn-c1-101"
# Same secret-provisioning namespace as src/api/server.py, so both entry points
# resolve through one chain. EnvProvider reads os.environ unscoped, so the
# registered Marketplace env vars are unaffected.
NAMESPACE = "cmn-c1-101"

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"


def load_this_agents_config() -> dict:
    """This template's own ``config/config.yaml``, or ``{}`` if absent.

    What agentcore 1.0.2's ``load_agent_config`` does, written out: that helper
    is absent from the vendored 1.0.1 wheel the CoE deploy tool installs, and
    ``load_config`` is present in both (measured).
    """
    return load_config(str(CONFIG_PATH)) if CONFIG_PATH.exists() else {}


def build_secret_provider():
    """The runner's own chain, built from this template's identity."""
    try:
        from shared.bootstrap.marketplace_app import build_marketplace_secret_provider
    except ImportError:
        # 1.0.1: reproduce the same chain rather than shorten it. Falling back to
        # `secrets_factory` alone would silently drop the EnvProvider in front —
        # the one that reads the registered Marketplace environment variables.
        from shared.secrets import factory as secrets_factory
        from shared.secrets.chained_provider import ChainedSecretProvider
        from shared.secrets.env_provider import EnvProvider

        return ChainedSecretProvider(
            EnvProvider(namespace=NAMESPACE, agent_name=AGENT_NAME),
            secrets_factory(namespace=NAMESPACE, agent_name=AGENT_NAME),
            namespace=NAMESPACE,
            agent_name=AGENT_NAME,
        )

    return build_marketplace_secret_provider(agent_name=AGENT_NAME, namespace=NAMESPACE)


def build_config(provider) -> dict:
    """config/config.yaml, plus the LLM iff this deployment is configured for one.

    Returns the file config unchanged when the provider has no Azure secrets —
    the absence of `llm` is the signal the graph reads to stay deterministic.
    """
    base = load_this_agents_config()
    from src.services.azure_llm import SecretsAzureLLM, secrets_available

    if not secrets_available(provider):
        return base
    return {**base, "llm": SecretsAzureLLM(provider)}


def wait_for_egress_before_start() -> bool:
    """Wait for the envoy egress sidecar, fail-open.

    The first thing a keys-present run does is call Azure, so the window in
    which the sidecar is not yet accepting connections matters more here than
    in a deterministic template. If the helper is absent or raises, the run
    continues: the worst case is exactly the pre-wait behaviour. The helper
    bounds itself at 15s and logs its own warning on timeout.
    """
    import asyncio

    try:
        from agenticstar_platform import wait_for_egress
    except ImportError:
        return False
    try:
        asyncio.run(wait_for_egress())
    except Exception:  # noqa: BLE001 - see the fail-open contract above
        return False
    return True


def main() -> None:
    wait_for_egress_before_start()
    # Imported here, not at module scope — see the module docstring.
    from framework.secrets.context import bound_secrets
    from shared.bootstrap.marketplace_app import run_agent_marketplace

    provider = build_secret_provider()
    config = build_config(provider)
    # Bound so the runner reuses this exact provider (its provider-first seam)
    # instead of building a second one — the LLM decision above and every later
    # secret read then come from one instance.
    with bound_secrets(provider):
        run_agent_marketplace(
            Agent,
            agent_name=AGENT_NAME,
            namespace=NAMESPACE,
            config=config,
        )


if __name__ == "__main__":
    main()
