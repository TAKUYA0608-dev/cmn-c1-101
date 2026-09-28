"""Azure OpenAI adapter for the Marketplace runtime (the `config["llm"]` seam).

This template carries its LLM through `config["llm"]` into the inner workflow.
The Marketplace entry point constructs the agent before secrets are
provisioned, so this client resolves its connection values
lazily — at the first call, not at construction. This is the per-template
shape identified as the conformant option until a shared lazy client
ships in `shared.services.llm` (the shape used in production).

Degradation semantics stay the TEMPLATE's: the entry point decides **at
construction, through the secret provider** whether to inject this client at
all — no keys -> no client -> this template's reviewed deterministic fallback,
exactly as on the HTTP path with no `config["llm"]`. A call that fails raises;
returning a fabricated string would assert an interpretation the model never
made, and every calling node here already treats an exception as a withheld
item.

Secrets are read through the provider chain (never `os.environ` directly —
the platform secrets rule); `EnvProvider` is the sanctioned reader of the registration env.
"""

from __future__ import annotations

from typing import Any, cast

_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)


def secrets_available(provider: Any) -> bool:
    """True when every Azure OpenAI value is present in the provider."""
    try:
        return all(bool(provider.get(k)) for k in _SECRET_KEYS)
    except Exception:
        return False


class SecretsAzureLLM:
    """Lazy Azure OpenAI client exposing both call shapes this template uses.

    `invoke(prompt) -> str` and `complete(prompt, **kwargs) -> str` return the
    model's text; keyword arguments the HTTP-path Anthropic client accepted
    (e.g. `api_key`) are ignored here because the key comes from the provider.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from shared.services.llm.azure_openai_client import AzureOpenAIClient

            values = {k: self._provider.get(k) for k in _SECRET_KEYS}
            missing = [k for k, v in values.items() if not v]
            if missing:
                raise RuntimeError(f"azure llm secrets missing: {sorted(missing)}")
            self._client = AzureOpenAIClient(
                {
                    "api_key": values["AZURE_OPENAI_API_KEY"],
                    "azure_endpoint": values["AZURE_OPENAI_ENDPOINT"],
                    "azure_deployment": values["AZURE_OPENAI_DEPLOYMENT"],
                }
            )
        return self._client

    def invoke(self, prompt: str) -> str:
        response = self._ensure_client().complete([{"role": "user", "content": prompt}])
        return cast(str, response["content"])

    def complete(self, prompt: str, **_ignored: Any) -> str:
        return self.invoke(prompt)
