"""
Settings of the Leonardo AI assistant.

API keys live in the operating system keychain (macOS Keychain, Windows Credential
Manager, Secret Service) through ``keyring``, with the usual environment variables as
fallback; they are never written to the settings file. Everything else (provider,
model, effort, approval policy, MCP server) is a small JSON file next to Leonardo's
other settings.
"""

import json
import os

KEYRING_SERVICE = "leonardo-ai-assistant"
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

# Pinned model lists (the Refresh button in Settings asks each provider for the live list).
ANTHROPIC_MODELS = [
    ("claude-opus-5", "Claude Opus 5"),
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-fable-5-1", "Claude Fable 5.1 (most capable)"),
    ("claude-sonnet-5", "Claude Sonnet 5"),
    ("claude-haiku-4-5", "Claude Haiku 4.5 (fast)"),
]
# OpenAI's catalogue changes often: the list comes from the API (Refresh); a model id can
# also be typed in.
OPENAI_MODELS = []
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

DEFAULTS = {
    "provider": "anthropic",
    "model_anthropic": "claude-opus-5",
    "model_openai": "",
    "effort": "high",
    "approval_policy": "heavy",       # see registry.APPROVAL_POLICIES
    "max_tool_calls": 300,            # pipelines need many calls; the settings dialog allows 5-400
    "mcp_server_enabled": False,
    "mcp_server_port": 8765,
}


def settings_path():
    base = os.environ.get("LEONARDO_SETTINGS_DIR") or os.path.join(os.path.expanduser("~"), ".leonardo")
    return os.path.join(base, "ai_assistant.json")


def load_settings(path=None):
    data = dict(DEFAULTS)
    try:
        with open(path or settings_path(), encoding="utf-8") as fh:
            saved = json.load(fh)
        if isinstance(saved, dict):
            data.update({k: v for k, v in saved.items() if k in DEFAULTS})
            if saved.get("max_tool_calls") == 60:        # the old default, too small for pipelines
                data["max_tool_calls"] = DEFAULTS["max_tool_calls"]
    except (OSError, ValueError):
        pass
    return data


def save_settings(data, path=None):
    target = path or settings_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    clean = {k: data.get(k, v) for k, v in DEFAULTS.items()}
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=1)
    os.replace(tmp, target)
    return target


def get_api_key(provider):
    """Keychain first, then the provider's environment variable; '' if none."""
    provider = str(provider)
    try:
        import keyring

        value = keyring.get_password(KEYRING_SERVICE, provider)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(ENV_KEYS.get(provider, ""), "")


def set_api_key(provider, value):
    """Store (or with an empty value, delete) a key in the keychain. Returns False if no keychain."""
    try:
        import keyring

        if value:
            keyring.set_password(KEYRING_SERVICE, str(provider), str(value))
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, str(provider))
            except Exception:
                pass
        return True
    except Exception:
        return False


def key_source(provider):
    """Where the key comes from, for the Settings dialog: 'keychain', 'environment' or ''."""
    try:
        import keyring

        if keyring.get_password(KEYRING_SERVICE, str(provider)):
            return "keychain"
    except Exception:
        pass
    return "environment" if os.environ.get(ENV_KEYS.get(str(provider), "")) else ""


def model_for(settings):
    provider = settings.get("provider", "anthropic")
    return settings.get("model_anthropic" if provider == "anthropic" else "model_openai", "") or ""


def make_provider(settings, system, tools):
    """Provider instance for the current settings (raises ProviderError when unusable)."""
    from .providers.base import ProviderError

    provider = settings.get("provider", "anthropic")
    model = model_for(settings)
    key = get_api_key(provider)
    if not key:
        raise ProviderError(f"No API key for {provider}: add it in AI Assistant > Settings.")
    if not model:
        raise ProviderError(f"No model selected for {provider}: choose one in AI Assistant > Settings.")
    if provider == "openai":
        from .providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model, key, system=system, tools=tools, effort=settings.get("effort", "high"))
    from .providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(model, key, system=system, tools=tools, effort=settings.get("effort", "high"))


# USD per million tokens (input, output); cache writes cost 1.25x the input price and cache reads 0.1x
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-fable-5-1": (10.0, 50.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}


def usage_cost(model, usage):
    """Cost in USD of a usage dict (input, output and prompt-cache tokens); None for an unknown model."""
    prices = CLAUDE_PRICES.get(str(model))
    if not prices:
        return None
    u = {k: float((usage or {}).get(k, 0) or 0) for k in
         ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    return (u["input_tokens"] + 1.25 * u["cache_creation_input_tokens"] + 0.1 * u["cache_read_input_tokens"]) \
        / 1e6 * prices[0] + u["output_tokens"] / 1e6 * prices[1]
