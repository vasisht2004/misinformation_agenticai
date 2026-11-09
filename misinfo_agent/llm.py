"""The only module that calls a language model.

Every other box imports this module instead of a provider SDK. The provider
is misinfo_agent.config.LLM_PROVIDER, so a swap does not touch the boxes.
Outputs are validated with Pydantic at the call site that owns the schema.
Keys are read from the environment and never printed.
"""


def complete(prompt: str, schema: type | None = None) -> dict:
    """One model call. schema, when given, is the Pydantic model to validate against."""
    raise NotImplementedError("LLM calls are not implemented yet.")


def call_structured(prompt: str, schema: type) -> dict:
    """One structured call. schema is the Pydantic model the output must satisfy.

    No provider is wired yet. Raising, rather than guessing a JSON object,
    means a missing key cannot silently decide that two claims are the same.
    The key stays in the environment. This function does not print it.
    """
    raise NotImplementedError("call_structured is not implemented yet.")
