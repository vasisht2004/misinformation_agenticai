"""Boxes 3–4: read the cleaned text and write the checkable claim.

Owner: ingestion teammate. One LLM call, through llm.py, validated with
Pydantic. The claim must follow prompts/prompt_v4a.txt so box 5 compares
like with like. Stub only, until that branch lands.
"""


def extract(state: dict) -> dict:
    """Boxes 3–4. Return claim, atomic_claims, entities, domain, red_flags."""
    raise NotImplementedError("Boxes 3–4 (extract) are not implemented on this branch.")
