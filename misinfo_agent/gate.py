"""Box 9: decide whether the evidence table is enough to assess.

Owner: Vash. Rules only. Sufficient when two independent groups agree, at
least one is tier 1–2, and no tier-1 source disagrees. Conflict goes to a
human. Otherwise search again, at most twice, then call it insufficient.
"""


def gate(state: dict) -> dict:
    """Box 9. Return the gate value and, when looping, the next round count."""
    raise NotImplementedError("Box 9 (gate) is not implemented yet.")
