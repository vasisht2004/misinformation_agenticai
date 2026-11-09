"""Boxes 8 and 10: write the explanation, then apply the verdict rule in code.

Owner: Vash. The model drafts per-atomic results, the true version and a
correction, citing evidence ids only. Code checks those citations exist and
computes confidence from evidence features. The model's own confidence is ignored.
"""


def assess(state: dict) -> dict:
    """Boxes 8+10. Return verdict, confidence and explanation."""
    raise NotImplementedError("Boxes 8+10 (assess) are not implemented yet.")
