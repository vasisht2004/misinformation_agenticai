"""Box 11: score how urgently a person should see this result.

Owner: Vash. Harm (domain, red flags), novelty (1 − top memory
cosine), reach (repeat debunks and resubmissions) and urgency (deadline words).
Weights are hand-set. Rules only.
"""


def score_priority(state: dict) -> dict:
    """Box 11. Return priority: High, Medium or Low."""
    raise NotImplementedError("Box 11 (priority) is not implemented yet.")
