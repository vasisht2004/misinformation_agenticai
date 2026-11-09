"""Box 16: write one SQLite row per run, and append approved verdicts to memory.

Owner: Vash. The row stores the input, evidence, verdict, human
action, model and prompt version, call counts and latency. Without it a
wrong verdict cannot be traced to the evidence that produced it.
"""


def audit(state: dict) -> dict:
    """Box 16. Return the audit record that was stored."""
    raise NotImplementedError("Box 16 (audit) is not implemented yet.")
