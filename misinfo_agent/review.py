"""Box 13: pause for a person when the rules will not decide alone.

Owner: Vash. LangGraph interrupt() pauses; the Streamlit queue
resumes with approve, edit, reject or a request for more evidence.
"""


def review(state: dict) -> dict:
    """Box 13. Return the human action once the run resumes."""
    raise NotImplementedError("Box 13 (human review) is not implemented yet.")
