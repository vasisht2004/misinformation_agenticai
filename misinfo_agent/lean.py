"""The model's lean, recorded only when the rules abstain.

One call, the same prompt and prompt_version the eval baseline uses.
The result is stored on model_lean. It is not a verdict: confidence,
priority, the decision, and the audit's verdict column stay as the
rules left them. A daily-cap error stops the run. Any other model
failure leaves the lean empty and the abstention in place.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import misinfo_agent.config as config
from misinfo_agent.llm import LLMError, LLMQuotaError, call_structured

_LABELS = {"False", "Misleading", "True"}


def assisted_lean(state: dict) -> dict:
    """Return {model_lean: ...} or {} when this run should not ask.

    An empty dict leaves the key alone. A decided claim and an Other
    claim do not get a lean, so the card stays without one.
    """
    if not config.ASSISTED_LEAN:
        return {}
    if state.get("verdict") != "Unverifiable":
        return {}
    claim = " ".join(str(state.get("claim") or "").split())
    if not claim:
        return {"model_lean": None}
    prompt, schema, version = _baseline_parts()
    try:
        answer = call_structured(
            prompt(claim),
            schema,
            purpose="eval_baseline",
            prompt_version=version,
        )
    except LLMQuotaError:
        raise
    except LLMError:
        return {"model_lean": None}
    label = getattr(answer, "verdict", None)
    if label not in _LABELS:
        return {"model_lean": None}
    try:
        confidence = float(answer.confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {"model_lean": {"label": label, "confidence": confidence}}


_CACHED = None


def _baseline_parts():
    """The eval's prompt, schema, and version. The text is not copied here."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("eval_baseline_prompt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("eval baseline prompt could not be loaded")
    module = importlib.util.module_from_spec(spec)
    # Pydantic resolves Literal while the class body runs. The module has
    # to be in sys.modules before exec, or BaselineVerdict stays unfinished.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BaselineVerdict.model_rebuild()
    _CACHED = (module.baseline_prompt, module.BaselineVerdict, module._BASELINE_VERSION)
    return _CACHED
