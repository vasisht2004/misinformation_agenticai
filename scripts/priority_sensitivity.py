"""Move each priority weight by 20% and count how many levels change.

The other weights stay where they are, so the four no longer sum to 1.
That is the question the hand-set numbers need answered: if harm was
set 20% high, or reach 20% low, how many of these claims would be
filed in a different band. Re-balancing after the move would hide a
pure scale change.

The two named fixtures are the garlic forward and the calm sports
rumour from tests/test_priority.py. The quiet health claim is there
because it sits near a cut; a weight check that only uses a 0.74 and
a 0.12 cannot show a crossing.

Run from the repo root:

    python scripts/priority_sensitivity.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.priority import components, level, priority  # noqa: E402

_WEIGHTS = ("harm", "novelty", "reach", "urgency")


def _fixtures() -> list[tuple[str, dict]]:
    """The states in tests/test_priority.py, loaded without running them.

    Imported inside the function so this file can be imported by a test
    that has not finished loading that module yet.
    """
    path = ROOT / "tests" / "test_priority.py"
    spec = importlib.util.spec_from_file_location("priority_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [
        ("garlic forward", module.GARLIC),
        ("calm sports rumour", module.SPORTS),
        ("quiet new health claim", module.QUIET_HEALTH),
    ]


def _score(parts: dict, weights: dict) -> float:
    total = 0.0
    for name in _WEIGHTS:
        total += float(weights[name]) * float(parts[name])
    if total < 0.0:
        return 0.0
    if total > 1.0:
        return 1.0
    return total


def sensitivity(weights: dict | None = None) -> list[dict]:
    """One row per weight and per direction.

    `n_changed` is how many fixtures land in a different High / Medium /
    Low band than they do under `weights` (the config weights, if omitted).
    """
    base = dict(config.PRIORITY_WEIGHTS if weights is None else weights)
    cases = []
    for name, state in _fixtures():
        parts = components(state)
        cases.append((name, parts, level(_score(parts, base))))
    rows = []
    for key in _WEIGHTS:
        for factor, direction in ((0.8, "-20%"), (1.2, "+20%")):
            moved = dict(base)
            moved[key] = moved[key] * factor
            changed = []
            for name, parts, original in cases:
                if level(_score(parts, moved)) != original:
                    changed.append(name)
            rows.append(
                {
                    "weight": key,
                    "change": direction,
                    "n_changed": len(changed),
                    "cases": changed,
                }
            )
    return rows


def base_levels() -> list[tuple[str, str, float]]:
    found = []
    for name, state in _fixtures():
        scored = priority(state)
        found.append((name, scored["priority"], float(scored["priority_score"])))
    return found


def main() -> None:
    print("base")
    for name, band, score in base_levels():
        print(f"  {name:<24} {band:<6} {score:.3f}")
    print()
    print(f"{'weight':<10} {'change':<8} {'levels changed':<16} cases")
    for row in sensitivity():
        names = ", ".join(row["cases"]) if row["cases"] else "-"
        print(f"{row['weight']:<10} {row['change']:<8} {row['n_changed']:<16} {names}")


if __name__ == "__main__":
    main()
