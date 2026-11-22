"""The split and the selection rule. No checkpoint and no network.

The arithmetic has to be true before the study script is trusted with
the 50 claims. Loading the script does not arm the zero-call guard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "gate_study"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "gate_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = _load()


def test_stratified_half_is_25_and_25_for_the_eval_mix():
    items = (
        [(f"f{i}", "False") for i in range(23)]
        + [(f"m{i}", "Misleading") for i in range(17)]
        + [(f"t{i}", "True") for i in range(10)]
    )
    dev, test = study.stratified_half(items, 42)
    assert len(dev) == 25
    assert len(test) == 25
    assert set(dev).isdisjoint(test)
    assert len(set(dev) | set(test)) == 50
    # Alphabetical labels. False is the first odd class, so dev takes the
    # extra. Misleading then gives its extra to test. True splits evenly.
    def count(ids, prefix):
        return sum(1 for item in ids if item.startswith(prefix))

    assert (count(dev, "f"), count(test, "f")) == (12, 11)
    assert (count(dev, "m"), count(test, "m")) == (8, 9)
    assert (count(dev, "t"), count(test, "t")) == (5, 5)


def test_selection_rule_requires_precision_and_no_extra_wrongs():
    def row(name, order, cov2, precision, wrong):
        return {
            "name": name,
            "order": order,
            "cov2": cov2,
            "true_precision": precision,
            "wrong": wrong,
        }

    g0 = row("G0", 0, 0.20, 0.80, 4)
    better = row("G2R", 2, 0.36, 0.80, 4)
    loose = row("G1", 1, 0.40, 0.50, 3)
    sloppy = row("G4", 4, 0.44, 1.0, 5)
    assert study.select_winner([g0, loose, better, sloppy]) == "G2R"
    # Never predicting True is not a precision of 0.75.
    never = row("G3", 3, 0.50, None, 2)
    assert study.select_winner([g0, never]) == "G0"
    # A tie on the coverage-adjusted rate keeps G0 rather than flipping a flag.
    tied = row("G2R", 2, 0.20, 0.80, 4)
    assert study.select_winner([g0, tied]) == "G0"
    # Nobody clears the bar.
    weak_g0 = row("G0", 0, 0.20, 0.40, 6)
    assert study.select_winner([weak_g0, loose]) is None


def test_theta_tie_prefers_the_stricter_value():
    scored = [
        {"theta": 0.8, "cov2": 0.3, "true_precision": 0.8, "wrong": 2},
        {"theta": 1.2, "cov2": 0.3, "true_precision": 0.8, "wrong": 2},
        {"theta": 2.0, "cov2": 0.1, "true_precision": 1.0, "wrong": 1},
    ]
    theta, eligible = study.choose_theta(scored, g0_wrong=2)
    assert eligible is True
    assert theta == 1.2
    missed = [
        {"theta": 0.8, "cov2": 0.4, "true_precision": 0.5, "wrong": 1},
        {"theta": 1.0, "cov2": 0.2, "true_precision": 0.5, "wrong": 1},
    ]
    theta, eligible = study.choose_theta(missed, g0_wrong=3)
    assert eligible is False
    assert theta == 0.8
