"""Offline checks for the blocked-eval diagnosis rules.

The arithmetic and the hand rules. Nothing here opens a checkpoint,
the parquet, or a socket. The script's import pops API keys; this
process is already offline because of conftest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    # scripts/ is not a package. The module snapshot of the quota files
    # is a read, and the test does not call main().
    name = "diagnose_eval"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "diagnose_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diag = _load()


def test_claim_type_first_match_and_no_image_rule():
    # A quoted number is a quote: the checkable act is the attribution,
    # and the statistic rule must not steal it.
    kind, hit = diag.claim_type('The minister said "prices rose 12%" yesterday')
    assert kind == "quote"
    assert hit == "quote_marks"
    kind, hit = diag.claim_type("Priya Sharma said the bridge was closed")
    assert (kind, hit) == ("quote", "said_named_person")
    # No named person, so "said" alone is not a quote. The percent is.
    kind, hit = diag.claim_type("He said unemployment is 12%")
    assert (kind, hit) == ("statistic", "percent")
    assert diag.claim_type("The fee is ₹500 today")[0] == "statistic"
    # A death count is a number before it is an event, so the hit is visible.
    assert diag.claim_type("Police killed 2 people")[1] == "number"
    assert diag.claim_type("The ministry banned the gathering")[0] == "official_action"
    assert diag.claim_type("Police killed the suspect")[0] == "event"
    assert diag.claim_type("The sky was clear") == ("other", "none")
    assert "image" not in diag._CLAIM_RULES
    assert "capped" not in diag._ABSTAIN_REASONS


def test_abstention_order_skips_capped_and_stops_at_the_first_hit():
    common = dict(
        decided=False,
        gate="insufficient",
        needs_human=False,
        errors_n=0,
        stance_failed=False,
        n_usable=3,
        n_support_groups=1,
        n_refute_groups=0,
        n_tier12_support=1,
        n_tier12_refute=0,
        quote_unverified=2,
        claim_kind="quote",
    )
    # A pause outranks a missing passage. The later rule would hide the pause.
    assert diag.abstention_reason(**{**common, "needs_human": True, "n_usable": 0}) == "human_pause"
    assert diag.abstention_reason(**{**common, "gate": "error", "needs_human": True}) == "error"
    assert diag.abstention_reason(**{**common, "n_usable": 0}) == "no_hits"
    assert diag.abstention_reason(**{
        **common, "n_tier12_support": 0, "n_support_groups": 2,
    }) == "only_tier34"
    # Exactly one credible group beats the quote rule, which is later.
    assert diag.abstention_reason(**common) == "single_credible"
    assert diag.abstention_reason(**{**common, "n_tier12_support": 2, "n_support_groups": 2}) == "fabricated_quote"
    assert diag.abstention_reason(**{
        **common, "n_tier12_support": 2, "n_support_groups": 2, "claim_kind": "event", "quote_unverified": 0,
    }) == "other"
    assert diag.abstention_reason(**{**common, "decided": True}) == ""


def test_buckets_cover_the_abstain_cells():
    assert diag.bucket_name(True, True, True, True) == "both_right"
    assert diag.bucket_name(True, True, True, False) == "system_only"
    assert diag.bucket_name(True, False, True, True) == "baseline_only"
    assert diag.bucket_name(True, False, True, False) == "both_wrong"
    assert diag.bucket_name(False, False, True, True) == "system_abstain_baseline_right"
    assert diag.bucket_name(False, False, True, False) == "system_abstain_baseline_wrong"
    assert diag.bucket_name(False, False, False, False) == "both_abstain"
    assert diag.bucket_name(True, True, False, False) == "system_right_baseline_abstain"
    assert diag.bucket_name(True, False, False, False) == "system_wrong_baseline_abstain"


def test_thread_match_is_exact_and_ambiguous_uses_latest():
    row = {"verdict": "False", "confidence": 0.5, "evidence": [{"id": "W1"}, {"id": "W2"}]}
    older = {"thread_id": "eval-abc-00000001", "checkpoint_id": "1", "state": {"verdict": "False", "confidence": 0.5, "evidence": [{"id": "W2"}, {"id": "W1"}]}}
    newer = {"thread_id": "eval-abc-00000002", "checkpoint_id": "9", "state": {"verdict": "True", "confidence": 0.5, "evidence": [{"id": "W1"}]}}
    chosen, status = diag.choose_thread([older, newer], row)
    assert status == "unique"
    assert chosen["thread_id"] == older["thread_id"]
    # Two states that both match: do not silently keep the first.
    twin = {**newer, "state": older["state"], "checkpoint_id": "8"}
    chosen, status = diag.choose_thread([older, twin], row)
    assert status == "ambiguous_many"
    assert chosen["checkpoint_id"] == "8"
    chosen, status = diag.choose_thread([newer], row)
    assert status == "ambiguous_0"
    assert chosen["checkpoint_id"] == "9"
    assert diag.choose_thread([], row) == (None, "ambiguous_0")
    # 1e-6 is the tolerance. A larger gap is a different run.
    drifted = {**older, "state": {**older["state"], "confidence": 0.5 + 1e-4}}
    assert not diag.thread_matches(drifted["state"], row)


def test_baseline_status_and_archive_blanks():
    assert diag.baseline_status({"error": "boom", "verdict": None, "needs_human": True}) == "error"
    assert diag.baseline_status({"verdict": "Unverifiable", "needs_human": False}) == "abstain_by_choice"
    assert diag.baseline_status({"verdict": "False", "needs_human": False}) == "decided"
    manipulation, form = diag.archive_labels("true_version", "Fabricated", "Text")
    assert manipulation != manipulation and form != form
    assert diag.archive_labels("primary_claim", "Fabricated", "Text") == ("Fabricated", "Text")


def test_true_rates_and_trivial_not_true_has_no_three_class():
    golds = ["False", "True", "True", "Misleading"]
    rates = diag.true_verdict_rates(golds, ["True", "True", "Unverifiable", "False"], [True, True, False, True])
    # One of two predicted Trues is right. Two gold Trues, one hit.
    assert rates["precision"] == 0.5
    assert rates["hits"] == 1 and rates["gold"] == 2
    scores = diag.policy_scores(golds, None)
    assert scores["acc3"] is None
    assert scores["correct2"] == 2


def test_mcnemar_matches_the_symmetric_tail():
    # Bin(10, 0.5), observing 2: both tails at distance >= 3 from 5.
    assert diag.mcnemar_pvalue(0, 0) is None
    got = diag.mcnemar_pvalue(2, 8)
    assert got == pytest.approx(112 / 1024)


def test_key_overlap_and_redact_do_not_need_a_model():
    line = diag.key_overlap_line(
        "RBI banned the ₹500 note",
        ["The central bank banned the note worth 500"],
    )
    assert "number=yes" in line
    assert "action=yes" in line
    line = diag.key_overlap_line("Priya Sharma left", ["no names here"])
    assert "name=no" in line
    claim = "The Kerala court protected a named person from arrest yesterday"
    assert diag.redact(claim, claim).startswith("(restates the claim")
    assert diag.redact("A1 is not yet supported or refuted.", claim) == "A1 is not yet supported or refuted."


def test_quarter_and_headline_guard():
    assert diag.quarter_label("2026-03-31") == "2026Q1"
    assert diag.quarter_label("2026-04-01") == "2026Q2"
    text = "\n".join(diag._HEADLINE_NEEDLES)
    diag.headline_holds(text, {"n": 50, "decided": 17, "correct": 8, "abstained": 33}, {"n": 50, "decided": 21, "correct": 13, "abstained": 29})
    with pytest.raises(SystemExit):
        diag.headline_holds(text, {"n": 50, "decided": 16, "correct": 8, "abstained": 33}, {"n": 50, "decided": 21, "correct": 13, "abstained": 29})
