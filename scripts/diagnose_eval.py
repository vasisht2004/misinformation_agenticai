"""Zero-call diagnosis of the blocked 50-claim eval.

The headline numbers stay in docs/eval_end_to_end.md. This script only
reads what that run already stored and writes a second view of the same
claims. It must not call Gemini or Tavily: a diagnosis that spends
quota is no longer a diagnosis of the run we have.

Usage:
  python scripts/diagnose_eval.py --cutoff YYYY-MM-DD
"""

from __future__ import annotations

import os

def _arm_zero_call_guard() -> None:
    """Drop API keys and freeze search to the cache before the package is imported.

    The three assignments are the whole guard. They are not run at import.
    pytest loads this file to test the rules, and a process-wide
    SEARCH_CACHE_ONLY would make every later search test read an empty cache.
    main() calls this before any misinfo_agent import.
    """
    global _COUNTER_SNAPSHOT
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)
    os.environ["SEARCH_CACHE_ONLY"] = "1"
    # The quota files are snapshotted here, still before the package import,
    # so a checkpoint read cannot be what first touches them.
    _COUNTER_SNAPSHOT = {
        path: path.read_bytes() if path.is_file() else None
        for path in (_LLM_COUNTER, _SEARCH_COUNTER)
    }

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover - scipy is installed in this env
    binomtest = None

ROOT = Path(__file__).resolve().parents[1]
# Running `python scripts/diagnose_eval.py` puts scripts/ on sys.path, not
# the repo root, so the package would be missing. Tests already have the
# root; inserting it here is a no-op for them.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_LLM_COUNTER = ROOT / "data" / "cache" / "llm_daily_count.json"
_SEARCH_COUNTER = ROOT / "data" / "cache" / "search_monthly_count.json"
_COUNTER_SNAPSHOT = None

# The eval checkpointer, copied in Stage 0. The script never opens
# data/checkpoints.db or data/eval/runs/checkpoints.db. Two processes on
# the live file is how a read turns into a write.
_CHECKPOINT_COPY = Path("/tmp/ckpt_eval_runs.db")
_TESTSET = ROOT / "data" / "eval" / "testset.csv"
_BLOCKED = ROOT / "data" / "eval" / "results_system_blocked.jsonl"
_BASELINE = ROOT / "data" / "eval" / "results_baseline.jsonl"
_PARQUET = ROOT / "data" / "raw" / "classified_v2.parquet"
_CACHE_DIR = ROOT / "data" / "cache" / "llm"
_HEADLINE = ROOT / "docs" / "eval_end_to_end.md"
_CSV_OUT = ROOT / "data" / "eval" / "diagnosis_blocked.csv"
_MD_OUT = ROOT / "docs" / "eval_diagnosis.md"
_PASSAGES_OUT = ROOT / "data" / "eval" / "false_trues_passages.md"

LABELS = ("False", "Misleading", "True")
# Same collapse the scorer uses. Abstention is not a third binary class.
_Z = 1.96
# First matching rule wins. Image/visual is absent on purpose: the archive
# form of every one of these 50 rows is Text, so that rule cannot fire.
_CLAIM_RULES = ("quote", "statistic", "official_action", "event", "other")
_BUCKETS = (
    "both_right",
    "system_only",
    "baseline_only",
    "both_wrong",
    "system_abstain_baseline_right",
    "system_abstain_baseline_wrong",
    "both_abstain",
    "system_right_baseline_abstain",
    # The listed buckets never cover "system decided wrong, baseline abstained".
    # Leaving those rows unnamed would make the crosstab sum to less than 50.
    "system_wrong_baseline_abstain",
)
# "capped" is not here. search_capped was never stored on the blocked
# jsonl or on the checkpoint, so a cap reason would be a guess.
_ABSTAIN_REASONS = (
    "error",
    "human_pause",
    "no_hits",
    "only_tier34",
    "single_credible",
    "fabricated_quote",
    "other",
)
_OFFICIAL = (
    "banned",
    "approved",
    "launched",
    "announced",
    "notification",
    "scheme",
    "rbi",
    "ministry",
)
_EVENTS = ("happened", "killed", "attacked", "arrested", "video shows")
_QUOTE_MARKS = re.compile(r"[\"“”«»]|‘[^’]{3,}’|'[^']{3,}'")
_SAID = re.compile(r"\b(?:said|says|stated)\b", re.IGNORECASE)
_PERSON = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|(?:Mr|Mrs|Ms|Dr|PM|President|Minister|Chief)\.?\s+[A-Z][a-z]+)\b"
)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
_CURRENCY = re.compile(
    r"(?:₹|Rs\.?|INR)\s*\d|\d[\d,]*(?:\.\d+)?\s*(?:₹|Rs\.?|INR|crore|crores|lakh|lakhs|rupees)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_HEADLINE_NEEDLES = (
    "| Decided | 17 | 5 | 21 |",
    "| Correct | 8 | 4 | 13 |",
    "0.471 (8/17)",
    "0.619 (13/21)",
    "0.660 (33/50)",
    "0.580 (29/50)",
    "0.160 (8/50)",
    "0.260 (13/50)",
)


def wilson(successes: int, n: int, z: float = _Z) -> tuple[float, float] | None:
    """95% Wilson interval. Same formula as scripts/score_eval.py.

    Copied rather than imported: importing the scorer is safe, but the
    interval has to stay readable next to the McNemar test, and a second
    import path is how two formulas drift.
    """
    if n <= 0:
        return None
    proportion = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (proportion + z2 / (2 * n)) / denom
    margin = z * ((proportion * (1 - proportion) / n + z2 / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_pvalue(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p-value for the discordant cells.

    Under the null, each discordant claim is equally likely to fall in
    either cell, which is a binomial with p = 0.5. n = 0 has no test.
    scipy's binomtest sums every outcome at least as unlikely as the one
    we saw. The math.comb version is that same sum, and it is exact for
    p = 0.5 because the probability depends only on the distance from n/2.
    """
    n = b + c
    if n <= 0:
        return None
    if binomtest is not None:
        return float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    observed = abs(2 * b - n)
    total = sum(math.comb(n, i) for i in range(n + 1) if abs(2 * i - n) >= observed)
    return total / (2**n)


def is_decided(row: dict) -> bool:
    """Scorer's rule: a gradable label, and nobody parked the run."""
    return (not row.get("needs_human")) and row.get("verdict") in LABELS


def is_abstained(row: dict) -> bool:
    """Scorer's rule. A pause and an Unverifiable on the same row count once."""
    return bool(row.get("needs_human")) or row.get("verdict") == "Unverifiable"


def collapsed(label: str | None) -> str | None:
    if label == "True":
        return "True"
    if label in {"False", "Misleading"}:
        return "not-true"
    return None


def baseline_status(row: dict) -> str:
    """Separate a crashed baseline call from a chosen Unverifiable.

    The scorer folds both into abstention. The three jsonl rows that
    carry an error field never received a label, so calling them a
    choice would blame the model for a call that failed.
    """
    if row.get("error"):
        return "error"
    if is_decided(row):
        return "decided"
    return "abstain_by_choice"


def claim_type(text: str) -> tuple[str, str]:
    """First matching hand rule, and the pattern that fired.

    The order is the order in the brief: a quoted number is a quote,
    because the thing to check is whether those words were said. The
    rule id is stored so a bad fire can be seen without re-deriving it.
    There is no image rule: form is Text for this whole test set.
    """
    raw = text or ""
    if _QUOTE_MARKS.search(raw):
        return "quote", "quote_marks"
    if _SAID.search(raw) and _PERSON.search(raw):
        return "quote", "said_named_person"
    if _PERCENT.search(raw):
        return "statistic", "percent"
    if _CURRENCY.search(raw):
        return "statistic", "currency"
    if _NUMBER.search(raw):
        return "statistic", "number"
    folded = raw.casefold()
    for word in _OFFICIAL:
        if re.search(rf"\b{re.escape(word)}\b", folded):
            return "official_action", f"official_action:{word}"
    for word in _EVENTS:
        if word == "video shows":
            if "video shows" in folded:
                return "event", "event:video shows"
            continue
        if re.search(rf"\b{re.escape(word)}\b", folded):
            return "event", f"event:{word}"
    return "other", "none"


def correct_flag(decided: bool, right: bool) -> float:
    """1 or 0 when the scorer would grade the row, else NaN."""
    if not decided:
        return float("nan")
    return 1.0 if right else 0.0


def paired_bit(decided: bool, right: bool) -> int:
    """McNemar's 0/1. An abstention is a miss, not a missing row."""
    return 1 if decided and right else 0


def bucket_name(system_decided: bool, system_right: bool, baseline_decided: bool, baseline_right: bool) -> str:
    """3-class partition. Right means decided and equal to the gold label."""
    if not system_decided and not baseline_decided:
        return "both_abstain"
    if not system_decided and baseline_decided:
        return "system_abstain_baseline_right" if baseline_right else "system_abstain_baseline_wrong"
    if system_decided and not baseline_decided:
        return "system_right_baseline_abstain" if system_right else "system_wrong_baseline_abstain"
    if system_right and baseline_right:
        return "both_right"
    if system_right and not baseline_right:
        return "system_only"
    if baseline_right and not system_right:
        return "baseline_only"
    return "both_wrong"


def deciding_side(n_support: int, n_refute: int, credible_support: int, credible_refute: int) -> tuple[str | None, int | None]:
    """Which side the gate was short of, and how many tier 1–2 groups it had.

    More independent groups wins. A tie falls through to more credible
    groups. A tie on both is not 'one side', so single_credible does not fire.
    """
    if n_support > n_refute:
        return "support", credible_support
    if n_refute > n_support:
        return "refute", credible_refute
    if credible_support > credible_refute:
        return "support", credible_support
    if credible_refute > credible_support:
        return "refute", credible_refute
    return None, None


def _truthy_flag(value) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return True
    if isinstance(value, str) and value.strip().casefold() in {"1", "true", "yes"}:
        return True
    return False


def abstention_reason(
    *,
    decided: bool,
    gate: str,
    needs_human: bool,
    errors_n: int,
    stance_failed: bool,
    n_usable: int,
    n_support_groups: int,
    n_refute_groups: int,
    n_tier12_support: int,
    n_tier12_refute: int,
    quote_unverified: int,
    claim_kind: str,
) -> str:
    """First match only. Later rules would hide the earlier failure.

    capped is not a branch: the run did not store search_capped.
    fabricated_quote is the stored unverified-quote counter on a claim
    the quote rule already marked, not a second reading of the passage.
    """
    if decided:
        return ""
    if gate == "error" or errors_n > 0 or stance_failed:
        return "error"
    if needs_human or gate == "conflicting":
        return "human_pause"
    if n_usable <= 0:
        return "no_hits"
    if n_tier12_support == 0 and n_tier12_refute == 0:
        return "only_tier34"
    _side, credible = deciding_side(n_support_groups, n_refute_groups, n_tier12_support, n_tier12_refute)
    if credible == 1:
        return "single_credible"
    if quote_unverified > 0 and claim_kind == "quote":
        return "fabricated_quote"
    return "other"


def evidence_ids(rows) -> set[str]:
    found = set()
    for item in rows or []:
        if isinstance(item, dict) and item.get("id"):
            found.add(str(item["id"]))
    return found


def _verdict(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _close_confidence(left, right) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return False


def thread_matches(state: dict, row: dict) -> bool:
    """The jsonl line is one final state: same label, same score, same ids.

    Confidence is a float the pipeline wrote twice, so 1e-6 is rounding,
    not a softer match. The id set ignores order and ignores quotes: the
    quote text is the claim's evidence, and comparing it would not change
    which thread is the run.
    """
    return (
        _verdict(state.get("verdict")) == _verdict(row.get("verdict"))
        and _close_confidence(state.get("confidence"), row.get("confidence"))
        and evidence_ids(state.get("evidence")) == evidence_ids(row.get("evidence"))
    )


def choose_thread(candidates: list[dict], row: dict) -> tuple[dict | None, str]:
    """One blocked thread, or the latest one with an explicit ambiguous flag.

    candidates carry thread_id, checkpoint_id, and state. checkpoint_id
    is LangGraph's time-ordered id, so the max is the latest write.
    Zero matches and several matches are both ambiguous: picking a match
    when two match would hide the collision.
    """
    if not candidates:
        return None, "ambiguous_0"
    matched = [item for item in candidates if thread_matches(item["state"], row)]
    latest = max(candidates, key=lambda item: item["checkpoint_id"])
    if len(matched) == 1:
        return matched[0], "unique"
    if len(matched) == 0:
        return latest, "ambiguous_0"
    return latest, "ambiguous_many"


def core_id(state: dict) -> str:
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("core") is True:
            return str(item.get("id") or "A1")
    return "A1"


def _tier(row: dict) -> int | None:
    try:
        return int(row.get("tier"))
    except (TypeError, ValueError):
        return None


def evidence_profile(state: dict | None) -> dict:
    """Counts the gate would have used, plus the raw row tallies.

    Group counts go through gate._groups so a 'single credible group'
    here is the same group the live rule counted. drop_after is on
    because the eval set EVAL_DROP_AFTER_CLAIM: a later page is not a
    passage the blocked run was allowed to use. Usable means a support
    or refute of the core claim with passage text still attached.
    """
    empty = {
        "n_evidence_rows": 0,
        "n_supports": 0,
        "n_refutes": 0,
        "n_independent_groups_support": 0,
        "n_independent_groups_refute": 0,
        "n_tier12_groups_support": 0,
        "n_tier12_groups_refute": 0,
        "n_usable": 0,
        "best_tier": None,
        "quote_unverified": 0,
        "errors_n": 0,
        "error_text": "",
        "stance_failed": False,
        "gate_reasons": [],
        "core_id": "A1",
        "atomic_claims": [],
    }
    if not isinstance(state, dict):
        return empty
    from misinfo_agent.gate import _groups, _rows

    evidence = [item for item in (state.get("evidence") or []) if isinstance(item, dict)]
    core = core_id(state)
    kept = _rows(evidence, core, drop_after=True)
    usable = [item for item in kept if str(item.get("passage") or "").strip()]
    grouped = _groups(kept)
    tiers = [tier for tier in (_tier(item) for item in evidence) if tier is not None]
    stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    try:
        unverified = int(stats.get("quote_unverified") or 0)
    except (TypeError, ValueError):
        unverified = 0
    errors = state.get("errors") or []
    if not isinstance(errors, list):
        errors = [errors]
    reasons = []
    for item in state.get("gate_reasons") or []:
        text = " ".join(str(item).split())
        if text:
            reasons.append(text)
    atomics = []
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict):
            atomics.append(item)
    return {
        "n_evidence_rows": len(evidence),
        "n_supports": sum(1 for item in evidence if item.get("stance") == "supports"),
        "n_refutes": sum(1 for item in evidence if item.get("stance") == "refutes"),
        "n_independent_groups_support": len(grouped["support"]),
        "n_independent_groups_refute": len(grouped["refute"]),
        "n_tier12_groups_support": len(grouped["support_credible"]),
        "n_tier12_groups_refute": len(grouped["refute_credible"]),
        "n_usable": len(usable),
        "best_tier": min(tiers) if tiers else None,
        "quote_unverified": unverified,
        "errors_n": len(errors),
        "error_text": " | ".join(str(item) for item in errors),
        "stance_failed": _truthy_flag(stats.get("stance_failed")),
        "gate_reasons": reasons,
        "core_id": core,
        "atomic_claims": atomics,
        "evidence": evidence,
    }


def archive_labels(source: str, manipulation, form) -> tuple[object, object]:
    """Donor articles are not a record of the true_version sentence.

    The parquet row is the fact-check the sentence was borrowed from.
    Its manipulation type and form describe that article, so they are
    left blank rather than copied onto a different claim.
    """
    if source == "true_version":
        return float("nan"), float("nan")
    return manipulation, form


def true_verdict_rates(golds: list[str], verdicts: list[str | None], decided: list[bool]) -> dict:
    """Precision is correct Trues over predicted Trues. Recall is over every gold True.

    An abstention is not a predicted True, so it does not enter precision.
    It does miss a gold True, so recall's denominator is the whole class.
    """
    predicted = [bool(ok) and verdict == "True" for ok, verdict in zip(decided, verdicts)]
    hits = sum(pred and gold == "True" for pred, gold in zip(predicted, golds))
    n_pred = sum(predicted)
    n_gold = sum(gold == "True" for gold in golds)
    return {
        "hits": hits,
        "predicted": n_pred,
        "gold": n_gold,
        "precision": (hits / n_pred) if n_pred else None,
        "recall": (hits / n_gold) if n_gold else None,
    }


def trivial_predictions(golds: list[str]) -> list[tuple[str, list[str] | None, str]]:
    """Fixed policies on all 50. None means the policy has no 3-class label.

    always-not-true is only a 2-class policy: it refuses to pick False
    versus Misleading, so a 3-class accuracy for it would be invented.
    """
    majority = max(LABELS, key=lambda label: sum(gold == label for gold in golds))
    return [
        ("always False", ["False"] * len(golds), "3-class label False"),
        ("always Misleading", ["Misleading"] * len(golds), "3-class label Misleading"),
        ("always True", ["True"] * len(golds), "3-class label True"),
        (f"majority ({majority})", [majority] * len(golds), "most common gold label"),
        ("always not-true", None, "2-class only"),
    ]


def policy_scores(golds: list[str], predicted: list[str] | None) -> dict:
    n = len(golds)
    if predicted is None:
        binary_correct = sum(collapsed(gold) == "not-true" for gold in golds)
        return {
            "acc3": None,
            "correct3": None,
            "acc2": binary_correct / n if n else None,
            "correct2": binary_correct,
            "coverage_adjusted": None,
            "true": {"hits": 0, "predicted": 0, "gold": sum(g == "True" for g in golds), "precision": None, "recall": 0.0},
        }
    correct3 = sum(gold == pred for gold, pred in zip(golds, predicted))
    correct2 = sum(collapsed(gold) == collapsed(pred) for gold, pred in zip(golds, predicted))
    return {
        "acc3": correct3 / n if n else None,
        "correct3": correct3,
        "acc2": correct2 / n if n else None,
        "correct2": correct2,
        "coverage_adjusted": correct3 / n if n else None,
        "true": true_verdict_rates(golds, predicted, [True] * n),
    }


def confusion(golds: list[str], verdicts: list[str | None], decided: list[bool], columns: list[str]) -> pd.DataFrame:
    """Gold by prediction. Anything the scorer would not grade is Abstain."""
    predicted = []
    for verdict, ok in zip(verdicts, decided):
        if not ok:
            predicted.append("Abstain")
        elif verdict in columns:
            predicted.append(verdict)
        else:
            predicted.append("Abstain")
    frame = pd.DataFrame({"gold": golds, "predicted": predicted})
    table = pd.crosstab(frame["gold"], frame["predicted"], dropna=False)
    return table.reindex(index=columns, columns=[*columns, "Abstain"], fill_value=0)


def quarter_label(day: str) -> str:
    year, month, _rest = day.split("-")
    quarter = (int(month) - 1) // 3 + 1
    return f"{year}Q{quarter}"


def key_overlap_line(claim: str, quotes: list[str]) -> str:
    """Whether a supporting quote still contains the claim's checkable tokens.

    This is a string test, not a reading of the passage. A number matches
    if its digits appear. A name matches as a whole phrase. An action word
    is one of the same verbs the claim-type rules use, and only if the
    claim itself used it.
    """
    numbers = _NUMBER.findall(claim or "")
    names = _PERSON.findall(claim or "")
    folded_claim = (claim or "").casefold()
    actions = []
    for word in (*_OFFICIAL, *_EVENTS):
        if word == "video shows":
            if "video shows" in folded_claim:
                actions.append(word)
            continue
        if re.search(rf"\b{re.escape(word)}\b", folded_claim):
            actions.append(word)
    blob = "\n".join(quotes).casefold()
    digits = re.sub(r"\D", "", blob)

    def number_hit(token: str) -> bool:
        return re.sub(r"\D", "", token) in digits and re.sub(r"\D", "", token) != ""

    def phrase_hit(token: str) -> bool:
        return token.casefold() in blob

    def part(found: bool | None, empty: bool) -> str:
        if empty:
            return "none in claim"
        return "yes" if found else "no"

    return (
        "Key overlap in supporting quotes: "
        f"number={part(any(number_hit(token) for token in numbers), not numbers)}; "
        f"name={part(any(phrase_hit(token) for token in names), not names)}; "
        f"action={part(any(phrase_hit(token) for token in actions), not actions)}."
    )


def redact(text: str, claim: str) -> str:
    """Keep a gate sentence out of the committable note when it repeats the claim."""
    flat = " ".join(str(text).split()).replace("|", "/")
    snippet = " ".join((claim or "").split()).casefold()
    if len(snippet) >= 24 and snippet[:24] in flat.casefold():
        return "(restates the claim; full text is in the csv only)"
    return flat


def rate_cell(successes: int | None, n: int) -> str:
    if successes is None or n <= 0:
        return "n/a"
    interval = wilson(successes, n)
    span = "n/a" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
    return f"{successes / n:.3f} ({successes}/{n}) {span}"


def fmt_rate(rate: float | None, successes: int | None, n: int) -> str:
    if rate is None or successes is None:
        return "n/a"
    return f"{rate:.3f} ({successes}/{n})"


def fmt_true(rates: dict) -> str:
    precision = "n/a" if rates["precision"] is None else f"{rates['precision']:.3f} ({rates['hits']}/{rates['predicted']})"
    recall = "n/a" if rates["recall"] is None else f"{rates['recall']:.3f} ({rates['hits']}/{rates['gold']})"
    return f"precision {precision}; recall {recall}"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def headline_holds(text: str, blocked: dict, baseline: dict) -> None:
    """Stop if our scorer replay or the written report is not the headline.

    The fractions are the ones the brief named (8/17, 13/21, 33/50, 29/50,
    8/50, 13/50). Rounding 0.660 to 0.66 is the brief's shorthand, not a
    second definition. Neither file is edited to force a match.
    """
    missing = [needle for needle in _HEADLINE_NEEDLES if needle not in text]
    if missing:
        raise SystemExit("HEADLINE MISMATCH: docs/eval_end_to_end.md is missing: " + "; ".join(missing))
    expected = {
        "blocked": {"n": 50, "decided": 17, "correct": 8, "abstained": 33},
        "baseline": {"n": 50, "decided": 21, "correct": 13, "abstained": 29},
    }
    for name, got in (("blocked", blocked), ("baseline", baseline)):
        for key, value in expected[name].items():
            if got[key] != value:
                raise SystemExit(f"HEADLINE MISMATCH: {name} {key} is {got[key]}, report expects {value}")


def condition_counts(rows: list[dict], gold_of: dict[str, str]) -> dict:
    decided_n = correct = binary = abstained = 0
    for row in rows:
        gold = gold_of[str(row["article_id"])]
        abstained += int(is_abstained(row))
        if not is_decided(row):
            continue
        decided_n += 1
        correct += int(row.get("verdict") == gold)
        binary += int(collapsed(row.get("verdict")) == collapsed(gold))
    return {
        "n": len(rows),
        "decided": decided_n,
        "correct": correct,
        "binary_correct": binary,
        "abstained": abstained,
    }


def model_note(cache_dir: Path) -> str:
    """Say the model is unknown unless a cache file actually names one.

    The cache body is the parsed answer. The filename hash is computed
    from the primary model even when the fallback model produced the
    text, and the baseline jsonl does not store usage. Reading the files
    is not a call. Printing an output value would copy a verdict into
    the log, so only the keys are inspected.
    """
    files = sorted(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "model" in payload or "model_used" in payload:
            return "model key present in cache; join not implemented"
        output = payload.get("output")
        if isinstance(output, dict) and ("model" in output or "model_used" in output):
            return "model key present in cache output; join not implemented"
    return "model per row not recoverable"


def load_blocked_threads(db_path: Path) -> dict[str, list[dict]]:
    """Final states of threads with no eval_sites mark, grouped by article id.

    score_eval treats a missing eval_sites as blocked. Allowed threads
    are marked and are not candidates, even when they share the article id.
    """
    if not db_path.is_file():
        raise SystemExit(f"checkpoint copy missing: {db_path}")
    from misinfo_agent.graph import build_graph

    graph = build_graph(db_path, "/tmp/diag_eval_audit_unused.db", "/tmp/diag_eval_additions_unused.parquet")
    connection = sqlite3.connect(db_path)
    try:
        threads = [row[0] for row in connection.execute("SELECT DISTINCT thread_id FROM checkpoints")]
        latest = dict(connection.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id"))
    finally:
        connection.close()
    grouped: dict[str, list[dict]] = {}
    for thread in threads:
        parts = str(thread).split("-")
        if len(parts) != 3 or parts[0] != "eval":
            continue
        snap = graph.get_state({"configurable": {"thread_id": thread}})
        values = getattr(snap, "values", None)
        if not isinstance(values, dict) or not values:
            continue
        marks = values.get("run_marks") if isinstance(values.get("run_marks"), dict) else {}
        if "eval_sites" in marks:
            continue
        grouped.setdefault(parts[1], []).append({
            "thread_id": thread,
            "checkpoint_id": str(latest.get(thread) or ""),
            "state": values,
        })
    return grouped


_COUNTERS_REPORTED = False


def assert_counters_unchanged() -> None:
    global _COUNTERS_REPORTED
    if _COUNTER_SNAPSHOT is None:
        raise SystemExit("counter snapshot missing")
    current = {
        path: path.read_bytes() if path.is_file() else None
        for path in _COUNTER_SNAPSHOT
    }
    if current != _COUNTER_SNAPSHOT:
        raise SystemExit("COUNTERS CHANGED")
    if not _COUNTERS_REPORTED:
        print("COUNTERS UNCHANGED")
        _COUNTERS_REPORTED = True


def _error_phrase(n: int) -> str:
    """'1 error' is not '1 errors'. Zero stays plural, which reads as a count."""
    word = "error" if n == 1 else "errors"
    return f"{n} {word}"


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split()).replace("|", "/")


def _md_table(frame: pd.DataFrame) -> str:
    show = frame.fillna("")
    header = "| " + " | ".join(_cell(col) for col in show.columns) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in show.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def _indexed_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    show.insert(0, show.index.name or "row", show.index)
    return _md_table(show.reset_index(drop=True))


def half_metrics(frame: pd.DataFrame) -> dict:
    n = len(frame)
    system_decided = frame["system_decided"].sum()
    baseline_decided = frame["baseline_decided"].sum()
    return {
        "n": int(n),
        "system_decided": int(system_decided),
        "system_correct3": int(frame["system_right3"].sum()),
        "system_correct2": int(frame["system_right2"].sum()),
        "system_abstain": int((~frame["system_decided"]).sum()),
        "baseline_decided": int(baseline_decided),
        "baseline_correct3": int(frame["baseline_right3"].sum()),
        "baseline_correct2": int(frame["baseline_right2"].sum()),
        "baseline_choice": int((frame["baseline_status"] == "abstain_by_choice").sum()),
        "baseline_errors": int((frame["baseline_status"] == "error").sum()),
        "small": n < 10,
    }


def render_half(title: str, metrics: dict) -> list[str]:
    flag = " Too small to conclude (n < 10)." if metrics["small"] else ""
    return [
        f"### {title}",
        "",
        f"n = {metrics['n']}.{flag}",
        "",
        f"System decided {metrics['system_decided']}. 3-class {fmt_rate(None if not metrics['system_decided'] else metrics['system_correct3'] / metrics['system_decided'], metrics['system_correct3'], metrics['system_decided'])}. 2-class {fmt_rate(None if not metrics['system_decided'] else metrics['system_correct2'] / metrics['system_decided'], metrics['system_correct2'], metrics['system_decided'])}. Abstention {metrics['system_abstain']}/{metrics['n']}.",
        "",
        f"Baseline decided {metrics['baseline_decided']}. 3-class {fmt_rate(None if not metrics['baseline_decided'] else metrics['baseline_correct3'] / metrics['baseline_decided'], metrics['baseline_correct3'], metrics['baseline_decided'])}. 2-class {fmt_rate(None if not metrics['baseline_decided'] else metrics['baseline_correct2'] / metrics['baseline_decided'], metrics['baseline_correct2'], metrics['baseline_decided'])}. Abstention {metrics['baseline_choice']} by choice + {_error_phrase(metrics['baseline_errors'])} (scorer {metrics['baseline_choice'] + metrics['baseline_errors']}/{metrics['n']}).",
        "",
    ]


def build_passages(false_trues: list[dict]) -> str:
    lines = [
        "# False Trues",
        "",
        f"Count: {len(false_trues)}.",
        "",
        "A false True is a blocked verdict of True whose gold label is False or Misleading.",
        "Supporting rows are evidence whose stance is supports. The overlap line is a string check.",
        "",
    ]
    for row in false_trues:
        lines.append(f"## {row['claim_id']}")
        lines.append("")
        lines.append(f"Gold: {row['gold']}")
        lines.append(f"Thread pick: {row['thread_pick']}")
        lines.append("")
        lines.append("Claim:")
        lines.append("")
        lines.append(row["claim"])
        lines.append("")
        lines.append("Atomic claims:")
        lines.append("")
        if not row["atomics"]:
            lines.append("- (none stored)")
        for atomic in row["atomics"]:
            mark = "core" if atomic.get("core") is True or str(atomic.get("id")) == row["core_id"] else "not core"
            lines.append(f"- {atomic.get('id')} ({mark}): {atomic.get('text')}")
        lines.append("")
        lines.append("Supporting evidence:")
        lines.append("")
        if not row["supports"]:
            lines.append("No supporting evidence row.")
            lines.append("")
        for item in row["supports"]:
            lines.extend([
                f"### {item.get('id')}",
                "",
                f"- domain: {item.get('netloc') or ''}",
                f"- tier: {item.get('tier')}",
                f"- group: {item.get('group_id') or ''}",
                f"- stance: {item.get('stance')}",
                f"- atomic: {item.get('atomic_id') or ''}",
                f"- quote: {item.get('quote') or ''}",
                "- passage:",
                "",
                str(item.get("passage") or ""),
                "",
            ])
        lines.append(row["overlap"])
        lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    """Committable note. Claim sentences and passages stay in data/eval."""
    lines = [
        "# Blocked-eval diagnosis",
        "",
        "Diagnostic analysis on the same 50 claims used for the headline eval. Nothing here is a held-out result.",
        "",
        "## How to read this",
        "",
        "Every archive `form` on these 50 claims is Text, so there is no image/visual claim type.",
        "",
        "manipulation_type and form are empty for the 10 true_version rows. That article_id is the donor fact-check, not a record of the sentence that was labelled True.",
        "",
        "search_capped is not on the blocked result rows and not on the checkpoint state, including investigate_stats. The abstention reason capped is not used.",
        "",
        f"Baseline abstention is {report['baseline_choice']} by choice + {_error_phrase(report['baseline_errors'])}, alongside the scorer's {report['baseline_abstained']}/{report['n']}.",
        "",
        report["model_note"] + ". The cache JSON has no model field, a fallback answer is stored under the primary model's cache key, and the baseline rows have no usage record.",
        "",
        f"Cutoff {report['cutoff']} is the date supplied on the command line (Gemini 3.5 Flash-Lite model card: knowledge cutoff March 2026).",
        "",
        f"THREAD AMBIGUOUS: {report['n_ambiguous']}.",
        "",
    ]
    if report["ambiguous_ids"]:
        lines.append("Article ids: " + ", ".join(
            f"{article_id} ({kind})" for article_id, kind in report["ambiguous_ids"]
        ) + ".")
        lines.append("")
        lines.append("Those rows use the latest blocked checkpoint. The match required the same verdict, confidence within 1e-6, and the same evidence id set. Evidence counts for an ambiguous id come from that latest state.")
        lines.append("")
    else:
        lines.append("Every article matched exactly one blocked thread.")
        lines.append("")
    lines.extend([
        "Abstention reasons are first-match, and only on rows the scorer did not decide: error, human_pause, no_hits, only_tier34, single_credible, fabricated_quote, other. fabricated_quote requires investigate_stats.quote_unverified > 0 and claim_type quote. Buckets are 3-class. system_wrong_baseline_abstain is the cell the other bucket names do not cover.",
        "",
        "Overlapping Wilson intervals are not a significance test. The McNemar p-value is the test.",
        "",
        "## 1. Claim type by bucket, and abstention reasons",
        "",
        _indexed_table(report["crosstab"]),
        "",
        "Abstention reasons (system rows that were not decided):",
        "",
        _md_table(report["reason_table"]),
        "",
        f"False True count: {report['n_false_true']}. Passages are in data/eval/false_trues_passages.md and are not copied here.",
        "",
        "## 2. Label distribution and trivial baselines",
        "",
        "Coverage is 100% because every trivial policy labels all 50 claims. Coverage-adjusted accuracy is the 3-class correct count over 50. always not-true has no 3-class label.",
        "",
        _md_table(report["label_table"]),
        "",
        _md_table(report["trivial_table"]),
        "",
        f"System True-verdict {fmt_true(report['system_true'])}. Baseline True-verdict {fmt_true(report['baseline_true'])}.",
        "",
        "## 3. Confusion matrices",
        "",
        "Rows are the gold label. Abstain is Unverifiable, needs_human, or a baseline error (verdict missing). A decided False/Misleading swap stays in its 3-class cell and collapses in the 2-class matrix.",
        "",
        "### System, 3-class",
        "",
        _indexed_table(report["system_cm3"]),
        "",
        "### Baseline, 3-class",
        "",
        _indexed_table(report["baseline_cm3"]),
        "",
        "### System, 2-class",
        "",
        _indexed_table(report["system_cm2"]),
        "",
        "### Baseline, 2-class",
        "",
        _indexed_table(report["baseline_cm2"]),
        "",
        "## 4. Paired comparison",
        "",
        "On all 50 claims, correct is 1 only when the row was decided and right. Otherwise it is 0. Wilson intervals below are on that 0/1 rate.",
        "",
        "### 3-class",
        "",
        _indexed_table(report["paired3"]),
        "",
        f"System {rate_cell(report['system_bits3'], report['n'])}. Baseline {rate_cell(report['baseline_bits3'], report['n'])}.",
        "",
        f"Exact two-sided McNemar p-value: {report['p3']}.",
        "",
        "### 2-class",
        "",
        _indexed_table(report["paired2"]),
        "",
        f"System {rate_cell(report['system_bits2'], report['n'])}. Baseline {rate_cell(report['baseline_bits2'], report['n'])}.",
        "",
        f"Exact two-sided McNemar p-value: {report['p2']}.",
        "",
        f"System True-verdict {fmt_true(report['system_true'])}. Baseline True-verdict {fmt_true(report['baseline_true'])}.",
        "",
        "Overlapping Wilson intervals are not a significance test.",
        "",
        "## 5. Where the two systems disagree",
        "",
        "3-class. Baseline right and system wrong or abstained:",
        "",
        _md_table(report["baseline_better"]),
        "",
        "System right and baseline wrong or abstained:",
        "",
        _md_table(report["system_better"]),
        "",
        "## 7. Assisted score",
        "",
        "If the system decided, that verdict is used. Otherwise a decided baseline verdict is used and shown as Unverifiable, model lean: that label. Otherwise the assisted verdict is Unverifiable. Baseline errors supply no lean.",
        "",
        _md_table(report["assisted_table"]),
        "",
        "## 8. Cutoff split",
        "",
        report["date_note"],
        "",
    ])
    lines.extend(render_half(f"claim_date <= {report['cutoff']}", report["half_le"]))
    lines.extend(render_half(f"claim_date > {report['cutoff']}", report["half_gt"]))
    if report["article_differs"]:
        lines.append("Fact-check article date differs from claim_date on some rows. The same split on the article date:")
        lines.append("")
        lines.extend(render_half(f"article_date <= {report['cutoff']}", report["article_le"]))
        lines.extend(render_half(f"article_date > {report['cutoff']}", report["article_gt"]))
    else:
        lines.append("The fact-check article date's calendar day equals claim_date on all 50 rows, including the donor articles behind true_version sentences. The article-date split is the same partition as the claim_date split.")
        lines.append("")
    lines.extend([
        "Baseline 3-class accuracy by quarter of claim_date. Accuracy is correct decided over decided. n is the cell size.",
        "",
        _md_table(report["quarter_table"]),
        "",
    ])
    return "\n".join(lines)


def _no_claim_text(markdown: str, claims: list[str]) -> None:
    folded = markdown.casefold()
    for claim in claims:
        snippet = " ".join(claim.split()).casefold()
        if len(snippet) >= 24 and snippet[:24] in folded:
            raise SystemExit("refusing to write docs/eval_diagnosis.md: a claim sentence would be included")


def main(argv: list[str] | None = None) -> None:
    _arm_zero_call_guard()
    parser = argparse.ArgumentParser(description="Diagnose the blocked eval without calling a model or a search API.")
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD knowledge cutoff. Required; there is no default.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.cutoff):
        raise SystemExit("--cutoff must be YYYY-MM-DD")
    if "GEMINI_API_KEY" in os.environ or "TAVILY_API_KEY" in os.environ:
        raise SystemExit("API key reappeared before the read; stopping")

    testset = load_testset(_TESTSET)
    blocked_rows = load_jsonl(_BLOCKED)
    baseline_rows = load_jsonl(_BASELINE)
    gold_of = {row["article_id"]: row["label"] for row in testset}
    blocked_counts = condition_counts(blocked_rows, gold_of)
    baseline_counts = condition_counts(baseline_rows, gold_of)
    headline_holds(_HEADLINE.read_text(encoding="utf-8"), blocked_counts, baseline_counts)

    archive = pd.read_parquet(_PARQUET, columns=["article_id", "date", "manipulation_type", "form"])
    archive["article_id"] = archive["article_id"].astype(str)
    archive = archive.drop_duplicates("article_id").set_index("article_id")

    threads = load_blocked_threads(_CHECKPOINT_COPY)
    if "GEMINI_API_KEY" in os.environ or "TAVILY_API_KEY" in os.environ:
        raise SystemExit("API key reappeared during checkpoint read; stopping")

    blocked_by_id = {str(row["article_id"]): row for row in blocked_rows}
    baseline_by_id = {str(row["article_id"]): row for row in baseline_rows}
    records = []
    false_trues = []
    ambiguous_ids = []
    for gold_row in testset:
        article_id = str(gold_row["article_id"])
        system = blocked_by_id[article_id]
        baseline = baseline_by_id[article_id]
        chosen, thread_pick = choose_thread(threads.get(article_id, []), system)
        if thread_pick != "unique":
            ambiguous_ids.append(article_id)
        profile = evidence_profile(chosen["state"] if chosen else None)
        kind, rule_hit = claim_type(gold_row["input"])
        source = "true_version-built True" if gold_row["source"] == "true_version" else "archive claim"
        manipulation, form = (float("nan"), float("nan"))
        article_day = ""
        if article_id in archive.index:
            stored = archive.loc[article_id]
            manipulation, form = archive_labels(gold_row["source"], stored["manipulation_type"], stored["form"])
            article_day = str(stored["date"])[:10]
        gold = gold_row["label"]
        system_decided = is_decided(system)
        baseline_decided = is_decided(baseline)
        system_right3 = bool(system_decided and system.get("verdict") == gold)
        baseline_right3 = bool(baseline_decided and baseline.get("verdict") == gold)
        system_right2 = bool(system_decided and collapsed(system.get("verdict")) == collapsed(gold))
        baseline_right2 = bool(baseline_decided and collapsed(baseline.get("verdict")) == collapsed(gold))
        reason = abstention_reason(
            decided=system_decided,
            gate=str(system.get("gate") or ""),
            needs_human=bool(system.get("needs_human")),
            errors_n=profile["errors_n"],
            stance_failed=profile["stance_failed"],
            n_usable=profile["n_usable"],
            n_support_groups=profile["n_independent_groups_support"],
            n_refute_groups=profile["n_independent_groups_refute"],
            n_tier12_support=profile["n_tier12_groups_support"],
            n_tier12_refute=profile["n_tier12_groups_refute"],
            quote_unverified=profile["quote_unverified"],
            claim_kind=kind,
        )
        gate_reasons = profile["gate_reasons"] or ([str(system.get("gate_reason"))] if system.get("gate_reason") else [])
        status = baseline_status(baseline)
        records.append({
            "claim_id": article_id,
            "claim": gold_row["input"],
            "gold_3c": gold,
            "gold_2c": collapsed(gold),
            "claim_date": gold_row["claim_date"],
            "article_date": article_day,
            "domain": gold_row["domain"],
            "manipulation_type": manipulation,
            "form": form,
            "source": source,
            "claim_type": kind,
            "claim_type_rule_hit": rule_hit,
            "system_verdict": system.get("verdict"),
            "system_confidence": system.get("confidence"),
            "gate": system.get("gate") or "",
            "gate_reasons": " | ".join(gate_reasons),
            "baseline_verdict": baseline.get("verdict"),
            "baseline_confidence": baseline.get("confidence"),
            "baseline_status": status,
            "system_correct_3c": correct_flag(system_decided, system_right3),
            "system_correct_2c": correct_flag(system_decided, system_right2),
            "baseline_correct_3c": correct_flag(baseline_decided, baseline_right3),
            "baseline_correct_2c": correct_flag(baseline_decided, baseline_right2),
            "n_evidence_rows": profile["n_evidence_rows"],
            "n_supports": profile["n_supports"],
            "n_refutes": profile["n_refutes"],
            "n_independent_groups_support": profile["n_independent_groups_support"],
            "n_independent_groups_refute": profile["n_independent_groups_refute"],
            "n_tier12_groups_support": profile["n_tier12_groups_support"],
            "n_tier12_groups_refute": profile["n_tier12_groups_refute"],
            "best_tier": profile["best_tier"],
            "search_capped": "not_recoverable",
            "n_searches": system.get("searches"),
            "errors": profile["errors_n"],
            "needs_human": bool(system.get("needs_human")),
            "abstention_reason": reason,
            "bucket": bucket_name(system_decided, system_right3, baseline_decided, baseline_right3),
            "thread_pick": thread_pick,
            "baseline_model": "not_recoverable",
            "system_decided": system_decided,
            "baseline_decided": baseline_decided,
            "system_right3": system_right3,
            "baseline_right3": baseline_right3,
            "system_right2": system_right2,
            "baseline_right2": baseline_right2,
            "system_bit3": paired_bit(system_decided, system_right3),
            "baseline_bit3": paired_bit(baseline_decided, baseline_right3),
            "system_bit2": paired_bit(system_decided, system_right2),
            "baseline_bit2": paired_bit(baseline_decided, baseline_right2),
        })
        if system.get("verdict") == "True" and gold in {"False", "Misleading"}:
            evidence = profile.get("evidence") or []
            supports = [item for item in evidence if item.get("stance") == "supports"]
            false_trues.append({
                "claim_id": article_id,
                "gold": gold,
                "claim": gold_row["input"],
                "thread_pick": thread_pick,
                "core_id": profile["core_id"],
                "atomics": profile["atomic_claims"],
                "supports": supports,
                "overlap": key_overlap_line(gold_row["input"], [str(item.get("quote") or "") for item in supports]),
            })

    frame = pd.DataFrame.from_records(records)
    if len(frame) != 50:
        raise SystemExit(f"expected 50 claims, built {len(frame)}")
    if set(frame["bucket"]) - set(_BUCKETS):
        raise SystemExit("unexpected bucket name")
    # Scoring uses the jsonl, not the checkpoint. If these drift, the
    # tables below are not the headline run and must not be written.
    if int(frame["system_decided"].sum()) != blocked_counts["decided"] or int(frame["system_right3"].sum()) != blocked_counts["correct"]:
        raise SystemExit("HEADLINE MISMATCH: rebuilt system rows do not match the scorer replay")
    if int(frame["baseline_decided"].sum()) != baseline_counts["decided"] or int(frame["baseline_right3"].sum()) != baseline_counts["correct"]:
        raise SystemExit("HEADLINE MISMATCH: rebuilt baseline rows do not match the scorer replay")
    choice_n = int((frame["baseline_status"] == "abstain_by_choice").sum())
    error_n = int((frame["baseline_status"] == "error").sum())
    if choice_n + error_n != baseline_counts["abstained"]:
        raise SystemExit(
            f"HEADLINE MISMATCH: baseline {choice_n} by choice + {error_n} errors "
            f"is not the scorer abstention {baseline_counts['abstained']}"
        )

    public = [
        "claim_id", "claim", "gold_3c", "gold_2c", "claim_date", "domain",
        "manipulation_type", "form", "source", "claim_type", "claim_type_rule_hit",
        "system_verdict", "system_confidence", "gate", "gate_reasons",
        "baseline_verdict", "baseline_confidence", "baseline_status",
        "system_correct_3c", "system_correct_2c", "baseline_correct_3c", "baseline_correct_2c",
        "n_evidence_rows", "n_supports", "n_refutes",
        "n_independent_groups_support", "n_independent_groups_refute",
        "n_tier12_groups_support", "n_tier12_groups_refute",
        "best_tier", "search_capped", "n_searches", "errors", "needs_human",
        "abstention_reason", "bucket", "thread_pick", "baseline_model",
    ]
    _CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, public].to_csv(_CSV_OUT, index=False)

    passages = build_passages(false_trues)
    _PASSAGES_OUT.write_text(passages, encoding="utf-8")
    n_false = len(false_trues)
    print(f"FALSE TRUES {n_false}")
    if n_false != 5:
        print(f"LOUD: false True count is {n_false}, the handoff said 5")

    golds = frame["gold_3c"].tolist()
    label_counts = frame["gold_3c"].value_counts().reindex(list(LABELS), fill_value=0)
    label_table = pd.DataFrame({"label": label_counts.index, "n": label_counts.to_numpy()})
    trivial_rows = []
    for name, predicted, _note in trivial_predictions(golds):
        scores = policy_scores(golds, predicted)
        trivial_rows.append({
            "policy": name,
            "3c acc": fmt_rate(scores["acc3"], scores["correct3"], 50),
            "2c acc": fmt_rate(scores["acc2"], scores["correct2"], 50),
            "coverage-adjusted": fmt_rate(scores["coverage_adjusted"], scores["correct3"], 50) if scores["coverage_adjusted"] is not None else "n/a (no 3-class label)",
            "True precision / recall": fmt_true(scores["true"]),
        })
    system_true = true_verdict_rates(golds, frame["system_verdict"].tolist(), frame["system_decided"].tolist())
    baseline_true = true_verdict_rates(golds, frame["baseline_verdict"].tolist(), frame["baseline_decided"].tolist())

    def _pair_table(left: pd.Series, right: pd.Series) -> pd.DataFrame:
        table = pd.crosstab(left.astype(int), right.astype(int), dropna=False)
        table = table.reindex(index=[1, 0], columns=[1, 0], fill_value=0)
        table.index.name = "system correct"
        table.columns.name = "baseline correct"
        return table

    paired3 = _pair_table(frame["system_bit3"], frame["baseline_bit3"])
    paired2 = _pair_table(frame["system_bit2"], frame["baseline_bit2"])
    # b is system-only correct, c is baseline-only correct. scipy tests either cell.
    p3 = mcnemar_pvalue(int(paired3.loc[1, 0]), int(paired3.loc[0, 1]))
    p2 = mcnemar_pvalue(int(paired2.loc[1, 0]), int(paired2.loc[0, 1]))

    system_cm3 = confusion(golds, frame["system_verdict"].tolist(), frame["system_decided"].tolist(), list(LABELS))
    baseline_cm3 = confusion(golds, frame["baseline_verdict"].tolist(), frame["baseline_decided"].tolist(), list(LABELS))
    system_cm2 = confusion(
        frame["gold_2c"].tolist(),
        [collapsed(value) for value in frame["system_verdict"].tolist()],
        frame["system_decided"].tolist(),
        ["not-true", "True"],
    )
    baseline_cm2 = confusion(
        frame["gold_2c"].tolist(),
        [collapsed(value) for value in frame["baseline_verdict"].tolist()],
        frame["baseline_decided"].tolist(),
        ["not-true", "True"],
    )

    crosstab = pd.crosstab(frame["claim_type"], frame["bucket"], dropna=False)
    crosstab = crosstab.reindex(index=list(_CLAIM_RULES), columns=list(_BUCKETS), fill_value=0)
    crosstab.index.name = "claim_type"
    held = frame.loc[~frame["system_decided"], "abstention_reason"]
    reason_counts = held.value_counts().reindex(list(_ABSTAIN_REASONS), fill_value=0)
    if int(reason_counts.sum()) != blocked_counts["abstained"]:
        raise SystemExit("abstention reasons do not cover every system abstention")
    if (held == "capped").any():
        raise SystemExit("capped is not a recoverable abstention reason")
    reason_table = pd.DataFrame({"abstention_reason": reason_counts.index, "n": reason_counts.to_numpy()})

    def disagreement(mask: pd.Series) -> pd.DataFrame:
        subset = frame.loc[mask, ["claim_id", "claim_type", "abstention_reason", "gate", "gate_reasons", "claim"]]
        rows = []
        for record in subset.to_dict(orient="records"):
            rows.append({
                "claim_id": record["claim_id"],
                "claim_type": record["claim_type"],
                "abstention_reason": record["abstention_reason"],
                "gate": record["gate"],
                "gate_reasons": redact(record["gate_reasons"], record["claim"]),
            })
        columns = ["claim_id", "claim_type", "abstention_reason", "gate", "gate_reasons"]
        return pd.DataFrame(rows, columns=columns)

    baseline_better = disagreement(frame["baseline_right3"] & ~frame["system_right3"])
    system_better = disagreement(frame["system_right3"] & ~frame["baseline_right3"])

    assisted_source = []
    assisted_verdict = []
    for record in frame.to_dict(orient="records"):
        if record["system_decided"]:
            assisted_source.append("system")
            assisted_verdict.append(record["system_verdict"])
        elif record["baseline_decided"]:
            assisted_source.append("baseline")
            assisted_verdict.append(record["baseline_verdict"])
        else:
            assisted_source.append("neither")
            assisted_verdict.append(None)
    assisted_decided = [source != "neither" for source in assisted_source]
    assisted_right3 = [
        bool(ok and verdict == gold) for ok, verdict, gold in zip(assisted_decided, assisted_verdict, golds)
    ]
    assisted_right2 = [
        bool(ok and collapsed(verdict) == collapsed(gold))
        for ok, verdict, gold in zip(assisted_decided, assisted_verdict, golds)
    ]
    assisted_rows = []
    for source in ("system", "baseline", "neither"):
        index = [item == source for item in assisted_source]
        n_source = sum(index)
        correct3 = sum(flag for flag, keep in zip(assisted_right3, index) if keep)
        correct2 = sum(flag for flag, keep in zip(assisted_right2, index) if keep)
        decided_n = n_source if source != "neither" else 0
        assisted_rows.append({
            "source": source,
            "n": n_source,
            "3c acc on its decisions": fmt_rate(None if source == "neither" else (correct3 / decided_n if decided_n else None), correct3 if source != "neither" else None, decided_n),
            "2c acc on its decisions": fmt_rate(None if source == "neither" else (correct2 / decided_n if decided_n else None), correct2 if source != "neither" else None, decided_n),
        })
    assisted_decided_n = sum(assisted_decided)
    assisted_rows.append({
        "source": "assisted total",
        "n": assisted_decided_n,
        "3c acc on its decisions": fmt_rate(
            sum(assisted_right3) / assisted_decided_n if assisted_decided_n else None,
            sum(assisted_right3),
            assisted_decided_n,
        ),
        "2c acc on its decisions": fmt_rate(
            sum(assisted_right2) / assisted_decided_n if assisted_decided_n else None,
            sum(assisted_right2),
            assisted_decided_n,
        ),
        "coverage-adjusted 3c": fmt_rate(sum(assisted_right3) / 50, sum(assisted_right3), 50),
    })
    assisted_table = pd.DataFrame(assisted_rows)

    frame["quarter"] = frame["claim_date"].map(quarter_label)
    quarter_rows = []
    for quarter, subset in frame.groupby("quarter", sort=True):
        decided_n = int(subset["baseline_decided"].sum())
        correct = int(subset["baseline_right3"].sum())
        correct2 = int(subset["baseline_right2"].sum())
        choice = int((subset["baseline_status"] == "abstain_by_choice").sum())
        errors = int((subset["baseline_status"] == "error").sum())
        quarter_rows.append({
            "quarter": quarter,
            "n": len(subset),
            "decided": decided_n,
            "correct 3c": correct,
            "3c acc": fmt_rate(correct / decided_n if decided_n else None, correct, decided_n),
            "2c acc": fmt_rate(correct2 / decided_n if decided_n else None, correct2, decided_n),
            "abstention": f"{choice} by choice + {_error_phrase(errors)}",
            "note": "n < 10, descriptive only" if len(subset) < 10 else "",
        })

    dates_equal = bool((frame["article_date"] == frame["claim_date"]).all())
    note = model_note(_CACHE_DIR)
    markdown = render_markdown({
        "n": 50,
        "cutoff": args.cutoff,
        "n_ambiguous": len(ambiguous_ids),
        "ambiguous_ids": list(zip(
            ambiguous_ids,
            frame.loc[frame["claim_id"].isin(ambiguous_ids), "thread_pick"].tolist(),
        )),
        "baseline_choice": int((frame["baseline_status"] == "abstain_by_choice").sum()),
        "baseline_errors": int((frame["baseline_status"] == "error").sum()),
        "baseline_abstained": baseline_counts["abstained"],
        "model_note": note,
        "crosstab": crosstab,
        "reason_table": reason_table,
        "n_false_true": n_false,
        "label_table": label_table,
        "trivial_table": pd.DataFrame(trivial_rows),
        "system_true": system_true,
        "baseline_true": baseline_true,
        "system_cm3": system_cm3,
        "baseline_cm3": baseline_cm3,
        "system_cm2": system_cm2,
        "baseline_cm2": baseline_cm2,
        "paired3": paired3,
        "paired2": paired2,
        "system_bits3": int(frame["system_bit3"].sum()),
        "baseline_bits3": int(frame["baseline_bit3"].sum()),
        "system_bits2": int(frame["system_bit2"].sum()),
        "baseline_bits2": int(frame["baseline_bit2"].sum()),
        "p3": "n/a" if p3 is None else f"{p3:.6f}",
        "p2": "n/a" if p2 is None else f"{p2:.6f}",
        "baseline_better": baseline_better,
        "system_better": system_better,
        "assisted_table": assisted_table,
        "date_note": f"Split at {args.cutoff}.",
        "half_le": half_metrics(frame.loc[frame["claim_date"] <= args.cutoff]),
        "half_gt": half_metrics(frame.loc[frame["claim_date"] > args.cutoff]),
        "article_differs": not dates_equal,
        "article_le": half_metrics(frame.loc[frame["article_date"] <= args.cutoff]) if not dates_equal else None,
        "article_gt": half_metrics(frame.loc[frame["article_date"] > args.cutoff]) if not dates_equal else None,
        "quarter_table": pd.DataFrame(quarter_rows),
    })
    _no_claim_text(markdown, frame["claim"].tolist())
    _MD_OUT.write_text(markdown, encoding="utf-8")
    print(f"THREAD AMBIGUOUS {len(ambiguous_ids)}")
    print(f"wrote {_CSV_OUT}")
    print(f"wrote {_MD_OUT}")
    print(f"wrote {_PASSAGES_OUT}")
    assert_counters_unchanged()


if __name__ == "__main__":
    try:
        main()
    finally:
        # A crash before the success print must still say whether the
        # quota files moved. A second "unchanged" line would look like
        # the run finished twice. Importing the module for tests never
        # arms the guard, so there is no snapshot to compare.
        if _COUNTER_SNAPSHOT is not None:
            current = {path: path.read_bytes() if path.is_file() else None for path in _COUNTER_SNAPSHOT}
            if current != _COUNTER_SNAPSHOT:
                print("COUNTERS CHANGED", file=sys.stderr)
            elif not _COUNTERS_REPORTED:
                print("COUNTERS UNCHANGED")
