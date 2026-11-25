"""Score the end-to-end results and write docs/eval_end_to_end.md.

Reads data/eval/testset.csv, results_system_blocked.jsonl,
results_system_allowed.jsonl and results_baseline.jsonl. A test row
with no result line has not been run. Blocked excludes fact-check
sites. Allowed may use other fact-checkers and still hides the source
article. The baseline is one Gemini call.
The numbers below are for the rows that have been run, which is the
whole file after a full eval and a prefix after a smoke run.

Decided claims are False, Misleading, or True with needs_human false.
Unverifiable and needs_human are abstentions. A provisional verdict
kept from a review pause has needs_human set, so it is not graded.
Coverage-adjusted accuracy is correct decided claims over every scored
claim, which is accuracy times coverage.

Usage:
  python scripts/score_eval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402

# The same quote check the pipeline used when it kept the row. A second
# implementation would disagree on curly quotes and call a real quote invalid.
from misinfo_agent.investigate import _quote_in_passage  # noqa: E402

LABELS = ["False", "Misleading", "True"]
# False and Misleading are both "not what the claim said". The 2-class
# score asks only whether the system told True from not-true.
BINARY = ["not-true", "True"]
# Normal 95% quantile. Wilson uses it so a rate of 0 or 1 does not print
# a bound outside 0–1 the way a plain normal interval does.
_Z = 1.96

_TESTSET = config.EVAL_DIR / "testset.csv"
_SYSTEM = config.EVAL_DIR / "results_system_blocked.jsonl"
_ALLOWED = config.EVAL_DIR / "results_system_allowed.jsonl"
_BASELINE = config.EVAL_DIR / "results_baseline.jsonl"
_RECYCLED = config.EVAL_DIR / "results_system_recycled.jsonl"
_ALLOWED_V2 = config.EVAL_DIR / "results_system_allowed_v2.jsonl"
_REPORT = ROOT / "docs" / "eval_end_to_end.md"


def wilson(successes: int, n: int, z: float = _Z) -> tuple[float, float] | None:
    """95% Wilson interval for successes/n. None when n is 0."""
    if n <= 0:
        return None
    proportion = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (proportion + z2 / (2 * n)) / denom
    margin = z * ((proportion * (1 - proportion) / n + z2 / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float | None:
    """ECE over equal-width bins. The last bin includes 1.0.

    A decided claim with no confidence is left out. The weight of a bin
    is its share of the claims that did have a confidence.
    """
    pairs = []
    for confidence, ok in zip(confidences, correct):
        if confidence is None:
            continue
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            continue
        pairs.append((min(1.0, max(0.0, value)), bool(ok)))
    n = len(pairs)
    if n == 0:
        return None
    total = 0.0
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            bucket = [pair for pair in pairs if low <= pair[0] <= high]
        else:
            bucket = [pair for pair in pairs if low <= pair[0] < high]
        if not bucket:
            continue
        accuracy = sum(ok for _, ok in bucket) / len(bucket)
        average = sum(value for value, _ in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(accuracy - average)
    return total


def calibration_bin_counts(confidences, n_bins: int = 10) -> list[int]:
    """How many decided confidences landed in each equal-width bin.

    The last bin includes 1.0, matching expected_calibration_error. A
    missing confidence is not a bin: the counts sum to the ECE sample.
    """
    counts = [0] * n_bins
    for confidence in confidences:
        value = _float_or_none(confidence)
        if value is None:
            continue
        value = min(1.0, max(0.0, float(value)))
        index = min(n_bins - 1, int(value * n_bins))
        counts[index] += 1
    return counts


def collapsed_label(label: str | None) -> str | None:
    """False and Misleading are not-true. True stays True."""
    if label == "True":
        return "True"
    if label in {"False", "Misleading"}:
        return "not-true"
    return None


def is_abstained(row: dict) -> bool:
    """Unverifiable, or a pause. A row that is both counts once."""
    return bool(row.get("needs_human")) or row.get("verdict") == "Unverifiable"


def is_provisionally_decided(row: dict) -> bool:
    """A gradable label, including one a person has not accepted yet.

    The strict column stays is_decided: needs_human false. This one
    counts False, Misleading, and True even when the review is pending.
    """
    return row.get("verdict") in LABELS


def is_pending_review(row: dict) -> bool:
    """A run the queue is still holding.

    review_status is the new flag. Older rows have no such key; a
    gradable label with needs_human was the old pause.
    """
    if row.get("review_status") == "pending":
        return True
    return bool(row.get("needs_human")) and row.get("verdict") in LABELS


def is_decided(row: dict) -> bool:
    """A label the scorer can mark right or wrong. A pause is not one."""
    return (not row.get("needs_human")) and row.get("verdict") in LABELS


def citation_validity(rows: list[dict]) -> dict:
    """Cited ids that exist, and web quotes that are in the stored passage.

    C1 is the archive prior, not a web quote. It exists when the run kept
    a prior. A web id exists when that evidence row was stored, and the
    quote verifies when the pipeline's own quote check finds it in the
    passage. The scorer does not fetch the page again.
    """
    web_n = web_exist = web_valid = 0
    archive_n = archive_valid = 0
    for row in rows:
        evidence = {}
        for item in row.get("evidence") or []:
            if isinstance(item, dict) and item.get("id"):
                evidence[str(item["id"])] = item
        for ident in row.get("citations") or []:
            ident = str(ident)
            if ident == "C1":
                archive_n += 1
                archive_valid += int(bool(row.get("prior_present")))
                continue
            web_n += 1
            item = evidence.get(ident)
            if item is None:
                continue
            web_exist += 1
            if _quote_in_passage(str(item.get("quote") or ""), str(item.get("passage") or "")):
                web_valid += 1
    return {
        "web_n": web_n,
        "web_exist": web_exist,
        "web_valid": web_valid,
        "archive_n": archive_n,
        "archive_valid": archive_valid,
    }


def condition_metrics(rows: list[dict]) -> dict:
    """One condition. rows already carry the gold label and the domain."""
    abstained = [row for row in rows if is_abstained(row)]
    decided = [row for row in rows if is_decided(row)]
    provisional = [row for row in rows if is_provisionally_decided(row)]
    correct = [row for row in decided if row.get("verdict") == row.get("label")]
    n = len(rows)
    metrics = {
        "n": n,
        "abstained": len(abstained),
        "decided": len(decided),
        "decided_incl_provisional": len(provisional),
        "pending_review": sum(1 for row in rows if is_pending_review(row)),
        "correct": len(correct),
        "accuracy": (len(correct) / len(decided)) if decided else None,
        "abstention": (len(abstained) / n) if n else None,
        "coverage_adjusted": (len(correct) / n) if n else None,
        "macro_f1": None,
        "binary_correct": 0,
        "binary_macro_f1": None,
        "per_class": [],
        "matrix": None,
        "ece": None,
        "ece_n": 0,
        "ece_bins": [0] * 10,
        "domains": [],
        "cost": _cost(rows),
        "citations": citation_validity(rows),
    }
    if not decided:
        metrics["domains"] = _domains(rows)
        return metrics
    y_true = [row["label"] for row in decided]
    y_pred = [row["verdict"] for row in decided]
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    metrics["matrix"] = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    per_class = []
    for index, label in enumerate(LABELS):
        # The diagonal is the true-positive count. Recall times support
        # is that same count, but the float can round off by one.
        per_class.append({
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "tp": int(metrics["matrix"][index][index]),
        })
    metrics["per_class"] = per_class
    confidences = [_float_or_none(row.get("confidence")) for row in decided]
    correct_flags = [row.get("verdict") == row.get("label") for row in decided]
    metrics["ece"] = expected_calibration_error(confidences, correct_flags, n_bins=10)
    metrics["ece_n"] = sum(value is not None for value in confidences)
    metrics["ece_bins"] = calibration_bin_counts(confidences, n_bins=10)
    # Same decided rows. A Misleading gold called False is wrong in the
    # three-class score and right in this one.
    y_true_bin = [collapsed_label(label) for label in y_true]
    y_pred_bin = [collapsed_label(label) for label in y_pred]
    metrics["binary_correct"] = sum(gold == pred for gold, pred in zip(y_true_bin, y_pred_bin))
    metrics["binary_macro_f1"] = float(
        f1_score(y_true_bin, y_pred_bin, average="macro", labels=BINARY, zero_division=0)
    )
    metrics["domains"] = _domains(rows)
    return metrics


def _allowed_superseded() -> str:
    """The allowed column in this file is the buggy run, kept for the record."""
    return "\n".join([
        "The allowed numbers in this file are superseded — evaluation setup bug, see diagnosis. They are not the condition the code runs now.",
        "",
        "## Diagnosis",
        "",
        "Eight claims, traced from the cached searches and the stored passages. No new model calls. Four were decided when fact-check sites were blocked and abstained when they were allowed (`fcaebc7ad5f3`, `2dcc928eca48`, `84a06ffe939b`, `0ab825964c1e`). The other four are `e2230f5774ce` and `90aad4c39280` (both decided) and `20eb7385815d` and `0179258fa67c` (both abstained).",
        "",
        "The phrase rule is not applied when blocking is off. `_quotes_factcheck` returns false as soon as the block flag is off. On `e2230f5774ce` the allowed pages contained 4 windows with a fact-check phrase, and the rule dropped 0 of them.",
        "",
        "Own-article exclusion matches the canonical URL, not the registrable domain. `_same_page` compares that one URL, ignoring scheme and tracking parameters. A different path on the same host is kept. Across the allowed caches, 1 fact-check hit was the claim's own URL.",
        "",
        "The after-claim rule removed 0 fact-check hits. The allowed caches hold 9 unique fact-check hits. One is the own article. None of the nine is dated after the claim. The search already asks for pages on or before the claim day, and no fact-check hit that came back was later.",
        "",
        "| Claim | Condition | Hits (fact-check / other) | What was removed | Passages | Stance |",
        "| --- | --- | --- | --- | --- | --- |",
        "| fcaebc7ad5f3 | Blocked | 19 (0 / 19) | circulation 7; fetch fail 0; cap of 25 dropped 0; own URL 0 | 12 | yes (llm 6) |",
        "| fcaebc7ad5f3 | Allowed | 26 (0 / 26) | circulation 6; cap of 25 dropped 1 other URL; 7 past the 12 fetches kept as snippets; phrase windows dropped 0; own URL 0; after-claim fact-check 0 | 22 | yes (llm 5) |",
        "| 2dcc928eca48 | Blocked | 25 (0 / 25) | circulation 10; fetch fail 0; cap 0 | 11 | yes (llm 6) |",
        "| 2dcc928eca48 | Allowed | 24 (0 / 24) | circulation 13; fetch fail 1; cap 0; fact-check hits 0 | 7 | yes (llm 6) |",
        "| 84a06ffe939b | Blocked | 25 (0 / 25) | circulation 1; cap 0 | 30 | yes (llm 7) |",
        "| 84a06ffe939b | Allowed | no cached open-web hits | per-claim search cap (8) discarded the gather | 0 | no (llm 1, extract only) |",
        "| 0ab825964c1e | Blocked | 27 (0 / 27) | circulation 6; fetch fail 2; cap of 25 dropped 2 | 19 | yes (llm 4) |",
        "| 0ab825964c1e | Allowed | no cached open-web hits | per-claim search cap (8) discarded the gather | 0 | no (llm 1, extract only) |",
        "| e2230f5774ce | Blocked | 18 (0 / 18) | circulation 2; cap 0 | 14 | yes (llm 4) |",
        "| e2230f5774ce | Allowed | 17 (1 / 16) | circulation 1; the 1 fact-check page is dated on or before the claim; 4 phrase windows, 0 dropped | 18 | yes (llm 4) |",
        "| 90aad4c39280 | Blocked | 0 | no web search; decided False from memory | 0 | no (llm 2) |",
        "| 90aad4c39280 | Allowed | 0 | no web search; decided False from memory | 0 | no (llm 2) |",
        "| 20eb7385815d | Blocked | 21 (0 / 21) | circulation 7; cap 0 | 11 | yes (llm 4) |",
        "| 20eb7385815d | Allowed | 22 (0 / 22) | circulation 2; 8 past the 12 fetches kept as snippets; fact-check hits 0 | 20 | yes (llm 4) |",
        "| 0179258fa67c | Blocked | 33 (0 / 33) | circulation 4; fetch fail 1; cap of 25 dropped 8 | 36 | yes (llm 4) |",
        "| 0179258fa67c | Allowed | 33 (0 / 33) | circulation 2; cap of 25 dropped 8; fact-check hits 0 | 32 | yes (llm 4) |",
        "",
        "Cause. Allowed did not reuse the blocked searches. The block flag is part of the cache key, so every allowed query was a new Tavily call on the same words. The blocked run had already used most of the monthly budget of 800. From the 22nd claim, 29 claims stopped on `monthly search cap reached (800 calls)` and kept no pages: one Gemini call, the extract, and no stance. Four claims before that, including `84a06ffe939b` and `0ab825964c1e`, hit the per-claim cap of 8, and that exception discards the pages the other queries had already fetched. On the claims that did finish a search, fact-check hits were rare: 9 unique in the allowed caches, and the open list is a different five pages, so one blocked evidence URL was missing for `fcaebc7ad5f3` and for `2dcc928eca48`. That gap is smaller than the caps. The caps are why allowed abstained on claims blocked had decided.",
        "",
        "The code now runs the blocked search first, which is a cache hit, and adds fact-check pages in their own budget of at most 8 fetches. The phrase rule stays off for that budget. The own article is still excluded by its canonical URL. After-claim dropping is unchanged, and on this run it removed 0 fact-check pages.",
        "",
        "A re-run was not started. The monthly search cap is full, so fact-check pages that were never cached cannot be fetched until it resets. Extract and stance on the blocked passages are cache hits. Five claims already have a fact-check page in the open-web cache that is not the own article and is not dated after the claim. Each of those is one new stance call if the page yields a passage, and an assess call only if the new row makes the gate pass. The other claims add no stance call while the search cap is full.",
    ])


def render(testset: list[dict], blocked: list[dict], baseline: list[dict], allowed: list[dict] | None = None) -> str:
    """The markdown report. Counts say how many result lines were present.

    `blocked` is the first system condition. `allowed` may be omitted
    when that file has not been written; its column is then empty.
    """
    if allowed is None:
        allowed = []
    blocked_rows = _join(testset, blocked)
    allowed_rows = _join(testset, allowed)
    baseline_rows = _join(testset, baseline)
    columns = [
        ("Blocked", blocked_rows, condition_metrics(blocked_rows)),
        ("Allowed", allowed_rows, condition_metrics(allowed_rows)),
        ("Baseline", baseline_rows, condition_metrics(baseline_rows)),
    ]
    blocked_metrics = columns[0][2]
    allowed_metrics = columns[1][2]
    baseline_metrics = columns[2][2]
    flips, flips_correct = became_decided(blocked_rows, allowed_rows)
    lines = [
        "# End-to-end evaluation",
        "",
        _allowed_superseded(),
        "",
        "Rewritten by `scripts/score_eval.py`. The numbers are the result lines that exist, joined to the test set on `article_id`. A test row with no result line has not been run.",
        "",
        f"Test set rows: {len(testset)}. Blocked results scored: {blocked_metrics['n']}. Allowed results scored: {allowed_metrics['n']}. Baseline results scored: {baseline_metrics['n']}.",
        "",
        f"Leak count, blocked: {_leak_total(blocked_rows)}. Expected 0. A leak there is the claim's own article, or a near-duplicate copy, in a memory match, a prior-match lead, or an evidence URL.",
        "",
        f"Leak count, allowed: {_leak_total(allowed_rows)}. Expected 0. A leak there is only the claim's own article. Another fact-checker's article is allowed.",
        "",
        _factcheck_share(allowed_rows),
        "",
        f"Blocked abstentions that became decided once fact-check sites were allowed: {flips}. Correct among those: {flips_correct}.",
        "",
        "Blocked excludes fact-check sites, drops pages published after the claim date, searches memory with `date_before` equal to the claim date, and excludes the source article plus the near-duplicate copies memory would collapse into `also_debunked_by`. Allowed uses the same date filter and still drops the source article's own URL from search results, fetched pages and evidence. Memory search in the allowed run excludes only that article's id, so an earlier fact-check of the same rumour can be retrieved. Submissions are not recorded. The LLM cache is on. A review pause is not answered: the provisional verdict is stored with `needs_human` and counted as an abstention.",
        "",
        "The baseline is one Gemini call on the claim text. It returns False, Misleading, True, or Unverifiable, plus its own confidence. That confidence is the model's self-report. The system's confidence is computed from the evidence.",
        "",
        "## Definitions",
        "",
        "A decided claim has verdict False, Misleading, or True and `needs_human` false. Abstention is Unverifiable or `needs_human`. A claim that is both counts once. The 3-class accuracy is correct decided claims over decided claims, and 3-class macro-F1 is the unweighted mean of the three class F1 scores; a class with no decided gold row scores 0. The 2-class score collapses False and Misleading to not-true and leaves True as True, on the same decided claims. Abstention is not part of that collapse. Coverage-adjusted accuracy is correct decided claims over every scored claim, using the 3-class correct count. ECE uses 10 equal-width confidence bins on decided claims. Wilson intervals are 95%.",
        "",
        "## Blocked, allowed, and baseline",
        "",
        _side_by_side([(name, metrics) for name, _rows, metrics in columns]),
        "",
    ]
    for title, builder in (
        ("Per class", _class_table),
        ("Per domain", _domain_table),
        ("Confusion matrix", _matrix_table),
        ("Citations", _citation_block),
    ):
        lines.extend(["", f"## {title}", ""])
        if title == "Confusion matrix":
            lines.extend(["Rows are the gold label. Columns are the predicted label. Decided claims only.", ""])
        if title == "Citations":
            lines.extend([
                "A web citation is valid when its id is one of the stored evidence rows and the quote is in that row's passage. C1 is the archive prior. It is valid when the run kept a prior. The baseline cites nothing.",
                "",
            ])
        for name, _rows, metrics in columns:
            lines.extend([f"### {name}", "", builder(metrics), ""])
    lines.extend([
        "## Calibration",
        "",
        _calibration_block([(name, metrics) for name, _rows, metrics in columns]),
        "",
    ])
    for name, _rows, metrics in columns:
        lines.extend([_bin_table(metrics, name), ""])
    lines.extend([
        "## What the system abstained on",
        "",
        "The gate reason is the sentence box 9 wrote. A review pause with no gate sentence shows the gate label.",
        "",
        "### Blocked",
        "",
        _abstention_table(blocked_rows),
        "",
        "### Allowed",
        "",
        _abstention_table(allowed_rows),
        "",
        "## Cost",
        "",
        "Median and 90th percentile per claim. The percentile is NumPy's linear interpolation.",
        "",
        _cost_table([(name, metrics) for name, _rows, metrics in columns]),
        "",
        "## Error analysis",
        "",
        "No new model calls. A decided claim is listed when the verdict differs from the gold label. The evidence rows are the ones on the side that produced the verdict. The root cause is a rule over those rows: a False/Misleading swap is `wrong refute_kind`; a True on a claim that is not True is `true kernel read as support`; a wrong verdict whose winning side has fewer than two tier 1–2 groups is `tier gap`; anything else is `other`.",
        "",
        "### Blocked",
        "",
        _error_table(blocked_rows),
        "",
        "### Allowed",
        "",
        _error_table(allowed_rows),
        "",
        "## Evidence availability",
        "",
        "Credible means tier 1 or 2. Fact-check pages are not counted, and the source article is not counted. One registrable domain is one group. The share is claims with zero such groups.",
        "",
        _availability_block("Blocked", blocked_rows),
        "",
        _availability_block("Allowed", allowed_rows),
        "",
        "## EXPLORATORY: two credible groups required to support",
        "",
        "EXPLORATORY. This rule was chosen after seeing these results. It is not a tuned headline and it was not used to pick the verdicts above.",
        "",
        "The live gate calls a claim supported when two independent groups take that side and at least one of them is tier 1 or 2. This section recounts the stored evidence with a stricter bar: supported requires at least two tier 1–2 groups on that side. A decided True that would no longer be supported is removed. A false True is a True whose gold label is not True. A correct True is a True whose gold label is True.",
        "",
        _exploratory_block("Blocked", blocked_rows),
        "",
        _exploratory_block("Allowed", allowed_rows),
        "",
        "## Limitations of this evaluation",
        "",
        _limitations(),
        "",
    ])
    return "\n".join(lines)


def later_sections(testset: list[dict], blocked: list[dict], baseline: list[dict]) -> str:
    """Sections A–C. The superseded allowed table above them stays."""
    from misinfo_agent.llm import calls_today
    from misinfo_agent.search import searches_this_month

    blocked_rows = _join(testset, blocked)
    baseline_rows = _join(testset, baseline)
    blocked_metrics = condition_metrics(blocked_rows)
    lines = [
        "## A. New rumour, fact-check sites blocked",
        "",
        "What this measures: a rumour checked on the day of the fact-check, with fact-check sites blocked and the source article hidden.",
        "",
        "No blocked claim hit the per-claim cap or the monthly cap, so none lost pages and none was re-run. The cap fix keeps pages already gathered when a later query is refused. It did not change this table.",
        "",
        _side_by_side([
            ("Blocked", blocked_metrics),
            ("Baseline", condition_metrics(baseline_rows)),
        ]),
        "",
        "## B. New rumour, fact-check sites allowed",
        "",
        "What this measures: the same new rumour, with other fact-check sites allowed to add pages on top of the blocked evidence.",
        "",
        "This condition was not run to the end. A cache-only replay of the open-web queries found an eligible fact-check page for 5 of 50 claims (`e2230f5774ce`, `dcd1974edddc`, `6486a37086dc`, `f62882a679bf`, `d5bc864c922a`). Four of those five had every fact-check query served from cache. One (`f62882a679bf`) also had a cache miss. Search asks for pages dated on or before the claim date (`end_date` is the day after the claim), so a fact-check published after the claim cannot appear. That is why most claims have no fact-check page in the cache.",
        "",
        "The news half of an allowed claim is supposed to be a cache hit, including the top-up. On `fcaebc7ad5f3` the top-up prompt missed the cache: the list of queries already run is appended in the order the searches finished, and that order is not the order from the blocked run. The new proposal is a Gemini call, and the new query strings are not in the search cache, so the pages that decided the blocked verdict are not fetched. 37 blocked claims used a top-up. At one new call each, plus one stance call for each of the 5 claims that have a fact-check page, the replay would pass 40 Gemini calls. It was stopped after that one probe. `results_system_allowed_v2.jsonl` has that probe only. The allowed column higher up in this file is still the superseded run.",
        "",
        "| Claim | What happened |",
        "| --- | --- |",
        "| fcaebc7ad5f3 | Probe. Blocked verdict False. Cache-only allowed verdict Unverifiable. 1 new Gemini call (top-up). 0 fact-check pages added. Own article excluded. |",
        "| other 49 | Not run. Projected Gemini calls would pass 40. |",
        "",
        "## C. Recycled rumour, 60 days later",
        "",
        "What this measures: the same claim text submitted again 60 days after its fact-check, when that debunk is already in memory.",
        "",
        "This is easier, because memory contains the debunk. The leak guard and the own-article exclusion are off for this condition only: the debunk now legitimately exists. The input is the testset sentence, not an extract paraphrase. Boxes 1–6 only. There is no web search. 40 claims were selected: `source` is not `true_version`, and `article_id` is in the memory index. The other 10 rows are the true-version sentences.",
        "",
        _recycled_table(load_jsonl(_RECYCLED)),
        "",
        "## Budget",
        "",
        f"Tavily month counter: {searches_this_month()}. Gemini calls_today(): {calls_today()}.",
        "",
    ]
    return "\n".join(lines)


def _recycled_table(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "No recycled rows."
    hit1 = sum(1 for row in rows if row.get("memory_rank") == 1)
    hit5 = sum(1 for row in rows if isinstance(row.get("memory_rank"), int) and row["memory_rank"] <= 5)
    reused = [row for row in rows if row.get("route") == "reuse_prior"]
    correct = sum(1 for row in reused if str(row.get("verdict")) == str(row.get("label")))
    wrong = [row for row in reused if str(row.get("verdict")) != str(row.get("label"))]
    different = sum(1 for row in reused if row.get("reused_from_different_claim") == "yes")
    interval = wilson(correct, len(reused))
    latencies = [float(row.get("latency") or 0) for row in rows]
    median = float(np.median(latencies)) if latencies else 0.0
    calls = sum(int(row.get("llm_calls") or 0) for row in rows)
    wrong_list = ", ".join(
        f"{row.get('article_id')} gold {row.get('label')} reused {row.get('verdict')} from {row.get('reused_from')}"
        for row in wrong
    ) or "none"
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Claims | {n} |",
        f"| Hit@1 | {_rate(hit1, n)} |",
        f"| Hit@5 | {_rate(hit5, n)} |",
        f"| Exact reuse | {_rate(len(reused), n)} |",
        f"| Reused-verdict accuracy | {_rate(correct, len(reused))} |",
        f"| Reused-verdict accuracy, 95% Wilson | {_interval(interval)} |",
        f"| Wrong reuses | {len(wrong)} ({wrong_list}) |",
        f"| Reused from a different claim | {different} |",
        f"| Median seconds | {median:.2f} |",
        f"| Gemini calls | {calls} |",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    del argv
    testset = load_csv(_TESTSET)
    blocked, allowed = enrich_from_checkpoints(load_jsonl(_SYSTEM), load_jsonl(_ALLOWED))
    baseline = load_jsonl(_BASELINE)
    text = render(testset, blocked, baseline, allowed)
    text = text + "\n" + later_sections(testset, blocked, baseline)
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {_REPORT}")


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    """Last line wins when an article id was written twice."""
    if not path.is_file():
        return []
    by_id = {}
    order = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            article_id = str(row.get("article_id") or "")
            if not article_id:
                continue
            if article_id not in by_id:
                order.append(article_id)
            by_id[article_id] = row
    return [by_id[article_id] for article_id in order]


def _join(testset: list[dict], results: list[dict]) -> list[dict]:
    by_id = {str(row.get("article_id")): row for row in results}
    joined = []
    for gold in testset:
        found = by_id.get(str(gold.get("article_id")))
        if found is None:
            continue
        joined.append({
            **found,
            "label": gold.get("label"),
            "domain": gold.get("domain") or "",
            "source": gold.get("source") or "",
            "input": gold.get("input") or "",
        })
    return joined


def _domains(rows: list[dict]) -> list[dict]:
    names = [domain for domain in config.DOMAINS if any(row.get("domain") == domain for row in rows)]
    extra = sorted({row.get("domain") or "" for row in rows} - set(names))
    table = []
    for domain in names + extra:
        subset = [row for row in rows if row.get("domain") == domain]
        decided = [row for row in subset if is_decided(row)]
        correct = [row for row in decided if row.get("verdict") == row.get("label")]
        macro = None
        if decided:
            macro = float(f1_score(
                [row["label"] for row in decided],
                [row["verdict"] for row in decided],
                average="macro",
                labels=LABELS,
                zero_division=0,
            ))
        table.append({
            "domain": domain,
            "n": len(subset),
            "abstained": sum(1 for row in subset if is_abstained(row)),
            "decided": len(decided),
            "correct": len(correct),
            "accuracy": (len(correct) / len(decided)) if decided else None,
            "macro_f1": macro,
        })
    return table


def _cost(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "latency_median": _percentile(rows, "latency", 50),
        "latency_p90": _percentile(rows, "latency", 90),
        "llm_calls_median": _percentile(rows, "llm_calls", 50),
        "llm_calls_p90": _percentile(rows, "llm_calls", 90),
        "searches_median": _percentile(rows, "searches", 50),
        "searches_p90": _percentile(rows, "searches", 90),
    }


def _percentile(rows: list[dict], key: str, q: float) -> float:
    values = [float(row.get(key) or 0) for row in rows]
    return float(np.percentile(values, q))


def _side_by_side(columns: list[tuple[str, dict]]) -> str:
    header = "| Metric | " + " | ".join(name for name, _metrics in columns) + " |"
    rule = "| --- | " + " | ".join("---" for _ in columns) + " |"
    specs = [
        ("Scored claims", lambda metrics: str(metrics["n"])),
        ("Decided", lambda metrics: str(metrics["decided"])),
        (
            "Decided incl. provisional",
            lambda metrics: str(metrics.get("decided_incl_provisional", "")),
        ),
        ("Pending review", lambda metrics: str(metrics.get("pending_review", ""))),
        ("Correct", lambda metrics: str(metrics["correct"])),
        ("Accuracy, 3-class (decided)", lambda metrics: _rate(metrics["correct"], metrics["decided"])),
        ("Macro-F1, 3-class (decided)", lambda metrics: _num(metrics["macro_f1"])),
        (
            "Accuracy, 2-class (not-true vs True)",
            lambda metrics: _rate(metrics["binary_correct"], metrics["decided"]),
        ),
        ("Macro-F1, 2-class (decided)", lambda metrics: _num(metrics["binary_macro_f1"])),
        ("Abstention", lambda metrics: _rate(metrics["abstained"], metrics["n"])),
        ("Coverage-adjusted accuracy", lambda metrics: _rate(metrics["correct"], metrics["n"])),
    ]
    body = []
    for name, cell in specs:
        body.append("| " + name + " | " + " | ".join(cell(metrics) for _title, metrics in columns) + " |")
    return "\n".join([header, rule, *body])


def _class_table(metrics: dict) -> str:
    if not metrics["per_class"]:
        return "No decided claims."
    lines = [
        "| Class | Precision | Recall | F1 | Support | Recall 95% Wilson |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics["per_class"]:
        interval = wilson(row["tp"], row["support"])
        lines.append(
            "| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support} | {interval} |".format(
                interval=_interval(interval),
                **row,
            )
        )
    return "\n".join(lines)


def _domain_table(metrics: dict) -> str:
    if not metrics["domains"]:
        return "No scored claims."
    lines = [
        "| Domain | n | Abstained | Decided | Correct | Accuracy | Macro-F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics["domains"]:
        lines.append(
            f"| {row['domain']} | {row['n']} | {row['abstained']} | {row['decided']} | "
            f"{row['correct']} | {_num(row['accuracy'])} | {_num(row['macro_f1'])} |"
        )
    return "\n".join(lines)


def _matrix_table(metrics: dict) -> str:
    matrix = metrics["matrix"]
    if not matrix:
        return "No decided claims."
    header = "| Gold \\ predicted | " + " | ".join(LABELS) + " |"
    rule = "| --- | " + " | ".join("---" for _ in LABELS) + " |"
    lines = [header, rule]
    for label, row in zip(LABELS, matrix):
        cells = " | ".join(str(int(value)) for value in row)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def _citation_block(metrics: dict) -> str:
    cites = metrics["citations"]
    if cites["web_n"] == 0 and cites["archive_n"] == 0:
        return "No citations."
    lines = [
        f"Web citations: {cites['web_n']}. Id present: {cites['web_exist']}. Id present and quote verified: {cites['web_valid']}.",
        f"Archive citations (C1): {cites['archive_n']}. Prior present: {cites['archive_valid']}.",
    ]
    web_rate = _rate(cites["web_valid"], cites["web_n"]) if cites["web_n"] else "n/a"
    lines.append(f"Web citation validity: {web_rate}.")
    return "\n".join(lines)


def _calibration_block(columns: list[tuple[str, dict]]) -> str:
    lines = []
    for name, metrics in columns:
        lines.append(
            f"{name} ECE (10 bins) on {_claims(metrics['ece_n'])} with a confidence: {_num(metrics['ece'])}."
        )
    return "\n".join(lines)


def _bin_table(metrics: dict, title: str) -> str:
    """Decided claims in each ECE bin. Empty bins stay, as 0."""
    counts = list(metrics.get("ece_bins") or [0] * 10)
    lines = [
        f"### {title}",
        "",
        "| Bin | Range | Decided claims |",
        "| --- | --- | --- |",
    ]
    for index, count in enumerate(counts):
        low = index / 10
        high = (index + 1) / 10
        span = f"[{low:.1f}, {high:.1f}]" if index == len(counts) - 1 else f"[{low:.1f}, {high:.1f})"
        lines.append(f"| {index} | {span} | {int(count)} |")
    return "\n".join(lines)


def _abstention_table(rows: list[dict]) -> str:
    held = [row for row in rows if is_abstained(row)]
    if not held:
        return "The system did not abstain."
    lines = [
        "| Claim | Gold | Gate reason |",
        "| --- | --- | --- |",
    ]
    for row in held:
        reason = row.get("gate_reason") or row.get("gate") or ""
        lines.append(f"| {_cell(row.get('input') or '')} | {_cell(row.get('label') or '')} | {_cell(reason)} |")
    return "\n".join(lines)


def _cell(text: str) -> str:
    """One markdown cell. Pipes would open a new column."""
    return " ".join(str(text).split()).replace("|", "/")


def _cost_table(columns: list[tuple[str, dict]]) -> str:
    heads = []
    for name, _metrics in columns:
        heads.extend([f"{name} median", f"{name} p90"])
    header = "| | " + " | ".join(heads) + " |"
    rule = "| --- | " + " | ".join("---" for _ in heads) + " |"
    rows = []
    for label, median_key, p90_key in (
        ("Latency (s)", "latency_median", "latency_p90"),
        ("LLM calls", "llm_calls_median", "llm_calls_p90"),
        ("Searches", "searches_median", "searches_p90"),
    ):
        cells = []
        for _name, metrics in columns:
            cells.append(_cost_cell(metrics, median_key))
            cells.append(_cost_cell(metrics, p90_key))
        rows.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def _cost_cell(metrics: dict, key: str) -> str:
    cost = metrics.get("cost") or {}
    if key not in cost:
        return "n/a"
    return f"{cost[key]:.2f}"


def _leak_total(rows: list[dict]) -> int:
    return sum(int(row.get("leak_count") or 0) for row in rows)


def _factcheck_share(rows: list[dict]) -> str:
    """Share of allowed claims whose decision used another fact-checker's page."""
    if not rows:
        return "Fact-check evidence used (allowed): no results yet."
    used = sum(1 for row in rows if int(row.get("factcheck_evidence_used") or 0) > 0)
    return (
        f"Fact-check evidence used (allowed): {_rate(used, len(rows))}. "
        "This is another outlet's fact-check in the evidence table, or a prior the route reused. "
        "The source article itself is not included."
    )


def became_decided(blocked: list[dict], allowed: list[dict]) -> tuple[int, int]:
    """Blocked abstentions that the allowed run decided, and how many were right.

    A claim present only on one side is not a flip. Correct means the
    allowed verdict matches the gold label.
    """
    other = {str(row.get("article_id")): row for row in allowed}
    flips = []
    for row in blocked:
        if not is_abstained(row):
            continue
        found = other.get(str(row.get("article_id")))
        if found is None or not is_decided(found):
            continue
        flips.append(found)
    correct = sum(1 for row in flips if row.get("verdict") == row.get("label"))
    return len(flips), correct


def root_cause(row: dict) -> str:
    """One label for a wrong decided claim. The rule is fixed, not a reread.

    A False/Misleading swap is the draft's refute_kind, which is the only
    thing that separates those two once the gate has said refuted. A True
    on any other gold label means a support row carried a claim the gold
    says is not true. A winning side with fewer than two tier 1–2 groups
    filled the rest of the bar with a weaker source.
    """
    gold = row.get("label")
    pred = row.get("verdict")
    if gold in {"False", "Misleading"} and pred in {"False", "Misleading"} and gold != pred:
        return "wrong refute_kind"
    if pred == "True" and gold != "True":
        return "true kernel read as support"
    core = str(row.get("core_id") or "A1")
    side = "supports" if pred == "True" else "refutes"
    tiers: dict[str, int] = {}
    for item in row.get("evidence") or []:
        if not isinstance(item, dict) or item.get("group_conflict"):
            continue
        if str(item.get("atomic_id") or "A1") != core or item.get("stance") != side:
            continue
        name = str(item.get("group_id") or item.get("id") or "")
        if not name:
            continue
        try:
            tier = int(item.get("tier"))
        except (TypeError, ValueError):
            tier = 4
        tiers[name] = min(tiers.get(name, 4), tier)
    credible = sum(1 for tier in tiers.values() if tier <= 2)
    if tiers and credible < 2:
        return "tier gap"
    return "other"


def stricter_support(row: dict) -> bool | None:
    """Whether the stored evidence would still support the core.

    None when a row has no tier, so the recount would be a guess. The
    bar is two tier 1–2 groups on the support side, and no conflict the
    live gate would already have blocked. The live gate's one-credible
    bar is not reapplied.
    """
    evidence = [item for item in (row.get("evidence") or []) if isinstance(item, dict)]
    if any(item.get("tier") is None or not item.get("stance") for item in evidence):
        return None
    from misinfo_agent.gate import _conflict, _groups, _rows

    core = str(row.get("core_id") or "A1")
    grouped = _groups(_rows(evidence, core, drop_after=True))
    blocks, _stray, _why = _conflict(grouped)
    if blocks:
        return False
    return len(grouped["support_credible"]) >= 2


def exploratory_counts(rows: list[dict]) -> dict:
    """False Trues removed and correct Trues lost under the stricter bar."""
    removed = lost = unknown = 0
    for row in rows:
        if not is_decided(row) or row.get("verdict") != "True":
            continue
        still = stricter_support(row)
        if still is None:
            unknown += 1
            continue
        if still:
            continue
        if row.get("label") == "True":
            lost += 1
        else:
            removed += 1
    return {"removed": removed, "lost": lost, "unknown": unknown}


def _error_table(rows: list[dict]) -> str:
    wrong = [row for row in rows if is_decided(row) and row.get("verdict") != row.get("label")]
    if not wrong:
        return "No decided claim was wrong."
    blocks = []
    for row in wrong:
        blocks.append(
            f"- **{_cell(row.get('input') or row.get('article_id') or '')}** "
            f"Gold `{row.get('label')}`. Predicted `{row.get('verdict')}`. "
            f"Root cause: `{root_cause(row)}`."
        )
        driving = _driving_evidence(row)
        if not driving:
            blocks.append("  - No stored evidence row on the side that produced the verdict.")
            continue
        for item in driving:
            quote = _cell(item.get("quote") or "")
            if len(quote) > 220:
                quote = quote[:217] + "..."
            site = _cell(item.get("netloc") or item.get("url") or "")
            blocks.append(
                f"  - {site}, tier {item.get('tier')}, {item.get('stance')}: {quote}"
            )
    return "\n".join(blocks)


def _driving_evidence(row: dict) -> list[dict]:
    """Rows on the stance that carried the verdict, for the core claim."""
    core = str(row.get("core_id") or "A1")
    side = "supports" if row.get("verdict") == "True" else "refutes"
    evidence = [item for item in (row.get("evidence") or []) if isinstance(item, dict)]
    driving = [
        item for item in evidence
        if str(item.get("atomic_id") or "A1") == core and item.get("stance") == side
    ]
    return driving or evidence


def _availability_block(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"### {name}\n\nNo results."
    known = [row for row in rows if row.get("credible_non_factcheck_groups") is not None]
    if len(known) != len(rows):
        return f"### {name}\n\nGroup counts were not stored for every claim."
    zero = sum(1 for row in known if int(row.get("credible_non_factcheck_groups") or 0) == 0)
    lines = [
        f"### {name}",
        "",
        f"Share with zero credible non-fact-check groups: {_rate(zero, len(known))}.",
        "",
        "| Claim | Credible non-fact-check groups |",
        "| --- | --- |",
    ]
    for row in known:
        lines.append(
            f"| {_cell(row.get('input') or '')} | {int(row.get('credible_non_factcheck_groups') or 0)} |"
        )
    return "\n".join(lines)


def _exploratory_block(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"### {name}\n\nNo results."
    counts = exploratory_counts(rows)
    return "\n".join([
        f"### {name}",
        "",
        f"False Trues removed: {counts['removed']}.",
        f"Correct Trues lost: {counts['lost']}.",
        f"Decided Trues whose stored evidence had no tier, so they were not recounted: {counts['unknown']}.",
    ])


def _limitations() -> str:
    return "\n".join([
        "Fifty claims. The Wilson intervals on accuracy and on abstention are wide, and a difference of a few claims moves the headline.",
        "",
        "The gold labels are the archive's `verdict_norm` on LLM-classified fact-checks, plus `true_version` sentences taken from other articles. They are not a second human coding of these fifty lines.",
        "",
        "The baseline is a Gemini call on the claim text alone. The model may have seen the rumour, or the fact-check, in its training data. A high baseline score can be memory of the article rather than reasoning from evidence.",
        "",
        "The blocked condition removes fact-check sites from web search. For many of these rumours that site is the main public correction, so the blocked number is the system working without the source a reader would actually open.",
        "",
        "The allowed condition may use another fact-checker's earlier verdict on the same rumour. A deployment that leaves fact-check sites in the search would use those pages too. The source article itself stays excluded, so the run is not reading the answer key it was drawn from.",
    ])


def _needs_checkpoint(row: dict) -> bool:
    """True when this result line cannot support the offline recount."""
    if "credible_non_factcheck_groups" not in row:
        return True
    for item in row.get("evidence") or []:
        if isinstance(item, dict) and "tier" not in item:
            return True
    return False


def enrich_from_checkpoints(blocked: list[dict], allowed: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fill tier, stance and group counts from the eval checkpoints.

    The blocked file was written before those fields were stored. The
    checkpoint is the same run: no model call, no fetch. A row that
    already has the fields is left as it was written.
    """
    if not any(_needs_checkpoint(row) for row in [*blocked, *allowed]):
        return blocked, allowed
    path = config.EVAL_DIR / "runs" / "checkpoints.db"
    if not path.is_file():
        return blocked, allowed
    index = _checkpoint_index(path)
    return (
        [_fill_from_checkpoint(row, index, "blocked") for row in blocked],
        [_fill_from_checkpoint(row, index, "allowed") for row in allowed],
    )


def _run_eval_module():
    """The harness module. scripts/ is not a package, so this loads it by path."""
    found = sys.modules.get("run_eval")
    if found is not None:
        return found
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval"] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_index(path: Path) -> dict[tuple[str, str], list[dict]]:
    """(article_id, sites) -> states. A thread with no eval_sites is blocked."""
    import sqlite3

    from misinfo_agent.graph import build_graph

    graph = build_graph(path, path.parent / "audit.db", path.parent / "memory_additions.parquet")
    threads = sqlite3.connect(path).execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    index: dict[tuple[str, str], list[dict]] = {}
    for (thread,) in threads:
        parts = str(thread).split("-")
        if len(parts) < 3 or parts[0] != "eval":
            continue
        snap = graph.get_state({"configurable": {"thread_id": thread}})
        values = getattr(snap, "values", None)
        if not isinstance(values, dict) or not values:
            continue
        marks = values.get("run_marks") if isinstance(values.get("run_marks"), dict) else {}
        site = str(marks.get("eval_sites") or "blocked")
        index.setdefault((parts[1], site), []).append(values)
    return index


def _fill_from_checkpoint(row: dict, index: dict, sites: str) -> dict:
    if not _needs_checkpoint(row):
        return row
    states = index.get((str(row.get("article_id")), sites), [])
    state = _best_checkpoint(states, row)
    if state is None:
        return row
    filled = dict(row)
    # An empty evidence list is what the run stored. Replacing it from a
    # different thread would invent citations. A list that has quotes but
    # no tier is the same rows, with the fields the later analysis needs.
    stored_missing_tier = any(
        isinstance(item, dict) and "tier" not in item for item in (row.get("evidence") or [])
    )
    if stored_missing_tier:
        filled["evidence"] = _slim_evidence(state)
    if "credible_non_factcheck_groups" not in row:
        module = _run_eval_module()
        credible_non_factcheck_groups = module.credible_non_factcheck_groups
        url_key = module.url_key

        marks = state.get("run_marks") if isinstance(state.get("run_marks"), dict) else {}
        banned = marks.get("eval_exclude_urls") or []
        own = url_key(banned[0]) if banned else ""
        filled["credible_non_factcheck_groups"] = credible_non_factcheck_groups(state, own)
    if not filled.get("route"):
        filled["route"] = str(state.get("route") or "")
    if not filled.get("core_id") or filled.get("core_id") == "A1":
        core, kind = _core_from_state(state)
        filled["core_id"] = core
        if not filled.get("refute_kind"):
            filled["refute_kind"] = kind
    return filled


def _best_checkpoint(states: list[dict], row: dict) -> dict | None:
    """The thread whose verdict and quotes match the result line."""
    if not states:
        return None
    stored = {str(item.get("quote") or "") for item in (row.get("evidence") or []) if isinstance(item, dict)}

    def score(state: dict) -> tuple:
        verdict_ok = str(state.get("verdict") or "") == str(row.get("verdict") or "")
        quotes = {str(item.get("quote") or "") for item in (state.get("evidence") or []) if isinstance(item, dict)}
        overlap = len(stored & quotes) if stored else 0
        return (verdict_ok, overlap, len(quotes))

    return max(states, key=score)


def _slim_evidence(state: dict) -> list[dict]:
    rows = []
    for item in state.get("evidence") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        rows.append({
            "id": str(item["id"]),
            "url": str(item.get("url") or ""),
            "netloc": str(item.get("netloc") or ""),
            "tier": item.get("tier"),
            "stance": str(item.get("stance") or ""),
            "quote": str(item.get("quote") or ""),
            "passage": str(item.get("passage") or ""),
            "group_id": str(item.get("group_id") or ""),
            "atomic_id": str(item.get("atomic_id") or "A1"),
            "group_conflict": bool(item.get("group_conflict")),
        })
    return rows


def _core_from_state(state: dict) -> tuple[str, str]:
    core = "A1"
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("core") is True:
            core = str(item.get("id") or "A1")
            break
    kind = ""
    for item in state.get("atomic_results") or []:
        if isinstance(item, dict) and str(item.get("atomic_id") or "") == core:
            kind = str(item.get("refute_kind") or "")
            break
    return core, kind


def _rate(successes: int, n: int) -> str:
    if n <= 0:
        return "n/a"
    interval = wilson(successes, n)
    return f"{successes / n:.3f} ({successes}/{n}) {_interval(interval)}"


def _interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _claims(n: int) -> str:
    word = "claim" if n == 1 else "claims"
    return f"{n} decided {word}"


def _num(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
