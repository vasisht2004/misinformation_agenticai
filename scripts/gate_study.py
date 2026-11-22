"""Dev/test study of gate variants. No model call and no live search.

The selection rule is written into docs/eval_diagnosis.md and flushed
before any claim is read and before any variant is scored. Test claims
are scored once, and only for G0 and the variant the dev rule picked.

Cache hits stay allowed: SEARCH_CACHE and LLM_CACHE are not turned off.
A miss cannot call out. The API keys are removed, SEARCH_CACHE_ONLY is
set, and the live call functions are replaced so a miss raises instead
of billing Gemini or Tavily.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ("False", "Misleading", "True")
HEADING = "## GATE STUDY (dev/test split; selection rule written before test was scored)"
RULE = (
    "Pick the variant with the best coverage-adjusted 2-class accuracy on dev "
    "among variants whose True precision >= 0.75 and whose number of wrong "
    "decisions on dev <= G0's. Score it once on test."
)
RESULTS_MARK = "<!-- gate-study-results -->"
CASE_CLAIM = "India has 29 states in total."
_COUNTER_FILES = (
    ROOT / "data" / "cache" / "llm_daily_count.json",
    ROOT / "data" / "cache" / "search_monthly_count.json",
)
_DOC = ROOT / "docs" / "eval_diagnosis.md"
_CONFIG = ROOT / "misinfo_agent" / "config.py"
_SNAPSHOT: dict | None = None

# G0 is the current gate: core match on, relaxed, single tier-1 denial on,
# and every flag below left off. Combos are those flags together. G4's
# theta is not a flag; it is chosen on dev from GATE_EVIDENCE_THETA_GRID.
VARIANTS = (
    ("G0", {}),
    ("G1", {"GATE_PRIMARY_SOURCE": True}),
    ("G2R", {"GATE_WEAK_REFUTE": True}),
    ("G3", {"GATE_SHARED_SOURCE": True}),
    ("G4", {"GATE_EVIDENCE_SCORE": True}),
    ("G2R+G3", {"GATE_WEAK_REFUTE": True, "GATE_SHARED_SOURCE": True}),
    ("G1+G2R+G3", {"GATE_PRIMARY_SOURCE": True, "GATE_WEAK_REFUTE": True, "GATE_SHARED_SOURCE": True}),
    ("G4+G3", {"GATE_EVIDENCE_SCORE": True, "GATE_SHARED_SOURCE": True}),
)
_FLAG_NAMES = (
    "GATE_PRIMARY_SOURCE",
    "GATE_WEAK_REFUTE",
    "GATE_SHARED_SOURCE",
    "GATE_EVIDENCE_SCORE",
)
_SENTENCE = {
    "G0": "Two independent groups, at least one of them tier 1 or 2, with the core-match rule and the single tier-1 denial already on.",
    "G1": "One tier-1 page from the office the claim names can support or refute when no tier 1-2 group is on the other side.",
    "G2R": "Three independent refutations and no support at any tier can refute, including tier 3 and 4, when each denial is about the claim; confidence is capped at 0.6 and the card says no tier 1-2 source was found. Support is never decided this way.",
    "G3": "Outlets that share a 12-word quotation, the same press-release URL, or a PTI, ANI, or PIB-release credit count as one source.",
    "G4": "Groups are summed as tier weight times stance, similarity, recency, and core match. Theta is chosen on dev from a fixed grid. Support still needs a tier 1-2 group, and the score does not overrule a count the current gate already settled.",
    "G2R+G3": "The weak-refute path runs after copied releases have been merged into one source.",
    "G1+G2R+G3": "The body-page path and the weak-refute path run after copied releases have been merged.",
    "G4+G3": "The evidence score runs after copied releases have been merged. Its theta is chosen on dev separately from G4 alone.",
}


def _arm() -> None:
    """Drop the keys and freeze search on the cache. Do not disable the caches.

    A cache hit is read before either client is built. A miss hits the
    tripwire installed in _run, not the network. The quota files are
    copied before the package import so opening a checkpoint cannot be
    what first touches them.
    """
    global _SNAPSHOT
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)
    os.environ["SEARCH_CACHE_ONLY"] = "1"
    _SNAPSHOT = {
        path: path.read_bytes() if path.is_file() else None
        for path in _COUNTER_FILES
    }


def _counters_unchanged() -> None:
    if _SNAPSHOT is None:
        raise SystemExit("counter snapshot missing")
    current = {
        path: path.read_bytes() if path.is_file() else None
        for path in _COUNTER_FILES
    }
    if current != _SNAPSHOT:
        raise SystemExit("COUNTERS CHANGED")
    print("COUNTERS UNCHANGED")


def _tripwire() -> None:
    """A cache miss must raise. A cache hit returns before these functions."""
    import misinfo_agent.llm as llm
    import misinfo_agent.search as search

    def _blocked(*_args, **_kwargs):
        raise SystemExit("live call blocked")

    llm._produce = _blocked
    search._real = _blocked


def stratified_half(items: list[tuple[str, str]], seed: int = 42) -> tuple[list[str], list[str]]:
    """Half of each gold label.

    Labels are shuffled with Random(seed) in alphabetical order, which for
    these labels is False, Misleading, True. An odd class gives its extra
    item to whichever side is currently smaller, so 23, 17, and 10 land on
    25 and 25 rather than 26 and 24.
    """
    buckets: dict[str, list[str]] = {}
    for article_id, label in items:
        buckets.setdefault(label, []).append(article_id)
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    for label in sorted(buckets):
        ids = list(buckets[label])
        rng.shuffle(ids)
        n_dev = len(ids) // 2
        if len(ids) % 2 == 1 and len(dev) <= len(test):
            n_dev += 1
        dev.extend(ids[:n_dev])
        test.extend(ids[n_dev:])
    return dev, test


def rates(golds: list[str], predicted: list[str]) -> dict:
    """Decided means a False, Misleading, or True. Anything else abstains.

    A wrong decision is a decided label that is not the gold label. A
    False/Misleading swap is wrong. An abstention is not a wrong decision.
    Coverage-adjusted accuracy counts an abstention as not correct.
    """
    n = len(golds)
    decided_at = [index for index, label in enumerate(predicted) if label in LABELS]
    decided = len(decided_at)
    correct3 = sum(1 for index in decided_at if predicted[index] == golds[index])
    correct2 = sum(
        1
        for index in decided_at
        if _collapsed(predicted[index]) == _collapsed(golds[index])
    )
    true_pred = [index for index in decided_at if predicted[index] == "True"]
    true_gold = sum(1 for gold in golds if gold == "True")
    true_hit = sum(1 for index in true_pred if golds[index] == "True")
    true_recalled = sum(
        1 for index, gold in enumerate(golds) if gold == "True" and predicted[index] == "True"
    )
    return {
        "n": n,
        "decided": decided,
        "correct3": correct3,
        "acc3": (correct3 / decided) if decided else None,
        "correct2": correct2,
        "acc2": (correct2 / decided) if decided else None,
        "cov3": (correct3 / n) if n else None,
        "cov2": (correct2 / n) if n else None,
        "abstain": ((n - decided) / n) if n else None,
        "true_precision": (true_hit / len(true_pred)) if true_pred else None,
        "true_hit": true_hit,
        "true_pred": len(true_pred),
        "true_recall": (true_recalled / true_gold) if true_gold else None,
        "true_gold_hit": true_recalled,
        "true_gold": true_gold,
        "wrong": decided - correct3,
    }


def select_winner(rows: list[dict]) -> str | None:
    """The selection rule. None when no row meets it.

    A tie on coverage-adjusted 2-class accuracy goes to fewer wrong
    decisions, then higher True precision, then G0, then the earlier row.
    Returning G0 means the current gate won and no new flag should change.
    """
    g0 = next(row for row in rows if row["name"] == "G0")
    eligible = []
    for row in rows:
        precision = row["true_precision"]
        if precision is None or precision < 0.75:
            continue
        if row["wrong"] > g0["wrong"]:
            continue
        eligible.append(row)
    if not eligible:
        return None

    def key(row: dict) -> tuple:
        return (
            row["cov2"] if row["cov2"] is not None else -1.0,
            -row["wrong"],
            row["true_precision"] if row["true_precision"] is not None else -1.0,
            1 if row["name"] == "G0" else 0,
            -row["order"],
        )

    return max(eligible, key=key)["name"]


def choose_theta(scored: list[dict], g0_wrong: int) -> tuple[float, bool]:
    """Theta on dev only. A tie takes the higher theta.

    The grid is fixed. When no theta keeps True precision and the wrong
    count inside the selection rule, the theta with the best dev
    coverage-adjusted 2-class accuracy is still reported, and the flag
    stays ineligible.
    """
    eligible = [
        row
        for row in scored
        if row["true_precision"] is not None
        and row["true_precision"] >= 0.75
        and row["wrong"] <= g0_wrong
    ]
    pool = eligible or scored
    best = max(pool, key=lambda row: (row["cov2"] if row["cov2"] is not None else -1.0, row["theta"]))
    return float(best["theta"]), bool(eligible)


def _collapsed(label: str | None) -> str | None:
    if label == "True":
        return "True"
    if label in {"False", "Misleading"}:
        return "not-true"
    return None


def _write_rule_first() -> None:
    """The rule hits disk before a checkpoint is opened.

    A second run finds the heading and does not write it again. Scoring
    starts only after this returns.
    """
    text = _DOC.read_text(encoding="utf-8")
    if HEADING not in text:
        block = (
            "\n\n"
            + HEADING
            + "\n\n"
            + "Selection rule, written before any variant was scored and before the test split was scored:\n\n"
            + RULE
            + "\n"
        )
        with _DOC.open("a", encoding="utf-8") as handle:
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    print("SELECTION RULE ON DISK")


def main() -> None:
    _arm()
    try:
        _write_rule_first()
        _run()
    finally:
        if _SNAPSHOT is not None:
            _counters_unchanged()


def _run() -> None:
    _tripwire()
    import misinfo_agent.config as config
    from misinfo_agent.gate import gate

    sys.path.insert(0, str(ROOT / "scripts"))
    import diagnose_eval as diag

    config.EVAL_DROP_AFTER_CLAIM = True
    _apply(config, {}, None)

    eval_copy = Path("/tmp/ckpt_gate_eval.db")
    live_copy = Path("/tmp/ckpt_live_gate.db")
    shutil.copy(ROOT / "data" / "eval" / "runs" / "checkpoints.db", eval_copy)
    shutil.copy(ROOT / "data" / "checkpoints.db", live_copy)

    testset = diag.load_testset(ROOT / "data" / "eval" / "testset.csv")
    blocked = {
        row["article_id"]: row
        for row in diag.load_jsonl(ROOT / "data" / "eval" / "results_system_blocked.jsonl")
    }
    grouped = diag.load_blocked_threads(eval_copy)
    records = []
    for item in testset:
        article_id = str(item["article_id"])
        gold = str(item["label"]).strip()
        if gold not in LABELS:
            raise SystemExit(f"unexpected gold label: {gold}")
        published = blocked[article_id]
        chosen, _how = diag.choose_thread(grouped.get(article_id, []), published)
        if not chosen:
            raise SystemExit(f"no checkpoint state for {article_id}")
        state = chosen["state"]
        graded = published.get("verdict") if diag.is_decided(published) else None
        records.append({
            "id": article_id,
            "gold": gold,
            "state": state,
            "reuse": state.get("route") == "reuse_prior",
            "published": graded if graded in LABELS else None,
        })
    if len(records) != 50:
        raise SystemExit(f"expected 50 eval claims, found {len(records)}")

    dev_ids, test_ids = stratified_half([(row["id"], row["gold"]) for row in records], 42)
    if len(dev_ids) != 25 or len(test_ids) != 25 or set(dev_ids) & set(test_ids):
        raise SystemExit("split was not 25/25")
    dev_set = set(dev_ids)
    dev_rows = [row for row in records if row["id"] in dev_set]
    test_rows = [row for row in records if row["id"] not in dev_set]
    _write_split(dev_rows, test_rows)

    case_state, case_thread, case_ckpt = _load_case(live_copy)
    _assert_case_shape(case_state)
    _write_features(records, case_state)

    # Dev only. test_rows is not passed in until a winner exists.
    dev_table, theta_notes = _score_dev(config, gate, dev_rows)
    winner = select_winner(dev_table)
    by_name = {row["name"]: row for row in dev_table}

    test_report = []
    if winner is None:
        test_report.append(_score_one(config, gate, test_rows, "G0", {}, None))
    else:
        winner_row = by_name[winner]
        flags = dict(next(flags for name, flags in VARIANTS if name == winner))
        theta = winner_row["theta"]
        test_report.append(_score_one(config, gate, test_rows, "G0", {}, None))
        if winner != "G0":
            test_report.append(_score_one(config, gate, test_rows, winner, flags, theta))

    case_lines = _case_study(config, gate, case_state, by_name)
    if case_lines["G2R"]["verdict"] not in {"False", "Misleading"}:
        raise SystemExit(
            "G2R did not refute the 29-states case: " + case_lines["G2R"]["verdict"]
        )

    turned = _turn_on(winner, by_name.get(winner, {}).get("theta") if winner else None)
    section = _render(
        diag,
        dev_rows,
        test_rows,
        dev_table,
        theta_notes,
        winner,
        test_report,
        case_lines,
        case_thread,
        case_ckpt,
        turned,
        sum(1 for row in records if row["reuse"]),
    )
    _write_results(section)
    print(section)
    if turned:
        print("FLAG ON: " + ", ".join(f"{name}={value}" for name, value in turned))
    else:
        print("FLAG ON: none")


def _apply(config, flags: dict, theta: float | None) -> None:
    for name in _FLAG_NAMES:
        setattr(config, name, bool(flags.get(name, False)))
    if theta is not None:
        config.GATE_EVIDENCE_THETA = float(theta)


def _score_dev(config, gate, dev_rows: list[dict]) -> tuple[list[dict], dict]:
    g0_pred = _predict(config, gate, dev_rows, {}, None)
    g0 = _pack("G0", 0, None, dev_rows, g0_pred)
    table = [g0]
    notes = {}
    for order, (name, flags) in enumerate(VARIANTS):
        if name == "G0":
            continue
        if flags.get("GATE_EVIDENCE_SCORE"):
            scored = []
            for theta in config.GATE_EVIDENCE_THETA_GRID:
                predicted = _predict(config, gate, dev_rows, flags, float(theta))
                metric = rates([row["gold"] for row in dev_rows], predicted)
                metric["theta"] = float(theta)
                scored.append(metric)
            theta, eligible = choose_theta(scored, g0["wrong"])
            notes[name] = {"grid": scored, "theta": theta, "eligible": eligible}
            predicted = _predict(config, gate, dev_rows, flags, theta)
            table.append(_pack(name, order, theta, dev_rows, predicted))
        else:
            predicted = _predict(config, gate, dev_rows, flags, None)
            table.append(_pack(name, order, None, dev_rows, predicted))
    _apply(config, {}, None)
    return table, notes


def _score_one(config, gate, rows, name, flags, theta) -> dict:
    predicted = _predict(config, gate, rows, flags, theta)
    packed = _pack(name, 0, theta, rows, predicted)
    packed["predicted"] = predicted
    packed["golds"] = [row["gold"] for row in rows]
    return packed


def _predict(config, gate, rows, flags, theta) -> list[str]:
    _apply(config, flags, theta)
    labels = []
    for row in rows:
        if row["reuse"]:
            gated = gate(row["state"])
            if gated.get("gate") != "skipped":
                raise SystemExit(f"reuse_prior was re-judged: {row['id']}")
            labels.append(row["published"] or "Unverifiable")
            continue
        gated = gate(row["state"])
        labels.append(_code_verdict(row["state"], gated))
    return labels


def _pack(name, order, theta, rows, predicted) -> dict:
    metric = rates([row["gold"] for row in rows], predicted)
    metric["name"] = name
    metric["order"] = order
    metric["theta"] = theta
    return metric


def _code_verdict(state: dict, gated: dict) -> str:
    """The live assess mapping. A missing refute_kind stays Misleading.

    This is not the post-hoc False used in the earlier replay. The study
    asked for the normal mapping.
    """
    gate_name = gated.get("gate")
    if gate_name in {"conflicting", "error", "skipped", "search_more"}:
        return "Unverifiable"
    core = _core_id(state)
    status = (gated.get("atomic_status") or {}).get(core, "insufficient")
    kind = _cached_kind(state, core)
    if status == "refuted":
        # Same split as assess._verdict. single_fact_wrong is False.
        # A missing kind stays Misleading.
        from misinfo_agent.assess import _FALSE_KINDS

        if kind in _FALSE_KINDS:
            return "False"
        return "Misleading"
    others = [
        value
        for key, value in (gated.get("atomic_status") or {}).items()
        if key != core and value != "not_checkable"
    ]
    if status == "supported" and any(value == "refuted" for value in others):
        return "Misleading"
    if status == "supported":
        return "True"
    if status == "not_checkable":
        return "Other"
    return "Unverifiable"


def _cached_kind(state: dict, core: str) -> str | None:
    for item in state.get("atomic_results") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("atomic_id") or "") != core:
            continue
        kind = item.get("refute_kind")
        # Same six values assess._verdict maps. A kind this list does
        # not know stays missing, and a missing kind stays Misleading.
        from misinfo_agent.assess import _FALSE_KINDS, _MISLEADING_KINDS

        if kind in _FALSE_KINDS or kind in _MISLEADING_KINDS:
            return kind
    return None


def _core_id(state: dict) -> str:
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("core") is True:
            return str(item.get("id") or "A1")
    return "A1"


def _core_text(state: dict) -> str:
    core = _core_id(state)
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and str(item.get("id") or "") == core:
            return str(item.get("text") or state.get("claim") or "")
    return str(state.get("claim") or "")


def _write_split(dev_rows: list[dict], test_rows: list[dict]) -> None:
    def side(rows: list[dict]) -> list[dict]:
        return [{"article_id": row["id"], "label": row["gold"]} for row in rows]

    def counts(rows: list[dict]) -> dict:
        found = {label: 0 for label in LABELS}
        for row in rows:
            found[row["gold"]] += 1
        return found

    payload = {
        "seed": 42,
        "n_dev": len(dev_rows),
        "n_test": len(test_rows),
        "method": (
            "Stratified by gold label. Within each label, shuffle with "
            "random.Random(42). Labels run in alphabetical order. An odd "
            "class gives the extra item to the side that is currently smaller."
        ),
        "dev_counts": counts(dev_rows),
        "test_counts": counts(test_rows),
        "dev": side(dev_rows),
        "test": side(test_rows),
        "case_study_in_scored_split": False,
    }
    path = ROOT / "data" / "eval" / "gate_split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_features(records: list[dict], case_state: dict) -> None:
    rows = []
    for record in records:
        rows.append(_feature_row(record["id"], record["gold"], "eval", record["state"]))
    rows.append(_feature_row("case-india-29-states", "", "case_study", case_state))
    path = ROOT / "data" / "eval" / "gate_features.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _feature_row(article_id: str, gold: str, role: str, state: dict) -> dict:
    from misinfo_agent.gate import _groups, _parse_day, _rows, refute_core_matches
    from misinfo_agent.prior_match import core_support_matches

    import misinfo_agent.config as config

    core = _core_text(state)
    rows = _rows(state.get("evidence"), _core_id(state), drop_after=True)
    grouped = _groups(rows)
    claim_day = _parse_day(state.get("claim_date"))
    stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    try:
        unverified = int(stats.get("quote_unverified") or 0)
    except (TypeError, ValueError):
        unverified = 0
    out = {
        "article_id": article_id,
        "role": role,
        "gold": gold,
        "domain": str(state.get("domain") or ""),
        "status_claim": _status_claim(str(state.get("claim") or core), config.STATUS_KEYWORDS),
        "quote_unverified": unverified,
        "claim_date": str(state.get("claim_date") or "")[:10],
    }
    for side, names in (("support", grouped["support"]), ("refute", grouped["refute"])):
        stance = "supports" if side == "support" else "refutes"
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for name in names:
            tier = grouped["tier"].get(name, 4)
            counts[tier if tier in counts else 4] += 1
        stance_rows = [row for row in rows if row.get("stance") == stance]
        cosines = []
        dated = []
        hits = 0
        for row in stance_rows:
            try:
                cosines.append(float(row.get("cos")))
            except (TypeError, ValueError):
                pass
            day = _parse_day(row.get("published_date"))
            if day is not None:
                dated.append(day)
            if stance == "supports":
                matched = core_support_matches(
                    str(row.get("quote") or ""),
                    core,
                    passage=str(row.get("passage") or ""),
                    mode="relaxed",
                )
            else:
                matched = refute_core_matches(
                    str(row.get("quote") or ""),
                    core,
                    passage=str(row.get("passage") or ""),
                )
            hits += int(bool(matched))
        newest = max(dated) if dated else None
        oldest = min(dated) if dated else None
        out[f"{side}_t1"] = counts[1]
        out[f"{side}_t2"] = counts[2]
        out[f"{side}_t3"] = counts[3]
        out[f"{side}_t4"] = counts[4]
        out[f"{side}_max_cos"] = f"{max(cosines):.6f}" if cosines else ""
        out[f"{side}_newest"] = newest.isoformat() if newest else ""
        out[f"{side}_oldest"] = oldest.isoformat() if oldest else ""
        out[f"{side}_newest_vs_claim_days"] = (
            (newest - claim_day).days if newest and claim_day else ""
        )
        out[f"{side}_oldest_vs_claim_days"] = (
            (oldest - claim_day).days if oldest and claim_day else ""
        )
        out[f"{side}_core_match_hits"] = hits
    return out


def _status_claim(text: str, words: list[str]) -> int:
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
            return 1
    return 0


def _assert_case_shape(state: dict) -> None:
    """The motivating table: three refuting groups, none of them tier 1-2, no support."""
    from misinfo_agent.gate import _groups, _rows

    rows = _rows(state.get("evidence"), _core_id(state), drop_after=True)
    grouped = _groups(rows)
    if grouped["support"] or len(grouped["refute"]) != 3 or grouped["refute_credible"]:
        raise SystemExit(
            "case study is not 3 lower-tier refutes and 0 support: "
            f"support={grouped['support']} refute={grouped['refute']}"
        )


def _load_case(db_path: Path) -> tuple[dict, str, str]:
    """The latest checkpoint whose claim is this sentence.

    A newer thread in the same file states a different sentence. That run
    is not this case, and it is not in the scored split.
    """
    from misinfo_agent.graph import build_graph

    graph = build_graph(db_path, "/tmp/gate_case_audit_unused.db", "/tmp/gate_case_additions_unused.parquet")
    connection = sqlite3.connect(db_path)
    try:
        latest = connection.execute(
            "SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id"
        ).fetchall()
    finally:
        connection.close()
    found = None
    for thread, checkpoint_id in latest:
        snap = graph.get_state({"configurable": {"thread_id": thread}})
        values = getattr(snap, "values", None) or {}
        if str(values.get("claim") or "").strip() != CASE_CLAIM:
            continue
        marker = str(checkpoint_id)
        if found is None or marker > found[0]:
            found = (marker, str(thread), values)
    if found is None:
        raise SystemExit("case-study claim was not in the live checkpoint copy")
    return found[2], found[1], found[0]


def _case_study(config, gate, state: dict, by_name: dict) -> dict:
    from misinfo_agent.assess import _confidence

    found = {}
    for name, flags in VARIANTS:
        theta = by_name.get(name, {}).get("theta")
        _apply(config, flags, theta)
        gated = gate(state)
        verdict = _code_verdict(state, gated)
        merged = dict(state)
        merged.update(gated)
        core = _core_id(state)
        status = (gated.get("atomic_status") or {}).get(core, "insufficient")
        try:
            confidence = float(_confidence(merged, verdict if verdict != "Other" else "Other", core, status))
        except Exception:
            confidence = None
        reasons = " ".join(str(item) for item in (gated.get("gate_reasons") or []))
        capped = config.WEAK_REFUTE_CAVEAT in reasons and verdict in {"False", "Misleading"}
        if capped and confidence is not None:
            confidence = min(confidence, float(config.WEAK_REFUTE_CONFIDENCE_CAP))
        found[name] = {
            "verdict": verdict,
            "status": status,
            "caveat": capped,
            "confidence": confidence,
        }
    _apply(config, {}, None)
    return found


def _turn_on(winner: str | None, theta: float | None) -> list[tuple[str, str]]:
    """Write the winning flags into config. G0 and a miss leave every flag off.

    The in-memory scoring above already finished. This edit is the default
    for the next process. A re-run sets every flag from the winner again,
    so an old winner cannot stay on after a later study picks someone else.
    """
    parts = set() if not winner or winner == "G0" else set(winner.split("+"))
    wanted = {
        "GATE_PRIMARY_SOURCE": "G1" in parts,
        "GATE_WEAK_REFUTE": "G2R" in parts,
        "GATE_SHARED_SOURCE": "G3" in parts,
        "GATE_EVIDENCE_SCORE": "G4" in parts,
    }
    text = _CONFIG.read_text(encoding="utf-8")
    turned = []
    for name, value in wanted.items():
        updated, count = re.subn(
            rf"^{re.escape(name)} = (?:True|False)\s*$",
            f"{name} = {'True' if value else 'False'}",
            text,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise SystemExit(f"could not set {name}")
        text = updated
        if value:
            turned.append((name, "True"))
    theta_text = f"{float(theta):.1f}" if ("G4" in parts and theta is not None) else "1.0"
    updated, count = re.subn(
        r"^GATE_EVIDENCE_THETA = [0-9.]+\s*$",
        f"GATE_EVIDENCE_THETA = {theta_text}",
        text,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit("could not set GATE_EVIDENCE_THETA")
    text = updated
    if "G4" in parts:
        turned.append(("GATE_EVIDENCE_THETA", theta_text))
    _CONFIG.write_text(text, encoding="utf-8")
    return turned


def _render(
    diag,
    dev_rows,
    test_rows,
    dev_table,
    theta_notes,
    winner,
    test_report,
    case_lines,
    case_thread,
    case_ckpt,
    turned,
    reused,
) -> str:
    lines = [
        HEADING,
        "",
        "Selection rule, written before any variant was scored and before the test split was scored:",
        "",
        RULE,
        "",
        "A wrong decision is a decided False, Misleading, or True that is not the gold label. A False/Misleading swap counts as wrong. An abstention does not. True precision is undefined, and fails the rule, when the variant never predicts True.",
        "",
        "Every column, including G0, is the current gate (core match on, relaxed mode, single tier-1 denial on) plus the flags in that column. Pages dated after the claim are dropped, which is the blocked-eval leakage guard. NATIONAL_OFFICIAL_DOMAINS is now on every official query, including Society & other. That changes later searches only. The eval cache and these checkpoints were stored before that list was added, so this table does not show pages from the new hosts.",
        "",
        f"Dev { _label_counts(dev_rows) }. Test { _label_counts(test_rows) }. Recycled path (reuse_prior), same published verdict on every variant: {reused}.",
        "",
        "### Dev",
        "",
        _metric_table(dev_table),
        "",
    ]
    lines.extend(_theta_lines(theta_notes))
    if winner is None:
        lines.append("No variant met the selection rule. Every new flag stays off. G0 was scored once on test. No other variant was scored on test.")
    elif winner == "G0":
        lines.append(
            "The selection rule's best dev result was a tie with G0, so no new flag is turned on. "
            "G0 was scored once on test. No other variant was scored on test."
        )
    else:
        lines.append(
            f"The selection rule picked {winner} on dev. That variant and G0 were then scored once on test. No other variant was scored on test."
        )
    lines.extend(["", "### Test", ""])
    lines.append(_metric_table(test_report))
    lines.append("")
    lines.extend(_test_intervals(diag, test_report))
    lines.extend([
        "",
        "### Case study",
        "",
        f"Claim: \"{CASE_CLAIM}\" Thread {case_thread}, checkpoint {case_ckpt}. This is the latest checkpoint with that sentence. A newer thread states a different sentence and is not this case. The claim is not in the scored split.",
        "",
        "The stored table has three independent refuting groups and no supporting group: Wikipedia tier 3, testbook tier 4, sarkarilist tier 4. The run was Unverifiable because none of those groups is tier 1 or 2. The cached refute_kind is empty, so the normal mapping is Misleading rather than False.",
        "",
        "| variant | verdict | caveat | confidence |",
        "| --- | --- | --- | --- |",
    ])
    for name, _flags in VARIANTS:
        item = case_lines[name]
        caveat = "yes" if item["caveat"] else "no"
        confidence = "—" if item["confidence"] is None else f"{item['confidence']:.3f}"
        lines.append(f"| {name} | {item['verdict']} | {caveat} | {confidence} |")
    lines.extend(["", "### What each variant is", ""])
    for name, _flags in VARIANTS:
        lines.append(f"{name}: {_SENTENCE[name]}")
    lines.extend([
        "",
        "The test set is 25 claims, so the intervals are wide: one claim moves a coverage-adjusted rate by 0.04.",
        "",
    ])
    if turned:
        shown = ", ".join(f"{name} = {value}" for name, value in turned)
        lines.append(f"Default now on: {shown}.")
    else:
        lines.append("Every new gate flag stays off.")
    lines.append("")
    return "\n".join(lines)


def _label_counts(rows: list[dict]) -> str:
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["gold"]] += 1
    parts = [f"{label} {counts[label]}" for label in LABELS]
    return f"n={len(rows)} (" + ", ".join(parts) + ")"


def _metric_table(rows: list[dict]) -> str:
    header = (
        "| variant | theta | decided | 3c acc | 2c acc | cov 3c | cov 2c | "
        "True precision | True recall | abstention | wrong |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for row in rows:
        theta = "—" if row["theta"] is None else f"{row['theta']:.1f}"
        body.append(
            "| "
            + " | ".join([
                row["name"],
                theta,
                str(row["decided"]),
                _fmt(row["acc3"], row["correct3"], row["decided"]),
                _fmt(row["acc2"], row["correct2"], row["decided"]),
                _fmt(row["cov3"], row["correct3"], row["n"]),
                _fmt(row["cov2"], row["correct2"], row["n"]),
                _fmt(row["true_precision"], row["true_hit"], row["true_pred"]),
                _fmt(row["true_recall"], row["true_gold_hit"], row["true_gold"]),
                _fmt_rate(row["abstain"]),
                str(row["wrong"]),
            ])
            + " |"
        )
    return "\n".join([header, rule, *body])


def _theta_lines(notes: dict) -> list[str]:
    lines = []
    for name in ("G4", "G4+G3"):
        note = notes.get(name)
        if not note:
            continue
        cells = ", ".join(
            f"{row['theta']:.1f} → cov2 {_fmt_rate(row['cov2'])} wrong {row['wrong']} prec {_fmt_rate(row['true_precision'])}"
            for row in note["grid"]
        )
        mark = "met the dev constraints" if note["eligible"] else "no theta met the dev constraints; the shown theta is the best coverage-adjusted 2-class accuracy and the variant is not eligible on that ground"
        lines.append(f"{name} theta grid on dev (not used to score the test): {cells}. Chosen {note['theta']:.1f} ({mark}).")
    if lines:
        lines.append("")
    return lines


def _test_intervals(diag, report: list[dict]) -> list[str]:
    lines = []
    if len(report) == 1:
        lines.append("McNemar was not run. There was no second system on the test.")
    else:
        left = report[0]
        right = report[1]
        lines.append(
            "Exact McNemar, winner against G0. A claim scores 1 only when it was decided and right. An abstention scores 0."
        )
        lines.append(
            "3-class (G0 right and winner wrong vs G0 wrong and winner right): "
            + _mcnemar(diag, _bits(left, False), _bits(right, False))
        )
        lines.append(
            "2-class (G0 right and winner wrong vs G0 wrong and winner right): "
            + _mcnemar(diag, _bits(left, True), _bits(right, True))
        )
    lines.append("Wilson 95% intervals on the coverage-adjusted rates:")
    for row in report:
        lines.append(
            f"{row['name']} 3c {_wilson(diag, row['correct3'], row['n'])}; "
            f"2c {_wilson(diag, row['correct2'], row['n'])}."
        )
    return lines


def _bits(row: dict, binary: bool) -> list[int]:
    found = []
    for gold, label in zip(row["golds"], row["predicted"]):
        if label not in LABELS:
            found.append(0)
        elif binary:
            found.append(1 if _collapsed(label) == _collapsed(gold) else 0)
        else:
            found.append(1 if label == gold else 0)
    return found


def _mcnemar(diag, left: list[int], right: list[int]) -> str:
    b = sum(1 for a, c in zip(left, right) if a == 1 and c == 0)
    c = sum(1 for a, c in zip(left, right) if a == 0 and c == 1)
    p = diag.mcnemar_pvalue(b, c)
    if p is None:
        return "n/a"
    return f"p={p:.6f} (discordant {b} vs {c})"


def _wilson(diag, successes: int, n: int) -> str:
    interval = diag.wilson(successes, n)
    if interval is None:
        return "—"
    rate = successes / n if n else 0.0
    return f"{rate:.3f} ({successes}/{n}) [{interval[0]:.3f}, {interval[1]:.3f}]"


def _fmt(rate, hit, n) -> str:
    if rate is None or not n:
        return "—"
    return f"{rate:.3f} ({hit}/{n})"


def _fmt_rate(rate) -> str:
    if rate is None:
        return "—"
    return f"{rate:.3f}"


def _write_results(section: str) -> None:
    text = _DOC.read_text(encoding="utf-8")
    if RESULTS_MARK in text:
        text = text.split(RESULTS_MARK)[0].rstrip() + "\n"
    body = section
    if body.startswith(HEADING):
        # The heading and the rule are already above the mark.
        parts = body.split(RULE, 1)
        body = parts[1].lstrip("\n") if len(parts) == 2 else body
    addition = "\n" + RESULTS_MARK + "\n\n" + body
    if not text.endswith("\n"):
        text += "\n"
    _DOC.write_text(text + addition, encoding="utf-8")


if __name__ == "__main__":
    main()
