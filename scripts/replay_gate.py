"""Re-judge the 50 blocked claims with the new gate rules. No model, no search.

The evidence is the checkpoint copy. The original column is the published
blocked eval, not a re-run. Fix 2 is TRUE_NEEDS_CORE_MATCH, run once in
CORE_MATCH_MODE strict and once in relaxed. Fix 3 is TIER1_SINGLE_REFUTE.
The assisted column is relaxed Fix 2 plus Fix 3, then the stored baseline
jsonl when the rules still abstain. It does not call Gemini.

A core that becomes refuted and has no cached refute_kind is scored False
and flagged refute_kind_defaulted. The live assess box still treats a
missing kind as Misleading; this script is the post-hoc mapping the
diagnosis asked for, and it says so in the section it appends.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ("False", "Misleading", "True")
_COUNTER_FILES = (
    ROOT / "data" / "cache" / "llm_daily_count.json",
    ROOT / "data" / "cache" / "search_monthly_count.json",
)
_SNAPSHOT: dict | None = None


def _arm() -> None:
    """Pop both keys and freeze the counters before any client can be built."""
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


def cached_kind(state: dict, core: str) -> str | None:
    for item in state.get("atomic_results") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("atomic_id") or "") != core:
            continue
        kind = item.get("refute_kind")
        if kind in {"entirely_false", "wrong_detail"}:
            return kind
    return None


def code_verdict(state: dict, gated: dict) -> tuple[str, bool]:
    """Map a re-gated table the way assess does, with one post-hoc exception.

    A core that was not refuted in the stored run, and is refuted now,
    and has no cached refute_kind, becomes False. The live box would say
    Misleading. The flag records that substitution.
    """
    if gated.get("gate") in {"conflicting", "error", "skipped"}:
        return "Unverifiable", False
    core = "A1"
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("core") is True:
            core = str(item.get("id") or "A1")
            break
    status = (gated.get("atomic_status") or {}).get(core, "insufficient")
    kind = cached_kind(state, core)
    previous = (state.get("atomic_status") or {}).get(core)
    if status == "refuted":
        if kind == "entirely_false":
            return "False", False
        if kind == "wrong_detail":
            return "Misleading", False
        if previous != "refuted":
            return "False", True
        return "Misleading", False
    others = [
        value
        for key, value in (gated.get("atomic_status") or {}).items()
        if key != core and value != "not_checkable"
    ]
    if status == "supported" and any(value == "refuted" for value in others):
        return "Misleading", False
    if status == "supported":
        return "True", False
    if status == "not_checkable":
        return "Other", False
    return "Unverifiable", False


def _graded(label: str | None, decided: bool) -> str | None:
    if decided and label in LABELS:
        return label
    return None


def _rates(golds: list[str], predicted: list[str | None]) -> dict:
    decided = [label for label in predicted if label in LABELS]
    correct = [
        gold == label
        for gold, label in zip(golds, predicted)
        if label in LABELS
    ]
    correct_n = sum(1 for item in correct if item)
    binary = [
        (gold == "True") == (label == "True")
        for gold, label in zip(golds, predicted)
        if label in LABELS
    ]
    n = len(golds)
    true_pred = [gold for gold, label in zip(golds, predicted) if label == "True"]
    true_gold = [label for gold, label in zip(golds, predicted) if gold == "True"]
    return {
        "decided": len(decided),
        "correct3": correct_n,
        "acc3": (correct_n / len(decided)) if decided else None,
        "correct2": sum(1 for item in binary if item),
        "acc2": (sum(1 for item in binary if item) / len(decided)) if decided else None,
        "cov3": correct_n / n if n else None,
        "cov2": sum(1 for item in binary if item) / n if n else None,
        "abstain": (n - len(decided)) / n if n else None,
        "true_precision": (sum(1 for gold in true_pred if gold == "True") / len(true_pred)) if true_pred else None,
        "true_recall": (sum(1 for label in true_gold if label == "True") / sum(1 for gold in golds if gold == "True")) if any(gold == "True" for gold in golds) else None,
        "true_hit": sum(1 for gold in true_pred if gold == "True"),
        "true_pred": len(true_pred),
        "true_gold_hit": sum(1 for label in true_gold if label == "True"),
        "true_gold": sum(1 for gold in golds if gold == "True"),
    }


def _fmt(rate, hit, n) -> str:
    if rate is None or n == 0:
        return "—"
    return f"{rate:.3f} ({hit}/{n})"


def _fmt_rate(rate) -> str:
    if rate is None:
        return "—"
    return f"{rate:.3f}"


def _bits(golds, predicted) -> list[int]:
    return [
        1 if label in LABELS and label == gold else 0
        for gold, label in zip(golds, predicted)
    ]


def _mcnemar(left: list[int], right: list[int]) -> str:
    from scripts_diagnose import mcnemar_pvalue  # set in main

    b = sum(1 for a, c in zip(left, right) if a == 1 and c == 0)
    c = sum(1 for a, c in zip(left, right) if a == 0 and c == 1)
    p = mcnemar_pvalue(b, c)
    if p is None:
        return "n/a"
    return f"p={p:.6f} (discordant {b} vs {c})"


def _plain_notes(rows: list[dict]) -> list[str]:
    """The sentences under the table. Ids only, no claim text."""
    correct = [row for row in rows if row["gold"] == "True" and row["original"] == "True"]
    strict_lost = [row["id"] for row in correct if row["fix2_strict"] != "True"]
    lines = []
    if strict_lost == ["0b6220e4a72d"]:
        lines.append(
            "Strict Fix 2 lost 1 correct True (0b6220e4a72d): quotes said 'Mamdani won', not the full name."
        )
    else:
        shown = ", ".join(strict_lost) if strict_lost else "none"
        lines.append(f"Strict Fix 2 lost {len(strict_lost)} correct True(s): {shown}.")
    false_trues = [
        row for row in rows
        if row["original"] == "True" and row["gold"] in {"False", "Misleading"}
    ]
    lines.append(f"False Trues in the published run: {len(false_trues)}.")
    for label, key in (
        ("Strict Fix 2", "fix2_strict"),
        ("Relaxed Fix 2", "fix2_relaxed"),
        ("Fix 3", "fix3"),
        ("Relaxed Fix 2 + Fix 3", "both"),
    ):
        removed = [row["id"] for row in false_trues if row[key] != "True"]
        listed = ", ".join(removed) if removed else "none"
        lines.append(f"{label} removes {len(removed)} of {len(false_trues)} false Trues: {listed}.")
    return lines


def _section(rows: list[dict], golds: list[str], baseline_pred: list[str | None], reused: int) -> str:
    from scripts_diagnose import mcnemar_pvalue  # noqa: F401  (imported for the closure)

    columns = [
        ("original", "original"),
        ("+Fix2 strict", "fix2_strict"),
        ("+Fix2 relaxed", "fix2_relaxed"),
        ("+Fix3", "fix3"),
        ("+Fix2 relaxed+Fix3", "both"),
        ("assisted (Fix2 relaxed+Fix3 + model lean)", "assisted"),
    ]
    baseline_bits = _bits(golds, baseline_pred)
    lines = [
        "",
        "## POST-HOC: rules tuned after seeing these 50 claims. Not a held-out result.",
        "",
        "Fix 2 drops a core support whose quote does not carry the core's key terms, and requires two tier 1-2 supporting groups. Strict uses the full name and the exact verb. Relaxed allows a surname, a stem, or a synonym, and reads the sentence that contains the quote. Numbers and dates stay exact in both. Fix 3 refutes the core on one tier-1 denial when no tier 1-2 group supports it. The assisted column keeps a decided label and otherwise uses the stored baseline jsonl. No new model call and no search.",
        "",
        f"Recycled path (reuse_prior): {reused} claim(s). The gate stays skipped and the verdict is the published one.",
        "",
        "| | " + " | ".join(name for name, _key in columns) + " |",
        "| --- | " + " | ".join("---" for _ in columns) + " |",
    ]
    packed = {key: _rates(golds, [row[key] for row in rows]) for _name, key in columns}
    specs = [
        ("Decided", lambda m: str(m["decided"])),
        ("3c acc", lambda m: _fmt(m["acc3"], m["correct3"], m["decided"])),
        ("2c acc", lambda m: _fmt(m["acc2"], m["correct2"], m["decided"])),
        ("Coverage-adjusted 3c", lambda m: _fmt(m["cov3"], m["correct3"], len(rows))),
        ("Coverage-adjusted 2c", lambda m: _fmt(m["cov2"], m["correct2"], len(rows))),
        ("Abstention", lambda m: _fmt_rate(m["abstain"])),
        ("True precision", lambda m: _fmt(m["true_precision"], m["true_hit"], m["true_pred"])),
        ("True recall", lambda m: _fmt(m["true_recall"], m["true_gold_hit"], m["true_gold"])),
    ]
    for title, cell in specs:
        lines.append("| " + title + " | " + " | ".join(cell(packed[key]) for _name, key in columns) + " |")
    mcnemar_cells = []
    for _name, key in columns:
        mcnemar_cells.append(_mcnemar(_bits(golds, [row[key] for row in rows]), baseline_bits))
    lines.append("| McNemar vs baseline (3c, exact) | " + " | ".join(mcnemar_cells) + " |")
    lines.append("")
    lines.extend(_plain_notes(rows))
    defaulted = [row["id"] for row in rows if row["defaulted"]]
    lines.append("")
    lines.append(
        f"Newly refuted with no cached refute_kind, scored False (refute_kind_defaulted): {len(defaulted)}."
    )
    if defaulted:
        lines.append(", ".join(defaulted))
    changed = [row for row in rows if row["original_shown"] != row["both_shown"]]
    lines.append("")
    lines.append("Changed claims (+Fix2 relaxed+Fix3 against the published verdict):")
    lines.append("")
    lines.append("| id | old -> new | right/wrong |")
    lines.append("| --- | --- | --- |")
    if not changed:
        lines.append("| — | — | — |")
    for row in changed:
        lines.append(f"| {row['id']} | {row['original_shown']} -> {row['both_shown']} | {row['mark']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    _arm()
    try:
        _run()
    finally:
        if _SNAPSHOT is not None:
            _counters_unchanged()


def _run() -> None:
    import misinfo_agent.config as config
    from misinfo_agent.gate import gate

    # Imported after the guard. diagnose_eval does not pop keys on import.
    sys.path.insert(0, str(ROOT / "scripts"))
    import diagnose_eval as diag

    # _mcnemar looks this up. Binding it here keeps the helper free of the
    # import at collection time.
    global_ns = globals()
    import types
    stub = types.ModuleType("scripts_diagnose")
    stub.mcnemar_pvalue = diag.mcnemar_pvalue
    sys.modules["scripts_diagnose"] = stub

    src = ROOT / "data" / "eval" / "runs" / "checkpoints.db"
    copy = Path("/tmp/ckpt_replay.db")
    if not src.is_file():
        raise SystemExit(f"checkpoint source missing: {src}")
    shutil.copy(src, copy)

    testset = diag.load_testset(ROOT / "data" / "eval" / "testset.csv")
    blocked = {row["article_id"]: row for row in diag.load_jsonl(ROOT / "data" / "eval" / "results_system_blocked.jsonl")}
    baseline = {row["article_id"]: row for row in diag.load_jsonl(ROOT / "data" / "eval" / "results_baseline.jsonl")}
    grouped = diag.load_blocked_threads(copy)

    rows = []
    reused = 0
    lost = []
    for item in testset:
        article_id = item["article_id"]
        gold = item["label"]
        published = blocked[article_id]
        chosen, _how = diag.choose_thread(grouped.get(article_id, []), published)
        state = chosen["state"] if chosen else {}
        original = published.get("verdict")
        decided = diag.is_decided(published)
        shown_old = original if original else "Unverifiable"
        record = {
            "id": article_id,
            "gold": gold,
            "original": _graded(original, decided),
            "original_shown": shown_old if decided else f"abstain:{shown_old}",
            "defaulted": False,
            "reuse": state.get("route") == "reuse_prior",
        }
        if record["reuse"]:
            reused += 1
            skipped = gate(state)
            if skipped.get("gate") != "skipped":
                raise SystemExit(f"reuse_prior was re-judged: {article_id}")
            for key in ("fix2_strict", "fix2_relaxed", "fix3", "both"):
                record[key] = record["original"] if record["original"] else "Unverifiable"
        else:
            record.update(_rerun(state, config, gate))
        record["assisted"] = _assisted(record["both"], baseline.get(article_id))
        record["both_shown"] = record["both"] if record["both"] in LABELS else f"abstain:{record['both']}"
        record["mark"] = _mark(gold, record["both"])
        # The assert is the relaxed matcher. Strict is allowed to drop the
        # one True whose quotes used the surname and not the full name.
        if gold == "True" and record["original"] == "True":
            if record["fix2_relaxed"] != "True" or record["both"] != "True":
                lost.append((article_id, state))
        rows.append(record)

    if lost:
        print("LOST CORRECT TRUES: " + ", ".join(item[0] for item in lost))
        _print_lost_quotes(lost)
        raise SystemExit(2)

    golds = [row["gold"] for row in rows]
    baseline_pred = [
        _graded(baseline[row["id"]].get("verdict"), diag.is_decided(baseline[row["id"]]))
        for row in rows
    ]
    # Score abstentions as not a label. _rates treats anything outside LABELS
    # as abstained, so Unverifiable and Other both abstain. A graded original
    # is already None when the published row was not decided.
    for row in rows:
        if row["original"] is None:
            row["original"] = None
    section = _section(rows, golds, baseline_pred, reused)
    path = ROOT / "docs" / "eval_diagnosis.md"
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" + section, encoding="utf-8")
    kept = sum(1 for row in rows if row["gold"] == "True" and row["original"] == "True" and row["both"] == "True")
    print(f"CORRECT TRUES KEPT: {kept}")
    print(section)


def _assisted(both: str, baseline_row: dict | None) -> str:
    if both in LABELS:
        return both
    if not baseline_row or baseline_row.get("error"):
        return "Unverifiable"
    verdict = baseline_row.get("verdict")
    if verdict in LABELS and not baseline_row.get("needs_human"):
        return verdict
    return "Unverifiable"


def _mark(gold: str, label: str) -> str:
    if label not in LABELS:
        return "abstain"
    return "right" if label == gold else "wrong"


def _print_lost_quotes(lost: list[tuple[str, dict]]) -> None:
    """The quotes the relaxed matcher refused. Printed, not written to the doc."""
    for article_id, state in lost:
        core = str(state.get("claim") or "")
        for item in state.get("atomic_claims") or []:
            if isinstance(item, dict) and item.get("core") is True:
                core = str(item.get("text") or core)
                break
        print(f"QUOTES {article_id}")
        print(f"CORE {core}")
        for row in state.get("evidence") or []:
            if not isinstance(row, dict) or row.get("stance") != "supports":
                continue
            quote = " ".join(str(row.get("quote") or "").split())
            print(f"  {row.get('id')} tier={row.get('tier')} {quote[:240]}")


def _rerun(state: dict, config, gate) -> dict:
    out = {}
    defaulted = False
    plans = {
        "fix2_strict": (True, False, "strict"),
        "fix2_relaxed": (True, False, "relaxed"),
        "fix3": (False, True, "relaxed"),
        "both": (True, True, "relaxed"),
    }
    for key, (core_match, single, mode) in plans.items():
        config.TRUE_NEEDS_CORE_MATCH = core_match
        config.TIER1_SINGLE_REFUTE = single
        config.CORE_MATCH_MODE = mode
        gated = gate(state)
        label, flagged = code_verdict(state, gated)
        out[key] = label
        if key == "both" and flagged:
            defaulted = True
    out["defaulted"] = defaulted
    return out


if __name__ == "__main__":
    main()
