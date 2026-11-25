"""Boxes 8 and 10. The model writes. The evidence table decides.

One call drafts an explanation for each atomic claim, a true version, and a
correction a person could forward. It is told not to pick a verdict and not
to add facts that are not in the rows. This module then does three things
the model is not allowed to do:

- Pick the verdict from box 9's status and, when the core was refuted, from
  the draft's refute_kind. A fluent paragraph cannot outvote two credible
  groups, and it cannot turn one group into a verdict.
- Check every cited id. A sentence that points at a row we do not have, or
  at another atomic claim's row, is dropped. An empty explanation is rebuilt
  from the rows, with no second model call.
- Score confidence from the table. The model's own certainty is never read.
  The weights are provisional. The eval reports calibration (ECE).

An exact fresh prior (route reuse_prior) copies the archive row and does
not call the model. The fact-check was already accepted by box 6.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

import misinfo_agent.config as config
from misinfo_agent import gate as gate_box
from misinfo_agent.gate import PHOTO_OUT_OF_CONTEXT
from misinfo_agent.llm import LLMError, LLMQuotaError, call_structured

# The sentence a "no" on the photograph adds. The card and the tests
# share it so a reworded caveat cannot pass one and fail the other.
CAPTION_MISMATCH = "The photo's details may not match the caption"
CAPTION_MISMATCH_FLAG = "Image-caption mismatch"

# The archive stores capitalised labels, and the sample fixture stores
# them in lower case. Both have to land on the same five words the card
# already uses. Blank is the archive's missing label: there is nothing
# to reuse, so a person has to look.
_FROM_ARCHIVE = {
    "false": "False",
    "misleading": "Misleading",
    "true": "True",
    "other": "Other",
    "unverifiable": "Unverifiable",
    "unverified": "Unverifiable",
}

# date_source values fetch.py keeps when a publication day survived.
# "untrusted" is the opposite: the only dates on the page were today or
# in the future, so that row does not count as dated.
_TRUSTED_DATE = {"htmldate", "url", "trafilatura", "tavily", "page", "text", "linkedin_id"}

_CITE = re.compile(r"\[(E\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Shareable lines used only when the model returned nothing usable.
# They restate the label the rule already chose. They do not add a fact.
# The draft names the kind. The code names the verdict. A claim whose
# whole point is one wrong date, number, or name is False: forwarding
# it forwards the false fact. A real event with a side detail wrong,
# or a fact that used to be true, is Misleading.
_FALSE_KINDS = frozenset({"entirely_false", "single_fact_wrong"})
_MISLEADING_KINDS = frozenset(
    {"wrong_detail", "outdated", "partly_true", "out_of_context"}
)

_CORRECTION = {
    "False": "This claim does not match the sources we found. Please do not forward it.",
    "Misleading": "Part of this claim does not match the sources. Please do not forward it as written.",
    "True": "The main claim matches the sources we found.",
    "Unverifiable": "We could not find two independent credible sources, so this is not confirmed.",
}

# A stance or model step failed. The table was not finished. The card
# must not say that no source addressed the claim: that sentence means
# the web was read and had nothing. The em dash is the wording the card
# shows, so a test and the UI cannot drift onto a different sentence.
_CHECK_FAILED = "Checking failed partway through — please run this check again."


class _AtomicDraft(BaseModel):
    """One atomic claim, as the model drafted it. Not a verdict."""

    atomic_id: str
    explanation: str = ""
    refute_kind: Literal[
        "entirely_false",
        "single_fact_wrong",
        "wrong_detail",
        "outdated",
        "partly_true",
        "out_of_context",
    ] | None = None

    @field_validator("refute_kind", mode="before")
    @classmethod
    def _blank_kind(cls, value):
        # The model often writes "" or "null" for "not refuted". Those are
        # not a third kind, and failing the schema would spend a retry on
        # a word we can treat as absent.
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


class _Draft(BaseModel):
    """The one structured answer. The verdict is not a field, on purpose."""

    atomic: list[_AtomicDraft] = Field(default_factory=list)
    true_version: str = Field(
        default="",
        description="At most 50 words. What the evidence rows say is the case. Cite ids like [E1]. Empty if the rows do not say it.",
    )
    correction_message: str = Field(
        default="",
        description="At most 60 words. Plain language a person could forward. No evidence ids.",
    )


def assess(state: dict) -> dict:
    """Draft the text, then apply the verdict and the confidence in code.

    Returns only the keys this box owns. A quota error is not caught:
    the run stops, the same way extract and investigate stop.
    """
    notes: list[str] = []
    # Not a fact. Extract already said so. There is nothing to draft
    # and nothing for a confidence score to be about.
    if state.get("checkable") is False:
        return _pack(
            verdict="Other",
            confidence=0.0,
            explanation="",
            true_version="",
            correction="",
            caveats=[],
            citations=[],
            atomic_results=[],
            notes=["This submission is not checkable, so it is Other."],
            needs_human=False,
        )

    # Box 6 already accepted an exact fresh fact-check. Calling the model
    # here would let it rewrite a verdict we decided not to reopen.
    if state.get("route") == "reuse_prior":
        return _from_archive(state, notes)

    gate_name = state.get("gate")
    if gate_name is None:
        notes.append("Box 9 has not run, so there is no verdict.")
        return _pack(
            verdict=None,
            confidence=0.0,
            explanation="",
            true_version="",
            correction=_CORRECTION["Unverifiable"],
            what_we_found="",
            caveats=[],
            citations=[],
            atomic_results=[],
            notes=notes,
            needs_human=True,
        )
    # The gate asked for another search and has not finished the table.
    # A verdict written now is the thing that search was meant to avoid.
    if gate_name == "search_more":
        notes.append("The gate asked for another search, so there is no verdict yet.")
        return _pack(
            verdict=None,
            confidence=0.0,
            explanation="",
            true_version="",
            correction=_CORRECTION["Unverifiable"],
            what_we_found="",
            caveats=[],
            citations=[],
            atomic_results=[],
            notes=notes,
            needs_human=False,
        )

    # Before the draft. A failed check must not call the model, and it
    # must not fall through to the empty-table sentence.
    if gate_name == "error":
        return _failed_check(state, notes)

    evidence = [row for row in (state.get("evidence") or []) if isinstance(row, dict)]
    draft = _draft(state, evidence, notes)
    explanations, kinds = _per_atomic(state, evidence, draft, notes)
    # refute_kind describes a refutation. On a claim the gate did not
    # call refuted, that field is the model picking a verdict the table
    # does not support, so it is dropped before the rule reads it.
    kinds = _kinds_for_refuted(state, kinds)
    # The photo rule already refuted the core. The model's kind would
    # otherwise call that entirely_false and the verdict would be False.
    # Out of context is Misleading: the event may be real, the caption is not.
    if _photo_context(state):
        kinds[_core_id(state)] = "out_of_context"
    true_version = _guard_claim_text(
        draft.true_version if draft is not None else "",
        evidence,
        notes,
        "true_version",
        50,
    )
    correction = _correction(
        draft.correction_message if draft is not None else "",
        notes,
    )
    verdict, caveats, needs_human = _verdict(state, kinds, notes)
    core = _core_id(state)
    true_version, correction, what_we_found = _settle_text(
        verdict,
        true_version,
        correction,
        explanations.get(core, ""),
        notes,
    )
    confidence = _confidence(state, verdict, core, _core_status(state))
    # A weak refute is real evidence and not a tier 1-2 record. The cap
    # is the ceiling for that path even when the logistic would sit higher.
    # The photo path has its own sentence: a tier 1-2 provenance refute
    # does not wear it, and is not capped.
    if _weak_caveat(state) and verdict in {"False", "Misleading"}:
        # G2R-2 uses the same caveat sentence and a lower ceiling. The
        # flag, not the number of groups, picks the ceiling: a three-group
        # refute under G2R-2 is still the looser rule.
        cap = (
            config.WEAK_REFUTE_2_CONFIDENCE_CAP
            if config.GATE_WEAK_REFUTE_2
            else config.WEAK_REFUTE_CONFIDENCE_CAP
        )
        confidence = min(confidence, float(cap))
    if _photo_weak_caveat(state) and verdict in {"False", "Misleading"}:
        confidence = min(confidence, float(config.PHOTO_WEAK_REFUTE_CONFIDENCE_CAP))
    citations = _citations(explanations, true_version, what_we_found)
    # Set on every Unverifiable, including a thin table. A conflict still
    # has verdict None here; the non-blocking review turns that into
    # Unverifiable and writes its own reason. Blocking mode keeps None.
    reason = gate_box.unverifiable_reason(state) if verdict == "Unverifiable" else ""
    caveats, needs_human, extra_flags = _caption_effects(state, caveats, needs_human)
    packed = _pack(
        verdict=verdict,
        confidence=confidence,
        explanation=explanations.get(core, ""),
        true_version=true_version,
        correction=correction,
        what_we_found=what_we_found,
        caveats=caveats,
        citations=citations,
        atomic_results=[
            {
                "atomic_id": aid,
                "explanation": explanations.get(aid, ""),
                "refute_kind": kinds.get(aid),
                "citations": _ids_in(explanations.get(aid, "")),
            }
            for aid in _atomic_ids(state)
        ],
        notes=notes,
        needs_human=needs_human,
        unverifiable_reason=reason,
    )
    if extra_flags is not None:
        packed["red_flags"] = extra_flags
    return packed


def _failed_check(state: dict, notes: list[str]) -> dict:
    """The card when the gate is a system error.

    explanation and the shareable correction are the same sentence.
    what_we_found stays empty so the unverifiable template, and the
    "no source we read" line, cannot be what a person forwards.
    """
    notes.append("The gate is a system error, so the card says the check failed.")
    core = _core_id(state)
    return _pack(
        verdict=None,
        confidence=_confidence(state, None, core, _core_status(state)),
        explanation=_CHECK_FAILED,
        true_version="",
        correction=_CHECK_FAILED,
        what_we_found="",
        caveats=[],
        citations=[],
        atomic_results=[
            {
                "atomic_id": aid,
                "explanation": _CHECK_FAILED,
                "refute_kind": None,
                "citations": [],
            }
            for aid in _atomic_ids(state)
        ],
        notes=notes,
        needs_human=True,
        unverifiable_reason="error",
    )


def _verdict(state: dict, kinds: dict[str, str | None], notes: list[str]) -> tuple:
    """The label. Each branch is a rule. The model does not pick it.

    Returns verdict, caveats, needs_human. A True verdict with a non-empty
    caveat list is the "True with caveats" outcome: the label stays True
    so box 12 does not treat an unconfirmed detail as a lie.
    """
    gate_name = state.get("gate")
    # A conflict, or a stance call that crashed, is not a lack of sources.
    # Either one is a table we refused to trust. The card must not show
    # False or True over it. A person reads the gate reasons instead.
    if gate_name in {"conflicting", "error"}:
        return None, [], True
    core = _core_id(state)
    core_status = _core_status(state)
    kind = kinds.get(core)
    weak = _weak_caveat(state) + _photo_weak_caveat(state)
    # entirely_false: the event did not happen. single_fact_wrong: the
    # claim's main assertion is one date, number, or name, and it is
    # wrong. Both are False. A side detail, an outdated fact, a partial
    # truth, or a quote out of context is Misleading.
    if core_status == "refuted" and kind in _FALSE_KINDS:
        return "False", weak, False
    if core_status == "refuted" and kind in _MISLEADING_KINDS:
        return "Misleading", weak, False
    # Refuted, but the draft never said which kind. False is the harsher
    # label and it needs the explicit entirely_false mark. A missing mark
    # must not become that harsher label by default.
    if core_status == "refuted":
        notes.append(
            f"{core} is refuted but refute_kind was missing, so the verdict is Misleading rather than False."
        )
        return "Misleading", weak, False
    others = _other_status(state, core)
    # The core and every other checkable part are supported. An empty
    # list of other parts is included: a single supported claim is True.
    if core_status == "supported" and all(item == "supported" for item in others):
        return "True", [], False
    # The headline is supported, but another checkable part was refuted.
    # The forward as written is Misleading. That check runs before the
    # caveat branch so a refuted detail is not softened into a caveat.
    if core_status == "supported" and any(item == "refuted" for item in others):
        return "Misleading", [], False
    # The headline is supported and something else was not confirmed.
    # A side claim the gate called conflicting is named here too: it was
    # not refuted, so it does not flip the label to Misleading.
    if core_status == "supported":
        return "True", _caveat_names(state, core), False
    # The core never reached two independent credible groups. A stronger
    # side claim does not promote that into a verdict about the headline.
    if core_status == "insufficient":
        return "Unverifiable", [], False
    # The core itself is a motive or a question. Same label as a submission
    # extract already called not checkable.
    if core_status == "not_checkable":
        return "Other", [], False
    # A core status the rule does not name (including a conflicting core
    # that the overall gate failed to catch) is not given a label.
    notes.append(f"{core} has status {core_status}, so no verdict was assigned.")
    return None, [], True


def _confidence(state: dict, verdict: str | None, core: str, core_status: str) -> float:
    """A probability in [0, 1] from the table, never from the model.

    The weights live in config and are provisional. The eval reports
    calibration (ECE); that is what should move them.
    """
    if state.get("route") == "reuse_prior":
        return _reuse_confidence(state)
    # Other is a routing label. Zero says "this is not a probability
    # that a claim is false", because there was no claim to score.
    if verdict == "Other":
        return 0.0
    # The number on an abstention is confidence that the claim is
    # unverifiable: we searched and still do not have two credible
    # groups. It is not confidence that the claim is false.
    if verdict == "Unverifiable" or core_status == "insufficient":
        return _squash(_unverifiable_z(state, core))
    if verdict == "False" or (verdict == "Misleading" and core_status == "refuted"):
        return _squash(_side_z(state, core, "refutes"))
    if verdict == "True":
        return _squash(_side_z(state, core, "supports"))
    if verdict == "Misleading":
        # The label needs the core to be supported AND another claim to
        # be refuted. It is only as sure as the weaker of those two.
        scores = [_side_z(state, core, "supports")]
        statuses = _statuses(state)
        for claim in _atomics(state):
            aid = str(claim.get("id") or "")
            if aid == core or claim.get("checkable") is False:
                continue
            if statuses.get(aid) == "refuted":
                scores.append(_side_z(state, aid, "refutes"))
        return _squash(min(scores))
    # No verdict (a conflict, an error, a missing status). The number
    # still describes the table, so a person can see the mix. It is not
    # a probability that the claim is false.
    if core_status == "refuted":
        return _squash(_side_z(state, core, "refutes"))
    if core_status == "supported":
        return _squash(_side_z(state, core, "supports"))
    return _squash(_unverifiable_z(state, core))


def _side_z(state: dict, atomic_id: str, side: str) -> float:
    """Log-odds for one side of one atomic claim."""
    rows = _rows_for(state, atomic_id)
    groups = gate_box._groups(rows)
    features = _side_features(state, rows, groups, side)
    return _linear(features, config.CONFIDENCE_WEIGHTS, config.CONFIDENCE_BIAS)


def _side_features(state: dict, rows: list[dict], groups: dict, side: str) -> dict:
    """The features, each in [0, 1], for the side the verdict rests on.

    Groups and the conflict drop come from the gate's own counter. A
    dissenter the gate would name has to be the same group this score
    treats as disagreement, or the two boxes would tell different stories.
    """
    if side == "refutes":
        win_c = groups["refute_credible"]
        lose_c = groups["support_credible"]
        win_names = set(groups["refute"])
    else:
        win_c = groups["support_credible"]
        lose_c = groups["refute_credible"]
        win_names = set(groups["support"])
    # Four credible groups is the top of this feature. The gate's bar is
    # two. Past four, another outlet should not keep lifting the score
    # the way the second one does.
    group_score = min(len(win_c), 4) / 4
    if win_c:
        best = min(groups["tier"][name] for name in win_c)
        # Tier 1 -> 1, tier 2 -> 2/3, tier 4 -> 0. Best means the lowest
        # tier number among the credible groups on this side.
        tier_score = (4 - best) / 3
    else:
        tier_score = 0.0
    # A named dissenter is a credible group on the other side. It lowers
    # the share that agree. A tier-4 stray does not: the gate does not
    # name that as a disagreement that blocks the result.
    credible_total = len(win_c) + len(lose_c)
    agreement = (len(win_c) / credible_total) if credible_total else 0.0
    dated_rows = [
        row
        for row in rows
        if row.get("stance") == side and _group_name(row) in win_names
    ]
    dated = (
        sum(1 for row in dated_rows if _trusted_date(row)) / len(dated_rows)
        if dated_rows
        else 0.0
    )
    stats = _stats(state)
    # Capped so a long run of snippet-only quotes cannot flatten the
    # groups and the tier, which are the features the rule actually uses.
    unverified = min(int(stats.get("quote_unverified") or 0), 5) / 5
    topup = 1.0 if stats.get("topup_used") else 0.0
    cap = max(int(config.MAX_SEARCH_ROUNDS), 1)
    rounds = min(max(int(state.get("rounds") or 1), 0), cap) / cap
    return {
        "groups": group_score,
        "tier": tier_score,
        "agreement": agreement,
        "dated": dated,
        "unverified": unverified,
        "topup": topup,
        "rounds": rounds,
    }


def _unverifiable_z(state: dict, core: str) -> float:
    """Log-odds that the abstention is the right call.

    More rounds and a thinner table raise it: we looked, and we are
    still short of two credible groups. A top-up means that extra query
    already ran. Unverified quotes lower it, because a failed quote check
    is a page we could not finish reading.
    """
    rows = _rows_for(state, core)
    groups = gate_box._groups(rows)
    either = len(set(groups["support_credible"]) | set(groups["refute_credible"]))
    # 0 credible groups -> 1, 1 group -> 0.5, 2 groups -> 0. Two is the
    # gate's bar, so being that far along is no longer "thin".
    thin = 1.0 - min(either, 2) / 2
    stats = _stats(state)
    cap = max(int(config.MAX_SEARCH_ROUNDS), 1)
    features = {
        "rounds": min(max(int(state.get("rounds") or 1), 0), cap) / cap,
        "thin": thin,
        "topup": 1.0 if stats.get("topup_used") else 0.0,
        "unverified": min(int(stats.get("quote_unverified") or 0), 5) / 5,
    }
    return _linear(
        features,
        config.CONFIDENCE_UNVERIFIABLE_WEIGHTS,
        config.CONFIDENCE_UNVERIFIABLE_BIAS,
    )


def _reuse_confidence(state: dict) -> float:
    """Confidence from the match strength, not from a new evidence table.

    dense_cos is the number box 6 already used. A near-copy of a stored
    fact-check is more reusable than a match that only just cleared the
    exact threshold. There is no group count on this path.
    """
    best = _best(state)
    try:
        cosine = float(best.get("dense_cos") or 0.0)
    except (TypeError, ValueError):
        cosine = 0.0
    cosine = min(1.0, max(0.0, cosine))
    return _squash(config.CONFIDENCE_REUSE_BIAS + config.CONFIDENCE_REUSE_COSINE * cosine)


def _linear(features: dict, weights: dict, bias: float) -> float:
    score = float(bias)
    for name, weight in weights.items():
        score += float(weight) * float(features.get(name) or 0.0)
    return score


def _squash(score: float) -> float:
    """Logistic into [0, 1]. Four decimals is enough for the card and for ECE bins."""
    # The positive and negative forms avoid overflow on a large score.
    if score >= 0:
        value = 1.0 / (1.0 + math.exp(-score))
    else:
        exp = math.exp(score)
        value = exp / (1.0 + exp)
    return round(value, 4)


def _draft(state: dict, evidence: list[dict], notes: list[str]) -> _Draft | None:
    """The one model call. None means the template path for every field.

    No rows: the model would have nothing it is allowed to say, and the
    citation guard would drop whatever it invented. Skip the call.
    A quota error propagates. Any other model failure becomes the template.
    """
    if not evidence:
        notes.append("No evidence rows, so the assess model was not called.")
        return None
    prompt = _prompt(state, evidence)
    try:
        return call_structured(
            prompt,
            _Draft,
            purpose="assess",
            prompt_version="assess_v2",
        )
    except LLMQuotaError:
        raise
    except LLMError as exc:
        notes.append(
            f"The assess call failed ({exc.reason}), so the text is built from the evidence rows."
        )
        return None


def _prompt(state: dict, evidence: list[dict]) -> str:
    template = (config.PROMPTS_DIR / "assess_v2.txt").read_text(encoding="utf-8")
    # replace, not format: the instructions mention JSON braces.
    return (
        template.replace("__CLAIM__", str(state.get("claim") or ""))
        .replace("__CLAIM_DATE__", str(state.get("claim_date") or ""))
        .replace("__ATOMICS__", _atomic_block(state))
        .replace("__EVIDENCE__", _evidence_block(evidence))
        .replace("__PRIOR__", _prior_block(state))
    )


def _atomic_block(state: dict) -> str:
    statuses = _statuses(state)
    lines = []
    for claim in _atomics(state):
        aid = str(claim.get("id") or "A1")
        role = "core" if claim.get("core") else "part"
        check = "checkable" if claim.get("checkable") is not False else "not checkable"
        status = statuses.get(aid, "not judged")
        lines.append(f"{aid} ({role}, {check}, status {status}): {claim.get('text') or ''}")
    return "\n".join(lines)


def _evidence_block(evidence: list[dict]) -> str:
    lines = []
    for row in evidence:
        site = row.get("netloc") or row.get("site") or "-"
        when = row.get("published_date") or row.get("date") or "-"
        quote = " ".join(str(row.get("quote") or "").split())
        lines.append(
            f"{row.get('id')}  atomic={_atomic_of(row)}  site={site}  "
            f"tier={row.get('tier')}  stance={row.get('stance')}  date={when}"
        )
        if quote:
            lines.append(quote)
    return "\n".join(lines) if lines else "(none)"


def _prior_block(state: dict) -> str:
    """Shown for a similar or exact hit. The prompt is told not to cite it.

    reuse_prior never reaches this function. A similar hit is context so
    the draft does not contradict a known fact-check, and it is still not
    the verdict: the evidence rule above this text is.
    """
    kind = state.get("match_type")
    best = _best(state)
    if kind not in {"exact", "similar"} or not best:
        return "none"
    return (
        f"match_type: {kind}\n"
        f"archive verdict: {best.get('verdict_norm') or best.get('verdict') or ''}\n"
        f"archive true_version: {best.get('true_version') or ''}\n"
        f"archive claim: {best.get('primary_claim') or ''}"
    )


def _per_atomic(
    state: dict,
    evidence: list[dict],
    draft: _Draft | None,
    notes: list[str],
) -> tuple[dict[str, str], dict[str, str | None]]:
    by_id = {}
    if draft is not None:
        for item in draft.atomic:
            by_id[str(item.atomic_id)] = item
    explanations: dict[str, str] = {}
    kinds: dict[str, str | None] = {}
    for claim in _atomics(state):
        aid = str(claim.get("id") or "A1")
        item = by_id.get(aid)
        raw = item.explanation if item is not None else ""
        kind = item.refute_kind if item is not None else None
        allowed = _ids_for(evidence, aid)
        kept = _guard_sentences(raw, allowed, notes, aid)
        if not kept:
            # Empty after the guard, or never written. The rows are the
            # only source the template is allowed to use.
            if raw.strip():
                notes.append(f"{aid}: every sentence was dropped, so the text is built from the evidence rows.")
            elif draft is not None:
                notes.append(f"{aid}: the model returned no explanation, so the text is built from the evidence rows.")
            kept = _template(evidence, aid)
        explanations[aid] = _trim(kept, 60, notes, aid)
        kinds[aid] = kind
    return explanations, kinds


def _guard_sentences(text: str, allowed: set[str], notes: list[str], label: str) -> str:
    """Drop a sentence whose bracketed id is missing or belongs elsewhere.

    A sentence with no citation is kept. The rule drops invalid ids, and
    a sentence that cites nothing has not cited a bad one. A sentence that
    cites one good id and one bad id is dropped: the bad id means the
    sentence is not grounded in this claim's rows.
    """
    kept = []
    for sentence in _sentences(text):
        cited = _CITE.findall(sentence)
        bad = [item for item in cited if item not in allowed]
        if bad:
            notes.append(
                f"{label}: dropped a sentence that cited {', '.join(bad)}, "
                "which is not evidence for this claim."
            )
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _guard_claim_text(
    text: str,
    evidence: list[dict],
    notes: list[str],
    label: str,
    limit: int,
) -> str:
    """true_version may cite any row in the table, and no id outside it."""
    allowed = {str(row.get("id")) for row in evidence if row.get("id")}
    kept = _guard_sentences(text, allowed, notes, label)
    return _trim(kept, limit, notes, label)


def _kinds_for_refuted(state: dict, kinds: dict[str, str | None]) -> dict[str, str | None]:
    """Keep a kind only on a claim the gate called refuted."""
    statuses = _statuses(state)
    return {
        aid: (kind if statuses.get(aid) == "refuted" else None)
        for aid, kind in kinds.items()
    }


def _photo_context(state: dict) -> bool:
    """True when the photo rule, not the model, refuted the core."""
    if _core_status(state) != "refuted":
        return False
    return any(PHOTO_OUT_OF_CONTEXT in str(line) for line in (state.get("gate_reasons") or []))


def _caption_effects(state: dict, caveats: list, needs_human: bool) -> tuple[list, bool, list | None]:
    """A caption mismatch is a caveat and a review, never a verdict.

    The verdict argument is whatever the evidence rule already chose.
    This only adds the sentence, the red flag, and pending review.
    """
    raw = state.get("image_consistency")
    if not isinstance(raw, dict) or raw.get("consistent") != "no":
        return caveats, needs_human, None
    updated = list(caveats)
    if CAPTION_MISMATCH not in updated:
        updated.append(CAPTION_MISMATCH)
    flags = [str(item) for item in (state.get("red_flags") or [])]
    if CAPTION_MISMATCH_FLAG not in flags:
        flags.append(CAPTION_MISMATCH_FLAG)
    return updated, True, flags


def _settle_text(
    verdict: str | None,
    true_version: str,
    correction: str,
    explanation: str,
    notes: list[str],
) -> tuple[str, str, str]:
    """Return true_version, correction_message, what_we_found.

    A written true version and a written correction exist only when the
    label is True, False, or Misleading. Unverifiable, and a verdict left
    empty for a person, both use one template. The rows' summary moves to
    what_we_found so it cannot be forwarded as the correction.
    """
    if verdict in {"True", "False", "Misleading"}:
        if not correction:
            fallback = _CORRECTION.get(verdict, "")
            if fallback:
                correction = fallback
                notes.append("correction_message was empty, so a template line was used.")
        return true_version, _trim(correction, 60, notes, "correction_message"), ""
    if verdict in {"Unverifiable", None}:
        template = _CORRECTION["Unverifiable"]
        # Prefer the draft that still cites ids. A shareable correction
        # has had its ids stripped, so it is the summary only when the
        # true version was empty.
        summary = true_version
        if not summary and correction and correction != template:
            summary = correction
        if not summary:
            summary = explanation
        if summary == template:
            summary = ""
        if true_version or (correction and correction != template):
            notes.append(
                "No decided verdict, so the evidence summary is what_we_found "
                "and the correction is the unverifiable line."
            )
        return "", template, summary
    return "", "", ""


def _cite_archive(text: str) -> str:
    """Mark the archive fact-check as C1. An empty line stays empty."""
    text = " ".join(text.split())
    if not text or "[C1]" in text:
        return text
    return f"{text} [C1]"


def _correction(text: str, notes: list[str]) -> str:
    """The shareable line. Evidence ids are stripped, not left for a reader."""
    if _CITE.search(text):
        notes.append("correction_message cited evidence ids; they were removed.")
        text = _CITE.sub("", text)
    return " ".join(text.split())


def _template(evidence: list[dict], atomic_id: str) -> str:
    """A sentence per row, from fields the row already has. No model, no new fact."""
    rows = [row for row in evidence if _atomic_of(row) == atomic_id and row.get("id")]
    if not rows:
        return "No source we read addressed this part of the claim."
    bits = []
    for row in rows[:4]:
        site = row.get("netloc") or row.get("site") or "A source"
        stance = row.get("stance") or "mentions"
        quote = _clip(str(row.get("quote") or row.get("display_quote") or ""), 25)
        cite = f"[{row.get('id')}]"
        if quote:
            bits.append(f"{site} {stance} it: {quote} {cite}.")
        else:
            bits.append(f"{site} {stance} it {cite}.")
    return " ".join(bits)


def _from_archive(state: dict, notes: list[str]) -> dict:
    """Copy the stored fact-check. No model call, no citation guard.

    The archive text is not in the [E#] scheme. Running the guard would
    delete a fact-checker's sentence for a bracket we never assigned.
    true_version is the field this dataset uses as the correction. The
    card's explanation is that same text, because the row has no separate
    explanation column. The shareable line is the same text, trimmed.
    """
    best = _best(state)
    notes.append("Reused the archive row. No assess model call.")
    label = _archive_verdict(best)
    needs_human = False
    if label is None:
        notes.append("The archive row has no verdict, so a person has to review it.")
        needs_human = True
    # The archive row is the source. C1 is that fact-check, not an
    # evidence id from this run, so the citation guard does not apply.
    # The bracket is added after the word trim so it is not the token
    # that gets cut. The shareable line stays free of ids.
    true_version = " ".join(str(best.get("true_version") or "").split())
    explanation = true_version or " ".join(str(best.get("primary_claim") or "").split())
    correction = _trim(_correction(true_version, notes), 60, notes, "correction_message")
    explanation = _cite_archive(_trim(explanation, 59, notes, "explanation"))
    true_version = _cite_archive(_trim(true_version, 49, notes, "true_version"))
    core = _core_id(state)
    kind = best.get("refute_kind")
    if kind not in _FALSE_KINDS and kind not in _MISLEADING_KINDS:
        kind = None
    true_version, correction, what_we_found = _settle_text(
        label,
        true_version,
        correction,
        explanation,
        notes,
    )
    citations = ["C1"] if label in {"True", "False", "Misleading"} or what_we_found or explanation else []
    return _pack(
        verdict=label,
        confidence=_reuse_confidence(state),
        explanation=explanation,
        true_version=true_version,
        correction=correction,
        what_we_found=what_we_found,
        caveats=[],
        citations=citations,
        atomic_results=[
            {
                "atomic_id": core,
                "explanation": explanation,
                "refute_kind": kind,
                "citations": ["C1"] if "[C1]" in explanation else [],
            }
        ],
        notes=notes,
        needs_human=needs_human,
        unverifiable_reason=gate_box.unverifiable_reason(state) if label == "Unverifiable" else "",
    )


def _archive_verdict(best: dict) -> str | None:
    raw = best.get("verdict_norm")
    if raw is None or str(raw).strip() == "":
        raw = best.get("verdict")
    key = str(raw or "").strip().casefold()
    if key in {"", "blank"}:
        return None
    return _FROM_ARCHIVE.get(key)


def _pack(
    *,
    verdict,
    confidence: float,
    explanation: str,
    true_version: str,
    correction: str,
    caveats: list,
    citations: list,
    atomic_results: list,
    notes: list,
    needs_human: bool,
    what_we_found: str = "",
    unverifiable_reason: str = "",
) -> dict:
    # Defaults for a run that never reaches review. Review overwrites
    # these when the decision is alert or human and the pause does not
    # stop the graph. needs_human stays the back-compat flag: pending
    # means a person has not answered yet.
    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "true_version": true_version,
        "correction_message": correction,
        "what_we_found": what_we_found,
        "caveats": caveats,
        "citations": citations,
        "atomic_results": atomic_results,
        "assess_notes": notes,
        "needs_human": needs_human,
        "provisional": False,
        "review_status": "pending" if needs_human else "none",
        "review_reason": "",
        "alert_pending_approval": False,
        "unverifiable_reason": unverifiable_reason,
        "model_lean": None,
    }


def _atomics(state: dict) -> list[dict]:
    """Extract's list, or one core claim when the list was never filled."""
    found = [
        item
        for item in (state.get("atomic_claims") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    if found:
        return found
    return [
        {
            "id": "A1",
            "text": state.get("claim") or "",
            "core": True,
            "checkable": True,
        }
    ]


def _atomic_ids(state: dict) -> list[str]:
    return [str(item.get("id") or "A1") for item in _atomics(state)]


def _core_id(state: dict) -> str:
    for item in _atomics(state):
        if item.get("core"):
            return str(item.get("id") or "A1")
    return _atomic_ids(state)[0]


def _statuses(state: dict) -> dict:
    raw = state.get("atomic_status")
    return raw if isinstance(raw, dict) else {}


def _core_status(state: dict) -> str:
    return str(_statuses(state).get(_core_id(state)) or "insufficient")


def _other_status(state: dict, core: str) -> list[str]:
    statuses = _statuses(state)
    found = []
    for item in _atomics(state):
        aid = str(item.get("id") or "")
        if aid == core or item.get("checkable") is False:
            continue
        found.append(str(statuses.get(aid) or "insufficient"))
    return found


def _caveat_names(state: dict, core: str) -> list[str]:
    """The unconfirmed checkable parts, in the claim's own words."""
    statuses = _statuses(state)
    names = []
    for item in _atomics(state):
        aid = str(item.get("id") or "")
        if aid == core or item.get("checkable") is False:
            continue
        if statuses.get(aid) == "supported":
            continue
        text = " ".join(str(item.get("text") or aid).split())
        names.append(text or aid)
    return names


def _best(state: dict) -> dict:
    direct = state.get("best_match")
    if isinstance(direct, dict) and direct:
        return direct
    hits = [item for item in (state.get("matches") or []) if isinstance(item, dict)]
    if not hits:
        return {}
    return max(hits, key=lambda item: float(item.get("dense_cos") or 0.0))


def _photo_weak_caveat(state: dict) -> list[str]:
    """The lower-tier photo sentence, only when that path decided.

    A tier 1-2 provenance refute does not insert the sentence, so it is
    not capped and the card does not claim that no such page was found.
    """
    if not config.GATE_PHOTO_WEAK_REFUTE:
        return []
    needle = config.PHOTO_WEAK_REFUTE_CAVEAT
    if any(needle in str(item) for item in (state.get("gate_reasons") or [])):
        return [needle]
    return []


def _weak_caveat(state: dict) -> list[str]:
    """The lower-tier sentence, only when that path actually decided.

    The flag alone is not enough. A normal two-newsroom refute must not
    wear this caveat or this cap.
    """
    if not (config.GATE_WEAK_REFUTE or config.GATE_WEAK_REFUTE_2):
        return []
    needle = config.WEAK_REFUTE_CAVEAT
    if any(needle in str(item) for item in (state.get("gate_reasons") or [])):
        return [needle]
    return []


def _rows_for(state: dict, atomic_id: str) -> list[dict]:
    """The same row filter the gate used, including the leakage switch.

    Confidence has to count the rows the verdict was built from. A page
    the gate ignored as after-claim must not raise the score. Shared-source
    merging is applied here too, on copies, so the score sees the groups
    the gate counted.
    """
    rows = gate_box._rows(
        list(state.get("evidence") or []),
        atomic_id,
        drop_after=bool(config.EVAL_DROP_AFTER_CLAIM),
    )
    return gate_box.counted_rows(rows)


def _stats(state: dict) -> dict:
    raw = state.get("investigate_stats")
    return raw if isinstance(raw, dict) else {}


def _atomic_of(row: dict) -> str:
    return str(row.get("atomic_id") or "A1")


def _group_name(row: dict) -> str:
    # Same fallback the gate uses, so a row with neither key is its own
    # group here too. An empty string must not glue those rows together.
    name = str(row.get("group_id") or row.get("group") or "").strip()
    if name:
        return name
    return f"row:{row.get('id')}"


def _trusted_date(row: dict) -> bool:
    source = row.get("date_source")
    if source == "untrusted":
        return False
    if source in _TRUSTED_DATE:
        return True
    # Rows built before date_source existed store the day in `date`.
    # An untrusted source already returned above, so a bare ISO day is
    # a date a person recorded on purpose.
    raw = row.get("published_date") or row.get("date")
    return isinstance(raw, str) and len(raw) >= 10 and raw[4] == "-" and raw[7] == "-"


def _ids_for(evidence: list[dict], atomic_id: str) -> set[str]:
    return {
        str(row.get("id"))
        for row in evidence
        if row.get("id") and _atomic_of(row) == atomic_id
    }


def _ids_in(text: str) -> list[str]:
    seen = []
    for item in _CITE.findall(text):
        if item not in seen:
            seen.append(item)
    return seen


def _citations(explanations: dict[str, str], true_version: str, what_we_found: str = "") -> list[str]:
    seen: list[str] = []
    for text in list(explanations.values()) + [true_version, what_we_found]:
        for item in _ids_in(text):
            if item not in seen:
                seen.append(item)
    return seen


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE.split(text.strip())
    return [part for part in parts if part.strip()]


def _trim(text: str, limit: int, notes: list[str], label: str) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    notes.append(f"{label}: cut from {len(words)} words to {limit}.")
    return " ".join(words[:limit])


def _clip(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit])
