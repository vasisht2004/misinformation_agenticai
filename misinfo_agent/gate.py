"""Box 9: is the evidence table enough to assess each atomic claim?

Rules only. No model call and no fetch. Box 7 has already assigned the
tier, the independent group, and the stance. This box counts those
groups. It does not re-decide what a page said.

A claim is enough when two independent groups take the same side and at
least one of them is tier 1 or 2. One credible voice on the other side
does not block that, unless it is a large minority or a tier 1 source.
TRUE_NEEDS_CORE_MATCH and TIER1_SINGLE_REFUTE tighten the core only.
With both off, the counts below are the older rule, unchanged.
GATE_WEAK_REFUTE, GATE_PRIMARY_SOURCE, GATE_SHARED_SOURCE, and
GATE_EVIDENCE_SCORE are extra paths. Each is off unless a study turned
it on. Off, they do not change a count.
A picture claim can be refuted by text and cannot be confirmed by it:
the text never saw the file.

The overall decision follows the core claim. A failed stance call is a
system failure. It is never reported as "not enough sources".
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import misinfo_agent.config as config

# The sentence the card must show when text would otherwise "confirm" a
# picture. Kept as one string so the test and the UI cannot drift.
_PICTURE_BLOCK = "Text sources cannot confirm what a picture shows."

# The sentence assess looks for when it forces refute_kind out_of_context.
# The wording is the product's, so the card and the kind cannot drift.
PHOTO_OUT_OF_CONTEXT = (
    "The photo is from a different event or an earlier time, so the caption is out of context."
)

# Credible means an editor or an office, not a blog. Tier 3 is a real
# organisation and still not this bar: the rule asks for tier 1 or 2.
_CREDIBLE_TIER = 2
_TIER1 = 1


def gate(state: dict) -> dict:
    """Judge each atomic claim, then the core claim.

    Returns the keys this box owns. `rounds` is included only when the
    decision is search_more, because that is the only time the count
    changes. `gate` is the gate_decision.
    """
    if _skipped(state):
        return _skip(state)

    # A crashed stance call can leave a partial table that looks
    # decisive. Counting it would turn an outage into a verdict.
    if _stance_failed(state):
        return _failed_stance(state)

    atomics = _atomics(state)
    drop_after = bool(config.EVAL_DROP_AFTER_CLAIM)
    statuses: dict[str, str] = {}
    reasons: list[str] = []
    judged: dict[str, dict] = {}
    core_detail_ids: list[str] = []
    for atomic in atomics:
        aid = str(atomic.get("id") or "A1")
        if atomic.get("checkable") is False:
            # A motive is not an event. Another search cannot make it one,
            # so it is recorded and then left out of the counts.
            statuses[aid] = "not_checkable"
            reasons.append(
                f"{aid} is not checkable, so no source is asked to confirm it."
            )
            continue
        status, lines, detail_ids = _judge(state, atomic, drop_after=drop_after)
        statuses[aid] = status
        reasons.extend(lines)
        judged[aid] = {"status": status}
        if aid == _core_id(atomics):
            core_detail_ids = detail_ids

    core_id = _core_id(atomics)
    core_status = statuses.get(core_id, "insufficient")
    decision, overall, rounds = _overall(state, core_id, core_status)
    reasons.append(overall)
    out = {
        "gate": decision,
        "gate_reasons": reasons,
        "atomic_status": statuses,
        # Empty when the flag is off, and when every support quote did
        # match the core. The card lists these so a detail is not read
        # as confirmation of the headline.
        "core_detail_only_ids": core_detail_ids,
    }
    if rounds is not None:
        out["rounds"] = rounds
    return out


def _skipped(state: dict) -> bool:
    """Box 6 already settled an exact fresh prior, or there is nothing to check.

    `route` is trusted. Box 6 sets reuse_prior only for exact and fresh;
    repeating that test here would let the two boxes disagree. A missing
    checkable flag is not a skip: extract has not said no, so the table
    is judged.
    """
    if state.get("route") == "reuse_prior":
        return True
    return state.get("checkable") is False


def _skip(state: dict) -> dict:
    if state.get("route") == "reuse_prior":
        reason = (
            "An exact, fresh fact-check already covers this, so the evidence gate is not applied."
        )
    else:
        reason = "This submission is not checkable, so the evidence gate is not applied."
    return {
        "gate": "skipped",
        "gate_reasons": [reason],
        "atomic_status": {},
        "core_detail_only_ids": [],
    }


def _stance_failed(state: dict) -> bool:
    stats = state.get("investigate_stats")
    if not isinstance(stats, dict):
        return False
    return bool(stats.get("stance_failed"))


def _failed_stance(state: dict) -> dict:
    """A system failure is never 'insufficient'.

    insufficient means the web was read and did not contain two credible
    groups. A stance error means the web may not have been read. Those
    two must not share a word in the decision, or the card would tell a
    person the sources were thin when the pipeline was down.
    """
    done = _rounds(state)
    if done < config.MAX_SEARCH_ROUNDS:
        return {
            "gate": "search_more",
            "rounds": done + 1,
            "gate_reasons": [
                "The stance check failed, so the evidence table cannot be trusted. "
                "Another search is allowed."
            ],
            "atomic_status": {},
            "core_detail_only_ids": [],
        }
    return {
        "gate": "error",
        "gate_reasons": [
            "The stance check failed and the search limit is already used. "
            "This is a system failure, not a lack of evidence."
        ],
        "atomic_status": {},
        "core_detail_only_ids": [],
    }


def _atomics(state: dict) -> list[dict]:
    """One core claim when extract did not split the forward.

    Rows with no atomic_id are A1, which is also what box 7 writes when
    the claim was not split. Inventing a second id here would orphan
    those rows.
    """
    items = state.get("atomic_claims")
    if not isinstance(items, list) or not items:
        return [
            {
                "id": "A1",
                "text": str(state.get("claim") or ""),
                "core": True,
                "checkable": True,
            }
        ]
    cleaned = []
    for item in items:
        if isinstance(item, dict):
            cleaned.append(item)
    return cleaned or [
        {
            "id": "A1",
            "text": str(state.get("claim") or ""),
            "core": True,
            "checkable": True,
        }
    ]


def _core_id(atomics: list[dict]) -> str:
    for item in atomics:
        if item.get("core") is True:
            return str(item.get("id") or "A1")
    # A list that forgot the flag still has a core. A1 is that claim
    # when extract followed the prompt, and the only claim otherwise.
    return str(atomics[0].get("id") or "A1")


def _judge(state: dict, atomic: dict, *, drop_after: bool) -> tuple[str, list[str], list[str]]:
    aid = str(atomic.get("id") or "A1")
    text = str(atomic.get("text") or "")
    # all_rows still holds a support that only confirms a side detail.
    # G2R's "nobody supports this" test has to see that page. The support
    # split below is what the support count uses. Off-subject refutes are
    # removed from both lists first.
    all_rows = counted_rows(_rows(state.get("evidence"), aid, drop_after=drop_after))
    detail_ids: list[str] = []
    rows = all_rows
    if config.TRUE_NEEDS_CORE_MATCH and _is_core(state, aid):
        # Off-subject denials are removed before either count. A stale
        # page about a different topic must not sit in refute_credible
        # and block the denials that do name this subject. all_rows keeps
        # every support, including one that only confirms a side detail,
        # so the weak-refute "anybody supports this" test still sees it.
        all_rows = _drop_off_subject_refutes(all_rows, text)
        rows, detail_ids = _split_core_supports(all_rows, text)
    groups = _groups(rows)
    reasons: list[str] = []
    for name in groups["conflict"]:
        # Named so a reader can see which outlet was dropped, not just
        # that the count is smaller than the row count.
        reasons.append(
            f"{name} argues both sides of {aid}, so it counts on neither side."
        )

    support = groups["support"]
    refute = groups["refute"]
    support_c = groups["support_credible"]
    refute_c = groups["refute_credible"]
    # A count or an office held "now" goes stale. An undated ministry
    # page, or one older than the other side's newest row, is still
    # evidence. It is not a current record, so it cannot be the only
    # credible vote. The note names the host so the card can show it.
    present = _present_status(text)
    stale_refute = _stale_sole_notes(rows, refute_c, "supports") if present else []
    stale_support = _stale_sole_notes(rows, support_c, "refutes") if present else []
    conflicting, stray, because = _conflict(groups)
    if stray:
        # The phrase is fixed for the one-source case the rule names.
        # A larger stray that still loses the fraction test uses the plural.
        if stray == 1:
            reasons.append("1 source disagrees.")
        else:
            reasons.append(f"{stray} sources disagree.")

    if conflicting:
        # A ministry on the small side is a different fact from a third
        # of the newsrooms. The card should say which one it was.
        if because == "tier1":
            line = (
                f"{aid} is conflicting: credible sources are on both sides, "
                "and the smaller side includes a tier 1 source."
            )
        else:
            line = (
                f"{aid} is conflicting: credible sources are on both sides, "
                "and the smaller side is at least a third of them."
            )
        reasons.insert(0, line)
        reasons.extend(stale_refute)
        reasons.extend(stale_support)
        return "conflicting", reasons, detail_ids

    # The early conflict test misses a majority of stale offices against
    # one fresh newsroom: the newsroom is a small stray, and the offices
    # would otherwise win. Both sides are still credible, so nobody wins.
    if (stale_refute and support_c) or (stale_support and refute_c):
        reasons.insert(
            0,
            f"{aid} is conflicting: credible sources are on both sides, "
            "and an official page on one side may be outdated.",
        )
        reasons.extend(stale_refute)
        reasons.extend(stale_support)
        return "conflicting", reasons, detail_ids

    # One office, and nobody on the other side at all. A tier-4 page is
    # support. It does not have to be tier 1-2 to stop a single denial.
    # Two blogs are still not enough to refute by themselves: the group
    # has to be tier 1, and a present-tense page has to be current.
    if (
        config.TIER1_SINGLE_REFUTE
        and _is_core(state, aid)
        and len(refute) == 1
        and len(support) == 0
        and groups["tier"].get(refute[0]) == _TIER1
        and not stale_refute
    ):
        host = _group_host(rows, refute[0])
        reasons.insert(0, f"single official denial: {host}")
        return "refuted", reasons, detail_ids

    if len(refute) >= 2 and len(refute_c) >= 1 and not stale_refute:
        reasons.insert(
            0,
            f"{aid} is refuted: {len(refute)} independent groups refute it, "
            f"and {len(refute_c)} of them are tier 1 or 2.",
        )
        return "refuted", reasons, detail_ids
    if stale_refute:
        # The only credible denial is a page that may no longer be current.
        # A lower-tier companion does not make that page a second office.
        reasons.extend(stale_refute)

    # Lower-tier denials, and not a single page on the other side.
    # Checked after the credible-refute rule so a tier 1-2 source still
    # uses the ordinary sentence and is not capped as a weak refute.
    # G2R asks for three groups. G2R-2 asks for two and caps confidence
    # lower. Either way a supporting page of any tier returns no groups,
    # and support is never decided on this path.
    if (config.GATE_WEAK_REFUTE or config.GATE_WEAK_REFUTE_2) and _is_core(state, aid):
        weak = _weak_refute_groups(all_rows, text)
        minimum = (
            config.WEAK_REFUTE_2_MIN_GROUPS
            if config.GATE_WEAK_REFUTE_2
            else config.WEAK_REFUTE_MIN_GROUPS
        )
        if len(weak) >= minimum:
            reasons.insert(
                0,
                f"{aid} is refuted: {len(weak)} independent groups refute it, "
                "and none of them is tier 1 or 2.",
            )
            reasons.insert(1, config.WEAK_REFUTE_CAVEAT)
            return "refuted", reasons, detail_ids

    if _picture(state, text) and config.GATE_PHOTO_CONTEXT:
        photo_lines = _photo_out_of_context(state, aid)
        if photo_lines:
            return "refuted", photo_lines + reasons, detail_ids

    picture = _picture(state, text)
    # The core needs two tier 1-2 supporters. A side claim, and the core
    # when the flag is off, still accepts one credible group plus any
    # second independent group.
    strict_support = config.TRUE_NEEDS_CORE_MATCH and _is_core(state, aid)
    supported = (
        len(support_c) >= 2 and not picture and not stale_support
        if strict_support
        else len(support) >= 2 and len(support_c) >= 1 and not picture and not stale_support
    )
    if supported:
        if strict_support:
            line = (
                f"{aid} is supported: {len(support_c)} independent tier 1 or 2 groups support it."
            )
        else:
            line = (
                f"{aid} is supported: {len(support)} independent groups support it, "
                f"and {len(support_c)} of them are tier 1 or 2."
            )
        reasons.insert(0, line)
        return "supported", reasons, detail_ids

    # One office, speaking about itself. The count rules already had
    # their chance. This does not fire when a tier 1-2 group is opposed,
    # and it does not let text confirm a picture.
    if config.GATE_PRIMARY_SOURCE and _is_core(state, aid):
        primary = _primary_side(rows, groups, text)
        if primary == "refutes" and not stale_refute:
            host = _primary_host(rows, groups["refute"], text, "refutes")
            reasons.insert(0, f"{aid} is refuted: {host} is the body this claim is about.")
            return "refuted", reasons, detail_ids
        if primary == "supports" and not picture and not stale_support:
            host = _primary_host(rows, groups["support"], text, "supports")
            reasons.insert(0, f"{aid} is supported: {host} is the body this claim is about.")
            return "supported", reasons, detail_ids
    if stale_support:
        reasons.extend(stale_support)

    # The score only decides a claim the counts left open. A conflict
    # already returned. Support still needs a tier 1-2 group, which is
    # the same asymmetry as the weak-refute path: a pile of blogs can
    # refute and cannot confirm.
    if config.GATE_EVIDENCE_SCORE and _is_core(state, aid):
        score = evidence_score(state, rows, text)
        theta = float(config.GATE_EVIDENCE_THETA)
        if score <= -theta and not stale_refute:
            reasons.insert(
                0,
                f"{aid} is refuted: the evidence score is {score:.2f}, "
                f"at or below -{theta:.1f}.",
            )
            return "refuted", reasons, detail_ids
        if score >= theta and len(support_c) >= 1 and not picture and not stale_support:
            reasons.insert(
                0,
                f"{aid} is supported: the evidence score is {score:.2f}, "
                f"at or above {theta:.1f}, and a tier 1 or 2 group supports it.",
            )
            return "supported", reasons, detail_ids

    if picture and config.GATE_PHOTO_SUPPORT:
        photo_lines = _photo_supported(state, aid)
        if photo_lines:
            # Two full matches of the same event. Text still cannot do this.
            return "supported", photo_lines + reasons, detail_ids

    picture_support = (
        len(support_c) >= 2
        if strict_support
        else len(support) >= 2 and len(support_c) >= 1
    )
    if picture and picture_support:
        # Text can still refute this claim, which is why the refute
        # branch above is allowed to win. It must not confirm the file.
        reasons.insert(0, _PICTURE_BLOCK)
        reasons.insert(
            1,
            f"{aid} has {len(support)} supporting groups, but they only describe the picture.",
        )
        return "insufficient", reasons, detail_ids

    reasons.insert(
        0,
        f"{aid} is not yet supported or refuted: "
        f"{_side(len(support), 'group supports', 'groups support')} it and "
        f"{_side(len(refute), 'group refutes', 'groups refute')} it.",
    )
    return "insufficient", reasons, detail_ids


def _rows(evidence, atomic_id: str, *, drop_after: bool) -> list[dict]:
    """Supports and refutes for this claim.

    after_claim rows are ignored only while the leakage guard is on.
    The eval turns that on so a page written after the rumour cannot
    count as something the system could have read. The demo leaves it
    off, and a later write-up is then ordinary evidence.
    """
    if not isinstance(evidence, list):
        return []
    kept = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        if str(row.get("atomic_id") or "A1") != atomic_id:
            continue
        if drop_after and row.get("after_claim") is True:
            continue
        if row.get("stance") not in {"supports", "refutes"}:
            continue
        # Provenance rows are judged by the photo rules, which read
        # image_provenance. Counting them here would let a reverse-image
        # page confirm a picture through the text rule, or refute it as
        # an ordinary false claim instead of out of context.
        if row.get("provenance") is True:
            continue
        kept.append(row)
    return kept


def _groups(rows: list[dict]) -> dict:
    """One vote per independent group, on one side.

    group_id is the independence key box 7 already computed (same
    registrable domain, or a MinHash near-copy). `group` is the older
    fixture name for the same idea. A row with neither is its own
    group: two undated snippets must not become one outlet just because
    both keys were empty.

    A group marked group_conflict has a support and a refute of this
    same claim. It is named and then dropped. Counting it on both sides
    would look like a disagreement between two outlets. Counting it on
    one side would hide the sentence that points the other way.
    """
    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        # Same label _row_group uses. A row with neither key stays its own
        # group: two undated snippets must not become one outlet.
        name = _row_group(row)
        if name == "row:":
            name = f"row:{len(order)}"
        if name not in buckets:
            order.append(name)
            buckets[name] = []
        buckets[name].append(row)

    support: list[str] = []
    refute: list[str] = []
    support_c: list[str] = []
    refute_c: list[str] = []
    conflict: list[str] = []
    tier_of: dict[str, int] = {}
    for name in order:
        members = buckets[name]
        if any(item.get("group_conflict") is True for item in members):
            conflict.append(name)
            continue
        stances = {item.get("stance") for item in members}
        # Both stances without the flag is the same disagreement. The
        # flag is what box 7 writes; this is the backstop for a row
        # that was built by hand and forgot it.
        if stances == {"supports", "refutes"}:
            conflict.append(name)
            continue
        best = min(_tier(item) for item in members)
        tier_of[name] = best
        if "supports" in stances:
            support.append(name)
            if best <= _CREDIBLE_TIER:
                support_c.append(name)
        elif "refutes" in stances:
            refute.append(name)
            if best <= _CREDIBLE_TIER:
                refute_c.append(name)
    return {
        "support": support,
        "refute": refute,
        "support_credible": support_c,
        "refute_credible": refute_c,
        "conflict": conflict,
        "tier": tier_of,
    }


def _conflict(groups: dict) -> tuple[bool, int, str]:
    """Whether the credible minority is large enough to block a result.

    One tier-2 source against four newsrooms is a stray. It is named and
    the majority still stands. The minority blocks when it is at least
    a third of the credible groups, or when it includes a tier 1 source:
    a ministry on the other side is not a stray, however many papers
    are lined up against it. When the two sides are the same size, each
    is the minority, so a tier 1 source on either side blocks.

    The comparison is integer. minority * 3 >= total is the same test
    as minority >= total / 3, without a float that makes 1 and 1/3 look
    equal on one machine and not on another.

    Returns (blocks, stray_count, why). stray_count is 0 when there is
    no credible minority left to name, including when that minority
    already blocked the result.
    """
    left = groups["support_credible"]
    right = groups["refute_credible"]
    if not left or not right:
        return False, 0, ""
    if len(left) < len(right):
        minority = left
    elif len(right) < len(left):
        minority = right
    else:
        # Same size: either side can hold the tier-1 source that blocks.
        minority = left + right
    total = len(left) + len(right)
    minority_n = min(len(left), len(right))
    by_share = minority_n * 3 >= total
    by_tier = _has_tier1(minority if len(left) != len(right) else left + right, groups["tier"])
    # On a tie the share test already blocks (n * 3 >= 2n). Tier 1 is
    # still reported when it is present, because that is the sharper reason.
    if by_tier and not by_share:
        return True, 0, "tier1"
    if by_share or by_tier:
        return True, 0, "share" if by_share else "tier1"
    return False, minority_n, ""


def _has_tier1(names: list[str], tier_of: dict) -> bool:
    return any(tier_of.get(name, 4) <= _TIER1 for name in names)


def _picture(state: dict, atomic_text: str) -> bool:
    """True when text must not be allowed to confirm the claim.

    An image submission is a picture even when the caption never says
    "photo": the file is the claim. A text forward that says "video
    shows" is the same problem, and box 6 already has the detector for
    that sentence. Using it here keeps "what is a picture" in one place.
    The import is local so this module can be loaded without pulling
    the memory encoder in, which that detector's module also imports.
    """
    form = str(state.get("input_type") or "").casefold()
    if form in {"image", "video"}:
        return True
    from misinfo_agent.prior_match import _claim_is_picture

    if _claim_is_picture(atomic_text, None):
        return True
    return _claim_is_picture(str(state.get("claim") or ""), None)


def _photo_out_of_context(state: dict, aid: str) -> list[str] | None:
    """Refuted as out of context, or None when the photo rule does not fire.

    The tier 1-2 rule is unchanged: one such group must place the file at
    an earlier event or a different place, and no tier 1-2 page may
    present it as this event. One lower-tier page is not enough.

    With GATE_PHOTO_WEAK_REFUTE, two groups of any tier also refute when
    no provenance group, at any tier, supports the caption. That path
    names the missing tier 1-2 page so the card can cap the confidence.
    A supporting page of any tier stops only this weaker path.

    PHOTO_RULE chooses who votes. v0 is full matches only. v1 adds
    partials. v2 and v3 keep a partial only when its date is trusted
    and at least a year before the event. v3 also lets a tier 1-2 full
    match of this event block the whole out-of-context path.
    """
    grouped = _photo_groups(state)
    if grouped is None:
        return None
    refute = [name for name, item in grouped.items() if item["refutes"] and not item["blocks"]]
    blocks = [name for name, item in grouped.items() if item["blocks"]]
    if blocks:
        return None
    credible = [name for name in refute if grouped[name]["tier"] <= _CREDIBLE_TIER]
    if credible:
        detail = (
            f"{aid} is refuted: {len(credible)} tier 1 or 2 provenance "
            f"{'group shows' if len(credible) == 1 else 'groups show'} an earlier or different event."
        )
        return [PHOTO_OUT_OF_CONTEXT, detail]
    supports = [name for name, item in grouped.items() if item["supports"]]
    if config.GATE_PHOTO_WEAK_REFUTE and len(refute) >= 2 and not supports:
        detail = (
            f"{aid} is refuted: {len(refute)} independent provenance groups "
            "show an earlier or different event."
        )
        return [PHOTO_OUT_OF_CONTEXT, detail, config.PHOTO_WEAK_REFUTE_CAVEAT]
    return None


def _photo_supported(state: dict, aid: str) -> list[str] | None:
    """True only from two independent full tier 1-2 matches of this event.

    A partial match, a tier 3 or 4 page, or a single newsroom is not
    enough. Otherwise a photograph is never True from provenance, and
    the text rule below still cannot confirm it.
    """
    grouped = _photo_groups(state)
    if grouped is None:
        return None
    # A tier 1-2 page that places the file at another event blocks True.
    # Two agreeing newsrooms do not outvote that page.
    if any(item["refutes"] and item["tier"] <= _CREDIBLE_TIER for item in grouped.values()):
        return None
    full = [name for name, item in grouped.items() if item["full_support"] and not item["refutes"]]
    if len(full) < 2:
        return None
    return [
        f"{aid} is supported: {len(full)} independent tier 1 or 2 groups "
        "have a full match of this photograph from the same event."
    ]


def _photo_groups(state: dict) -> dict | None:
    """One vote per provenance group. None when there is no photo table."""
    raw = state.get("image_provenance")
    if not isinstance(raw, dict):
        return None
    from misinfo_agent.provenance import claim_event_date, is_dated_earlier, photo_match_counts

    # The caption's event, not today and not a bare claim_date when the
    # caption named a month or a day. Partials vote only under the rule.
    event = claim_event_date(state)
    matches = [
        row
        for row in (raw.get("matches") or [])
        if isinstance(row, dict) and photo_match_counts(row, event)
    ]
    if not matches:
        return None
    claim_date = event
    buckets: dict[str, dict] = {}
    for index, match in enumerate(matches):
        name = str(match.get("group_id") or "").strip() or f"row:{index}"
        item = buckets.setdefault(
            name,
            {
                "refutes": False,
                "blocks": False,
                "supports": False,
                "full_support": False,
                "tier": 4,
            },
        )
        tier = _tier(match)
        item["tier"] = min(item["tier"], tier)
        early = is_dated_earlier(match, event)
        stance = match.get("stance")
        # An earlier date is a different time even when the model said
        # the page agrees. The date is the part a caption cannot talk past.
        if early or stance == "refutes":
            item["refutes"] = True
        # Any tier. The weak photo rule needs to know that somebody,
        # including a blog, presented the file as this event. The tier
        # 1-2 block below is the older rule and is not this flag.
        if stance == "supports" and not early:
            item["supports"] = True
        if (
            stance == "supports"
            and not early
            and tier <= _CREDIBLE_TIER
        ):
            item["blocks"] = True
        if (
            stance == "supports"
            and match.get("match") == "full"
            and not early
            and tier <= _CREDIBLE_TIER
            and _photo_on_event(match.get("date"), claim_date)
        ):
            item["full_support"] = True
    return buckets


def _photo_earlier(page_date, claim_date) -> bool:
    """True when the page is older than claim_date minus the margin."""
    page = _photo_day(page_date)
    claim = _photo_day(claim_date)
    if page is None or claim is None:
        return False
    return page < claim - timedelta(days=config.PHOTO_EARLIER_DAYS)


def _photo_on_event(page_date, claim_date) -> bool:
    """Dated on or after the event, allowing the same margin before the forward.

    A story often runs a day or two before the forward. Older than that
    margin is an earlier event, which is the refute side of this rule.
    A missing date is not "on or after": True requires a date.
    """
    page = _photo_day(page_date)
    claim = _photo_day(claim_date)
    if page is None or claim is None:
        return False
    return page >= claim - timedelta(days=config.PHOTO_EARLIER_DAYS)


def _photo_day(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _overall(state: dict, core_id: str, core_status: str) -> tuple[str, str, int | None]:
    """Map the core status to the gate, and advance rounds only to search again.

    A not-checkable core does not get another search. The sentence is a
    motive or an opinion; a second query cannot produce a page that
    confirms it. Conflict does not get another search either: more pages
    will not make a tier-1 disagreement disappear, and box 12 sends it
    to a person.
    """
    done = _rounds(state)
    if core_status == "not_checkable":
        return (
            "insufficient",
            f"The core claim {core_id} is not something a source can confirm or refute.",
            None,
        )
    if core_status == "conflicting":
        return (
            "conflicting",
            f"The core claim {core_id} is conflicting, so a person has to judge it.",
            None,
        )
    if core_status in {"supported", "refuted"}:
        return (
            "sufficient",
            f"The core claim {core_id} is {core_status}, so the evidence is enough to assess.",
            None,
        )
    if done < config.MAX_SEARCH_ROUNDS:
        return (
            "search_more",
            f"The core claim {core_id} does not yet have enough evidence, and another search is allowed.",
            done + 1,
        )
    return (
        "insufficient",
        f"The core claim {core_id} does not have enough evidence, and the search limit is used.",
        None,
    )


def _side(n: int, one: str, many: str) -> str:
    """'1 group supports' or '2 groups support'. Zero uses the plural."""
    return f"{n} {one if n == 1 else many}"


def _rounds(state: dict) -> int:
    try:
        return max(0, int(state.get("rounds") or 0))
    except (TypeError, ValueError):
        return 0


def _tier(row: dict) -> int:
    """A missing or garbled tier is 4, which is not credible.

    JSON and a hand-built fixture sometimes store the tier as a string.
    Treating that as missing would drop a ministry out of the credible
    count. Failing the cast is the unknown-host case, not a crash.
    """
    try:
        return int(row.get("tier"))
    except (TypeError, ValueError):
        return 4


def _is_core(state: dict, atomic_id: str) -> bool:
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and str(item.get("id") or "A1") == atomic_id:
            return item.get("core") is not False
    return atomic_id == "A1"


def _split_core_supports(rows: list[dict], core_text: str) -> tuple[list[dict], list[str]]:
    """Drop support quotes that do not carry the core's key terms.

    CORE_MATCH_MODE picks the matcher. "strict" demands the full name and
    the exact verb on the quote. "relaxed" allows a surname or a synonym
    and may use the sentence around the quote. Both still require the
    core's numbers and dates, and both still need two tier 1-2 groups
    before the core counts as supported. A refute is not filtered here.
    _drop_off_subject_refutes applies the subject rule, which does not
    ask the quote to repeat the claim's figure.
    """
    from misinfo_agent.prior_match import core_support_matches

    kept = []
    detail_ids = []
    for row in rows:
        if row.get("stance") == "supports" and not core_support_matches(
            str(row.get("quote") or ""),
            core_text,
            passage=str(row.get("passage") or ""),
        ):
            evidence_id = str(row.get("id") or "").strip()
            if evidence_id:
                detail_ids.append(evidence_id)
            continue
        kept.append(row)
    return kept, detail_ids


def _group_host(rows: list[dict], group_name: str) -> str:
    """The site named in the single-denial sentence.

    netloc is what a reader can check. The group id is the fallback
    because older rows stored the host there and left netloc empty.
    """
    for row in rows:
        name = str(row.get("group_id") or row.get("group") or "").strip()
        if name != group_name:
            continue
        host = str(row.get("netloc") or "").strip()
        if host:
            return host
        url = str(row.get("url") or "")
        if "://" in url:
            return url.split("/")[2]
        return group_name
    return group_name


def counted_rows(rows: list[dict]) -> list[dict]:
    """The rows the gate and the confidence score both count.

    Shared-source merging rewrites group_id on copies. The stored evidence
    is left alone, so a later variant or a replay of the same state still
    sees the groups box 7 wrote. With the flag off this is the same list.
    """
    if not config.GATE_SHARED_SOURCE:
        return rows
    return _merge_shared_sources([dict(row) for row in rows if isinstance(row, dict)])


def evidence_score(state: dict, rows: list[dict], core_text: str) -> float:
    """S = sum over groups of tier weight, stance, cosine, recency, core match.

    Stance is +1 for a support and -1 for a refute. A group that fails the
    core match contributes nothing: a page about a side detail is not a
    vote on the headline. Theta is applied by the caller. The weights live
    in config and are not fitted.
    """
    grouped = _groups(rows)
    claim_date = state.get("claim_date")
    total = 0.0
    for name in grouped["support"]:
        total += _group_term(rows, name, 1.0, core_text, claim_date, grouped)
    for name in grouped["refute"]:
        total += _group_term(rows, name, -1.0, core_text, claim_date, grouped)
    return total


def refute_core_matches(quote: str, core: str, *, passage: str = "") -> bool:
    """True when a denial is about this subject, whatever figure it states."""
    return refute_core_failure(quote, core, passage=passage) == ""


def refute_core_failure(quote: str, core: str, *, passage: str = "") -> str:
    """The subject term a denial failed to carry, or "" when it matches.

    A correction states a different figure. "28 states" is how a page
    refutes "45 states", and "established in 1935" is how a page refutes
    "founded in 1947". Requiring the quote to repeat 45 or 1947 would
    drop every numeric correction. The subject still has to be the same
    one: the main entity and the attribute word ("India" and "states").
    A page about Nepal, or about a different scheme, fails that and the
    returned word is the term that was missing.

    A claim with no name and no attribute keeps the older overlap test.
    Those sentences have no subject to protect, and a fixture row whose
    quote is only the stance label has nothing in it to reject.
    """
    entity, attribute, applicable = _refute_subject(str(core or ""))
    if not applicable:
        return "" if _refute_legacy(quote, core, passage) else _legacy_miss(quote, core, passage)
    from misinfo_agent.prior_match import _quote_and_sentence

    haystack = _quote_and_sentence(quote, passage)
    if entity and not _entity_in_refute(entity, haystack):
        return entity
    if attribute and not _attribute_in_refute(attribute, haystack):
        return attribute
    return ""


def _drop_off_subject_refutes(rows: list[dict], core_text: str) -> list[dict]:
    """Remove refute quotes that name a different subject.

    A row whose quote has no real words is kept. The stance label is all
    box 7 stored, and there is no subject in that quote to disagree with.
    Support rows are not touched.
    """
    if not _refute_subject(core_text)[2]:
        return rows
    kept = []
    for row in rows:
        if row.get("stance") != "refutes":
            kept.append(row)
            continue
        quote = str(row.get("quote") or "")
        if not _quote_has_prose(quote):
            kept.append(row)
            continue
        if refute_core_matches(quote, core_text, passage=str(row.get("passage") or "")):
            kept.append(row)
    return kept


def _quote_has_prose(quote: str) -> bool:
    """True when the quote contains a word that could name a subject.

    "refutes 1" is the placeholder a count-only fixture stores. Treating
    that label as a topic would drop every such row, including the ones
    the group tests are about. A real quotation has some other word.
    """
    words = [
        word
        for word in re.findall(r"[A-Za-z]{4,}", quote or "")
        if word.casefold() not in {"refutes", "supports", "neutral"}
    ]
    return bool(words)


def _refute_legacy(quote: str, core: str, passage: str) -> bool:
    """The overlap test used when the claim names no subject.

    Numbers are blanked, same as before. One shared content word is
    enough, because there is no entity or attribute to demand. Any
    denial of any sentence must still fail when the claim has words
    and the quote shares none of them.
    """
    from misinfo_agent.prior_match import (
        _action_words,
        _quote_and_sentence,
        _relaxed_entity,
        core_support_matches,
    )

    blanked = _NUMBER_TOKEN.sub(" ", str(core or ""))
    if not core_support_matches(quote, blanked, passage=passage, mode="relaxed"):
        return False
    if _relaxed_entity(blanked) or _action_words(blanked):
        return True
    haystack = _quote_and_sentence(quote, passage).casefold()
    words = _legacy_words(core)
    if not words:
        return bool(str(quote or "").strip())
    return any(re.search(rf"\b{re.escape(word)}\b", haystack) for word in words)


def _legacy_miss(quote: str, core: str, passage: str) -> str:
    from misinfo_agent.prior_match import _quote_and_sentence

    haystack = _quote_and_sentence(quote, passage).casefold()
    missing = [
        word
        for word in _legacy_words(core)
        if re.search(rf"\b{re.escape(word)}\b", haystack) is None
    ]
    if missing:
        return missing[0]
    return "subject"


def _legacy_words(core: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z]{4,}", str(core or "").casefold())
        if word not in _CONTENT_STOP
    ]


def _refute_subject(core: str) -> tuple[str, str, bool]:
    """(entity, attribute, applicable).

    Applicable means the sentence has a name, an action, or a figure
    with a noun, so a denial can be held to that subject. A sentence
    with none of those is not given a new rule.
    """
    text = str(core or "")
    entity = _named_org(text) or _refute_entity_name(text)
    # The noun next to the figure is the attribute ("45 states", "₹500
    # notes"). A framing verb such as "announced" is not, or a denial
    # that says "invalid" and "notes" would fail for using the wrong
    # synonym of a verb that was not the subject. The verb is the
    # attribute only when the claim has no such noun ("founded in 1947").
    noun = _quantity_noun(text, entity)
    if noun:
        return entity, noun, True
    actions = _refute_actions(text)
    if actions:
        return entity, actions[0], True
    if entity:
        return entity, "", True
    return "", "", False


def _named_org(text: str) -> str:
    """A known organisation written out, or its acronym in capitals.

    The short form is matched only in capitals. "who" in a question is
    not the World Health Organization. The long name is what a quote
    that says "Reserve Bank of India" and a quote that says "RBI" share.
    """
    from misinfo_agent.prior_match import _KNOWN_ACRONYMS

    folded = str(text or "").casefold()
    for short, long in _KNOWN_ACRONYMS.items():
        if re.search(rf"\b{re.escape(long)}\b", folded):
            return long
    for short, long in _KNOWN_ACRONYMS.items():
        if re.search(rf"\b{re.escape(short.upper())}\b", text or ""):
            return short.upper()
    return ""


def _refute_entity_name(text: str) -> str:
    """The capital name, including a sentence-initial one.

    The support matcher skips the first word of the sentence so that
    "The" is not a person. "India has 45 states" then has no name at
    all, and a quote about Nepal would pass on the word "states" alone.
    Here the first word is the subject when nothing else was capitalised.
    """
    from misinfo_agent.prior_match import _NOT_A_NAME, _relaxed_entity

    found = _relaxed_entity(text)
    if found:
        return found
    raw = str(text or "").lstrip()
    match = re.match(r"([A-Z][A-Za-z]{2,})\b", raw)
    if not match:
        return ""
    word = match.group(1)
    if word.casefold() in _NOT_A_NAME or word.casefold() in {"the", "this", "that", "there"}:
        return ""
    return word


def _refute_actions(text: str) -> list[str]:
    from misinfo_agent.prior_match import _action_words

    return _action_words(text)


def _quantity_noun(text: str, entity: str) -> str:
    """The noun the figure is about, not the figure.

    "45 states" gives "states". A year at the end of "founded in 1947"
    gives "founded", because the word after the year is not the topic.
    A month is part of the date, so "on 26 January 1950" looks backward
    to the verb rather than taking "January" as the attribute.
    """
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    if not match:
        return ""
    after = re.findall(r"[A-Za-z]+", text[match.end():])
    if after and after[0].casefold() in _MONTHS:
        return _word_before(text[: match.start()], entity)
    for word in after:
        if _content_token(word, entity):
            return word.casefold()
    return _word_before(text[: match.start()], entity)


def _word_before(text: str, entity: str) -> str:
    words = re.findall(r"[A-Za-z]+", text or "")
    for word in reversed(words):
        if _content_token(word, entity):
            return word.casefold()
    return ""


def _content_token(word: str, entity: str) -> bool:
    raw = word.casefold()
    if raw in _REFUTE_FRAME or raw in _MONTHS:
        return False
    if len(raw) < 4 and not word.isupper():
        return False
    ent = (entity or "").casefold()
    if ent and (raw == ent or raw in ent.split()):
        return False
    return True


def _entity_in_refute(entity: str, text: str) -> bool:
    """The name, its acronym, the short org name, or the adjective.

    "Reserve Bank" is the Reserve Bank of India when the quote stops
    before "of India". "Indian" is India. "Indiana" is not: the only
    extra endings are the short ones that turn a country into its adjective.
    """
    from misinfo_agent.prior_match import _KNOWN_ACRONYMS, _entity_matches, _phrase_in, _word_in

    if _entity_matches(entity, text):
        return True
    folded_ent = entity.casefold()
    for short, long in _KNOWN_ACRONYMS.items():
        if folded_ent not in {short, long}:
            continue
        if _word_in(text, short) or _phrase_in(text, long):
            return True
        head = long.split(" of ")[0]
        if " " in head and _phrase_in(text, head):
            return True
    ent = folded_ent
    if " " in ent or len(ent) < 4:
        return False
    for token in re.findall(r"[A-Za-z]+", text or ""):
        folded = token.casefold()
        if not folded.startswith(ent):
            continue
        if folded[len(ent):] in {"n", "ian", "ese", "i", "ish", "an"}:
            return True
    return False


def _attribute_in_refute(word: str, text: str) -> bool:
    """The same stem, an action synonym, or founded/established.

    "states" and "state" share a stem. "founded" and "established" do
    not, and the support synonym table does not list either, so the
    founding class lives here. Adding it to the support list would make
    a support quote say "founded" when the claim said "established".
    """
    from misinfo_agent.prior_match import _action_satisfied, _stem

    if _action_satisfied(word, text):
        return True
    tokens = re.findall(r"[A-Za-z]+", text or "")
    stem = _stem(word)
    if any(_stem(token) == stem or token.casefold() == word.casefold() for token in tokens):
        return True
    group_stems = {_stem(item) for item in _FOUNDING}
    if word.casefold() in _FOUNDING or stem in group_stems:
        return any(token.casefold() in _FOUNDING or _stem(token) in group_stems for token in tokens)
    return False


# "total" is how a count is phrased ("in total"). It is not the thing
# being counted. Months are part of a date, not the attribute.
_REFUTE_FRAME = frozenset({
    "total", "overall", "altogether", "currently", "present", "today",
    "there", "their", "about", "after", "before", "which", "would",
    "could", "should", "have", "been", "were", "will", "into", "over",
    "under", "than", "then", "with", "from", "that", "this", "also",
    "just", "only", "more", "most", "many", "some", "such", "other",
    "first", "being", "every", "each", "where", "when", "what", "your",
    "they", "them", "these", "those", "does", "did",
})
_MONTHS = frozenset({
    "january", "jan", "february", "feb", "march", "mar", "april", "apr",
    "may", "june", "jun", "july", "jul", "august", "aug", "september",
    "sept", "sep", "october", "oct", "november", "nov", "december", "dec",
})
# Not in the support synonym table on purpose. See _attribute_in_refute.
_FOUNDING = frozenset({
    "found", "founded", "establish", "established", "formed", "form",
    "created", "create", "creation",
})


def _weak_refute_groups(rows: list[dict], core_text: str) -> list[str]:
    """Refuting groups that may convict the core without a tier 1-2 source.

    Empty when any group supports, at any tier, or when a tier 1-2 group
    already refutes (that case belongs to the ordinary rule). Every row in
    a counted group has to pass the relaxed refute match. One off-topic
    sentence inside the group drops the group.
    """
    grouped = _groups(rows)
    if grouped["support"] or grouped["refute_credible"]:
        return []
    buckets = _named_buckets(rows)
    passed = []
    for name in grouped["refute"]:
        members = [row for row in buckets.get(name, []) if row.get("stance") == "refutes"]
        if not members:
            continue
        if all(
            refute_core_matches(
                str(row.get("quote") or ""),
                core_text,
                passage=str(row.get("passage") or ""),
            )
            for row in members
        ):
            passed.append(name)
    return passed


def _group_term(rows, name, sign, core_text, claim_date, grouped) -> float:
    members = [row for row in rows if _row_group(row) == name]
    if not members:
        return 0.0
    tier = grouped["tier"].get(name, 4)
    weight = _tier_weight(tier)
    cosines = [_row_cos(row) for row in members]
    known = [value for value in cosines if value is not None]
    # No stored cosine means this row was not a picked passage. It does
    # not get a free similarity. Real evidence rows carry `cos`.
    cosine = max(known) if known else 0.0
    recency = _group_recency(members, claim_date)
    if sign > 0:
        matched = any(
            _support_core_matches(row, core_text) for row in members if row.get("stance") == "supports"
        )
    else:
        matched = any(
            refute_core_matches(
                str(row.get("quote") or ""),
                core_text,
                passage=str(row.get("passage") or ""),
            )
            for row in members
            if row.get("stance") == "refutes"
        )
    core_match = 1.0 if matched else 0.0
    return weight * sign * cosine * recency * core_match


def _support_core_matches(row: dict, core_text: str) -> bool:
    from misinfo_agent.prior_match import core_support_matches

    return core_support_matches(
        str(row.get("quote") or ""),
        core_text,
        passage=str(row.get("passage") or ""),
        mode="relaxed",
    )


def _tier_weight(tier: int) -> float:
    weights = config.GATE_TIER_WEIGHTS
    try:
        number = int(tier)
    except (TypeError, ValueError):
        number = 4
    if number in weights:
        return float(weights[number])
    if number < 1:
        return float(weights[1])
    return float(weights[4])


def _row_cos(row: dict) -> float | None:
    try:
        number = float(row.get("cos"))
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return min(1.0, max(0.0, number))


def _group_recency(members: list[dict], claim_date) -> float:
    """1 on the claim's own day, about half a year later, 0.5 if undated.

    Half a year is not a fitted half-life. STALE_DAYS is already how long
    a status claim may stand before box 6 asks for a fresh check, so a
    page that old is half credit here too. An unknown date is half credit
    rather than a zero: dropping it would treat "we could not date the
    page" as "the page is irrelevant".
    """
    scored = []
    for row in members:
        value = _recency(row.get("published_date"), claim_date)
        if value is not None:
            scored.append(value)
    if not scored:
        return 0.5
    return max(scored)


def _recency(published, claim_date) -> float | None:
    page = _parse_day(published)
    claim = _parse_day(claim_date)
    if page is None or claim is None:
        return None
    days = abs((claim - page).days)
    window = float(config.STALE_DAYS) or 365.0
    return 1.0 / (1.0 + days / window)


def _parse_day(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


# Present-tense status, count, and office claims. "has 28" is a count.
# "in total", "currently", "now", and "is the current" ask what is true
# today. A founding year ("was established in 1935") does not match.
_PRESENT_STATUS = re.compile(
    r"\bhas\s+\d|\bis the current\b|\bnow\b|\btotal\b|\bcurrently\b",
    re.IGNORECASE,
)


def _present_status(text: str) -> bool:
    """True when the sentence is about a current count, status, or office."""
    return _PRESENT_STATUS.search(str(text or "")) is not None


def _stale_sole_notes(rows, credible_names, opposing_stance) -> list[str]:
    """Notes when every credible group on this side is a stale tier-1 page.

    Empty when any tier 1-2 group on this side is current. A newsroom
    dated after the claim, sitting beside an old office page, means the
    office is not the only credible vote, so this returns nothing.
    """
    if not credible_names:
        return []
    notes = []
    for name in credible_names:
        stale, host, label = _official_page_stale(rows, name, opposing_stance)
        if not stale:
            return []
        notes.append(f"official page may be outdated: {host}, {label}")
    return notes


def _official_page_stale(rows, name, opposing_stance) -> tuple[bool, str, str]:
    """Whether this group is a tier-1 page that cannot decide a present claim.

    Undated is stale. A date older than the newest row on the other side
    is stale. A tier 2-4 group is never this case. The label is the date
    or the word undated, which is what the gate reason quotes.
    """
    members = [row for row in rows if _row_group(row) == name]
    if not members or min(_tier(row) for row in members) > _TIER1:
        return False, "", ""
    host = _group_host(rows, name)
    days = [
        day
        for day in (_parse_day(row.get("published_date") or row.get("date")) for row in members)
        if day is not None
    ]
    if not days:
        return True, host, "undated"
    newest = max(days)
    opposed = []
    for row in rows:
        if row.get("stance") != opposing_stance:
            continue
        day = _parse_day(row.get("published_date") or row.get("date"))
        if day is not None:
            opposed.append(day)
    if opposed and newest < max(opposed):
        return True, host, newest.isoformat()
    return False, host, newest.isoformat()


def _primary_side(rows: list[dict], groups: dict, claim_text: str) -> str | None:
    """'supports' or 'refutes' when one tier-1 body-page stands alone.

    A support stands alone when no tier 1-2 group refutes it. A refute
    stands alone only when nothing supports the claim at any tier. A
    blog on the supporting side is enough to stop a single official
    denial; the two-group rule is what may still refute. A tier-1 page
    whose host is not the body the sentence names does not count.
    """
    support_hit = _primary_host(rows, groups["support"], claim_text, "supports")
    refute_hit = _primary_host(rows, groups["refute"], claim_text, "refutes")
    if support_hit and not groups["refute_credible"]:
        return "supports"
    if refute_hit and not groups["support"]:
        return "refutes"
    return None


def _primary_host(rows: list[dict], names: list[str], claim_text: str, stance: str) -> str | None:
    """The host of a tier-1 group that is the body, and whose quote is about the claim.

    A page from the right office about a different event does not count.
    Support uses the relaxed support match. A refute may state a different
    figure, which is refute_core_matches.
    """
    for name in names:
        if not _group_tier_is_primary(rows, name):
            continue
        host = _group_host(rows, name)
        if not _host_is_body(host, claim_text):
            continue
        members = [row for row in rows if _row_group(row) == name and row.get("stance") == stance]
        if stance == "supports" and any(_support_core_matches(row, claim_text) for row in members):
            return host
        if stance == "refutes" and any(
            refute_core_matches(
                str(row.get("quote") or ""),
                claim_text,
                passage=str(row.get("passage") or ""),
            )
            for row in members
        ):
            return host
    return None


def _group_tier_is_primary(rows: list[dict], name: str) -> bool:
    tiers = [_tier(row) for row in rows if _row_group(row) == name]
    return bool(tiers) and min(tiers) <= _TIER1


def _host_is_body(host: str, claim_text: str) -> bool:
    """True when the claim names the office that publishes this host."""
    names = _body_names(host)
    if not names or not claim_text:
        return False
    for name in names:
        if " " in name or len(name) >= 4:
            if re.search(rf"\b{re.escape(name)}\b", claim_text, flags=re.IGNORECASE):
                return True
        elif re.search(rf"\b{re.escape(name)}\b", claim_text):
            return True
    return False


def _body_names(host: str) -> tuple[str, ...]:
    key = _bare_host(host)
    table = config.PRIMARY_BODY_NAMES
    if key in table:
        return tuple(table[key])
    for known, names in table.items():
        if key.endswith("." + known):
            return tuple(names)
    return ()


def _bare_host(host: str) -> str:
    text = str(host or "").strip().lower()
    if text.startswith("www."):
        text = text[4:]
    return text


def _merge_shared_sources(rows: list[dict]) -> list[dict]:
    """One group when two outlets repeat the same release.

    Three signals, any one of them is enough, and they chain: a shared
    12-word quotation, the same press-release URL, or the same wire
    credit (PTI, ANI, or a PIB release) in the passage. Different wires
    stay different sources. The earliest group id becomes the merged id
    so the reason text does not depend on hash order.
    """
    if len(rows) < 2:
        return rows
    names = [_row_group(row) for row in rows]
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        ra, rb = find(left), find(right)
        if ra == rb:
            return
        # Keep the name that appeared first. find() walks toward it.
        if names.index(ra) <= names.index(rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    by_signal: dict[str, list[str]] = {}

    def add(signal: str, name: str) -> None:
        bucket = by_signal.setdefault(signal, [])
        if name not in bucket:
            bucket.append(name)

    for row, name in zip(rows, names):
        blob = f"{row.get('passage') or ''}\n{row.get('quote') or ''}"
        for shingle in _quoted_shingles(row):
            add("q:" + shingle, name)
        for url in _press_urls(blob) | _press_urls(str(row.get("url") or "")):
            add("u:" + url, name)
        for agency in _agencies(blob):
            add("a:" + agency, name)
    for group_names in by_signal.values():
        if len(group_names) < 2:
            continue
        head = group_names[0]
        for other in group_names[1:]:
            union(head, other)
    merged = []
    for row, name in zip(rows, names):
        copied = dict(row)
        copied["group_id"] = find(name)
        merged.append(copied)
    return merged


def _quoted_shingles(row: dict) -> set[str]:
    """12-word runs inside a quotation, not inside the whole article.

    The evidence quote is a quotation the stance step already cut out.
    A span in quotation marks inside the passage is the other place a
    copied sentence shows up. The rest of the article can share a topic
    without being the same release.
    """
    width = int(config.GATE_SHARED_QUOTE_WORDS)
    texts = []
    quote = str(row.get("quote") or "").strip()
    if len(_words(quote)) >= width:
        texts.append(quote)
    passage = str(row.get("passage") or "")
    for match in _QUOTED_SPAN.finditer(passage):
        span = match.group(1) or match.group(2) or ""
        if len(_words(span)) >= width:
            texts.append(span)
    found = set()
    for text in texts:
        words = _words(text)
        for index in range(len(words) - width + 1):
            found.add(" ".join(words[index:index + width]))
    return found


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


def _press_urls(text: str) -> set[str]:
    found = set()
    for match in _URL.finditer(str(text or "")):
        raw = match.group(0).rstrip(".,);]")
        canonical = _canonical_url(raw)
        if canonical and _is_press_url(canonical):
            found.add(canonical)
    return found


def _canonical_url(raw: str) -> str:
    try:
        parts = urlsplit(raw)
    except ValueError:
        return ""
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if not key.lower().startswith("utm")
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, urlencode(query), ""))


def _is_press_url(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = (parts.path or "").casefold()
    if host == "pib.gov.in" or host.endswith(".pib.gov.in"):
        return True
    if "press-release" in path or "pressrelease" in path:
        return True
    return "prid=" in (parts.query or "").casefold()


def _agencies(text: str) -> set[str]:
    """Wire credits. PIB counts only when the sentence calls it a release.

    PTI and ANI are the credits a newsroom puts on a copied wire. A passing
    mention of the bureau, without 'release', is not that credit: plenty of
    pages name PIB without reprinting one note.
    """
    found = set()
    if re.search(r"\bPTI\b", text or ""):
        found.add("PTI")
    if re.search(r"\bANI\b", text or ""):
        found.add("ANI")
    if re.search(r"\bPIB\b", text or "") and re.search(r"\brelease\b", text or "", flags=re.IGNORECASE):
        found.add("PIB")
    if re.search(r"\bpress information bureau\b", text or "", flags=re.IGNORECASE):
        found.add("PIB")
    return found


def _named_buckets(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(_row_group(row), []).append(row)
    return buckets


def _row_group(row: dict) -> str:
    name = str(row.get("group_id") or row.get("group") or "").strip()
    if name:
        return name
    return f"row:{row.get('id') or id(row)}"


# "total" stays. It is a content word of a count ("29 states in total").
# The function words are how the sentence is glued together.
_CONTENT_STOP = frozenset({
    "that", "this", "with", "from", "have", "been", "were", "will",
    "into", "over", "under", "than", "then", "their", "there", "about",
    "after", "before", "which", "would", "could", "should",
})
_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_QUOTED_SPAN = re.compile(r"[\"“]([^\"”]{0,500})[\"”]|[«]([^»]{0,500})[»]")
_URL = re.compile(r"https?://[^\s<>\"]+")


def unverifiable_reason(state: dict) -> str:
    """Why a finished Unverifiable was not graded.

    First match, same order as the diagnosis: a broken stance call, a
    real disagreement, no usable pages, no tier 1-2 after the official
    query, only weaker pages, or exactly one credible group. A picture
    the text cannot confirm has no code of its own; it falls through to
    the same thin-evidence labels.
    """
    gate_name = state.get("gate")
    if gate_name == "error":
        return "error"
    if gate_name == "conflicting":
        return "conflicting_credible_sources"
    atomics = _atomics(state)
    core = _core_id(atomics)
    drop_after = bool(config.EVAL_DROP_AFTER_CLAIM)
    raw = _rows(state.get("evidence"), core, drop_after=drop_after)
    # No support and no refute at all. A detail-only row is still a page,
    # so it must not be reported as "nothing came back".
    if not raw:
        return "no_hits"
    rows = raw
    if config.TRUE_NEEDS_CORE_MATCH:
        rows, _ignored = _split_core_supports(rows, _core_text(state, core))
    usable = [
        row for row in rows
        if str(row.get("passage") or row.get("quote") or "").strip()
    ]
    groups = _groups(rows)
    credible = len(groups["support_credible"]) + len(groups["refute_credible"])
    if credible == 0:
        if official_query_ran(state):
            return "no_credible_record"
        # Pages came back, and none of them is tier 1 or 2 on this claim.
        # A quote that only matched a detail is in this bucket too.
        return "only_tier34"
    support_c = len(groups["support_credible"])
    refute_c = len(groups["refute_credible"])
    larger = max(support_c, refute_c)
    smaller = min(support_c, refute_c)
    if larger == 1 and smaller == 0:
        return "single_credible"
    if not usable:
        return "no_hits"
    return "only_tier34"


def _core_text(state: dict, core_id: str) -> str:
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and str(item.get("id") or "") == core_id:
            return str(item.get("text") or state.get("claim") or "")
    return str(state.get("claim") or "")


def official_query_ran(state: dict) -> bool:
    """True when the official line was actually searched.

    A passage tagged official means a page came back. A stored query
    whose text is the official line means the search was issued even
    when it returned nothing. Planning the line is not the same thing:
    a cap can stop the run before that query starts.
    """
    for passage in state.get("passages") or []:
        if isinstance(passage, dict) and "official" in (passage.get("query_kinds") or []):
            return True
    stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    texts = []
    for item in stats.get("queries") or []:
        if isinstance(item, dict) and item.get("kind") == "official":
            return True
        if isinstance(item, str) and item.strip():
            texts.append(item)
    if not texts:
        return False
    from misinfo_agent.investigate import build_queries

    for query in build_queries(state):
        if query.kind == "official" and query.text in texts:
            return True
    return False
