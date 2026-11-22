"""Box 9: is the evidence table enough to assess each atomic claim?

Rules only. No model call and no fetch. Box 7 has already assigned the
tier, the independent group, and the stance. This box counts those
groups. It does not re-decide what a page said.

A claim is enough when two independent groups take the same side and at
least one of them is tier 1 or 2. One credible voice on the other side
does not block that, unless it is a large minority or a tier 1 source.
TRUE_NEEDS_CORE_MATCH and TIER1_SINGLE_REFUTE tighten the core only.
With both off, the counts below are the older rule, unchanged.
A picture claim can be refuted by text and cannot be confirmed by it:
the text never saw the file.

The overall decision follows the core claim. A failed stance call is a
system failure. It is never reported as "not enough sources".
"""

from __future__ import annotations

import misinfo_agent.config as config

# The sentence the card must show when text would otherwise "confirm" a
# picture. Kept as one string so the test and the UI cannot drift.
_PICTURE_BLOCK = "Text sources cannot confirm what a picture shows."

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
    rows = _rows(state.get("evidence"), aid, drop_after=drop_after)
    # Side claims keep the old count. The false Trues were pages that
    # confirmed a detail of the headline, not the headline.
    detail_ids: list[str] = []
    if config.TRUE_NEEDS_CORE_MATCH and _is_core(state, aid):
        rows, detail_ids = _split_core_supports(rows, text)
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
        return "conflicting", reasons, detail_ids

    # One office, and nobody credible on the other side. Two blogs are
    # still not enough: the group has to be tier 1.
    if (
        config.TIER1_SINGLE_REFUTE
        and _is_core(state, aid)
        and len(refute) == 1
        and len(support_c) == 0
        and groups["tier"].get(refute[0]) == _TIER1
    ):
        host = _group_host(rows, refute[0])
        reasons.insert(0, f"single official denial: {host}")
        return "refuted", reasons, detail_ids

    if len(refute) >= 2 and len(refute_c) >= 1:
        reasons.insert(
            0,
            f"{aid} is refuted: {len(refute)} independent groups refute it, "
            f"and {len(refute_c)} of them are tier 1 or 2.",
        )
        return "refuted", reasons, detail_ids

    picture = _picture(state, text)
    # The core needs two tier 1-2 supporters. A side claim, and the core
    # when the flag is off, still accepts one credible group plus any
    # second independent group.
    strict_support = config.TRUE_NEEDS_CORE_MATCH and _is_core(state, aid)
    supported = (
        len(support_c) >= 2 and not picture
        if strict_support
        else len(support) >= 2 and len(support_c) >= 1 and not picture
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
        name = str(row.get("group_id") or row.get("group") or "").strip()
        if not name:
            name = f"row:{row.get('id') or len(order)}"
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
    before the core counts as supported. A refute is unchanged.
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
