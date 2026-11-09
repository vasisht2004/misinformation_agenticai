"""Box 6: thresholds propose a match; details, subjects and age can overrule.

The lemon/cancer pair is the one the cosine misses. At a forced 0.95 it
would be "exact" on the threshold alone, and neither the amounts nor the
capitalised words differ. It still must not be reused.
"""

from datetime import date

from misinfo_agent.prior_match import _normalise_quantities, decide_prior, llm_context_check
import pytest


def _hit(primary: str, cosine: float, **extra) -> dict:
    row = {
        "article_id": "t1",
        "site": "Alt News",
        "date": "2026-06-01",
        "primary_claim": primary,
        "true_version": "",
        "verdict": "False",
        "verdict_norm": "false",
        "url": "https://example.test/claim",
        "domain": "Health",
        "manipulation_type": "Fabricated",
        "form": "Text",
        "dense_cos": cosine,
        "dense_rank": 1,
        "bm25_rank": 1,
        "rrf": 0.03,
        "also_debunked_by": [],
    }
    row.update(extra)
    return row


def test_rupee_amount_conflict_is_never_exact():
    decision = decide_prior(
        "RBI says all ₹500 notes will be withdrawn.",
        [_hit("RBI says all ₹2000 notes will be withdrawn.", 0.96)],
        now=date(2026, 11, 9),
    )
    assert decision["match_type"] != "exact"
    assert decision["match_type"] == "similar"
    assert any("₹500" in reason and "₹2000" in reason for reason in decision["reasons"])


def test_stale_status_claim_needs_recheck():
    # Covaxin was "not WHO-approved" until November 2021. A 2020 fact-check
    # of that sentence cannot be reused in 2026: the status moved.
    decision = decide_prior(
        "Covaxin is not WHO-approved.",
        [_hit("Covaxin is not WHO-approved.", 0.97, date="2020-01-15")],
        now=date(2026, 11, 9),
    )
    assert decision["needs_recheck"] is True
    assert decision["route"] == "investigate"


def test_old_out_of_context_image_does_not_need_recheck():
    decision = decide_prior(
        "Photo shows the banned protest in Delhi.",
        [
            _hit(
                "Photo shows the banned protest in Delhi.",
                0.97,
                date="2018-03-01",
                form="Image",
                manipulation_type="Out of context",
            )
        ],
        now=date(2026, 11, 9),
    )
    assert decision["needs_recheck"] is False


def test_lemon_cancer_swap_is_never_exact():
    # Top IDF on the cancer sentence is "cells", not "cancer" (cancer is
    # common here). "cells" is not close to any word in the lemon sentence,
    # and that is what blocks reuse at a forced 0.95.
    decision = decide_prior(
        "Drinking hot lemon water kills coronavirus",
        [_hit("Drinking hot lemon water kills cancer cells", 0.95)],
        now=date(2026, 11, 9),
    )
    assert decision["match_type"] != "exact"
    assert decision["match_type"] == "similar"
    text = " ".join(decision["reasons"]).lower()
    assert "cells" in text


def test_five_lakh_matches_digit_lakh():
    assert _normalise_quantities("5 lakh") == _normalise_quantities("five lakh")
    assert _normalise_quantities("Rs. 5 lakh") == _normalise_quantities("five lakh rupees")
    decision = decide_prior(
        "The rally drew 5 lakh people.",
        [_hit("The rally drew five lakh people.", 0.96)],
        now=date(2026, 11, 9),
    )
    assert decision["match_type"] == "exact"
    assert not any("differs" in reason for reason in decision["match_reasons"])


def test_extra_number_is_not_a_contradiction():
    decision = decide_prior(
        "A mother giving birth experiences up to 57 DEL of pain, equal to 20 bones fracturing at once.",
        [
            _hit(
                "Women giving birth experience 57 del of pain, equal to 20 bones fracturing at once, and men die at 45 del.",
                0.96,
            )
        ],
        now=date(2026, 11, 9),
    )
    assert not any("number differs" in reason for reason in decision["match_reasons"])


def test_hindu_manager_is_never_exact():
    decision = decide_prior(
        "A Hindu restaurant manager in Bahrain denied entry to a hijab-wearing Muslim woman.",
        [_hit("An Indian manager of a restaurant in Bahrain barred a veiled woman from entering.", 0.95)],
        now=date(2026, 11, 9),
    )
    assert decision["match_type"] != "exact"
    assert any(
        "The claim names a community ('Hindu')" in reason for reason in decision["match_reasons"]
    )


def test_picture_claim_is_never_exact_from_text_alone():
    decision = decide_prior(
        "Video shows flood in Chennai",
        [
            _hit(
                "Video shows the Chennai airport flooded due to heavy rains.",
                0.95,
                domain="Disasters & accidents",
                form="Video",
                manipulation_type="Out of context",
                date="2021-11-14",
            )
        ],
        now=date(2026, 11, 9),
    )
    assert decision["match_type"] != "exact"
    assert decision["match_type"] == "similar"
    assert any("same picture" in reason for reason in decision["match_reasons"])
    # The airport video is old, and it is out of context. Age alone still
    # must not set needs_recheck. There is no status word here either.
    assert decision["needs_recheck"] is False


def test_dates_are_compared_before_plain_numbers():
    decision = decide_prior(
        "₹500 notes invalid from 1 December",
        [_hit("Banks will stop disbursing ₹500 notes by 30 September 2025.", 0.96)],
        now=date(2026, 11, 9),
    )
    text = " ".join(decision["match_reasons"])
    assert "The date differs: 1 December vs 30 September 2025" in text
    assert "1 vs 30" not in text
    assert decision["match_summary"] == decision["match_reasons"][:2]
    assert 1 <= len(decision["match_summary"]) <= 2


def test_percent_is_one_value_and_blocks_exact():
    decision = decide_prior(
        "Unemployment fell from 18% to 9.2%.",
        [_hit("Unemployment came down to 2 percent.", 0.96)],
        now=date(2026, 11, 9),
    )
    text = " ".join(decision["match_reasons"])
    assert decision["match_type"] != "exact"
    assert "The percent differs: 9.2% vs 2%." in text
    assert "9 vs 2" not in text


def test_llm_context_check_is_not_implemented():
    with pytest.raises(NotImplementedError):
        llm_context_check("a claim", {})


def test_llm_veto_turns_exact_into_similar(monkeypatch):
    monkeypatch.setattr("misinfo_agent.prior_match.LLM_CONTEXT_CHECK", True)
    monkeypatch.setattr(
        "misinfo_agent.prior_match.call_structured",
        lambda prompt, schema: {"same_claim": False, "differing_detail": "the city"},
    )
    decision = decide_prior(
        "RBI says the note was withdrawn.",
        [_hit("RBI says the note was withdrawn.", 0.96)],
        now=date(2026, 11, 9),
    )
    assert decision["match_type"] == "similar"
    assert any(
        "A careful read found a difference: the city." in reason
        for reason in decision["match_reasons"]
    )


def test_llm_veto_does_not_run_unless_the_pair_is_exact(monkeypatch):
    calls = []
    monkeypatch.setattr("misinfo_agent.prior_match.LLM_CONTEXT_CHECK", True)
    monkeypatch.setattr(
        "misinfo_agent.prior_match.call_structured",
        lambda prompt, schema: calls.append(prompt),
    )
    decide_prior(
        "A rumour about a new comet over Pune.",
        [_hit("The city opened a new metro line.", 0.40)],
        now=date(2026, 11, 9),
    )
    decide_prior(
        "RBI says all ₹500 notes will be withdrawn.",
        [_hit("RBI says all ₹2000 notes will be withdrawn.", 0.96)],
        now=date(2026, 11, 9),
    )
    assert calls == []


def test_llm_veto_is_skipped_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr("misinfo_agent.prior_match.LLM_CONTEXT_CHECK", False)
    monkeypatch.setattr(
        "misinfo_agent.prior_match.call_structured",
        lambda prompt, schema: calls.append(prompt),
    )
    decision = decide_prior(
        "RBI says the note was withdrawn.",
        [_hit("RBI says the note was withdrawn.", 0.96)],
        now=date(2026, 11, 9),
    )
    assert calls == []
    assert decision["match_type"] == "exact"
