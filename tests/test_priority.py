"""Box 11 on fixture states. No model and no network.

Style signals are allowed to move the score. They are not allowed to
move a verdict; that is box 9's job and is asserted in test_gate.py.
"""

import pytest

import misinfo_agent.config as config
from misinfo_agent.priority import components, priority

# The garlic forward from the ingest tests, after extract would have
# labelled it. Breaking, a share request, a borrowed authority, and a
# health cure. The first copy, with nothing in memory.
GARLIC = {
    "domain": "Health",
    "claim": "Drinking raw garlic juice every morning cures Type 2 diabetes in 30 days.",
    "text": (
        "AIIMS Delhi doctors have confirmed that drinking raw garlic juice "
        "every morning cures Type 2 diabetes in 30 days."
    ),
    "red_flags": [
        "Urgency",
        "Fake or vague authority",
        "Miracle or absolute claim",
        "Call to forward",
        "Sensational style",
    ],
    "style_signals": {
        "urgency_words": ["breaking"],
        "share_request": True,
        "forwarded_label": "Forwarded as received",
        "caps_ratio": 0.2,
        "exclaim_count": 4,
        "emoji_count": 2,
        "authority_mentions": ["AIIMS"],
        "conspiracy_phrases": ["hiding"],
        "certainty_words": ["confirmed", "cures"],
    },
    "submission_count": 1,
    "matches": [],
    "investigate_stats": {"circulation_sightings": 0},
}

# A practice-match score. No banner, no forward request, no red flag.
# Sports is the low-harm end of the scale on purpose.
SPORTS = {
    "domain": "Sports",
    "claim": "Virat Kohli scored a double century in a practice match yesterday.",
    "text": "Virat Kohli scored a double century in a practice match yesterday.",
    "red_flags": [],
    "style_signals": {
        "urgency_words": [],
        "share_request": False,
        "forwarded_label": None,
        "caps_ratio": 0.02,
        "exclaim_count": 0,
        "emoji_count": 0,
        "authority_mentions": [],
        "conspiracy_phrases": [],
        "certainty_words": [],
    },
    "submission_count": 1,
    "matches": [{"dense_cos": 0.62, "also_debunked_by": []}],
    "investigate_stats": {"circulation_sightings": 0},
}

# Used by the sensitivity script as well as the bounds test. A new
# health claim with no shout and no spread sits near the middle, which
# is where a 20% weight change can cross a level.
QUIET_HEALTH = {
    "domain": "Health",
    "claim": "A new hospital in Pune has run out of insulin.",
    "red_flags": [],
    "style_signals": {
        "urgency_words": [],
        "share_request": False,
        "forwarded_label": None,
    },
    "submission_count": 1,
    "matches": [],
    "investigate_stats": {"circulation_sightings": 0},
}


def test_garlic_forward_is_high():
    out = priority(GARLIC)
    assert out["priority"] == "High"
    assert 0.0 <= out["priority_score"] <= 1.0
    assert len(out["priority_reasons"]) == 4


def test_calm_sports_rumour_is_low():
    out = priority(SPORTS)
    assert out["priority"] == "Low"
    assert out["priority_score"] < config.PRIORITY_MEDIUM


def test_novelty_is_one_minus_the_top_cosine():
    parts = components(SPORTS)
    assert parts["novelty"] == pytest.approx(1.0 - 0.62)
    fresh = components(GARLIC)
    assert fresh["novelty"] == 1.0


def test_forwarded_many_times_counts_more_than_as_received():
    many = dict(GARLIC)
    many["style_signals"] = dict(GARLIC["style_signals"])
    many["style_signals"]["forwarded_label"] = "Forwarded many times"
    received = components(GARLIC)["reach"]
    viral = components(many)["reach"]
    assert viral > received


def test_scores_stay_inside_the_unit_interval():
    for state in (GARLIC, SPORTS, QUIET_HEALTH):
        parts = components(state)
        for name in ("harm", "novelty", "reach", "urgency"):
            assert 0.0 <= parts[name] <= 1.0
        out = priority(state)
        assert 0.0 <= out["priority_score"] <= 1.0


def test_weights_and_domains_match_the_config():
    assert set(config.DOMAIN_HARM) == set(config.DOMAINS)
    assert config.PRIORITY_WEIGHTS == {
        "harm": 0.40,
        "novelty": 0.15,
        "reach": 0.25,
        "urgency": 0.20,
    }
    assert sum(config.PRIORITY_WEIGHTS.values()) == pytest.approx(1.0)
    assert config.PRIORITY_HIGH == 0.60
    assert config.PRIORITY_MEDIUM == 0.35


def test_sensitivity_script_counts_level_changes():
    # Loaded lazily: the script imports this module to reuse the fixtures,
    # so a top-level import here would run that load while this file is
    # still being defined.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "priority_sensitivity.py"
    spec = importlib.util.spec_from_file_location("priority_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = module.base_levels()
    by_name = {name: band for name, band, _score in base}
    assert by_name["garlic forward"] == "High"
    assert by_name["calm sports rumour"] == "Low"
    rows = module.sensitivity()
    assert len(rows) == 8
    assert all(isinstance(row["n_changed"], int) for row in rows)


def test_urgency_and_sensational_style_do_not_raise_harm():
    # Those two are packaging. Harm moves for the four dangerous flags
    # only, or the score would count one shout in two components.
    plain = dict(QUIET_HEALTH)
    shouted = dict(QUIET_HEALTH)
    shouted["red_flags"] = ["Urgency", "Sensational style"]
    assert components(plain)["harm"] == components(shouted)["harm"]
    flagged = dict(QUIET_HEALTH)
    flagged["red_flags"] = ["Fear or outrage", "Call to forward"]
    assert components(flagged)["harm"] > components(plain)["harm"]
