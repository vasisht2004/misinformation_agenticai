"""Box 12: alert, ask a person, search once more, or store.

Rules only. The verdict and the confidence are already on the state.
This box does not reopen the evidence and does not read style signals.
Style already had its say in the priority level. Reading it again would
let a banner word change the action after the verdict was fixed.

The bar is p* = a / (a + bH). H is harm, not the full priority score.
A higher H lowers the bar: the cost of missing a harmful false claim
is larger than the cost of missing a sports rumour, so we act on a
lower confidence. A viral sports rumour raises priority (box 11) and
does not move this bar.

An alert is still a request. While ALERT_NEEDS_APPROVAL is on, box 13
has to approve it before anything is sent. The decision stays "alert"
so the card can say what was asked; the reason says the pause is there.
"""

from __future__ import annotations

import misinfo_agent.config as config
from misinfo_agent.priority import harm, priority

# False and Misleading are the same action. Misleading still sends
# someone to the wrong conclusion; the grain of truth does not make it
# safe to leave in a chat. The other three labels do not alert.
_ACT = frozenset({"False", "Misleading"})


def decide(state: dict) -> dict:
    """Choose the action and record p*.

    `rounds` is returned only for search_more. That is the only decision
    that spends another investigate pass, and the counter has to move or
    the next call would ask again forever.
    """
    harm_score = harm(state)
    bar = _p_star(harm_score)
    gate = str(state.get("gate") or "")
    verdict = str(state.get("verdict") or "").strip()
    confidence = _confidence(state)
    band = _priority_level(state)
    high_harm = harm_score >= config.HARM_HIGH

    # A conflict or a crashed stance check is not a verdict, however
    # confident the rest of the state looks. Box 9 already refused to
    # call that "not enough sources".
    if gate in {"error", "conflicting"}:
        why = (
            "The gate is conflicting, so a person has to judge it."
            if gate == "conflicting"
            else "The gate is a system error, so a person has to look before any action."
        )
        return _out("human", [why], bar)

    if verdict in _ACT and band == "High" and confidence >= bar:
        reasons = [
            f"The verdict is {verdict}, the priority is High, and the confidence "
            f"{confidence:.2f} clears the bar of {bar:.2f}."
        ]
        if config.ALERT_NEEDS_APPROVAL:
            # The flag is the pause. Renaming the decision to "human"
            # would hide that an alert was the thing being approved.
            reasons.append(
                "This alert pauses for a person to approve it at box 13 before anything is sent."
            )
        return _out("alert", reasons, bar)

    if verdict in _ACT and high_harm and confidence < bar:
        # Close to the bar: one more search can still move the confidence
        # across it. Far below, another page will not rescue a claim we
        # already do not believe, and a person should see the harm.
        close = confidence >= bar - config.P_STAR_MARGIN
        done = _rounds(state)
        if close and done < config.MAX_SEARCH_ROUNDS:
            return _out(
                "search_more",
                [
                    f"The verdict is {verdict} and the harm is high, but the confidence "
                    f"{confidence:.2f} is under the bar of {bar:.2f} and still within "
                    f"{config.P_STAR_MARGIN:.2f} of it. Another search is allowed."
                ],
                bar,
                rounds=done + 1,
            )
        return _out(
            "human",
            [
                f"The verdict is {verdict} and the harm is high, but the confidence "
                f"{confidence:.2f} does not clear the bar of {bar:.2f}."
            ],
            bar,
        )

    if verdict == "Unverifiable" and high_harm:
        return _out(
            "human",
            [
                "The claim is unverifiable and the harm is high, so a person should see it "
                "rather than leaving it stored."
            ],
            bar,
        )
    if verdict == "Unverifiable":
        return _out(
            "store",
            [
                "The claim is unverifiable and the harm is not in the high band, so it is stored."
            ],
            bar,
        )
    if verdict == "True":
        return _out(
            "store",
            ["The claim checks out, so it is stored and not sent as an alert."],
            bar,
        )
    if verdict == "Other":
        return _out(
            "store",
            ["This is not a checkable claim, so it is stored."],
            bar,
        )
    if verdict in _ACT and confidence >= bar:
        # The bar is cleared, and the priority is not High: the alert
        # branch above is the only one that sends. A false claim can be
        # stored when it is not the thing a person should be pulled
        # away for.
        return _out(
            "store",
            [
                f"The verdict is {verdict} and the confidence clears the bar of {bar:.2f}, "
                "but the priority is not High, so it is stored rather than sent as an alert."
            ],
            bar,
        )
    if verdict in _ACT:
        return _out(
            "store",
            [
                f"The verdict is {verdict}, and the harm is not in the high band, so it is stored."
            ],
            bar,
        )
    return _out(
        "store",
        ["There is no verdict to act on, so it is stored."],
        bar,
    )


def _p_star(harm_score: float) -> float:
    """a / (a + bH). H is clamped by harm() to [0, 1].

    b > 0 is what makes the bar fall as H rises. At H = 0 the bar is 1:
    nothing is acted on without certainty when nobody can get hurt. At
    H = 1 the bar is a/(a+b) = 0.25 with the current constants.
    """
    denominator = config.P_STAR_A + config.P_STAR_B * harm_score
    if denominator <= 0:
        return 1.0
    return config.P_STAR_A / denominator


def _priority_level(state: dict) -> str:
    """The stored level wins.

    Tests and a paused run set it directly. Recomputing when it is
    already there would let a later edit of the weights change an
    action the graph had already chosen. When the key is missing, box
    11's function fills it so a partial state is not treated as Low
    by accident.
    """
    stored = state.get("priority")
    if stored in {"High", "Medium", "Low"}:
        return stored
    return priority(state)["priority"]


def _confidence(state: dict) -> float:
    try:
        value = float(state.get("confidence"))
    except (TypeError, ValueError):
        # A missing confidence is not a high one. Alerting on it would
        # treat "we did not score this" as "we are sure".
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _rounds(state: dict) -> int:
    try:
        return max(0, int(state.get("rounds") or 0))
    except (TypeError, ValueError):
        return 0


def _out(decision: str, reasons: list[str], bar: float, rounds: int | None = None) -> dict:
    result = {
        "decision": decision,
        "decision_reasons": reasons,
        "p_star": bar,
    }
    if rounds is not None:
        result["rounds"] = rounds
    return result
