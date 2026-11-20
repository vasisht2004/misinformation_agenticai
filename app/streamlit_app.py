"""The checker people use: one claim in, a card or a review pause out.

Started with `streamlit run app/streamlit_app.py`. This page calls
run_pipeline and draws what that already decided. It does not score a
verdict and it does not search on its own.

Three pages share one process. A run sets a busy flag and the next
pass is the run; the page radio and the action buttons are not drawn
while that pass is inside the pipeline, so a second claim cannot start
beside it. BGE-M3 and the memory index are loaded once per process.
A check is not a forward: RECORD_SUBMISSIONS stays off unless the
sidebar switch is on, because box 11 reads that count as reach.
"""

from __future__ import annotations

import html
import io
import logging
import socket
import sys
import urllib.error
import uuid
from pathlib import Path
from urllib.parse import urlsplit

# Streamlit puts the script's folder, app/, on sys.path. The package
# lives in the repo root. Without this, the page cannot import the
# pipeline it is supposed to call.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

import misinfo_agent.config as config
from misinfo_agent.fetch import FetchError
from misinfo_agent.graph import paused_review, run_pipeline
from misinfo_agent.llm import LLMError, LLMQuotaError
from misinfo_agent.output import past_factcheck_heading
from misinfo_agent.search import SearchError, SearchQuotaError

# Off before any widget. A rerun that fails before the sidebar is drawn
# must not inherit "on" from a previous experiment in this process.
config.RECORD_SUBMISSIONS = False

log = logging.getLogger("misinfo_app")

_VERDICTS = ("False", "Misleading", "True", "Unverifiable", "Other")

_TIER_LABELS = {
    1: ("Tier 1 Official", "tier1"),
    2: ("Tier 2 Major outlet", "tier2"),
    3: ("Tier 3 Other", "tier3"),
    4: ("Tier 4 Unverified", "tier4"),
}

_NETWORK = (
    TimeoutError,
    ConnectionError,
    socket.timeout,
    socket.gaierror,
    urllib.error.URLError,
    requests.RequestException,
)

_GARLIC = (
    "‼️BREAKING‼️ Forwarded as received — AIIMS Delhi doctors have CONFIRMED "
    "that drinking 2 spoons of raw garlic juice every morning cures Type 2 "
    "diabetes in 30 days. Govt is hiding this to protect pharma companies!! "
    "Send to 10 groups 🙏🙏"
)

_TEXT_EXAMPLES = (
    {"label": "Garlic cure forward", "text": _GARLIC},
    {
        "label": "Metro and campus visit",
        "text": (
            "PM Narendra Modi inaugurated the Pink Line of Namma Metro in "
            "Bengaluru and then visited the IIM Bangalore campus"
        ),
    },
    {"label": "HSBC leaves Sri Lanka", "text": "HSBC Bank is completely withdrawing from Sri Lanka"},
    {"label": "Hot lemon water and coronavirus", "text": "Drinking hot water with lemon kills coronavirus"},
    {"label": "Hot lemon water and cancer", "text": "Drinking hot lemon water kills cancer cells"},
)


def _css() -> None:
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #f6f4f0; color: #1c2430; }
        [data-testid="stHeader"] { background: #f6f4f0; }
        .block-container { max-width: 860px; padding-top: 2.2rem; padding-bottom: 4rem; }
        h1, h2, h3, p, label, li { color: #1c2430; }
        h1 { letter-spacing: -0.03em; font-weight: 650; }
        .lede { color: #3d4450 !important; font-size: 1.05rem; margin: 0 0 1rem 0; }
        .verdict {
          display: inline-block; font-size: 1.65rem; font-weight: 750; letter-spacing: -0.02em;
          padding: 0.4rem 1rem; border-radius: 999px; margin: 0.15rem 0 0.7rem;
        }
        .verdict.false { background: #f8d0d0; color: #9f1239; }
        .verdict.misleading { background: #ffedd5; color: #c2410c; }
        .verdict.true { background: #dcfce7; color: #166534; }
        .verdict.unverifiable { background: #e5e7eb; color: #4b5563; }
        .verdict.other, .verdict.undecided { background: #f4f1ea; color: #44403c; border: 1px solid #e3dbd0; }
        .verdict.provisional { box-shadow: inset 0 0 0 2px currentColor; }
        .banner { border-radius: 12px; padding: 0.85rem 1rem; font-weight: 650; margin: 0.2rem 0 0.8rem; }
        .banner.awaiting { background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; }
        .chip {
          display: inline-block; margin: 0.15rem 0.4rem 0.15rem 0; padding: 0.28rem 0.72rem;
          border-radius: 999px; background: #fff; border: 1px solid #e3dbd0; color: #3f3428;
          font-size: 0.92rem;
        }
        .tier {
          display: inline-block; border-radius: 999px; padding: 0.1rem 0.5rem;
          font-size: 0.78rem; font-weight: 650; margin-left: 0.35rem;
        }
        .tier1 { background: #e5f4ee; color: #0f6b4c; }
        .tier2 { background: #e7f0fb; color: #0b4f8a; }
        .tier3 { background: #fff4df; color: #8a5a00; }
        .tier4 { background: #f1f2f4; color: #5c6570; }
        .main-badge {
          display: inline-block; margin-left: 0.4rem; padding: 0.05rem 0.45rem;
          border-radius: 999px; background: #1f3a5f; color: #fff; font-size: 0.75rem; font-weight: 650;
        }
        .quote { margin: 0.35rem 0 0.2rem; color: #1f2933; }
        .muted { color: #5c6570; font-size: 0.92rem; }
        .evidence-row { padding: 0.7rem 0; border-top: 1px solid #ece7e0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def friendly(exc: BaseException) -> str:
    """A sentence for the page. The exception text stays in the terminal.

    Quota is named on its own so a person does not keep retrying a day
    that is already spent. A search failure and a model failure look
    the same on the page: both mean this check did not finish, and the
    page must not show a verdict for it.
    """
    if isinstance(exc, LLMQuotaError):
        return "The checker has reached its limit for today. Please try again tomorrow."
    if isinstance(exc, SearchQuotaError):
        return "Web search has reached its limit. Please try again later."
    if isinstance(exc, (LLMError, SearchError, FetchError)) or isinstance(exc, _NETWORK):
        return "The checker couldn't finish this one. Please try again in a moment."
    if isinstance(exc, FileNotFoundError):
        return "The checker's memory of past fact-checks isn't available on this machine."
    if isinstance(exc, (ValueError, OSError, TypeError)):
        return "That input couldn't be read. Check the message, link, or screenshot and try again."
    return "Something went wrong while checking this. Please try again."


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan", "nat"}:
        return ""
    return text


def _shown_date(value) -> str:
    text = _clean(value)
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text or "date unknown"


def _safe_href(url: str) -> str | None:
    text = _clean(url)
    if not text.startswith(("http://", "https://")):
        return None
    if any(ch in text for ch in " \"'<>"):
        return None
    return text


def _normalize_url(text: str) -> str | None:
    raw = text.strip()
    if not raw or any(ch.isspace() for ch in raw):
        return None
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parts = urlsplit(raw)
    if not parts.netloc or "." not in parts.netloc:
        return None
    return raw


def _confidence_text(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    if 0.0 <= number <= 1.0:
        return f"{number:.0%}"
    return f"{number:.2f}"


def _latency_text(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.1f} s" if number < 10 else f"{round(number)} s"


def _verdict_kind(verdict) -> str:
    text = _clean(verdict).casefold()
    if text in {"false", "misleading", "true", "unverifiable", "other"}:
        return text
    return "undecided"


def _badge(verdict, *, provisional: bool) -> None:
    kind = _verdict_kind(verdict)
    label = _clean(verdict) or "Undecided"
    # The word is in the badge, not only in a caption. A red "False"
    # with no marker is what a finished card looks like, and a paused
    # run must not be readable as that.
    if provisional:
        label = f"Provisional · {label}"
    classes = f"verdict {kind}"
    if provisional:
        classes += " provisional"
    st.markdown(f"<div class='{classes}'>{html.escape(label)}</div>", unsafe_allow_html=True)


def _host(row: dict) -> str:
    host = _clean(row.get("site") or row.get("netloc")).lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown source"


def _norm_row(row: dict) -> dict:
    return {
        "id": _clean(row.get("id")),
        "site": _clean(row.get("site") or row.get("netloc")),
        "tier": row.get("tier"),
        "stance": _clean(row.get("stance")),
        "date": row.get("date") or row.get("published_date") or "",
        "link": _clean(row.get("link") or row.get("url")),
        "quote": _clean(row.get("quote") or row.get("display_quote")),
        "atomic_id": _clean(row.get("atomic_id")) or "A1",
    }


@st.cache_resource(show_spinner=False)
def load_checker():
    """One BGE-M3 and one memory index for this process.

    Streamlit reruns the script on every click. The module globals
    would usually survive that, and this cache is what still holds the
    model when a rerun imports the package again. A second encoder
    would use the memory twice and might not be the 256-token window
    the index was built with.
    """
    from misinfo_agent.memory import get_encoder, get_memory

    get_encoder()
    return get_memory()


@st.cache_data(show_spinner=False)
def _schools_screenshot() -> bytes:
    """The same one-sentence card the examples use. Pixels, not a file in git."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 460), "#f4f1ea")
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 70, 1136, 390), fill="white", outline="#d9d3c7", width=2)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    lines = ["All Bengaluru schools closed till Monday", "due to heavy rain"]
    y = 150
    for line in lines:
        width = draw.textlength(line, font=font)
        draw.text(((1200 - width) / 2, y), line, font=font, fill="#1c1c1c")
        y += 68
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _evidence_columns(rows: list[dict]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Evidence against**")
        _render_side([row for row in rows if row.get("stance") == "refutes"])
    with right:
        st.markdown("**Evidence for**")
        _render_side([row for row in rows if row.get("stance") == "supports"])


def _render_side(rows: list[dict]) -> None:
    if not rows:
        st.caption("None found.")
        return
    ranked = sorted(rows, key=lambda row: (int(row.get("tier") or 4), _host(row)))
    blocks = []
    for row in ranked:
        try:
            tier = int(row.get("tier"))
        except (TypeError, ValueError):
            tier = 4
        label, css = _TIER_LABELS.get(tier, _TIER_LABELS[4])
        host = html.escape(_host(row))
        href = _safe_href(row.get("link") or "")
        if href:
            source = f"<a href='{html.escape(href, quote=True)}'>{host}</a>"
        else:
            source = host
        quote = row.get("quote") or ""
        when = html.escape(_shown_date(row.get("date")))
        blocks.append(
            "<div class='evidence-row'>"
            f"<div>{source} <span class='tier {css}'>{html.escape(label)}</span></div>"
            f"<div class='muted'>{when}</div>"
            f"<div class='quote'>{html.escape(quote) if quote else 'No quote recorded.'}</div>"
            "</div>"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def _render_parts(parts: list[dict]) -> None:
    if not parts:
        return
    st.markdown("### Explanation")
    for part in parts:
        badge = "<span class='main-badge'>main claim</span>" if part.get("core") else ""
        text = _clean(part.get("text"))
        if text or badge:
            st.markdown(f"{html.escape(text)} {badge}", unsafe_allow_html=True)
        explanation = _clean(part.get("explanation"))
        if explanation:
            st.write(explanation)


def _render_evidence(rows: list[dict], parts: list[dict]) -> None:
    st.markdown("### Evidence")
    if not rows:
        st.caption("No evidence rows.")
        return
    order = []
    labels = {}
    for part in parts:
        atomic_id = _clean(part.get("atomic_id")) or "A1"
        if atomic_id not in order:
            order.append(atomic_id)
            labels[atomic_id] = _clean(part.get("text"))
    for row in rows:
        atomic_id = row.get("atomic_id") or "A1"
        if atomic_id not in order:
            order.append(atomic_id)
    many = len(order) > 1
    for atomic_id in order:
        if many:
            badge = ""
            for part in parts:
                if (_clean(part.get("atomic_id")) or "A1") == atomic_id and part.get("core"):
                    badge = " <span class='main-badge'>main claim</span>"
            title = labels.get(atomic_id) or atomic_id
            st.markdown(f"**{html.escape(title)}**{badge}", unsafe_allow_html=True)
        group = [row for row in rows if (row.get("atomic_id") or "A1") == atomic_id]
        _evidence_columns(group)


def _render_seen(seen) -> None:
    # match_type is on the card so this page does not reopen the
    # checkpoint to learn whether the archive hit was the same claim.
    kind = ""
    if isinstance(seen, dict):
        kind = _clean(seen.get("match_type"))
    st.markdown(f"### {past_factcheck_heading(kind)}")
    if not isinstance(seen, dict) or not (_clean(seen.get("link")) or _clean(seen.get("claim")) or _clean(seen.get("site"))):
        st.write("No past fact-check matched.")
        return
    site = _clean(seen.get("site")) or "Fact-check"
    when = _shown_date(seen.get("date"))
    st.markdown(f"**{html.escape(site)}**  ·  {html.escape(when)}")
    claim = _clean(seen.get("claim"))
    if claim:
        st.write(claim)
    href = _safe_href(seen.get("link") or "")
    if href:
        st.markdown(f"[Read the fact-check]({href})")


def _render_chips(flags) -> None:
    st.markdown("### Warning signs")
    chips = [_clean(flag) for flag in (flags or []) if _clean(flag)]
    if chips:
        st.markdown(
            "".join(f"<span class='chip'>{html.escape(chip)}</span>" for chip in chips),
            unsafe_allow_html=True,
        )
    else:
        st.write("No warning signs recorded.")
    st.caption("Warning signs don't decide whether a claim is true.")


def _render_details(queries, cost, run_id: str) -> None:
    with st.expander("Details"):
        st.markdown("**Search queries**")
        found = [_clean(item) for item in (queries or []) if _clean(item)]
        if found:
            for query in found:
                st.write(query)
        else:
            st.write("None.")
        st.markdown("**Cost**")
        cost = cost if isinstance(cost, dict) else {}
        st.write(f"LLM calls: {int(cost.get('llm_calls') or 0)}")
        st.write(f"Searches: {int(cost.get('searches') or 0)}")
        st.markdown("**Run id**")
        st.code(run_id or "—")


def _render_correction(text: str, *, provisional: bool) -> None:
    message = _clean(text)
    if not message:
        return
    st.markdown("### Correction")
    if provisional:
        st.caption("Provisional. A reviewer has not accepted this wording yet.")
    # st.code has a copy button. A disabled text area looks like a box
    # and does not offer one.
    st.code(message, language=None)


def _render_truth(card: dict) -> None:
    true_version = _clean(card.get("true_version"))
    found = _clean(card.get("what_we_found"))
    if true_version:
        st.markdown("### What is actually true")
        st.write(true_version)
        return
    if found:
        # true_version is only written for True, False, and Misleading.
        # On an unverifiable claim the sources still said something,
        # and that text is what_we_found, not a correction.
        st.markdown("### What we found")
        st.write(found)
        return
    st.markdown("### What is actually true")
    st.caption("Not recorded.")


def _render_result(result: dict, *, run_id: str = "") -> None:
    """A finished card. The verdict is the one the rules stored."""
    claim = _clean(result.get("claim"))
    if not claim:
        for part in result.get("parts") or []:
            if isinstance(part, dict) and part.get("core") and _clean(part.get("text")):
                claim = _clean(part.get("text"))
                break
    if claim:
        st.markdown("**Claim**")
        st.write(claim)
    _badge(result.get("verdict"), provisional=False)
    st.write(f"Confidence {_confidence_text(result.get('confidence'))}")
    action = _clean(result.get("human_action"))
    if action:
        st.caption(f"Human action: {action}")
    if _clean(result.get("note")):
        st.write(result.get("note"))
    _render_correction(result.get("correction_message") or "", provisional=False)
    _render_truth(result)
    parts = [part for part in (result.get("parts") or []) if isinstance(part, dict)]
    if not parts:
        parts = []
        for item in result.get("explanations") or []:
            if isinstance(item, dict):
                parts.append({
                    "atomic_id": item.get("atomic_id") or "A1",
                    "text": "",
                    "core": True,
                    "explanation": item.get("explanation") or "",
                })
    _render_parts(parts)
    rows = [_norm_row(row) for row in (result.get("evidence") or []) if isinstance(row, dict)]
    _render_evidence(rows, parts)
    _render_seen(result.get("seen_before"))
    _render_chips(result.get("warning_signs"))
    st.markdown("### Priority")
    st.write(_clean(result.get("priority")) or "—")
    _render_details(result.get("queries"), result.get("cost"), _clean(result.get("run_id")) or run_id)


def _render_paused(review: dict, thread_id: str) -> None:
    """The check page when the run parked at box 13.

    There is no result card yet: output runs after a person answers.
    The badge says provisional so it cannot be read as the stored verdict.
    """
    st.markdown(
        "<div class='banner awaiting'>Awaiting human review</div>",
        unsafe_allow_html=True,
    )
    st.caption("This verdict is provisional. It is in the review queue and is not final.")
    claim = _clean(review.get("claim"))
    if claim:
        st.markdown("**Claim**")
        st.write(claim)
    _badge(review.get("verdict"), provisional=True)
    st.write(f"Confidence {_confidence_text(review.get('confidence'))}")
    why = _clean(review.get("why"))
    if why:
        st.write(why)
    _render_correction(review.get("correction_message") or "", provisional=True)
    _render_truth(review)
    rows = [_norm_row(row) for row in (review.get("evidence") or []) if isinstance(row, dict)]
    _render_evidence(rows, [])
    _render_chips(review.get("warning_signs"))
    st.markdown("### Priority")
    st.write(_clean(review.get("priority")) or "—")
    _render_details(review.get("queries"), review.get("cost"), thread_id)


def _apply_example() -> None:
    pending = st.session_state.pop("pending_fill", None)
    if not pending:
        return
    # Applied before the widgets exist. Setting a widget key after the
    # widget is drawn is rejected, so the example click only stores a
    # request and the next run fills the field.
    st.session_state.input_mode = pending["mode"]
    if pending["mode"] == "Text":
        st.session_state.claim_text = pending["text"]
        st.session_state.example_image = None
    elif pending["mode"] == "Screenshot":
        st.session_state.example_image = pending["image"]
        st.session_state.caption_text = ""


def _page_check() -> None:
    st.session_state.setdefault("input_mode", "Text")
    st.session_state.setdefault("claim_text", "")
    st.session_state.setdefault("link_url", "")
    st.session_state.setdefault("caption_text", "")
    _apply_example()
    st.title("Misinformation Checker")
    st.markdown(
        "<p class='lede'>Paste a message, a link, or a screenshot. The checker returns a verdict, or pauses for review.</p>",
        unsafe_allow_html=True,
    )
    mode = st.segmented_control(
        "Input type",
        ["Text", "Link", "Screenshot"],
        key="input_mode",
        label_visibility="collapsed",
        width="stretch",
        required=True,
    )
    if mode not in {"Text", "Link", "Screenshot"}:
        mode = "Text"
    uploaded = None
    if mode == "Text":
        st.text_area(
            "Message",
            key="claim_text",
            height=220,
            placeholder="Paste a claim or a WhatsApp forward",
        )
    elif mode == "Link":
        st.text_input("Link", key="link_url", placeholder="https://")
    else:
        if st.session_state.get("example_image"):
            st.image(st.session_state.example_image, width=420)
            st.caption("Example screenshot ready.")
            if st.button("Remove example screenshot"):
                st.session_state.example_image = None
                st.rerun()
        uploaded = st.file_uploader(
            "Screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            key="screenshot_upload",
        )
        st.text_input(
            "Caption (optional)",
            key="caption_text",
            placeholder="Add a caption if the image needs one",
        )
    exclude = st.toggle(
        "Exclude fact-check websites",
        key="exclude_factcheck",
        value=True,
        help="The system has to find the evidence itself instead of copying a fact-checker.",
    )
    if st.button("Check", type="primary", width="stretch"):
        image = None
        url = ""
        text = ""
        problem = None
        if mode == "Text":
            text = (st.session_state.get("claim_text") or "").strip()
            if not text:
                problem = "Paste a message first."
        elif mode == "Link":
            url = _normalize_url(st.session_state.get("link_url") or "")
            if url is None:
                problem = "That doesn't look like a link."
        else:
            if st.session_state.get("example_image"):
                image = st.session_state.example_image
            elif uploaded is not None:
                image = uploaded.getvalue()
            if not image:
                problem = "Add a screenshot first."
        if problem:
            st.session_state.form_error = problem
        else:
            st.session_state.form_error = None
            _start_job({
                "page": "Check",
                "action": "check",
                "mode": mode,
                "text": text,
                "url": url or "",
                "caption": st.session_state.get("caption_text") or "",
                "image": image,
                "exclude": bool(exclude),
            })
    st.caption("A typical check takes 30–60 seconds. One check runs at a time.")
    if st.session_state.get("form_error"):
        st.caption(st.session_state.form_error)
    with st.expander("Examples"):
        st.caption("These fill the box. They do not run until you press Check.")
        for index, example in enumerate(_TEXT_EXAMPLES):
            if st.button(example["label"], key=f"example_{index}", width="stretch"):
                st.session_state.pending_fill = {"mode": "Text", "text": example["text"]}
                st.rerun()
        if st.button("Schools closed (screenshot)", key="example_screenshot", width="stretch"):
            st.session_state.pending_fill = {"mode": "Screenshot", "image": _schools_screenshot()}
            st.rerun()
    _show_error("Check")
    outcome = st.session_state.get("check_outcome")
    if not outcome:
        return
    st.divider()
    if outcome.get("kind") == "paused":
        _render_paused(outcome.get("review") or {}, outcome.get("thread_id") or "")
    else:
        _render_result(outcome.get("result") or {}, run_id=outcome.get("thread_id") or "")


def _queue_rows() -> list[dict]:
    from misinfo_agent.review import list_pending

    return list_pending(config.DATA_DIR / "audit.db")


def _page_review() -> None:
    st.title("Review queue")
    st.caption("Paused runs, High priority first. Approving, editing, rejecting, or asking for more evidence resumes that run.")
    _show_error("Review queue")
    outcome = st.session_state.get("review_outcome")
    if isinstance(outcome, dict) and outcome.get("status") == "done":
        st.success("Review saved. The card below is the finished run.")
        _render_result(outcome.get("result") or {}, run_id=outcome.get("thread_id") or "")
        st.divider()
    elif isinstance(outcome, dict) and outcome.get("status") == "interrupt":
        st.info("Still awaiting review. The updated card is in the queue.")
    try:
        rows = _queue_rows()
    except Exception as exc:
        log.exception("review queue failed")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.error(friendly(exc))
        return
    if not rows:
        st.write("Nothing is waiting for review.")
        return
    for row in rows:
        thread_id = row.get("thread_id") or ""
        payload = {}
        try:
            payload = paused_review(thread_id)
        except Exception as exc:
            # The table row is enough to approve. A checkpoint that
            # cannot be read should not hide the pause.
            log.exception("paused review unreadable")
            print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
            payload = {}
        with st.container(border=True):
            st.markdown("**Claim**")
            st.write(row.get("claim") or "—")
            _badge(row.get("provisional_verdict"), provisional=True)
            st.write(
                f"Confidence {_confidence_text(row.get('confidence'))}"
                f"  ·  Priority {row.get('priority') or '—'}"
            )
            reason = _clean(row.get("reason")) or _clean(payload.get("why"))
            if reason:
                st.write(reason)
            evidence = payload.get("evidence") if isinstance(payload, dict) else []
            table = []
            for item in evidence or []:
                if not isinstance(item, dict):
                    continue
                norm = _norm_row(item)
                try:
                    tier = int(norm.get("tier"))
                except (TypeError, ValueError):
                    tier = ""
                else:
                    tier = _TIER_LABELS.get(tier, _TIER_LABELS[4])[0]
                table.append({
                    "Id": norm["id"],
                    "Site": _host(norm),
                    "Tier": tier,
                    "Stance": norm["stance"],
                    "Date": _shown_date(norm["date"]),
                    "Link": norm["link"],
                    "Quote": norm["quote"],
                })
            st.markdown("**Evidence**")
            if table:
                st.dataframe(table, hide_index=True, width="stretch")
            else:
                st.caption("No evidence rows.")
            approve, reject, more = st.columns(3)
            with approve:
                if st.button("Approve", key=f"approve_{thread_id}", type="primary", width="stretch"):
                    _start_job({
                        "page": "Review queue",
                        "action": "resume",
                        "thread_id": thread_id,
                        "resume": {"action": "approve"},
                    })
            with reject:
                if st.button("Reject", key=f"reject_{thread_id}", width="stretch"):
                    _start_job({
                        "page": "Review queue",
                        "action": "resume",
                        "thread_id": thread_id,
                        "resume": {"action": "reject"},
                    })
            with more:
                if st.button("More evidence", key=f"more_{thread_id}", width="stretch"):
                    _start_job({
                        "page": "Review queue",
                        "action": "resume",
                        "thread_id": thread_id,
                        "resume": {"action": "more_evidence"},
                    })
            with st.expander("Edit"):
                # Set once, before the widgets. Passing index= or value=
                # on later reruns would put the provisional verdict back
                # and the reviewer could not keep a different choice.
                # The box starts from the draft so an edit that only
                # changes the label does not wipe the correction.
                verdict_key = f"verdict_{thread_id}"
                correction_key = f"correction_{thread_id}"
                current = _clean(row.get("provisional_verdict"))
                if verdict_key not in st.session_state:
                    st.session_state[verdict_key] = current if current in _VERDICTS else "False"
                if correction_key not in st.session_state:
                    st.session_state[correction_key] = _clean(payload.get("correction_message"))
                verdict = st.selectbox("Verdict", list(_VERDICTS), key=verdict_key)
                correction = st.text_area("Correction", key=correction_key, height=100)
                if st.button("Save edit", key=f"save_{thread_id}", width="stretch"):
                    _start_job({
                        "page": "Review queue",
                        "action": "resume",
                        "thread_id": thread_id,
                        "resume": {
                            "action": "edit",
                            "verdict": verdict,
                            "correction": correction,
                        },
                    })


def _page_history() -> None:
    st.title("History")
    st.caption("Finished runs, newest first.")
    _show_error("History")
    from misinfo_agent.audit import list_runs

    try:
        rows = list_runs(config.DATA_DIR / "audit.db")
    except Exception as exc:
        log.exception("history failed")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.error(friendly(exc))
        return
    if not rows:
        st.write("No checks have been recorded yet.")
        return
    table = []
    for row in rows:
        calls = row.get("llm_calls")
        table.append({
            "Time": row.get("created_at") or "",
            "Claim": row.get("claim") or "",
            "Verdict": row.get("verdict") or "",
            "Confidence": _confidence_text(row.get("confidence")),
            "Decision": row.get("decision") or "",
            "Human action": row.get("human_action") or "",
            "Calls": "" if calls is None else calls,
            "Latency": _latency_text(row.get("latency_s")),
        })
    st.dataframe(table, hide_index=True, width="stretch")


def _show_error(page: str) -> None:
    message = st.session_state.get("error")
    if message and st.session_state.get("error_page") == page:
        st.error(message)


def _start_job(job: dict) -> None:
    job["record"] = bool(st.session_state.get("record_submissions", False))
    st.session_state.job = job
    st.session_state.busy = True
    st.session_state.error = None
    st.rerun()


def _stage_image(data: bytes):
    """A path for run_pipeline. The checkpointer must not store the pixels.

    Ingest reads the file and writes its own copy under UPLOADS_DIR.
    This submit file is removed when the call returns. The path is not
    shown on the card.
    """
    folder = config.UPLOADS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"submit-{uuid.uuid4().hex}.png"
    path.write_bytes(data)
    return path


def _apply_run_flags(job: dict) -> None:
    # Captured when the button was pressed. The sidebar switch is not
    # on screen during the run, and a rerun must not flip reach counting
    # halfway through ingest.
    config.RECORD_SUBMISSIONS = bool(job.get("record"))
    if job.get("action") == "check":
        config.BLOCK_FACTCHECK_SITES = bool(job.get("exclude"))
        return
    # More evidence searches again. It has to keep the block the first
    # search used. A restarted process loses the map; excluding is the
    # check page's default, and the honest number.
    remembered = st.session_state.exclude_by_thread.get(job.get("thread_id"), True)
    config.BLOCK_FACTCHECK_SITES = bool(remembered)


def _run_job() -> None:
    job = st.session_state.get("job") or {}
    action = job.get("action")
    if action == "resume":
        label = {
            "approve": "Saving the approval",
            "edit": "Saving the edit",
            "reject": "Saving the rejection",
            "more_evidence": "Gathering more evidence",
        }.get((job.get("resume") or {}).get("action"), "Saving the review")
    else:
        label = "Checking"
    image_path = None
    try:
        _apply_run_flags(job)
        with st.status(label, expanded=True) as status:
            status.write("One check runs at a time. This usually takes 30–60 seconds.")
            if action == "resume":
                # run_pipeline sends Command(resume=...) to the checkpointer
                # on this thread id. The page does not build a second way
                # to resume, or a pause could be answered twice.
                out = run_pipeline(resume=job.get("resume") or {}, thread_id=job.get("thread_id"))
            else:
                if job.get("mode") == "Link":
                    out = run_pipeline(job.get("url") or "", input_type="url")
                elif job.get("mode") == "Screenshot":
                    image_path = _stage_image(job.get("image") or b"")
                    out = run_pipeline(
                        str(image_path),
                        input_type="image",
                        caption=job.get("caption") or None,
                    )
                else:
                    out = run_pipeline(job.get("text") or "", input_type="text")
            status.update(label="Finished", state="complete", expanded=False)
        if out.get("status") not in {"done", "interrupt"}:
            raise RuntimeError("the checker returned an unexpected result")
        thread_id = out.get("thread_id") or ""
        if action == "check" and thread_id:
            st.session_state.exclude_by_thread[thread_id] = bool(job.get("exclude"))
        if action == "resume":
            st.session_state.review_outcome = out
        elif out.get("status") == "interrupt":
            st.session_state.check_outcome = {
                "kind": "paused",
                "thread_id": thread_id,
                "review": out.get("review") or {},
            }
        else:
            st.session_state.check_outcome = {
                "kind": "done",
                "thread_id": thread_id,
                "result": out.get("result") or {},
            }
        verdict = (out.get("result") or out.get("review") or {}).get("verdict")
        print(f"APP {out.get('status')} thread={thread_id} verdict={verdict}", flush=True)
        st.session_state.error = None
    except Exception as exc:
        log.exception("check failed")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.session_state.error = friendly(exc)
        st.session_state.error_page = job.get("page") or "Check"
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        # Back to the switch. A failed run must not leave reach counting on.
        config.RECORD_SUBMISSIONS = bool(st.session_state.get("record_submissions", False))
        st.session_state.job = None
        st.session_state.busy = False
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Misinformation Checker", layout="centered")
    _css()
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("job", None)
    st.session_state.setdefault("exclude_by_thread", {})
    st.session_state.setdefault("check_outcome", None)
    st.session_state.setdefault("review_outcome", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("form_error", None)
    st.session_state.setdefault("example_image", None)
    busy = bool(st.session_state.busy)
    with st.sidebar:
        st.markdown("**Pages**")
        page = st.radio(
            "Page",
            ["Check", "Review queue", "History"],
            key="page",
            disabled=busy,
            label_visibility="collapsed",
        )
        st.toggle(
            "Count this check as a submission",
            key="record_submissions",
            value=False,
            disabled=busy,
            help=(
                "Off unless you turn it on. A check here is not another person "
                "forwarding the message. Priority reads that count as reach."
            ),
        )
    if not busy:
        config.RECORD_SUBMISSIONS = bool(st.session_state.get("record_submissions", False))
    try:
        with st.spinner("Loading the checker…"):
            load_checker()
    except Exception as exc:
        log.exception("checker failed to load")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.error(friendly(exc))
        return
    if busy and st.session_state.get("job"):
        _run_job()
        return
    if page == "Review queue":
        _page_review()
    elif page == "History":
        _page_history()
    else:
        _page_check()


if __name__ == "__main__":
    main()
