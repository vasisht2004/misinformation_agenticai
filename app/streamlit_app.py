"""The checker people use. One claim in, a card or a review pause out.

Started with `streamlit run app/streamlit_app.py`. The page calls the
pipeline and draws what that already decided. It does not score a
verdict and it does not search on its own.

Four pages share one process. A run sets a busy flag and the next pass
is the run, so a second claim cannot start beside it. BGE-M3 and the
memory index load once per process, and only when a real check starts:
a fixture URL must not pay for the encoder or touch the shared index.

`?fixture=` draws a recorded card and does not call the pipeline.
SEARCH_CACHE_ONLY defaults on so a demo does not spend the remaining
November credits. The counter over-counts; the cap is 1000 until 1 Dec.
"""

from __future__ import annotations

import io
import logging
import os
import socket
import sqlite3
import sys
import urllib.error
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import requests
import streamlit as st

import misinfo_agent.config as config
from app import components, theme
from app.pipeline_stream import stream_pipeline
from misinfo_agent.fetch import FetchError
from misinfo_agent.llm import LLMError, LLMQuotaError
from misinfo_agent.search import MONTHLY_COUNT_PATH, SearchError, SearchQuotaError

# Off before any widget. A rerun that fails before the switch is drawn
# must not inherit "on" from a previous experiment in this process.
# Box 11 reads the submission count as reach. A check is not a forward.
config.RECORD_SUBMISSIONS = False

log = logging.getLogger("misinfo_app")

_VERDICTS = ("False", "Misleading", "True", "Unverifiable", "Other")
_PAGES = {"check", "review", "history", "how"}
_STANCES = ("Supports", "Refutes", "Neutral")
_TIERS = ("Tier 1", "Tier 2", "Tier 3", "Tier 4")

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

_VERDICT_RANGE = ["False", "Misleading", "True", "Unverifiable", "Other", "Withheld", "Undecided"]
_VERDICT_COLORS = ["#E05A47", "#E8A33D", "#3FB27F", "#7C8AA5", "#5B6475", "#5B6475", "#5B6475"]
_TIER_RANGE = ["T1", "T2", "T3", "T4"]
_TIER_COLORS = ["#D4A84B", "#C5CDD6", "#8E99A8", "#5C6778"]


def friendly(exc: BaseException) -> str:
    """A sentence for the page. The exception text stays in the terminal.

    Quota is named on its own so a person does not keep retrying a day
    or a month that is already spent. A search failure and a model
    failure look the same on the page: both mean this check did not
    finish, and the page must not show a verdict for it.
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


def _query_value(name: str) -> str:
    raw = st.query_params.get(name, "")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    return "" if raw is None else str(raw).strip()


def _page() -> str:
    page = _query_value("page") or "check"
    return page if page in _PAGES else "check"


def _fixture() -> str:
    return _query_value("fixture")


@st.cache_resource(show_spinner=False)
def load_checker():
    """One BGE-M3 and one memory index for this process.

    Streamlit reruns the script on every click. The cache is what still
    holds the model when a rerun imports the package again. A second
    encoder would use the memory twice and might not be the 256-token
    window the index was built with. Called from a real check only: a
    fixture must not load it.
    """
    from misinfo_agent.memory import get_encoder, get_memory

    get_encoder()
    return get_memory()


def _review_api():
    """The new review helpers, or nothing when this process is on the old module.

    apply_review and get_card are landing with the engine. Importing
    them by name would break the page until that merge. getattr is the
    same idea as a try/except around the import: absent means use the
    pause-and-resume path the graph already has.
    """
    try:
        from misinfo_agent import review as review_mod
    except Exception:
        return None, None, None
    return (
        getattr(review_mod, "list_pending", None),
        getattr(review_mod, "apply_review", None),
        getattr(review_mod, "get_card", None),
    )


def _call_list_pending():
    list_pending, _apply, _card = _review_api()
    if list_pending is None:
        return []
    try:
        rows = list_pending()
    except TypeError:
        rows = list_pending(config.DATA_DIR / "audit.db")
    return rows if isinstance(rows, list) else []


def _pending_count() -> int:
    try:
        return len(_call_list_pending())
    except Exception:
        log.exception("pending count failed")
        return 0


def _get_card(thread_id: str) -> dict:
    """The stored card for one thread. Empty when it cannot be read.

    A lock on the checkpoint is a dash, not a crash: the other process
    may be writing the same file. The new get_card is preferred. The
    old path reads the pause summary, then the checkpoint's result.
    """
    if not thread_id:
        return {}
    _list, _apply, get_card = _review_api()
    if get_card is not None:
        try:
            card = get_card(thread_id)
        except Exception:
            log.exception("get_card failed")
            card = None
        if isinstance(card, dict):
            return card
    try:
        from misinfo_agent.graph import build_graph, paused_review

        paused = paused_review(thread_id)
        if isinstance(paused, dict) and paused:
            return paused
        graph = build_graph()
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
        values = getattr(snap, "values", None) or {}
        if isinstance(values, dict):
            return values
    except Exception:
        log.exception("checkpoint card failed")
    return {}


def _apply_review(job: dict):
    _list, apply_review, _card = _review_api()
    action = job.get("review_action") or ""
    thread_id = job.get("thread_id") or ""
    if apply_review is not None:
        kwargs = {}
        if action == "edit":
            kwargs["verdict"] = job.get("verdict")
            kwargs["correction"] = job.get("correction")
            if job.get("note"):
                kwargs["note"] = job.get("note")
        return apply_review(thread_id, action, **kwargs)
    # The old review() only resumes through the graph. The payload keys
    # are the ones that function already reads: action, verdict,
    # correction, note.
    from misinfo_agent.graph import run_pipeline

    resume = {"action": action}
    if action == "edit":
        if job.get("verdict") is not None:
            resume["verdict"] = job.get("verdict")
        if job.get("correction") is not None:
            resume["correction"] = job.get("correction")
        if job.get("note"):
            resume["note"] = job.get("note")
    return run_pipeline(resume=resume, thread_id=thread_id)


def _search_used() -> int:
    """This UTC month's Tavily count. A missing file is zero, not an error.

    The file is the bill. Reading it does not increment it. A lock or a
    half-written file shows zero rather than a guessed number.
    """
    try:
        from datetime import datetime, timezone

        payload = __import__("json").loads(MONTHLY_COUNT_PATH.read_text(encoding="utf-8"))
        key = datetime.now(timezone.utc).strftime("%Y-%m")
        return int(payload.get(key, 0))
    except Exception:
        return 0


def _model_used() -> int:
    try:
        from misinfo_agent.llm import calls_today

        return int(calls_today())
    except Exception:
        return 0


def _quota_html() -> str:
    return components.quota_html(
        _model_used(),
        int(config.LLM_DAILY_MAX),
        _search_used(),
        int(config.TAVILY_MONTHLY_MAX),
    )


def _apply_cache_only() -> None:
    """Demo mode is on unless the switch says otherwise.

    The search client reads the environment at call time. Setting it
    here, and not only inside the run, means a stray call in this
    process still misses closed rather than billing a search. The
    monthly cap is already the reason the default is on.
    """
    if st.session_state.get("demo_cache_only", True):
        os.environ["SEARCH_CACHE_ONLY"] = "1"
    else:
        os.environ.pop("SEARCH_CACHE_ONLY", None)


def _normalize_url(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw or any(ch.isspace() for ch in raw):
        return None
    if not raw.lower().startswith(("http://", "https://")):
        raw = "https://" + raw
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc or "." not in parts.netloc:
        return None
    return raw


def _init_state() -> None:
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("job", None)
    st.session_state.setdefault("check_view", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("error_page", "check")
    st.session_state.setdefault("form_error", None)
    st.session_state.setdefault("example_image", None)
    st.session_state.setdefault("demo_cache_only", True)
    st.session_state.setdefault("record_submissions", False)
    st.session_state.setdefault("input_mode", "Text")
    st.session_state.setdefault("claim_text", "")
    st.session_state.setdefault("link_url", "")
    st.session_state.setdefault("caption_text", "")
    st.session_state.setdefault("review_toast", None)
    st.session_state.setdefault("review_result", None)


def _apply_example() -> None:
    pending = st.session_state.pop("pending_fill", None)
    if not pending:
        return
    # Applied before the widgets exist. Setting a widget key after the
    # widget is drawn is rejected, so the chip only stores a request.
    st.session_state.input_mode = pending["mode"]
    if pending["mode"] == "Text":
        st.session_state.claim_text = pending["text"]
        st.session_state.example_image = None
    elif pending["mode"] == "Image":
        st.session_state.example_image = pending["image"]
        st.session_state.caption_text = pending.get("caption") or ""
    elif pending["mode"] == "Link":
        st.session_state.link_url = pending["url"]


@st.cache_data(show_spinner=False)
def _schools_screenshot() -> bytes:
    """The same one-sentence card the examples use. Pixels, not a file in git."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 460), "#0F1B36")
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 70, 1136, 390), fill="#0A0B0E", outline="#1E2A45", width=2)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    lines = ["All Bengaluru schools closed till Monday", "due to heavy rain"]
    y = 150
    for line in lines:
        width = draw.textlength(line, font=font)
        draw.text(((1200 - width) / 2, y), line, font=font, fill="#F2F0EA")
        y += 68
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _start_job(job: dict) -> None:
    if st.session_state.get("busy"):
        return
    job["cache_only"] = bool(st.session_state.get("demo_cache_only", True))
    job["record"] = bool(st.session_state.get("record_submissions", False))
    st.session_state.job = job
    st.session_state.busy = True
    st.session_state.error = None
    st.rerun()


def _stage_image(data: bytes) -> Path:
    """A path for the pipeline. The checkpointer must not store the pixels."""
    folder = config.UPLOADS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"submit-{uuid.uuid4().hex}.png"
    path.write_bytes(data)
    return path


def _run_flags(job: dict) -> None:
    config.RECORD_SUBMISSIONS = bool(job.get("record"))
    if job.get("cache_only", True):
        os.environ["SEARCH_CACHE_ONLY"] = "1"
    else:
        os.environ.pop("SEARCH_CACHE_ONLY", None)


def _clear_flags() -> None:
    config.RECORD_SUBMISSIONS = False
    _apply_cache_only()


def _run_job() -> None:
    job = st.session_state.get("job") or {}
    try:
        _run_flags(job)
        if job.get("action") == "review":
            _run_review(job)
        else:
            _run_check(job)
    except Exception as exc:
        log.exception("check failed")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.session_state.error = friendly(exc)
        st.session_state.error_page = job.get("page") or "check"
    finally:
        _clear_flags()
        st.session_state.job = None
        st.session_state.busy = False
    st.rerun()


def _run_check(job: dict):
    with st.spinner("Loading the checker…"):
        load_checker()
    outcome: dict = {}
    ran: list = []
    placeholder = st.empty()
    placeholder.markdown(
        components.stepper_html([], {}, settled=False),
        unsafe_allow_html=True,
    )
    # Removed even when the stream raises. The checkpointer must not
    # keep the pixels, and a failed run must not leave the file behind.
    image_path = None
    try:
        if job.get("mode") == "Link":
            raw, kind = job.get("url") or "", "url"
        elif job.get("mode") == "Image":
            image_path = _stage_image(job.get("image") or b"")
            raw, kind = str(image_path), "image"
        else:
            raw, kind = job.get("text") or "", "text"
        for name, elapsed in stream_pipeline(
            raw,
            input_type=kind,
            caption=job.get("caption") or None,
            outcome=outcome,
        ):
            ran.append((name, elapsed))
            placeholder.markdown(
                components.stepper_html(ran, outcome.get("state") or {}, settled=False),
                unsafe_allow_html=True,
            )
        if outcome.get("status") not in {"done", "interrupt"}:
            raise RuntimeError("the checker returned an unexpected result")
        st.session_state.check_view = {
            "status": outcome.get("status"),
            "state": outcome.get("state") or {},
            "result": outcome.get("result") or {},
            "review": outcome.get("review") or {},
            "ran": ran,
            "thread_id": outcome.get("thread_id") or "",
            "mode": outcome.get("mode") or "",
        }
        verdict = (outcome.get("result") or outcome.get("review") or {}).get("verdict")
        print(
            f"APP {outcome.get('status')} thread={outcome.get('thread_id')} verdict={verdict}",
            flush=True,
        )
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def _run_review(job: dict) -> None:
    with st.status("Saving the review", expanded=False):
        result = _apply_review(job)
    st.session_state.review_toast = job.get("review_action") or "saved"
    st.session_state.review_result = result if isinstance(result, dict) else {}


def _show_error(page: str) -> None:
    message = st.session_state.get("error")
    if message and st.session_state.get("error_page") == page:
        st.error(message)


def _cite_href(fixture: str, ident: str) -> str:
    parts = ["page=check"]
    if fixture:
        parts.append("fixture=" + fixture)
    parts.append("cite=" + ident)
    safe = re_ident(ident)
    return "?" + "&".join(parts) + "#ev-" + safe


def re_ident(ident: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_-]", "", ident or "row")


def _dark(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#F2F0EA",
            titleColor="#A7B0C2",
            gridColor="#1E2A45",
            domainColor="#1E2A45",
        )
        .configure_legend(labelColor="#F2F0EA", titleColor="#A7B0C2")
    )


def _filter_rows(rows, stances, tiers, highlight: str) -> list[dict]:
    allowed_stance = {item.lower() for item in (stances or [])}
    allowed_tier = set()
    for item in tiers or []:
        text = str(item).lower().replace(" ", "")
        if text.startswith("tier") and text[-1:].isdigit():
            allowed_tier.add(int(text[-1]))
    kept = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if highlight and str(row.get("id") or "") == highlight:
            kept.append(row)
            continue
        stance = str(row.get("stance") or "").lower()
        try:
            tier = int(row.get("tier"))
        except (TypeError, ValueError):
            tier = 4
        if stance in allowed_stance and tier in allowed_tier:
            kept.append(row)
    return kept


def _render_card(model: dict, *, fixture: str, highlight: str) -> None:
    ids = [row.get("id") for row in model.get("evidence") or [] if isinstance(row, dict) and row.get("id")]
    # C1 is the reused fact-check. A similar or none hit is not that chip.
    if str(model.get("match_type") or "").lower() == "exact":
        ids.append("C1")

    def href_for(ident: str) -> str:
        return _cite_href(fixture, ident)

    banner = components.search_banner_html(model)
    if banner:
        st.markdown(banner, unsafe_allow_html=True)
    st.markdown(components.verdict_html(model), unsafe_allow_html=True)
    image_block = components.image_check_html(model)
    if image_block:
        st.markdown(image_block, unsafe_allow_html=True)
    # The evidence lean is computed from the rows. It sits above the
    # model lean, which stays the dashed, weaker guess.
    source_lean = components.evidence_lean_html(model)
    if source_lean:
        st.markdown(source_lean, unsafe_allow_html=True)
    lean = components.lean_html(model)
    if lean:
        st.markdown(lean, unsafe_allow_html=True)
    seen = components.seen_html(model)
    if seen:
        st.markdown(seen, unsafe_allow_html=True)
    explanation = components.explanation_html(model.get("explanation", ""), ids, href_for)
    if explanation:
        st.markdown(components.eyebrow("Explanation"), unsafe_allow_html=True)
        st.markdown(explanation, unsafe_allow_html=True)
    rows = model.get("evidence") or []
    st.markdown(components.eyebrow("Evidence"), unsafe_allow_html=True)
    if rows:
        stance_pick = st.pills(
            "Stance",
            list(_STANCES),
            selection_mode="multi",
            default=list(_STANCES),
            key="stance_filter",
        )
        tier_pick = st.pills(
            "Tier",
            list(_TIERS),
            selection_mode="multi",
            default=list(_TIERS),
            key="tier_filter",
        )
        # None means the widget has not answered yet. An empty list is
        # a reader who turned every pill off, and that should hide rows.
        if stance_pick is None:
            stance_pick = list(_STANCES)
        else:
            stance_pick = list(stance_pick)
        if tier_pick is None:
            tier_pick = list(_TIERS)
        else:
            tier_pick = list(tier_pick)
        shown = _filter_rows(rows, stance_pick, tier_pick, highlight)
        frame = components.stance_tier_rows(shown)
        if frame:
            chart = (
                alt.Chart(pd.DataFrame(frame))
                .mark_bar()
                .encode(
                    x=alt.X("stance:N", sort=["supports", "refutes", "neutral"], title="Stance"),
                    y=alt.Y("groups:Q", title="Independent groups"),
                    color=alt.Color(
                        "tier:N",
                        scale=alt.Scale(domain=_TIER_RANGE, range=_TIER_COLORS),
                        title="Tier",
                    ),
                )
                .properties(height=220)
            )
            st.altair_chart(_dark(chart), width="stretch", theme=None)
        st.markdown(
            components.evidence_html(
                shown,
                highlight=highlight,
                core_detail_only_ids=model.get("core_detail_only_ids") or [],
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No evidence rows.")
    st.markdown(components.eyebrow("Warning signs"), unsafe_allow_html=True)
    st.markdown(components.flags_html(model.get("red_flags") or []), unsafe_allow_html=True)
    true_version = str(model.get("true_version") or "").strip()
    found = str(model.get("what_we_found") or "").strip()
    if true_version:
        st.markdown(components.eyebrow("What's actually true"), unsafe_allow_html=True)
        st.markdown(f"<p>{components.esc(true_version)}</p>", unsafe_allow_html=True)
    elif found:
        st.markdown(components.eyebrow("What we found"), unsafe_allow_html=True)
        st.markdown(f"<p>{components.esc(found)}</p>", unsafe_allow_html=True)
    correction = components.correction_text(model)
    if correction:
        st.markdown(components.eyebrow("Correction"), unsafe_allow_html=True)
        if model.get("provisional"):
            st.caption("Provisional. A reviewer has not accepted this wording yet.")
        # st.code has a copy button. The text is not interpreted as HTML.
        st.code(correction, language=None)
    queries = model.get("queries") or []
    with st.expander("Queries & cost"):
        if queries:
            for query in queries:
                st.write(query)
        else:
            st.write("No queries recorded.")
        st.write(f"Model calls: {_count_text(model.get('llm_calls'))}")
        st.write(f"Searches: {_count_text(model.get('searches'))}")
        st.write(f"Latency: {_latency_text(model.get('latency_s'))}")


def _count_text(value) -> str:
    if value is None or value == "":
        return "—"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "—"


def _latency_text(value) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.1f} s" if number < 10 else f"{round(number)} s"


def _interrupt_model(saved: dict) -> dict:
    """A pause is not a finished card, even when the new flag has not landed yet.

    The engine may already set provisional and alert_pending_approval.
    When those keys are absent, a paused alert is still held: publishing
    the draft word as final is the failure this band exists to prevent.
    An explicit False is kept. That is the engine saying the hold is off.
    """
    state = dict(saved.get("state") or {})
    review = saved.get("review") or {}
    if not isinstance(review, dict):
        review = {}
    for key in (
        "verdict",
        "claim",
        "confidence",
        "priority",
        "correction_message",
        "true_version",
        "explanation",
        "evidence",
    ):
        if key not in state and key in review:
            state[key] = review[key]
    if "provisional" not in state:
        state["provisional"] = True
    if not state.get("review_reason"):
        state["review_reason"] = review.get("review_reason") or review.get("why") or ""
    if "alert_pending_approval" not in state and (state.get("decision") == "alert" or review.get("decision") == "alert"):
        state["alert_pending_approval"] = True
    return components.view_model(state, None)


def _page_check(fixture: str) -> None:
    _apply_example()
    highlight = _query_value("cite")
    st.markdown(components.hero_html() + components.marquee_html(config.DOMAINS), unsafe_allow_html=True)
    st.markdown(components.eyebrow("Check a forward"), unsafe_allow_html=True)
    with st.container(border=True):
        mode = st.segmented_control(
            "Input",
            ["Text", "Link", "Image"],
            key="input_mode",
            label_visibility="collapsed",
        )
        if mode not in {"Text", "Link", "Image"}:
            mode = "Text"
        uploaded = None
        if mode == "Text":
            st.text_area(
                "Message",
                key="claim_text",
                height=180,
                placeholder="Paste a claim or a WhatsApp forward",
            )
        elif mode == "Link":
            st.text_input("Link", key="link_url", placeholder="https://")
        else:
            st.markdown(components.image_prompt_html(), unsafe_allow_html=True)
            preview = st.session_state.get("example_image")
            if preview:
                st.image(preview, width=420)
                st.caption("Example screenshot ready.")
                if st.button("Remove example screenshot"):
                    st.session_state.example_image = None
                    st.rerun()
            uploaded = st.file_uploader(
                "Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="image_upload",
            )
            if uploaded is not None and not preview:
                st.image(uploaded.getvalue(), width=420)
            st.text_input("Caption (optional)", key="caption_text", placeholder="Add a caption if the image needs one")
        st.markdown(_quota_html(), unsafe_allow_html=True)
        with st.expander("Run options"):
            st.toggle(
                "Demo mode (cache only)",
                key="demo_cache_only",
                help=(
                    "On by default. A search that is not already cached returns nothing "
                    "instead of calling Tavily. The counter is ahead of the real balance; the cap is 1000 until 1 Dec."
                ),
            )
            st.toggle(
                "Record submission",
                key="record_submissions",
                help=(
                    "Off unless you turn it on. A check here is not another person "
                    "forwarding the message. Priority reads that count as reach."
                ),
            )
        if st.button("Check this", type="primary", key="check_this"):
            if fixture:
                st.session_state.form_error = (
                    "This screen is a recorded fixture. Remove ?fixture= from the address to run a check."
                )
            else:
                _submit_check(mode, uploaded)
        if st.session_state.get("form_error"):
            st.caption(st.session_state.form_error)
        st.caption("Examples fill the box. They do not run until you press Check this.")
        with st.container(horizontal=True, gap="small"):
            if mode == "Image":
                if st.button("Schools closed", key="example_image_btn"):
                    st.session_state.pending_fill = {"mode": "Image", "image": _schools_screenshot()}
                    st.rerun()
            elif mode == "Link":
                if st.button("RBI home", key="example_link"):
                    st.session_state.pending_fill = {"mode": "Link", "url": "https://www.rbi.org.in/"}
                    st.rerun()
            else:
                for index, example in enumerate(_TEXT_EXAMPLES):
                    if st.button(example["label"], key=f"example_{index}"):
                        st.session_state.pending_fill = {"mode": "Text", "text": example["text"]}
                        st.rerun()
    _show_error("check")
    if fixture == "pipeline":
        _show_replay()
        return
    if fixture:
        _show_fixture(fixture, highlight)
        return
    saved = st.session_state.get("check_view")
    if not saved:
        return
    st.markdown(components.eyebrow("The 16 steps"), unsafe_allow_html=True)
    if saved.get("mode") == "fallback":
        st.caption(
            "Per-step times were not available. Steps are filled from the final state, "
            "and a clock is shown only where that state recorded one."
        )
    st.markdown(
        components.stepper_html(saved.get("ran") or [], saved.get("state") or {}, settled=True),
        unsafe_allow_html=True,
    )
    if saved.get("status") == "interrupt":
        model = _interrupt_model(saved)
    else:
        model = components.view_model(saved.get("state") or {}, saved.get("result") or {})
    _render_card(model, fixture="", highlight=highlight)


def _submit_check(mode: str, uploaded) -> None:
    problem = None
    text = ""
    url = ""
    image = None
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
            problem = "Add an image first."
        elif components.image_too_large(len(image)):
            problem = "That image is over the 10 MB limit."
    if problem:
        st.session_state.form_error = problem
        return
    st.session_state.form_error = None
    _start_job({
        "page": "check",
        "action": "check",
        "mode": mode,
        "text": text,
        "url": url or "",
        "caption": st.session_state.get("caption_text") or "",
        "image": image,
    })


def _show_replay() -> None:
    try:
        payload = components.load_fixture("pipeline")
    except (OSError, ValueError) as exc:
        st.error(friendly(exc))
        return
    ran = []
    for item in payload.get("nodes") or []:
        if isinstance(item, dict) and item.get("node"):
            ran.append((item.get("node"), item.get("elapsed_s")))
    st.markdown(components.eyebrow("The 16 steps"), unsafe_allow_html=True)
    st.markdown(
        components.stepper_html(ran, payload, settled=True, replay=True),
        unsafe_allow_html=True,
    )


def _show_fixture(name: str, highlight: str) -> None:
    try:
        payload = components.load_fixture(name)
    except (OSError, ValueError):
        st.error("That fixture is not on this page.")
        return
    if payload.get("kind") == "replay":
        _show_replay()
        return
    st.markdown(
        f"<p class='fl-replay'>fixture · {components.esc(name)} · pipeline not run</p>",
        unsafe_allow_html=True,
    )
    model = components.view_model(payload)
    _render_card(model, fixture=name, highlight=highlight)


def _evidence_count(row: dict) -> int | None:
    """Prefer a count the queue row already has. Open the card only if it does not.

    list_pending today stores the claim and the reason, not the rows.
    get_card is the read that has them. A failure leaves the count blank
    so a locked checkpoint does not hide the rest of the queue.
    """
    if not isinstance(row, dict):
        return None
    if row.get("evidence_count") is not None:
        try:
            return int(row.get("evidence_count"))
        except (TypeError, ValueError):
            return None
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        return len(evidence)
    ids = row.get("evidence_ids")
    if isinstance(ids, list):
        return len(ids)
    thread_id = str(row.get("thread_id") or "")
    if not thread_id:
        return None
    card = _get_card(thread_id)
    if not isinstance(card, dict):
        return None
    for key in ("evidence",):
        if isinstance(card.get(key), list):
            return len(card.get(key))
    result = card.get("result") if isinstance(card.get("result"), dict) else {}
    if isinstance(result.get("evidence"), list):
        return len(result.get("evidence"))
    return None


def _page_review() -> None:
    st.markdown(components.eyebrow("Review queue"), unsafe_allow_html=True)
    st.markdown(components.wordmark_html(huge=False), unsafe_allow_html=True)
    st.caption("Paused runs, High priority first. Approving, editing, or rejecting resumes that run.")
    toast = st.session_state.pop("review_toast", None)
    if toast == "approve":
        st.toast("Approved")
    elif toast == "reject":
        st.toast("Rejected")
    elif toast == "edit":
        st.toast("Edit saved")
    elif toast:
        st.toast("Review saved")
    _show_error("review")
    outcome = st.session_state.get("review_result")
    if isinstance(outcome, dict) and outcome.get("status") == "done":
        st.success("Review saved. The card below is the finished run.")
        model = components.view_model(outcome.get("state") or {}, outcome.get("result") or outcome)
        _render_card(model, fixture="", highlight="")
        st.divider()
    try:
        rows = _call_list_pending()
    except Exception as exc:
        log.exception("review queue failed")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.error(friendly(exc))
        return
    if not rows:
        st.write("Nothing is waiting for review.")
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        thread_id = str(row.get("thread_id") or "")
        count = _evidence_count(row)
        st.markdown(components.queue_summary_html(row, count), unsafe_allow_html=True)
        approve, edit, reject = st.columns(3)
        with approve:
            if st.button("Approve", key=f"approve_{thread_id}", type="primary"):
                _start_job({
                    "page": "review",
                    "action": "review",
                    "review_action": "approve",
                    "thread_id": thread_id,
                })
        with reject:
            if st.button("Reject", key=f"reject_{thread_id}"):
                _start_job({
                    "page": "review",
                    "action": "review",
                    "review_action": "reject",
                    "thread_id": thread_id,
                })
        with edit:
            st.write("")
        with st.expander("Edit"):
            verdict_key = f"verdict_{thread_id}"
            correction_key = f"correction_{thread_id}"
            note_key = f"note_{thread_id}"
            current = str(row.get("provisional_verdict") or row.get("verdict") or "")
            if verdict_key not in st.session_state:
                st.session_state[verdict_key] = current if current in _VERDICTS else "False"
            if correction_key not in st.session_state:
                st.session_state[correction_key] = str(row.get("correction_message") or row.get("correction") or "")
            if note_key not in st.session_state:
                st.session_state[note_key] = ""
            verdict = st.selectbox("Verdict", list(_VERDICTS), key=verdict_key)
            correction = st.text_area("Correction", key=correction_key, height=100)
            note = st.text_input("Note for the audit row (optional)", key=note_key)
            if st.button("Save edit", key=f"save_{thread_id}"):
                _start_job({
                    "page": "review",
                    "action": "review",
                    "review_action": "edit",
                    "thread_id": thread_id,
                    "verdict": verdict,
                    "correction": correction,
                    "note": note,
                })


def _history_rows() -> list[dict]:
    from misinfo_agent.audit import list_runs

    rows = list_runs(config.DATA_DIR / "audit.db")
    extra = _priority_by_run()
    found = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if not item.get("priority") and item.get("run_id") in extra:
            item["priority"] = extra[item["run_id"]]
        found.append(item)
    return found


def _priority_by_run() -> dict:
    """priority if the audit table has grown that column. Read-only, one second.

    list_runs does not return priority today. A later engine may add the
    column without this page knowing the name in advance. A missing
    column is an empty map, and the mix chart says so.
    """
    path = config.DATA_DIR / "audit.db"
    if not path.exists():
        return {}
    try:
        uri = f"file:{path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1) as conn:
            columns = [info[1] for info in conn.execute("PRAGMA table_info(runs)")]
            if "priority" not in columns:
                return {}
            queried = conn.execute("SELECT run_id, priority FROM runs")
            return {row[0]: row[1] for row in queried if row[0]}
    except Exception:
        log.exception("priority column unreadable")
        return {}


def _page_history() -> None:
    st.markdown(components.eyebrow("History"), unsafe_allow_html=True)
    st.markdown(components.wordmark_html(huge=False), unsafe_allow_html=True)
    st.caption("Finished runs, newest first.")
    _show_error("history")
    try:
        rows = _history_rows()
    except Exception as exc:
        log.exception("history failed")
        print(f"APP_ERROR {type(exc).__name__}: {exc}", flush=True)
        st.error(friendly(exc))
        return
    if not rows:
        st.write("No checks have been recorded yet.")
        return
    frame = pd.DataFrame([
        {
            "Time": row.get("created_at") or "",
            "Claim": row.get("claim") or "",
            "Verdict": row.get("verdict") or "",
            "Confidence": _confidence_text(row.get("confidence")),
            "Priority": row.get("priority") or "",
            "Decision": row.get("decision") or "",
            "Human action": row.get("human_action") or "",
            "Calls": "" if row.get("llm_calls") is None else row.get("llm_calls"),
            "Latency": _latency_text(row.get("latency_s")),
            "run_id": row.get("run_id") or "",
        }
        for row in rows
    ])
    event = st.dataframe(
        frame.drop(columns=["run_id"]),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="history_table",
    )
    _history_charts(frame)
    selected = []
    if event is not None and getattr(event, "selection", None) is not None:
        selected = list(getattr(event.selection, "rows", []) or [])
    if not selected:
        st.caption("Select a row to open its card.")
        return
    index = selected[0]
    if index < 0 or index >= len(frame):
        return
    thread_id = str(frame.iloc[index]["run_id"] or "")
    card = _get_card(thread_id)
    if not card:
        st.caption("That run has no card to open.")
        return
    st.markdown(components.eyebrow("Card"), unsafe_allow_html=True)
    model = components.view_model(card, card.get("result") if isinstance(card.get("result"), dict) else card)
    _render_card(model, fixture="", highlight="")


def _confidence_text(value) -> str:
    number = components._confidence_float(value)
    if number is None:
        return "—"
    return f"{number:.0%}"


def _history_charts(frame: pd.DataFrame) -> None:
    plotted = frame.copy()
    plotted["when"] = pd.to_datetime(plotted["Time"], utc=True, errors="coerce")
    known = plotted.dropna(subset=["when"])
    if not known.empty:
        over_time = (
            alt.Chart(known)
            .mark_circle(size=90)
            .encode(
                x=alt.X("when:T", title="When"),
                y=alt.Y("Verdict:N", title="Verdict"),
                color=alt.Color(
                    "Verdict:N",
                    scale=alt.Scale(domain=_VERDICT_RANGE, range=_VERDICT_COLORS),
                    legend=None,
                ),
                tooltip=["Time", "Verdict", "Claim"],
            )
            .properties(height=220, title="Verdicts over time")
        )
        st.altair_chart(_dark(over_time), width="stretch", theme=None)
    mix = (
        alt.Chart(plotted)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Runs"),
            y=alt.Y("Verdict:N", title=""),
            color=alt.Color(
                "Verdict:N",
                scale=alt.Scale(domain=_VERDICT_RANGE, range=_VERDICT_COLORS),
                legend=None,
            ),
        )
        .properties(height=220, title="Verdict mix")
    )
    st.altair_chart(_dark(mix), width="stretch", theme=None)
    priority = plotted[plotted["Priority"].astype(str).str.strip() != ""]
    if priority.empty:
        st.caption("Priority is not on the history row yet, so the priority mix stays empty until the audit list includes it.")
        return
    bars = (
        alt.Chart(priority)
        .mark_bar(color="#E05A47")
        .encode(
            x=alt.X("count():Q", title="Runs"),
            y=alt.Y("Priority:N", title="", sort=["High", "Medium", "Low"]),
        )
        .properties(height=180, title="Priority mix")
    )
    st.altair_chart(_dark(bars), width="stretch", theme=None)


def _page_how() -> None:
    st.markdown(components.how_it_works_html(), unsafe_allow_html=True)


def _chrome(fixture: str) -> None:
    pending = 0 if fixture else _pending_count()
    st.markdown(
        components.utility_html() + components.tabs_html(_page(), pending, fixture),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=components.APP_NAME, page_icon="✦", layout="wide", initial_sidebar_state="collapsed")
    theme.inject()
    _init_state()
    _apply_cache_only()
    fixture = _fixture()
    _chrome(fixture)
    if st.session_state.get("busy") and st.session_state.get("job"):
        _run_job()
        return
    page = _page()
    if page == "review":
        _page_review()
    elif page == "history":
        _page_history()
    elif page == "how":
        _page_how()
    else:
        _page_check(fixture)


if __name__ == "__main__":
    main()
