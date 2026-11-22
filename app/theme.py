"""Visual theme for the checker page.

Colours and type only. The reference is a dark editorial site: a navy
field, a condensed wordmark, coral used as a single accent. Nothing
here copies a logo, a crest, a photograph, or that site's words.

Streamlit's own chrome (menu, header, footer, deploy button) is hidden
so the page reads as one product. Motion is real CSS. A reader who asks
the browser for less motion gets the same layout with the animations
switched off, because a glow that will not stop is worse than no glow.
"""

from __future__ import annotations

# Imported once, in the first paint. A later rerun does not need another
# download; the fallback stack is what shows if the request is blocked.
_FONTS = (
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700"
    "&family=Barlow:wght@400;500;600&family=Inter:wght@400;500;600&display=swap"
)

# Display is condensed and tight. Body is a text face. Arial Narrow is
# the fallback that still looks condensed when Google Fonts does not load.
_DISPLAY = '"Barlow Condensed", "Arial Narrow", "Arial", system-ui, sans-serif'
_BODY = '"Barlow", "Inter", system-ui, sans-serif'

CSS = f"""
@import url('{_FONTS}');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.stApp, [data-testid="stMain"], [data-testid="stHeader"] {{
  background: #0A0B0E !important;
  color: #F2F0EA;
  font-family: {_BODY};
}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stDeployButton"],
.stDeployButton, header [data-testid="stToolbar"] {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
}}
header[data-testid="stHeader"] {{
  display: none !important;
}}
[data-testid="stSidebar"] {{
  display: none !important;
}}
.block-container {{
  max-width: 1200px;
  padding-top: 0.4rem;
  padding-bottom: 4rem;
}}
h1, h2, h3, h4, p, li, label, span, div {{
  font-family: {_BODY};
}}
a {{
  color: #F2F0EA;
}}
.fl-tab {{
  text-decoration: none !important;
}}
.fl-seen a, .fl-ev a {{
  color: #F2F0EA !important;
}}
.fl-cite {{
  color: #E05A47 !important;
  text-decoration: none !important;
}}

.fl-utility {{
  width: 100vw;
  position: relative;
  left: 50%;
  margin-left: -50vw;
  background: #0C1A33;
  color: #A7B0C2;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  padding: 0.45rem 1.2rem;
  margin-bottom: 0.8rem;
}}
.fl-utility-inner {{
  max-width: 1200px;
  margin: 0 auto;
}}

.fl-tabs {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.2rem 0 1.1rem;
}}
.fl-tab {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  text-decoration: none;
  color: #F2F0EA;
  border: 1px solid #F2F0EA44;
  border-radius: 999px;
  padding: 0.38rem 0.9rem;
  font-size: 0.92rem;
  background: transparent;
}}
.fl-tab:hover {{
  border-color: #E05A47;
  transform: translateY(-1px);
}}
.fl-tab.is-on {{
  background: #B8352A;
  border-color: #B8352A;
  color: #F2F0EA;
}}
.fl-badge {{
  background: #E05A47;
  color: #0A0B0E;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.05rem 0.4rem;
  min-width: 1.1rem;
  text-align: center;
}}
.fl-tab.is-on .fl-badge {{
  background: #F2F0EA;
  color: #B8352A;
}}

.fl-hero {{
  position: relative;
  overflow: hidden;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 2.2rem 1.6rem 1.8rem;
  margin-bottom: 0.9rem;
  background:
    radial-gradient(900px 420px at 12% -10%, rgba(224, 90, 71, 0.22), transparent 55%),
    radial-gradient(700px 380px at 100% 0%, rgba(12, 26, 51, 0.95), transparent 50%),
    #0A0B0E;
  background-size: 140% 140%;
}}
.fl-hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.45;
  background-image: radial-gradient(rgba(242, 240, 234, 0.16) 0.55px, transparent 0.55px);
  background-size: 3px 3px;
  mix-blend-mode: overlay;
}}
.fl-kicker {{
  position: relative;
  color: #A7B0C2;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0 0 0.6rem;
}}
.fl-kicker .fl-dot {{
  color: #E05A47;
  margin: 0 0.35rem;
}}
.fl-wordmark {{
  position: relative;
  font-family: {_DISPLAY};
  font-weight: 700;
  letter-spacing: -0.045em;
  line-height: 0.84;
  color: #F2F0EA;
  margin: 0;
}}
.fl-wordmark.huge {{
  font-size: clamp(4.4rem, 12vw, 8.4rem);
}}
.fl-wordmark.small {{
  font-size: clamp(2rem, 4vw, 2.6rem);
}}
.fl-star {{
  color: #E05A47;
}}
.fl-lede {{
  position: relative;
  margin: 0.85rem 0 0;
  font-size: 1.35rem;
  color: #F2F0EA;
}}
.fl-eyebrow {{
  color: #E05A47;
  font-variant: small-caps;
  letter-spacing: 0.16em;
  font-size: 0.78rem;
  font-weight: 600;
  margin: 1.4rem 0 0.7rem;
}}

.fl-marquee {{
  overflow: hidden;
  border-top: 1px solid #1E2A45;
  border-bottom: 1px solid #1E2A45;
  padding: 0.55rem 0;
  margin: 0.2rem 0 1rem;
}}
.fl-marquee-track {{
  display: flex;
  gap: 0.75rem;
  width: max-content;
  align-items: center;
  color: #A7B0C2;
  font-size: 0.92rem;
}}
.fl-marquee-item {{
  white-space: nowrap;
}}

.fl-panel {{
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 1rem 1.1rem 1.1rem;
  margin: 0.7rem 0;
}}
.fl-panel:hover {{
  transform: translateY(-2px);
}}

.fl-band {{
  background: linear-gradient(90deg, #E05A47, #B8352A);
  color: #0A0B0E;
  border-radius: 10px;
  padding: 0.7rem 1rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  margin: 0.4rem 0 0.7rem;
}}
.fl-band p {{
  margin: 0.2rem 0 0;
  font-weight: 500;
  letter-spacing: 0;
  color: #0A0B0E;
}}
.fl-alert-held {{
  margin: 0.45rem 0 0.2rem;
  color: #F2F0EA;
  font-weight: 600;
}}

.fl-verdict-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}}
.fl-verdict-word {{
  font-family: {_DISPLAY};
  font-weight: 700;
  font-size: clamp(4rem, 10vw, 7.2rem);
  line-height: 0.84;
  letter-spacing: -0.04em;
  margin: 0;
}}
.fl-priority {{
  display: inline-block;
  border-radius: 999px;
  padding: 0.2rem 0.7rem;
  border: 1px solid #F2F0EA44;
  color: #F2F0EA;
  font-size: 0.85rem;
  margin-top: 0.4rem;
}}
.fl-priority.high {{ border-color: #E05A47; color: #E05A47; }}
.fl-priority.medium {{ border-color: #E8A33D; color: #E8A33D; }}
.fl-priority.low {{ border-color: #A7B0C2; color: #A7B0C2; }}
.fl-claim {{
  font-size: 1.15rem;
  margin: 0.8rem 0 0;
}}
.fl-plain-reason {{
  color: #A7B0C2;
  margin: 0.4rem 0 0;
}}
.fl-caveat {{
  color: #E8A33D;
  margin: 0.55rem 0 0;
  font-size: 1.02rem;
}}
.fl-strength {{ text-align: center; }}
.fl-strength-label {{
  color: #A7B0C2;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  margin-top: 0.1rem;
}}
.fl-search-banner {{
  background: #3A3018;
  color: #E3C27A;
  border: 1px solid #E8A33D;
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  margin: 0 0 0.8rem;
}}

.fl-src-lean {{
  border: 1px solid #F2F0EA;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  margin-top: 0.8rem;
  color: #F2F0EA;
  background: #0F1B36;
}}
.fl-src-lean strong {{
  color: #F2F0EA;
  font-weight: 600;
}}
.fl-lean {{
  border: 1px dashed #F2F0EA44;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  margin-top: 0.8rem;
  color: #A7B0C2;
  background: transparent;
}}
.fl-lean strong {{
  color: #A7B0C2;
  font-weight: 600;
  font-size: 0.92rem;
}}
.fl-lean .fl-lean-label {{
  color: #7C8AA5;
  font-size: 1.05rem;
  margin: 0.25rem 0;
}}
.fl-lean p {{
  margin: 0.15rem 0;
  font-size: 0.88rem;
}}

.fl-seen {{
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin: 0.7rem 0;
}}
.fl-related {{
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 0.7rem 1rem;
  margin: 0.7rem 0;
}}
.fl-related summary {{ cursor: pointer; color: #F2F0EA; }}
.fl-muted {{ color: #A7B0C2; font-size: 0.92rem; }}
.fl-explain {{
  font-size: 1.05rem;
  line-height: 1.55;
}}
.fl-cite {{
  display: inline-block;
  margin: 0 0.15rem;
  padding: 0.02rem 0.4rem;
  border-radius: 999px;
  border: 1px solid #E05A47;
  color: #E05A47;
  text-decoration: none;
  font-size: 0.82rem;
  font-weight: 600;
}}
.fl-cite:hover {{ background: #E05A4722; }}

.fl-filters-note {{ color: #A7B0C2; font-size: 0.85rem; margin: 0 0 0.4rem; }}
.fl-group {{
  margin: 0.8rem 0 0.3rem;
  color: #A7B0C2;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.fl-ev {{
  display: grid;
  grid-template-columns: 6px 1fr;
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  margin: 0.45rem 0;
  overflow: hidden;
}}
.fl-ev:hover {{ transform: translateY(-2px); }}
.fl-ev.is-hot {{
  border-color: #E05A47;
  box-shadow: 0 0 0 1px #E05A47;
}}
.fl-bar.refutes {{ background: #E05A47; }}
.fl-bar.supports {{ background: #3FB27F; }}
.fl-bar.neutral {{ background: #7C8AA5; }}
.fl-bar.other {{ background: #5B6475; }}
.fl-ev-body {{ padding: 0.75rem 0.9rem 0.85rem; }}
.fl-ev-top {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  align-items: center;
}}
.fl-tier {{
  border-radius: 999px;
  padding: 0.05rem 0.45rem;
  font-size: 0.75rem;
  font-weight: 700;
}}
.fl-tier.t1 {{ background: #3A3018; color: #E3C27A; }}
.fl-tier.t2 {{ background: #2C3340; color: #D5DCE6; }}
.fl-tier.t3, .fl-tier.t4 {{ background: #1A2336; color: #A7B0C2; }}
.fl-detail-tag {{
  border-radius: 999px;
  border: 1px solid #E8A33D;
  color: #E8A33D;
  font-size: 0.75rem;
  padding: 0.05rem 0.45rem;
}}
.fl-ev details {{ margin-top: 0.45rem; }}
.fl-ev summary {{ cursor: pointer; color: #A7B0C2; }}
.fl-passage {{ color: #F2F0EA; line-height: 1.5; }}
.fl-passage mark {{
  background: #E05A4733;
  color: #F2F0EA;
  padding: 0 0.1rem;
}}

.fl-chip {{
  display: inline-block;
  margin: 0.15rem 0.35rem 0.15rem 0;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  border: 1px solid #F2F0EA44;
  color: #F2F0EA;
  font-size: 0.88rem;
}}
.fl-chip.flag {{ border-color: #E05A47; color: #F2F0EA; }}
.fl-chip.is-warn {{ border-color: #E8A33D; color: #E8A33D; }}
.fl-chip.is-cap {{ border-color: #E05A47; color: #E05A47; }}
.fl-quota {{ margin: 0.35rem 0 0.2rem; }}

.fl-stepper {{ margin: 0.4rem 0 0.8rem; }}
.fl-replay {{
  color: #A7B0C2;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  margin: 0 0 0.4rem;
}}
.fl-step-group {{
  color: #E05A47;
  font-variant: small-caps;
  letter-spacing: 0.14em;
  font-size: 0.75rem;
  margin: 0.7rem 0 0.25rem;
}}
.fl-step {{
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 0.6rem;
  align-items: baseline;
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 0.45rem 0.7rem;
  margin: 0.28rem 0;
}}
.fl-step-box {{
  color: #A7B0C2;
  font-size: 0.75rem;
  width: 1.6rem;
}}
.fl-step-title {{ color: #F2F0EA; }}
.fl-step-reason {{
  grid-column: 2 / -1;
  color: #A7B0C2;
  font-size: 0.82rem;
}}
.fl-step-time {{ color: #A7B0C2; font-variant-numeric: tabular-nums; font-size: 0.85rem; }}
.fl-step.is-running {{
  border-color: #E05A47;
  box-shadow: 0 0 0 1px #E05A47, 0 0 18px rgba(224, 90, 71, 0.45);
}}
.fl-step.is-skipped {{ opacity: 0.48; }}
.fl-step.is-skipped .fl-step-title {{ color: #A7B0C2; }}
.fl-step.is-wait .fl-step-title {{ color: #A7B0C2; }}
.fl-dot-done {{
  display: inline-block;
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: #E05A47;
  margin-right: 0.35rem;
}}

.fl-flow {{
  display: flex;
  flex-direction: column;
  gap: 0;
}}
.fl-flow-card {{
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 0.8rem 0.95rem;
}}
.fl-flow-card:hover {{ transform: translateY(-2px); }}
.fl-flow-kicker {{
  color: #E05A47;
  font-variant: small-caps;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
}}
.fl-flow-title {{
  font-family: {_DISPLAY};
  font-weight: 700;
  font-size: 1.45rem;
  letter-spacing: -0.02em;
  margin: 0.15rem 0;
}}
.fl-arrow {{
  color: #E05A47;
  text-align: center;
  line-height: 1;
  padding: 0.15rem 0;
  font-size: 1.1rem;
}}
.fl-principle {{
  font-family: {_DISPLAY};
  font-weight: 700;
  font-size: clamp(1.8rem, 4vw, 2.6rem);
  letter-spacing: -0.03em;
  line-height: 1;
  margin: 0.2rem 0 0.8rem;
}}
.fl-honest {{
  background: #0F1B36;
  border: 1px solid #1E2A45;
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin: 0.5rem 0;
}}

div[data-testid="stButton"] > button,
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"] {{
  border-radius: 999px !important;
}}
button[data-testid="stBaseButton-primary"] {{
  background: #B8352A !important;
  border: 1px solid #B8352A !important;
  color: #F2F0EA !important;
}}
button[data-testid="stBaseButton-primary"]:hover {{
  background: #E05A47 !important;
  border-color: #E05A47 !important;
}}
button[data-testid="stBaseButton-secondary"] {{
  background: transparent !important;
  border: 1px solid #F2F0EA44 !important;
  color: #F2F0EA !important;
}}
button[data-testid="stBaseButton-secondary"]:hover {{
  border-color: #F2F0EA !important;
  transform: translateY(-1px);
}}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
  background: #0A0B0E !important;
  color: #F2F0EA !important;
  border: 1px solid #1E2A45 !important;
  border-radius: 10px !important;
}}
[data-testid="stExpander"] details {{
  background: #0F1B36 !important;
  border: 1px solid #1E2A45 !important;
  border-radius: 10px !important;
}}
[data-testid="stExpander"] details summary:hover {{
  color: #F2F0EA;
}}
[data-testid="stCode"], pre, code {{
  background: #0A0B0E !important;
  color: #F2F0EA !important;
  border-radius: 10px !important;
}}
[data-testid="stFileUploader"] {{
  background: #0F1B36;
  border-radius: 10px;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: #0F1B36 !important;
  border: 1px solid #1E2A45 !important;
  border-radius: 10px !important;
}}

@media (max-width: 800px) {{
  .fl-hero {{ padding: 1.4rem 1rem; }}
}}

@media (prefers-reduced-motion: no-preference) {{
  .fl-hero {{
    animation: fl-drift 18s ease-in-out infinite alternate;
  }}
  .fl-hero::before {{
    animation: fl-grain 0.7s steps(2) infinite;
  }}
  .fl-marquee-track {{
    animation: fl-marquee 42s linear infinite;
  }}
  .fl-marquee:hover .fl-marquee-track {{
    animation-play-state: paused;
  }}
  .fl-verdict-word {{
    animation: fl-reveal 0.7s cubic-bezier(.2, .8, .2, 1) both;
  }}
  .fl-stepper.is-replay .fl-step.is-done {{
    animation: fl-replay 8s ease-in-out infinite;
    animation-delay: calc(var(--i) * 0.42s);
  }}
  .fl-ev details[open] .fl-passage {{
    animation: fl-expand 0.25s ease;
  }}
}}

@media (prefers-reduced-motion: reduce) {{
  .fl-ev, .fl-flow-card, .fl-panel, .fl-tab, button {{
    transition: none !important;
  }}
  .fl-ev:hover, .fl-flow-card:hover, .fl-panel:hover, .fl-tab:hover {{
    transform: none !important;
  }}
  .fl-marquee-copy {{
    display: none;
  }}
  .fl-marquee-track {{
    flex-wrap: wrap;
    width: auto;
    white-space: normal;
  }}
}}

@keyframes fl-drift {{
  from {{ background-position: 0% 0%; }}
  to {{ background-position: 40% 20%; }}
}}
@keyframes fl-grain {{
  0% {{ background-position: 0 0; }}
  100% {{ background-position: 3px 2px; }}
}}
@keyframes fl-marquee {{
  from {{ transform: translateX(0); }}
  to {{ transform: translateX(-50%); }}
}}
@keyframes fl-reveal {{
  from {{ opacity: 0; transform: translateY(16px); filter: blur(5px); }}
  to {{ opacity: 1; transform: none; filter: none; }}
}}
@keyframes fl-replay {{
  0%, 100% {{ box-shadow: none; border-color: #1E2A45; }}
  8%, 16% {{ box-shadow: 0 0 18px rgba(224, 90, 71, 0.5); border-color: #E05A47; }}
}}
@keyframes fl-expand {{
  from {{ opacity: 0; transform: translateY(-4px); }}
  to {{ opacity: 1; transform: none; }}
}}
"""


def inject() -> None:
    """Paint the theme. Called after set_page_config and before any widget.

    Streamlit drops a <style> block that is not marked unsafe, so the
    theme has to go through the HTML path. The CSS itself is ours; it
    contains no claim text.
    """
    import streamlit as st

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
