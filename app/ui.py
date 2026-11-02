from __future__ import annotations

from html import escape as html_escape
from textwrap import dedent
from functools import wraps
import math

# FitMaxxx visual system: soft rose, warm white, and fitness-inspired track texture.
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
:root {
  --fm-ink: #252334;
  --fm-muted: #827985;
  --fm-pink: #e85d91;
  --fm-rose: #f5a8c2;
  --fm-blush: #fde7ef;
  --fm-pale: #fff7fa;
  --fm-line: #f1dce4;
  --fm-white: rgba(255,255,255,.88);
  --fm-shadow: 0 12px 34px rgba(129, 73, 96, .08);
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--fm-ink); }
h1,h2,h3,h4,[data-testid="stMetricValue"] { font-family: 'Outfit', sans-serif !important; letter-spacing: -.025em; }
.stApp {
  background-color: #fffafc;
  background-image:
    radial-gradient(ellipse at 8% 5%, rgba(248,190,209,.20), transparent 34%),
    radial-gradient(ellipse at 92% 16%, rgba(252,220,231,.22), transparent 30%),
    repeating-linear-gradient(132deg, rgba(224,164,185,.045) 0px, rgba(224,164,185,.045) 1px, transparent 1px, transparent 34px),
    repeating-linear-gradient(48deg, rgba(255,255,255,.68) 0px, rgba(255,255,255,.68) 2px, transparent 2px, transparent 38px);
  color: var(--fm-ink);
}
[data-testid="stHeader"] { background: rgba(255,250,252,.72); }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stMain"] .block-container { max-width: 1480px; padding-top: 1.45rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fff0f5 0%, #fff8fb 55%, #fff5f8 100%);
  border-right: 1px solid #f2dce5;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #675a66; }
[data-testid="stSidebar"] hr { border-color: #efd5df; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--fm-ink); }
[data-testid="stSidebar"] div[data-testid="stRadio"] > label { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: .28rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
  border: 1px solid transparent; border-radius: 13px; padding: .48rem .72rem;
  min-height: 42px; transition: .18s ease; color: #554957; font-weight: 600;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, #f9d6e3, #fff3f7); border-color: #f2bfd0; color: #b83d6c;
  box-shadow: 0 4px 12px rgba(203,104,142,.08);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #fff5f8; border-color: #f5d7e1; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] .stButton button {
  border-radius: 12px; border: 1px solid #e982a7; color: #b83d6c; background: #fff7fa; font-weight: 700;
}
[data-testid="stSidebar"] .stButton button:hover { background: #fce2eb; border-color: #df6792; color: #9f315c; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
  background: linear-gradient(135deg, #e85d91, #d9437d); border: 0; color: white;
  border-radius: 12px; font-weight: 700; box-shadow: 0 7px 16px rgba(220,75,127,.17);
}
.stButton button[kind="primary"]:hover { background: linear-gradient(135deg, #d94b82, #c93770); color: white; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
  border-radius: 11px !important; border-color: #efd6e0 !important; background: rgba(255,255,255,.88) !important;
}
.stRadio > div { gap: .55rem; }
[data-testid="stMetric"] { background: rgba(255,255,255,.78); border: 1px solid #f2dce5; padding: .8rem 1rem; border-radius: 16px; box-shadow: 0 5px 20px rgba(129,73,96,.045); }
[data-testid="stMetricLabel"] { color: #837481 !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #2d2635 !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #f1dce4 !important; border-radius: 18px !important; background: rgba(255,255,255,.68); }
div[data-testid="stExpander"] { border: 1px solid #f1dce4; border-radius: 14px; background: rgba(255,255,255,.72); }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #f2a6bf, #e4538a); }
[data-testid="stDataFrame"] { border: 1px solid #f1dce4; border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 13px; }
#MainMenu, footer { visibility: hidden; }
.fm-brand { display:flex; align-items:center; gap:11px; margin: .2rem 0 1.2rem; }
.fm-brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:15px; background:linear-gradient(145deg,#f7bfd1,#fcecf2); box-shadow: inset 0 0 0 1px #f3c8d7; color:#d6447d; font-size:22px; }
.fm-brand-name { font:800 1.45rem 'Outfit',sans-serif; letter-spacing:-.05em; color:#2b2534; line-height:1; }
.fm-brand-sub { font-size:.72rem; color:#8e7784; margin-top:4px; }
.fm-eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.67rem; font-weight:800; color:#d34f82; }
.fm-hero { position:relative; overflow:hidden; padding:1.05rem 1.25rem 1.1rem; border:1px solid #f3dce5; border-radius:24px; margin-bottom:1rem; background:linear-gradient(105deg,rgba(255,255,255,.93) 0%,rgba(255,247,250,.90) 55%,rgba(255,230,239,.68) 100%); box-shadow:var(--fm-shadow); }
.fm-hero:before { content:""; position:absolute; inset:0; opacity:.58; pointer-events:none; background:repeating-radial-gradient(ellipse at 88% 120%, transparent 0 24px, rgba(226,119,158,.13) 25px 27px, transparent 28px 43px); }
.fm-hero:after { content:"✦   ·     ·      ✧"; position:absolute; right:8%; top:16%; color:#efb4c9; letter-spacing:1.5rem; font-size:1.5rem; opacity:.55; }
.fm-hero-inner { position:relative; z-index:1; display:flex; justify-content:space-between; align-items:center; gap:1rem; }
.fm-greeting { font:700 clamp(1.5rem,2.6vw,2.15rem) 'Outfit',sans-serif; color:#292333; margin:0; letter-spacing:-.04em; }
.fm-greeting span { color:#d94e83; }
.fm-subtitle { color:#817380; font-size:.92rem; margin:.25rem 0 0; }
.fm-date-pill { white-space:nowrap; padding:.55rem .85rem; border:1px solid #f0d4df; background:rgba(255,255,255,.78); border-radius:999px; color:#9f476b; font-size:.8rem; font-weight:700; }
.fm-runner { width:150px; max-width:18%; opacity:.72; flex-shrink:0; }
.fm-card { background:rgba(255,255,255,.85); border:1px solid #f1dce5; border-radius:22px; padding:1rem 1.1rem; box-shadow:var(--fm-shadow); height:100%; }
.fm-card-title { font:700 1.08rem 'Outfit',sans-serif; color:#302838; margin:0 0 .12rem; }
.fm-card-sub { color:#958693; font-size:.76rem; margin:0 0 .7rem; }
.fm-ring-layout { display:flex; align-items:center; gap:1.1rem; min-height:255px; }
.fm-rings { width:min(44%,230px); min-width:160px; flex-shrink:0; }
.fm-ring-legend { flex:1; min-width:0; }
.fm-ring-row { display:flex; justify-content:space-between; align-items:center; padding:.56rem 0; border-bottom:1px solid #f5e7ed; gap:.5rem; }
.fm-ring-row:last-child { border-bottom:0; }
.fm-ring-label { display:flex; align-items:center; gap:8px; color:#6d5f6d; font-size:.82rem; font-weight:600; }
.fm-dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.fm-ring-value { font:700 .91rem 'Outfit',sans-serif; color:#302838; white-space:nowrap; }
.fm-watch-row { display:flex; align-items:center; gap:.75rem; margin:.1rem 0 .8rem; }
.fm-watch-art { width:110px; min-width:82px; max-width:32%; }
.fm-device-pill { background:#fff0f5; color:#a94d70; border:1px solid #f3d6e1; padding:.48rem .72rem; border-radius:999px; font-size:.76rem; font-weight:700; }
.fm-status { margin-left:auto; color:#3d895f; background:#edf8f0; border:1px solid #d7efde; border-radius:999px; padding:.38rem .62rem; font-size:.72rem; font-weight:700; white-space:nowrap; }
.fm-device-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.5rem; }
.fm-device-stat { text-align:center; background:linear-gradient(180deg,#fff8fb,#fff0f5); border:1px solid #f4e1e8; padding:.75rem .35rem; border-radius:15px; }
.fm-device-icon { color:#df5b8c; font-size:1.08rem; }
.fm-device-value { font:700 1.05rem 'Outfit',sans-serif; color:#312a36; margin-top:.15rem; }
.fm-device-label { color:#8d7c88; font-size:.69rem; }
.fm-section-head { display:flex; justify-content:space-between; gap:.7rem; align-items:baseline; margin:.45rem 0 .6rem; }
.fm-section-head h2 { font:700 1.22rem 'Outfit',sans-serif; color:#302838; margin:0; }
.fm-section-head span { color:#9a8995; font-size:.76rem; }
.fm-focus { display:flex; align-items:center; justify-content:space-between; gap:1rem; overflow:hidden; padding:1rem 1.2rem; border:1px solid #f0d5df; border-radius:22px; background:linear-gradient(100deg,#fff 0%,#fff6f9 60%,#fce3ec 100%); box-shadow:var(--fm-shadow); position:relative; }
.fm-focus-copy { position:relative; z-index:1; max-width:66%; }
.fm-focus-title { font:700 1.23rem 'Outfit',sans-serif; color:#2e2735; margin:0 0 .35rem; }
.fm-focus-text { color:#706371; font-size:.86rem; line-height:1.55; margin:0; }
.fm-focus-art { width:220px; max-width:32%; flex-shrink:0; }
.fm-profile { display:flex; align-items:center; gap:.7rem; margin:.65rem 0 .9rem; }
.fm-avatar { width:46px; height:46px; border-radius:50%; background:linear-gradient(145deg,#f7bfd1,#fcebf2); border:2px solid white; display:grid; place-items:center; color:#a83e69; font:800 1rem 'Outfit',sans-serif; box-shadow:0 3px 10px rgba(180,90,125,.13); }
.fm-profile-name { font:700 1rem 'Outfit',sans-serif; color:#302838; }
.fm-profile-note { font-size:.72rem; color:#907f8b; }
.fm-sidebar-small { color:#9c8994; font-size:.69rem; line-height:1.45; }
.fm-soft-note { padding:.85rem 1rem; background:#fff3f7; border:1px solid #f2dce5; border-radius:15px; color:#786674; font-size:.82rem; }
.fm-page-head { margin:.2rem 0 1rem; }
.fm-page-head h1 { font:800 1.8rem 'Outfit',sans-serif; margin:0; color:#2c2634; }
.fm-page-head p { color:#8d7d89; margin:.2rem 0 0; font-size:.86rem; }
.fm-badge { display:inline-block; padding:.32rem .62rem; border-radius:999px; background:#fff0f5; color:#b54d76; border:1px solid #f3d5e0; font-size:.69rem; font-weight:800; letter-spacing:.04em; }
/* Keep every custom component inside the available Streamlit content width. */
.fm-hero-copy, .fm-card, .fm-focus, .fm-ring-layout, .fm-ring-legend { min-width:0; }
.fm-rings svg, .fm-watch-art, .fm-focus-art, .fm-runner { display:block; height:auto; }
.fm-rings svg { width:100%; }

/* Tablet and narrow desktop: the sidebar reduces the usable content width, so
   this breakpoint intentionally activates before the browser itself feels tiny. */
@media (max-width: 1050px) {
  [data-testid="stMain"] .block-container {
    padding-left:1.05rem;
    padding-right:1.05rem;
    padding-top:1.1rem;
  }
  .fm-hero-inner {
    display:grid;
    grid-template-columns:minmax(0,1fr) auto;
    align-items:center;
    column-gap:.8rem;
    row-gap:.65rem;
  }
  .fm-hero-copy { grid-column:1; grid-row:1; }
  .fm-runner {
    grid-column:2;
    grid-row:1 / span 2;
    width:105px;
    max-width:100%;
  }
  .fm-date-pill {
    grid-column:1;
    grid-row:2;
    justify-self:start;
    font-size:.74rem;
  }
  .fm-section-head {
    flex-direction:column;
    align-items:flex-start;
    gap:.12rem;
  }
  .fm-section-head span { line-height:1.4; }
  .fm-ring-layout {
    flex-direction:column;
    align-items:stretch;
    gap:.45rem;
    min-height:0;
  }
  .fm-rings { width:min(100%,220px); min-width:0; margin:0 auto; }
  .fm-ring-legend { width:100%; }
  .fm-device-stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .fm-focus-copy { max-width:100%; }
  .fm-focus-art { display:none; }
}

/* Phone landscape and portrait. Stack the hero so the date never competes
   with the greeting for the same narrow row. */
@media (max-width: 900px) {
  [data-testid="stMain"] .block-container {
    padding-left:.85rem;
    padding-right:.85rem;
    padding-top:.85rem;
  }
  .fm-hero { padding:.95rem 1rem; border-radius:20px; }
  .fm-hero-inner {
    display:flex;
    flex-direction:column;
    align-items:flex-start;
    gap:.62rem;
  }
  .fm-hero-copy { width:100%; }
  .fm-greeting { font-size:clamp(1.55rem,4.8vw,2rem); line-height:1.12; }
  .fm-subtitle { font-size:.86rem; line-height:1.45; }
  .fm-runner { display:none; }
  .fm-date-pill { align-self:flex-start; font-size:.73rem; padding:.48rem .72rem; }
  .fm-hero:after { right:5%; top:10%; font-size:1.1rem; letter-spacing:.8rem; }
  .fm-section-head { margin:.3rem 0 .55rem; }
  .fm-section-head h2 { font-size:1.1rem; }
  .fm-card { padding:.9rem; border-radius:18px; }
  .fm-card-title { font-size:1.02rem; }
  .fm-ring-row { padding:.5rem 0; }
  .fm-ring-value { font-size:.86rem; }
  .fm-watch-row { gap:.55rem; }
  .fm-watch-art { width:88px; min-width:64px; }
  .fm-device-pill { display:inline-block; white-space:normal; line-height:1.35; }
  .fm-status { margin-left:0; display:inline-block; }
}

@media (max-width: 560px) {
  [data-testid="stMain"] .block-container { padding-left:.7rem; padding-right:.7rem; }
  .fm-hero { padding:.85rem .9rem; border-radius:18px; }
  .fm-eyebrow { font-size:.62rem; }
  .fm-greeting { font-size:clamp(1.45rem,6vw,1.8rem); }
  .fm-subtitle { font-size:.82rem; }
  .fm-date-pill { font-size:.7rem; }
  .fm-card { padding:.82rem; }
  .fm-ring-row { gap:.35rem; }
  .fm-ring-label { font-size:.76rem; }
  .fm-ring-value { font-size:.8rem; }
  .fm-device-stat { padding:.62rem .25rem; border-radius:12px; }
  .fm-device-value { font-size:.96rem; }
  .fm-device-label { font-size:.65rem; }
  .fm-page-head h1 { font-size:1.55rem; }
  .fm-focus { padding:.9rem; }
}
</style>
"""


def escape(value) -> str:
    return html_escape(str(value), quote=True)


def sidebar_brand_html() -> str:
    return """
    <div class='fm-brand'>
      <div class='fm-brand-mark'>♡</div>
      <div><div class='fm-brand-name'>FitMaxxx</div><div class='fm-brand-sub'>Your fitness companion</div></div>
    </div>
    """


def initials(name: str) -> str:
    parts = [p for p in str(name).strip().split() if p]
    return "".join(p[0].upper() for p in parts[:2]) or "FM"


def profile_html(name: str, subtitle: str = "Fitness enthusiast") -> str:
    return f"""
    <div class='fm-profile'>
      <div class='fm-avatar'>{escape(initials(name))}</div>
      <div><div class='fm-profile-name'>{escape(name)}</div><div class='fm-profile-note'>{escape(subtitle)}</div></div>
    </div>
    """


def runner_svg() -> str:
    return """
    <svg class='fm-runner' viewBox='0 0 180 128' role='img' aria-label='Runner illustration'>
      <g fill='none' stroke='#e9a1bb' stroke-width='3' stroke-linecap='round' opacity='.75'>
        <path d='M5 104 C48 74 90 74 168 16'/><path d='M0 119 C52 88 100 89 177 31'/><path d='M15 126 C62 101 111 99 180 48'/>
      </g>
      <g fill='#e9a1bb' opacity='.92' transform='translate(49 9) rotate(-8 44 53)'>
        <circle cx='81' cy='15' r='9'/><path d='M73 27 Q61 36 65 52 L77 65 L96 52 L91 35 Z'/>
        <path d='M70 34 L50 43 L39 33 L34 38 L49 55 L72 47 Z'/><path d='M87 34 L104 42 L113 30 L119 34 L109 54 L86 50 Z'/>
        <path d='M77 61 L62 83 L39 91 L42 98 L69 95 L91 71 Z'/><path d='M86 62 L98 82 L122 91 L119 98 L91 93 L74 76 Z'/>
      </g>
    </svg>
    """


def hero_html(name: str = "there") -> str:
    return f"""
    <div class='fm-hero'><div class='fm-hero-inner'>
      <div class='fm-hero-copy'>
        <div class='fm-eyebrow'>YOUR DAILY MOVEMENT EDIT</div>
        <h1 class='fm-greeting'>Good day, <span>{escape(name)}</span> ✨</h1>
        <p class='fm-subtitle'>A little movement goes a long way.</p>
      </div>
      {runner_svg()}
      <div class='fm-date-pill'>♡ &nbsp;{__import__('datetime').datetime.now().strftime('%a, %d %b %Y')}</div>
    </div></div>
    """


def progress_rings_html(steps_pct: float, active_pct: float, sleep_pct: float, score: int) -> str:
    rings = [
        (98, 13, max(0, min(100, steps_pct)), "#e65a90"),
        (78, 12, max(0, min(100, active_pct)), "#f39cba"),
        (58, 11, max(0, min(100, sleep_pct)), "#c94d83"),
    ]
    svg_parts = []
    for radius, width, pct, color in rings:
        circumference = 2 * math.pi * radius
        offset = circumference * (1 - pct / 100)
        svg_parts.append(
            f"<circle cx='120' cy='120' r='{radius}' fill='none' stroke='#f9e8ee' stroke-width='{width}'/>"
            f"<circle cx='120' cy='120' r='{radius}' fill='none' stroke='{color}' stroke-width='{width}' "
            f"stroke-linecap='round' stroke-dasharray='{circumference:.2f}' stroke-dashoffset='{offset:.2f}' "
            f"transform='rotate(-90 120 120)'/>"
        )
    svg = "".join(svg_parts)
    return f"""
    <div class='fm-rings'><svg viewBox='0 0 240 240' width='100%' role='img' aria-label='Daily progress rings'>
      {svg}
      <circle cx='120' cy='120' r='43' fill='#fff9fb'/>
      <text x='120' y='111' text-anchor='middle' font-family='Outfit' font-size='12' fill='#8c6e7c' font-weight='600'>DAILY SCORE</text>
      <text x='120' y='140' text-anchor='middle' font-family='Outfit' font-size='31' fill='#302838' font-weight='800'>{int(score)}</text>
    </svg></div>
    """


def watch_art_html() -> str:
    return """
    <svg class='fm-watch-art' viewBox='0 0 130 190' role='img' aria-label='Illustration of a rose-pink smartwatch'>
      <defs><linearGradient id='strap' x1='0' x2='1'><stop stop-color='#d88da3'/><stop offset='.5' stop-color='#f4b8c9'/><stop offset='1' stop-color='#c87891'/></linearGradient><linearGradient id='case' x1='0' x2='1'><stop stop-color='#bd9a91'/><stop offset='.5' stop-color='#f0d9d0'/><stop offset='1' stop-color='#9d7b75'/></linearGradient></defs>
      <path d='M43 6 Q65 -1 87 6 L93 55 L37 55 Z' fill='url(#strap)'/><path d='M37 135 L93 135 L87 184 Q65 192 43 184 Z' fill='url(#strap)'/>
      <rect x='28' y='42' width='74' height='105' rx='23' fill='url(#case)'/><rect x='33' y='47' width='64' height='95' rx='19' fill='#191821'/>
      <rect x='39' y='54' width='52' height='81' rx='14' fill='#242230'/><circle cx='65' cy='72' r='3' fill='#f58eae'/>
      <text x='65' y='96' text-anchor='middle' font-family='DM Sans' font-size='19' font-weight='700' fill='#fff'>82</text><text x='65' y='108' text-anchor='middle' font-family='DM Sans' font-size='7' fill='#f9b6cb'>BPM</text>
      <path d='M42 120 L49 120 L53 112 L59 127 L65 116 L70 120 L76 120 L80 115 L85 120 L90 120' stroke='#f36b9b' stroke-width='1.7' fill='none' stroke-linecap='round'/>
      <circle cx='65' cy='151' r='2.5' fill='#c9a39e'/>
    </svg>
    """


def focus_art_html() -> str:
    return """
    <svg class='fm-focus-art' viewBox='0 0 260 130' role='img' aria-label='Pink fitness weights, towel and water bottle'>
      <ellipse cx='128' cy='111' rx='112' ry='12' fill='#f7dce6'/>
      <path d='M33 89 Q86 70 156 89 L178 108 Q105 119 34 103 Z' fill='#f6c7d7'/>
      <g transform='translate(35 40) rotate(-13 62 35)' fill='#df7d9e'><rect x='41' y='28' width='61' height='9' rx='4'/><rect x='28' y='16' width='16' height='34' rx='6'/><rect x='99' y='16' width='16' height='34' rx='6'/></g>
      <g transform='translate(175 13) rotate(9 22 49)'><rect x='7' y='5' width='28' height='11' rx='3' fill='#66515e'/><rect x='4' y='15' width='34' height='88' rx='12' fill='#f2b8ca'/><rect x='10' y='28' width='22' height='61' rx='8' fill='#fff1f6' opacity='.72'/><path d='M10 69 Q20 63 32 69 L32 86 L10 86 Z' fill='#e982a7'/></g>
      <path d='M89 76 Q121 62 152 79 L166 99 Q140 110 108 100 Z' fill='#fff8fa' stroke='#f1ccda' stroke-width='2'/><path d='M102 79 Q123 72 146 83' stroke='#efc3d2' stroke-width='2' fill='none'/>
    </svg>
    """


def section_head(title: str, subtitle: str = "") -> str:
    return f"<div class='fm-section-head'><h2>{escape(title)}</h2><span>{escape(subtitle)}</span></div>"


def focus_html(goal: str, steps_goal: int) -> str:
    return f"""
    <div class='fm-focus'>
      <div class='fm-focus-copy'><div class='fm-eyebrow'>A SMALL GOAL FOR TODAY</div>
      <h3 class='fm-focus-title'>Today's focus</h3>
      <p class='fm-focus-text'>Your focus is <strong>{escape(goal)}</strong>. Build a steady routine and aim for <strong>{int(steps_goal):,} steps</strong> today.</p></div>
      {focus_art_html()}
    </div>
    """


def device_card_html(device_name: str, subtitle: str = "Simulated device") -> str:
    return f"""
    <div class='fm-watch-row'>{watch_art_html()}<div style='flex:1;min-width:0'>
      <span class='fm-device-pill'>⌚ &nbsp;{escape(device_name)}</span><p class='fm-card-sub' style='margin-top:.5rem'>{escape(subtitle)}</p>
      <span class='fm-status'>● &nbsp;Demo feed active</span></div></div>
    """


# Streamlit's Markdown parser treats four-space-indented HTML as code blocks.
# Normalize every HTML fragment returned by this module before it reaches st.markdown.
def _dedented_html(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, str):
            return dedent(result).strip()
        return result
    return wrapper


_HTML_FUNCTIONS = (
    "sidebar_brand_html",
    "profile_html",
    "runner_svg",
    "hero_html",
    "progress_rings_html",
    "watch_art_html",
    "focus_art_html",
    "section_head",
    "focus_html",
    "device_card_html",
)

for _function_name in _HTML_FUNCTIONS:
    globals()[_function_name] = _dedented_html(globals()[_function_name])

del _function_name
