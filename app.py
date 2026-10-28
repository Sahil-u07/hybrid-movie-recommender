"""Streamlit front end. Run: streamlit run app.py"""
import html
import json
import re
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

import recommender as R

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide",
                   initial_sidebar_state="collapsed")

ss = st.session_state
for key, default in dict(onboarded=False, genres=[], liked=[], disliked=[], page="Home").items():
    ss.setdefault(key, default)

DIALS = {"Comfort zone": "comfort", "A little different": "different", "Surprise me": "surprise"}
PAGES = ["Home", "My taste", "How it works"]
SIGNAL_LABEL = {"preference": "Your genres", "content": "Similar films", "collab": "People like you",
                "context": "Tonight's mood", "quality": "Rating", "discovery": "Discovery"}
# Deterministic poster backgrounds, so a film keeps the same colours between reruns.
PALETTES = [("#3A1C4A", "#8E2C48"), ("#12324A", "#2E8B84"), ("#4A2B18", "#C97B3C"),
            ("#1E2140", "#4F6D8F"), ("#3F1726", "#B9324F"), ("#17352B", "#5E8C5A"),
            ("#2C2340", "#6D5C8F"), ("#43321A", "#F2B33D")]

ARTICLE = re.compile(r"^(?P<body>.+), (?P<art>The|A|An)(?P<tail> \(.*)?$")


def pretty(title):
    """MovieLens stores "Matrix, The (1999)"; read it back as "The Matrix (1999)"."""
    m = ARTICLE.match(title)
    return f"{m['art']} {m['body']}{m['tail'] or ''}" if m else title


MODEL = R.get_model()
TITLES = {int(i): pretty(t) for i, t in zip(MODEL.ids, MODEL.movies.title)}
RAW = {int(i): t for i, t in zip(MODEL.ids, MODEL.movies.title)}
LINKS = pd.read_csv(R.DATA / "links.csv").set_index("movieId").tmdbId.to_dict()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#0B0A0F; --panel:#16141F; --panel2:#1E1B29;
  --ink:#F3F1F7; --muted:#9F9BAE; --line:#2A2635;
  --wine:#C9365A; --wine-dk:#8E2C48; --gold:#F2B33D;
  --display:'Bebas Neue','Haettenschweiler','Arial Narrow',Impact,sans-serif;
  --sans:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
}
html,body,.stApp{background:var(--bg);color:var(--ink);font-family:var(--sans);}
header[data-testid="stHeader"]{background:transparent;}
.stMainBlockContainer{padding:3.2rem 2.2rem 4rem;max-width:1680px;}
#MainMenu,footer{visibility:hidden;}
::selection{background:var(--wine);color:#fff;}
*:focus-visible{outline:2px solid var(--gold);outline-offset:2px;border-radius:6px;}

/* ---------- top bar ---------- */
.st-key-nav{border-bottom:1px solid var(--line);margin-bottom:1.1rem;}
.st-key-nav .stHorizontalBlock{align-items:center;gap:6px;}
.wordmark{font-family:var(--display);font-size:30px;letter-spacing:.055em;line-height:1;
  color:var(--wine);margin:2px 0 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wordmark em{color:var(--ink);font-style:normal;}
.st-key-nav .stButton>button{background:transparent;border:0;color:var(--muted);
  font-weight:600;font-size:14px;padding:6px 10px;box-shadow:none;}
.st-key-nav .stButton>button:hover{color:var(--ink);background:transparent;}
.st-key-nav .stButton>button[kind="primary"]{color:#fff;background:transparent;border:0;
  box-shadow:inset 0 -2px 0 var(--wine);border-radius:0;}
.st-key-nav .stButton>button[kind="primary"]:hover{background:transparent;color:#fff;}
.stagepill{text-align:right;color:var(--muted);font-size:12px;padding-top:4px;}
.stagepill b{color:var(--gold);font-weight:600;}

/* ---------- hero ---------- */
.hero{position:relative;border-radius:16px;overflow:hidden;border:1px solid var(--line);
  min-height:330px;display:flex;align-items:flex-end;margin-bottom:.4rem;}
.hero .wash{position:absolute;inset:0;}
.hero .wash.img{background-size:cover;background-position:center top;}
.hero .wash.img.blur{filter:blur(26px) saturate(1.3);transform:scale(1.12);}
.hero .veil.img{position:absolute;inset:0;
  background:linear-gradient(90deg,rgba(11,10,15,.97) 0%,rgba(11,10,15,.88) 46%,rgba(11,10,15,.45) 100%);}
.hero .art{position:absolute;right:30px;top:50%;transform:translateY(-50%);width:160px;aspect-ratio:2/3;
  border-radius:10px;overflow:hidden;border:1px solid var(--line);box-shadow:0 14px 34px rgba(0,0,0,.6);}
.hero .art img{width:100%;height:100%;object-fit:cover;display:block;}
.hero .veil{position:absolute;inset:0;
  background:linear-gradient(90deg,rgba(11,10,15,.95) 0%,rgba(11,10,15,.72) 42%,rgba(11,10,15,.25) 100%);}
.hero .inner{position:relative;padding:28px 32px;max-width:720px;}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);
  font-weight:600;margin-bottom:8px;}
.hero h1{font-family:var(--display);font-size:clamp(30px,3.4vw,46px);line-height:1.02;
  letter-spacing:.01em;margin:0 0 10px;color:#fff;}
.hero h1 a,.hero h1 svg{display:none!important;}
.hero .why{font-size:16px;color:#E7E4EE;margin:0 0 12px;max-width:60ch;}
.facts{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.chip{font-size:11.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);
  background:rgba(255,255,255,.05);color:var(--muted);}
.chip.score{border-color:rgba(242,179,61,.45);color:var(--gold);}

/* ---------- rows ---------- */
.rowhead{font-size:17px;font-weight:700;color:var(--ink);margin:18px 0 2px;letter-spacing:.01em;}
.rowsub{font-size:12.5px;color:var(--muted);margin:0 0 2px;}
[class*="st-key-row_"] .stHorizontalBlock{flex-wrap:nowrap!important;overflow-x:auto;
  gap:14px!important;padding:8px 2px 16px;scroll-snap-type:x proximity;}
[class*="st-key-row_"] .stColumn{flex:0 0 190px!important;min-width:190px!important;
  width:190px!important;scroll-snap-align:start;}
[class*="st-key-row_"] .stHorizontalBlock::-webkit-scrollbar{height:8px;}
[class*="st-key-row_"] .stHorizontalBlock::-webkit-scrollbar-thumb{background:#2E2A3B;border-radius:8px;}
[class*="st-key-row_"] .stHorizontalBlock::-webkit-scrollbar-thumb:hover{background:var(--wine-dk);}
[class*="st-key-row_"] .stHorizontalBlock::-webkit-scrollbar-track{background:transparent;}

/* ---------- card ---------- */
.card{transition:transform .18s ease;}
.card:hover{transform:translateY(-4px);}
.poster{position:relative;aspect-ratio:2/3;border-radius:10px;overflow:hidden;
  border:1px solid var(--line);background:var(--panel);
  box-shadow:0 6px 18px rgba(0,0,0,.45);}
.poster img{width:100%;height:100%;object-fit:cover;display:block;}
.poster .ph{width:100%;height:100%;display:flex;align-items:flex-end;padding:12px;}
.poster .ph span{font-family:var(--display);font-size:20px;line-height:1.05;color:#fff;
  text-shadow:0 2px 10px rgba(0,0,0,.6);display:-webkit-box;-webkit-line-clamp:4;
  -webkit-box-orient:vertical;overflow:hidden;}
.poster .rate{position:absolute;top:7px;right:7px;background:rgba(11,10,15,.82);
  color:var(--gold);font-size:11px;font-weight:600;padding:2px 6px;border-radius:6px;
  border:1px solid rgba(242,179,61,.3);}
.card:hover .poster{border-color:var(--wine);box-shadow:0 10px 26px rgba(201,54,90,.28);}
.ctitle{font-size:13.5px;font-weight:600;color:var(--ink);margin:9px 0 2px;line-height:1.25;
  height:2.5em;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.cgen{font-size:11px;color:var(--muted);margin-bottom:5px;height:1.2em;overflow:hidden;white-space:nowrap;
  text-overflow:ellipsis;}
.cwhy{font-size:11.5px;color:#C7C2D4;line-height:1.35;border-left:2px solid var(--wine-dk);
  padding-left:7px;height:48px;margin-bottom:8px;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}

/* ---------- buttons ---------- */
.stButton>button{border-radius:8px;font-weight:600;font-size:12px;padding:5px 8px;
  border:1px solid var(--line);background:var(--panel2);color:var(--ink);width:100%;
  transition:background .15s ease,border-color .15s ease;}
.stButton>button:hover{border-color:var(--wine);color:#fff;background:#241f30;}
.stButton>button[kind="primary"]{background:var(--wine);border-color:var(--wine);color:#fff;}
.stButton>button[kind="primary"]:hover{background:#DD3F64;border-color:#DD3F64;}

/* ---------- why-this panel ---------- */
.stExpander{border:0;}
.stExpander details{background:transparent;border:1px solid var(--line);border-radius:8px;}
.stExpander summary{font-size:11.5px;color:var(--muted);padding:4px 8px;}
.stExpander summary:hover{color:var(--ink);}
.bars{display:flex;flex-direction:column;gap:6px;padding:2px 2px 4px;}
.sig{display:grid;grid-template-columns:84px 1fr 34px;gap:7px;align-items:center;font-size:11px;}
.sig span{color:var(--muted);}
.sig i{display:block;height:6px;border-radius:4px;
  background:linear-gradient(90deg,var(--wine-dk),var(--wine));}
.sig b{color:var(--ink);text-align:right;font-weight:600;}

/* ---------- controls ---------- */
.st-key-controls{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:12px 16px 4px;margin:14px 0 2px;}
.st-key-controls .stHorizontalBlock{flex-wrap:wrap;}
.st-key-controls .stButtonGroup>div{flex-wrap:wrap!important;overflow:visible!important;gap:6px;}
.st-key-controls label{color:var(--muted)!important;font-size:12px!important;font-weight:600!important;}

/* ---------- quiz ---------- */
.st-key-quiz{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:24px 26px 8px;}
.quizhead{font-family:var(--display);font-size:clamp(32px,4.6vw,54px);line-height:1;color:#fff;margin:0 0 8px;}
.quizsub{color:var(--muted);font-size:15px;margin:0 0 18px;max-width:62ch;}
.stMultiSelect div[data-baseweb="tag"]{background:var(--wine-dk)!important;}

/* ---------- misc ---------- */
.secthead{font-family:var(--display);font-size:34px;color:#fff;margin:4px 0 10px;letter-spacing:.02em;}
.note{color:var(--muted);font-size:12.5px;}
.stDataFrame{border:1px solid var(--line);border-radius:10px;}
hr{border-color:var(--line);}
@media (max-width:760px){
  .stMainBlockContainer{padding:2.8rem 1rem 3rem;}
  .st-key-nav .stHorizontalBlock{flex-wrap:wrap!important;gap:0;}
  .st-key-nav .stColumn{min-width:0!important;flex:0 0 32%!important;}
  .st-key-nav .stColumn:has(.wordmark){flex:0 0 100%!important;}
  .st-key-nav .stButton>button{font-size:13px;padding:6px 4px;}
  .wordmark{font-size:24px;margin:0 0 8px;}
  .stagepill{display:none;}
  [class*="st-key-row_"] .stColumn{flex:0 0 150px!important;min-width:150px!important;width:150px!important;}
  .hero .inner{padding:20px;}
  .hero{min-height:280px;}
  .hero .art{display:none;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def tmdb_key():
    try:
        return os.environ.get("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY")
    except Exception:  # no secrets file
        return None


# Overridable so the artwork paths can be tested against local stand-ins. ARTWORK=off skips both.
TMDB_BASE = os.environ.get("TMDB_BASE", "https://api.themoviedb.org/3")
TMDB_IMG = os.environ.get("TMDB_IMG", "https://image.tmdb.org/t/p")
WIKI_API = os.environ.get("WIKI_API", "https://en.wikipedia.org/w/api.php")
ARTWORK = os.environ.get("ARTWORK", "on") != "off"
WIKI_UA = ("HybridMovieRecommender/1.0 (student project; "
           "https://github.com/Sahil-u07/hybrid-movie-recommender)")
# Drop a parenthetical that is not the year: "Seven (a.k.a. Se7en) (1995)" -> "Seven (1995)".
_ALIAS = re.compile(r"\s*\((?!\d{4}\))[^)]*\)")


@st.cache_resource
def _art_cache():
    """movie_id -> (poster url, backdrop url). Shared: artwork is not per-user."""
    return {}


def _fetch_art(tid, key):
    """One TMDB lookup. Touches no Streamlit API, so it is safe to run in a thread."""
    try:
        with urllib.request.urlopen(f"{TMDB_BASE}/movie/{tid}?api_key={key}", timeout=6) as r:
            d = json.load(r)
        pick = lambda p, size: f"{TMDB_IMG}/{size}{p}" if p else None
        return pick(d.get("poster_path"), "w342"), pick(d.get("backdrop_path"), "w1280")
    except Exception:
        return None, None


def wiki_query(title):
    """"Seven (a.k.a. Se7en) (1995)" -> "Seven 1995 film", which Wikipedia search resolves."""
    t = _ALIAS.sub("", title).replace("(", "").replace(")", "")
    return f"{' '.join(t.split())} film"


def _fetch_wiki(title):
    """Lead image of the film's English Wikipedia article. No key, but no wide backdrop either."""
    q = urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": wiki_query(title),
        "gsrlimit": 1, "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 400,
        "pilicense": "any", "format": "json"})  # posters are non-free, so "any" is required
    try:
        req = urllib.request.Request(f"{WIKI_API}?{q}", headers={"User-Agent": WIKI_UA})
        with urllib.request.urlopen(req, timeout=6) as r:
            pages = json.load(r).get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return page.get("thumbnail", {}).get("source"), None
    except Exception:
        return None, None


def prefetch(movie_ids):
    """Load every poster a row needs at once; one at a time would stall the first paint."""
    if not ARTWORK:
        return
    cache = _art_cache()
    todo = [m for m in movie_ids if m not in cache]
    if not todo:
        return
    key = tmdb_key()
    if key:
        jobs = {m: int(LINKS[m]) for m in todo if not pd.isna(LINKS.get(m))}
        fn = lambda tid: _fetch_art(tid, key)
    else:
        jobs = {m: TITLES[m] for m in todo}
        fn = _fetch_wiki
    if jobs:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for m, found in zip(jobs, ex.map(fn, jobs.values())):
                cache[m] = found
    for m in todo:
        cache.setdefault(m, (None, None))  # a miss is cached too, or every rerun retries it


def art(movie_id):
    """(poster, backdrop), each None when there is no key or no image. Never guessed."""
    return _art_cache().get(movie_id, (None, None))


def poster(movie_id):
    return art(movie_id)[0]


def esc(s):
    return html.escape(str(s))


def wash(movie_id, angle="145deg"):
    a, b = PALETTES[movie_id % len(PALETTES)]
    return f"linear-gradient({angle},{a} 0%,{b} 100%)"


def poster_html(it):
    url = poster(it["movie_id"])
    rate = f"<span class='rate'>{it['avg_rating']:.1f}</span>" if it["avg_rating"] else ""
    if url:
        art = f"<img src='{esc(url)}' alt='{esc(TITLES[it['movie_id']])} poster' loading='lazy'>"
    else:
        art = (f"<div class='ph' style='background:{wash(it['movie_id'])}'>"
               f"<span>{esc(TITLES[it['movie_id']])}</span></div>")
    return f"<div class='poster'>{art}{rate}</div>"


def bars_html(breakdown):
    """The score breakdown as bars, so the shown reason is traceable to real numbers."""
    total = sum(breakdown.values()) or 1
    top = max(breakdown.values()) or 1
    rows = "".join(
        f"<div class='sig'><span>{esc(SIGNAL_LABEL.get(k, k))}</span>"
        f"<i style='width:{max(v / top * 100, 3):.0f}%'></i>"
        f"<b>{v / total * 100:.0f}%</b></div>"
        for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1]))
    return f"<div class='bars'>{rows}</div>"


def like(m):
    if m in ss.disliked: ss.disliked.remove(m)
    if m not in ss.liked: ss.liked.append(m)
    st.toast(f"Liked {TITLES[m]}", icon=":material/thumb_up:")


def dislike(m):
    if m in ss.liked: ss.liked.remove(m)
    if m not in ss.disliked: ss.disliked.append(m)
    st.toast(f"Hidden {TITLES[m]}", icon=":material/visibility_off:")


def card(it, slot, omit=None):
    m = it["movie_id"]
    why = it["explanation"]
    if omit and why.startswith(omit):  # the row header already says this
        why = why[len(omit):].strip() or it["explanation"]
    st.markdown(
        f"<div class='card'>{poster_html(it)}"
        f"<div class='ctitle'>{esc(TITLES[m])}</div>"
        f"<div class='cgen'>{esc(', '.join(it['genres'][:2]) or 'no genre listed')}</div>"
        f"<div class='cwhy'>{esc(why)}</div></div>", unsafe_allow_html=True)
    st.button("Like", key=f"{slot}_like_{m}", on_click=like, args=(m,), use_container_width=True)
    st.button("Not for me", key=f"{slot}_no_{m}", on_click=dislike, args=(m,), use_container_width=True)
    with st.expander("Why this?"):
        st.markdown(bars_html(it["breakdown"]), unsafe_allow_html=True)
        st.caption(f"Score {it['score']:.2f} · {it['n_ratings']} ratings")


def show_row(title, subtitle, items, slot, omit=None):
    if not items:
        return
    prefetch([it["movie_id"] for it in items])
    st.markdown(f"<div class='rowhead'>{esc(title)}</div>"
                f"<div class='rowsub'>{esc(subtitle)}</div>", unsafe_allow_html=True)
    with st.container(key=f"row_{slot}"):
        for col, it in zip(st.columns(len(items)), items):
            with col:
                card(it, slot, omit)


def hero(it):
    prefetch([it["movie_id"]])
    pos, back = art(it["movie_id"])
    genres = ", ".join(it["genres"][:3]) or "no genre listed"
    dup = "Rated" in it["explanation"]  # the reason already carries the rating
    rating = ("" if dup or not it["avg_rating"]
              else f"<span class='chip'>{it['avg_rating']:.1f}/5 from {it['n_ratings']} viewers</span>")
    st.markdown(
        f"<div class='hero'>"
        + (f"<div class='wash img{'' if back else ' blur'}' style=\"background-image:url('{esc(back or pos)}')\"></div>"
           f"<div class='veil img'></div>" if (back or pos) else
           f"<div class='wash' style='background:{wash(it['movie_id'], '110deg')}'></div><div class='veil'></div>")
        + (f"<div class='art'><img src='{esc(pos)}' alt='{esc(TITLES[it['movie_id']])} poster'></div>" if pos else "")
        + f"<div class='inner'>"
        f"<div class='kicker'>Tonight's top pick</div>"
        f"<h1>{esc(TITLES[it['movie_id']])}</h1>"
        f"<p class='why'>{esc(it['explanation'])}</p>"
        f"<div class='facts'><span class='chip'>{esc(genres)}</span>{rating}</div>"
        f"</div></div>", unsafe_allow_html=True)
    m = it["movie_id"]
    a, b, _ = st.columns([1, 1, 4])
    a.button("Like this", key=f"hero_like_{m}", on_click=like, args=(m,), type="primary",
             use_container_width=True)
    b.button("Not for me", key=f"hero_no_{m}", on_click=dislike, args=(m,), use_container_width=True)
    with st.expander("Why this?"):
        st.markdown(bars_html(it["breakdown"]), unsafe_allow_html=True)


def nav():
    with st.container(key="nav"):
        cols = st.columns([2.2, 1, 1, 1.2, 2.4])
        cols[0].markdown("<div class='wordmark'>HYBRID<em>REC</em></div>", unsafe_allow_html=True)
        for col, name in zip(cols[1:4], PAGES):
            if col.button(name, key=f"nav_{name}", type="primary" if ss.page == name else "secondary"):
                ss.page = name
                st.rerun()
        stage = R.stage_of(dict(liked=ss.liked, disliked=ss.disliked))
        cols[4].markdown(f"<div class='stagepill'>{len(ss.liked)} liked · {len(ss.disliked)} hidden · "
                         f"profile <b>{stage}</b></div>", unsafe_allow_html=True)


def onboarding():
    st.markdown("<div class='wordmark' style='font-size:34px'>HYBRID<em>REC</em></div>", unsafe_allow_html=True)
    with st.container(key="quiz"):
        st.markdown("<h1 class='quizhead'>Find something worth watching tonight</h1>"
                    "<p class='quizsub'>Two optional questions. Skip either one and you still get picks, "
                    "and every recommendation tells you why it is there.</p>", unsafe_allow_html=True)
        popular = MODEL.movies.assign(n=MODEL.count).nlargest(1500, "n").movieId.tolist()
        with st.form("quiz"):
            genres = st.multiselect("Genres you like", MODEL.gnames)
            favs = st.multiselect("Up to 5 favourite films (type to search)", popular,
                                  max_selections=5, format_func=TITLES.get)
            go, skip, _ = st.columns([1.2, 1, 3])
            start = go.form_submit_button("Show my picks", type="primary", use_container_width=True)
            skipped = skip.form_submit_button("Skip", use_container_width=True)
    if start or skipped:
        if start:
            ss.genres, ss.liked = genres, list(favs)
        ss.onboarded = True
        st.rerun()


def home():
    with st.container(key="controls"):
        c1, c2 = st.columns([3, 2])
        mood = c1.pills("Mood tonight", [m.capitalize() for m in R.MOODS], key="mood")
        dial = DIALS[c2.radio("Discovery dial", list(DIALS), index=1, horizontal=True)]
    ctx = {"mood": mood.lower() if mood else None, "discovery": dial}
    profile = dict(genres=ss.genres, liked=ss.liked, disliked=ss.disliked)
    main = MODEL.recommend(profile, ctx, 11)
    hero(main[0])
    taken = [it["movie_id"] for it in main]

    def more(extra_profile, extra_ctx, k=8):
        p = dict(extra_profile, seen=ss.liked + ss.disliked + taken)
        items = MODEL.recommend(p, extra_ctx, k)
        taken.extend(it["movie_id"] for it in items)
        return items

    stage = R.stage_of(profile)
    note = {"cold": "Popular and well rated, until you tell the app what you like",
            "warming": "Still learning your taste",
            "established": "Built from your likes and from viewers who agree with you"}[stage]
    show_row("Picked for you", note, main[1:], "main")
    if ss.liked:
        last = ss.liked[-1]
        show_row(f"Because you liked {TITLES[last]}", "Closest matches to that one",
                 more(dict(genres=[], liked=[last], disliked=ss.disliked), {"discovery": dial}), "because",
                 omit=f"Popular with people who liked {RAW[last]}.")
    if mood:
        show_row(f"Fits a {mood.lower()} night", "Matched to tonight's mood", more(profile, ctx), "mood")
    show_row("Try something different", "Outside your usual genres, still close to your taste",
             more(profile, {"mood": ctx["mood"], "discovery": "surprise"}), "diff")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB."
               if tmdb_key() else
               "Posters come from the films' English Wikipedia articles. Set a TMDB API key as "
               "TMDB_API_KEY in the app's secrets to use TMDB artwork instead.")


def profile_page():
    st.markdown("<div class='secthead'>Your taste profile</div>", unsafe_allow_html=True)
    ss.genres = st.multiselect("Genres you like", MODEL.gnames, default=ss.genres)
    for label, key in [("Liked", "liked"), ("Not for me", "disliked")]:
        st.markdown(f"<div class='rowhead'>{label} ({len(ss[key])})</div>", unsafe_allow_html=True)
        if not ss[key]:
            st.markdown("<p class='note'>Nothing here yet.</p>", unsafe_allow_html=True)
        for m in list(ss[key]):
            a, b = st.columns([6, 1])
            a.write(TITLES[m])
            if b.button("Remove", key=f"rm_{key}_{m}"):
                ss[key].remove(m)
                st.rerun()
    st.divider()
    if st.button("Clear everything and restart the quiz"):
        for k in ("genres", "liked", "disliked"): ss[k] = []
        ss.onboarded = False
        ss.page = "Home"
        st.rerun()


def about():
    st.markdown("<div class='secthead'>How it works</div>", unsafe_allow_html=True)
    st.write("Six signals are min-max normalised and combined with weights that depend on how much you have "
             "interacted: **preference** (quiz genres), **content** (TF-IDF on genres and tags), **collaborative** "
             "(item-item, adjusted cosine), **context** (mood), **quality** (Bayesian average and popularity), "
             "**discovery** (unfamiliar genres that still fit). An MMR re-rank then trades score against similarity "
             "to films already in the row. The discovery dial sets that trade-off.")
    f = Path(__file__).parent / "results" / "eval_results.csv"
    if f.exists():
        st.markdown("<div class='rowhead'>Offline evaluation (held-out last 20% of each user's ratings)</div>",
                    unsafe_allow_html=True)
        df = pd.read_csv(f, index_col=[0, 1]).round(4)
        for part in df.index.levels[0]:
            st.caption(part.replace("_", " "))
            st.dataframe(df.loc[part], use_container_width=True)


if not ss.onboarded:
    onboarding()
else:
    nav()
    {"Home": home, "My taste": profile_page, "How it works": about}[ss.page]()
