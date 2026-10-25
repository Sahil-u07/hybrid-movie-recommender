"""Streamlit front end. Run: streamlit run app.py"""
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

import recommender as R

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
ss = st.session_state
for key, default in dict(onboarded=False, genres=[], liked=[], disliked=[]).items():
    ss.setdefault(key, default)

DIALS = {"Comfort zone": "comfort", "A little different": "different", "Surprise me": "surprise"}
MODEL = R.get_model()  # lru_cache keeps one copy across reruns
TITLES = dict(zip(MODEL.ids, MODEL.movies.title))
LINKS = pd.read_csv(R.DATA / "links.csv").set_index("movieId").tmdbId.to_dict()


@st.cache_resource
def tmdb_key():
    try:
        return os.environ.get("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY")
    except Exception:  # no secrets file
        return None


@st.cache_data(show_spinner=False)
def poster(movie_id):
    """Poster URL from TMDB, or None when there is no key or no poster (never guessed)."""
    key, tid = tmdb_key(), LINKS.get(movie_id)
    if not key or pd.isna(tid):
        return None
    try:
        with urllib.request.urlopen(f"https://api.themoviedb.org/3/movie/{int(tid)}?api_key={key}", timeout=5) as r:
            path = json.load(r).get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


def like(m):
    if m in ss.disliked: ss.disliked.remove(m)
    if m not in ss.liked: ss.liked.append(m)


def dislike(m):
    if m in ss.liked: ss.liked.remove(m)
    if m not in ss.disliked: ss.disliked.append(m)


def card(it, row):
    m = it["movie_id"]
    url = poster(m)
    if url:
        st.image(url, use_container_width=True)
    else:
        st.markdown(f"<div style='aspect-ratio:2/3;background:#1e2430;border-radius:8px;padding:12px;"
                    f"display:flex;align-items:flex-end;font-weight:600'>{it['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"**{it['title']}**")
    rating = f"{it['avg_rating']:.1f}/5 · " if it["avg_rating"] else ""
    st.caption(f"{rating}{', '.join(it['genres'][:3]) or 'no genre listed'}")
    st.caption(it["explanation"])
    a, b = st.columns(2)
    a.button("Like", key=f"{row}_like_{m}", on_click=like, args=(m,), use_container_width=True)
    b.button("Not for me", key=f"{row}_no_{m}", on_click=dislike, args=(m,), use_container_width=True)
    with st.expander("Why this?"):
        st.caption(f"Score {it['score']:.2f}. Contribution of each signal:")
        st.bar_chart(pd.Series(it["breakdown"]), height=140)


def show_row(title, items, row, per_line=5):
    stage_note = "  ·  still learning your taste" if items and items[0]["stage"] != "established" else ""
    st.subheader(title)
    if stage_note:
        st.caption(stage_note.strip(" ·"))
    for start in range(0, len(items), per_line):
        for col, it in zip(st.columns(per_line), items[start:start + per_line]):
            with col:
                card(it, f"{row}{start}")


def onboarding():
    st.title("Pick a movie you will enjoy tonight")
    st.write("Two optional questions. Skip either one and you still get picks.")
    popular = MODEL.movies.assign(n=MODEL.count).nlargest(1500, "n").movieId.tolist()
    with st.form("quiz"):
        genres = st.multiselect("Genres you like", MODEL.gnames)
        favs = st.multiselect("Up to 5 favourite films (type to search)", popular, max_selections=5,
                              format_func=TITLES.get)
        go, skip = st.columns(2)
        start = go.form_submit_button("Show my picks", type="primary")
        skipped = skip.form_submit_button("Skip")
    if start or skipped:
        if start:
            ss.genres, ss.liked = genres, list(favs)
        ss.onboarded = True
        st.rerun()


def home():
    st.title("Picked for you")
    c1, c2 = st.columns([3, 2])
    mood = c1.pills("Mood tonight", [m.capitalize() for m in R.MOODS], key="mood")
    dial = DIALS[c2.radio("Discovery dial", list(DIALS), index=1, horizontal=True)]
    ctx = {"mood": mood.lower() if mood else None, "discovery": dial}
    profile = dict(genres=ss.genres, liked=ss.liked, disliked=ss.disliked)
    main = MODEL.recommend(profile, ctx, 10)
    taken = [it["movie_id"] for it in main]

    def more(extra_profile, extra_ctx, k=5):
        p = dict(extra_profile, seen=ss.liked + ss.disliked + taken)
        items = MODEL.recommend(p, extra_ctx, k)
        taken.extend(it["movie_id"] for it in items)
        return items

    show_row("Picked for you", main, "main")
    if ss.liked:
        last = ss.liked[-1]
        show_row(f"Because you liked {TITLES[last]}",
                 more(dict(genres=[], liked=[last], disliked=ss.disliked), {"discovery": dial}), "because")
    if mood:
        show_row(f"Fits a {mood.lower()} night", more(profile, ctx), "mood")
    show_row("Try something different", more(profile, {"mood": ctx["mood"], "discovery": "surprise"}), "diff")
    if tmdb_key():
        st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")


def profile_page():
    st.title("Your taste profile")
    ss.genres = st.multiselect("Genres you like", MODEL.gnames, default=ss.genres)
    for label, key in [("Liked", "liked"), ("Not for me", "disliked")]:
        st.subheader(f"{label} ({len(ss[key])})")
        for m in list(ss[key]):
            a, b = st.columns([6, 1])
            a.write(TITLES[m])
            if b.button("Remove", key=f"rm_{key}_{m}"):
                ss[key].remove(m)
                st.rerun()
    if st.button("Clear everything and restart the quiz"):
        for k in ("genres", "liked", "disliked"): ss[k] = []
        ss.onboarded = False
        st.rerun()


def about():
    st.title("How it works")
    st.write("Six signals are min-max normalised and combined with weights that depend on how much you have "
             "interacted: **preference** (quiz genres), **content** (TF-IDF on genres and tags), **collaborative** "
             "(item-item, adjusted cosine), **context** (mood), **quality** (Bayesian average and popularity), "
             "**discovery** (unfamiliar genres that still fit). An MMR re-rank then trades score against similarity "
             "to films already in the row. The discovery dial sets that trade-off.")
    f = Path(__file__).parent / "results" / "eval_results.csv"
    if f.exists():
        st.subheader("Offline evaluation (held-out last 20% of each user's ratings)")
        df = pd.read_csv(f, index_col=[0, 1]).round(4)
        for part in df.index.levels[0]:
            st.caption(part.replace("_", " "))
            st.dataframe(df.loc[part])


if not ss.onboarded:
    onboarding()
else:
    page = st.sidebar.radio("Go to", ["Home", "Taste profile", "How it works"])
    st.sidebar.caption(f"{len(ss.liked)} liked · {len(ss.disliked)} not for me · stage: "
                       f"{R.stage_of(dict(liked=ss.liked, disliked=ss.disliked))}")
    {"Home": home, "Taste profile": profile_page, "How it works": about}[page]()
