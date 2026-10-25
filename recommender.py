"""Hybrid movie recommender on MovieLens ml-latest-small.

Six signals (preference, content, collaborative, context, quality, discovery) are
min-max normalised inside the candidate set, weighted by profile stage, then
re-ranked with MMR. Public entry point: recommend(profile, context, k).
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse

DATA = Path(__file__).parent / "data"
SIGNALS = ["preference", "content", "collab", "context", "quality", "discovery"]

# Weights per profile stage, from section 2 of the project spec (untuned starting values).
SPEC_W = {
    "cold":        dict(preference=.45, content=.15, collab=0.0, context=.15, quality=.15, discovery=.10),
    "warming":     dict(preference=.30, content=.25, collab=.10, context=.15, quality=.10, discovery=.10),
    "established": dict(preference=.20, content=.25, collab=.25, context=.15, quality=.05, discovery=.10),
}
STAGE_W = {k: dict(v) for k, v in SPEC_W.items()}
# Tuned by tune.py on a validation split of the training data (floor 0.05 per signal, quality capped at 0.15). Cold stage keeps the spec values.
STAGE_W["warming"].update(preference=.13, content=.057, collab=.424, quality=.139)
STAGE_W["established"].update(preference=.098, content=.065, collab=.448, quality=.139)
# Discovery dial: (MMR lambda, discovery weight)
DIAL = {"comfort": (0.85, 0.05), "different": (0.70, 0.10), "surprise": (0.55, 0.20)}
EVENT_W = {"liked": 1.0, "disliked": -0.7}  # event weights from spec section 2
MOODS = {
    "funny": ["Comedy"],
    "thrilling": ["Thriller", "Crime", "Mystery", "Horror"],
    "comforting": ["Animation", "Children", "Comedy", "Romance", "Musical", "Fantasy"],
    "thought-provoking": ["Drama", "Documentary", "Sci-Fi", "Mystery"],
    "romantic": ["Romance"],
    "high-energy": ["Action", "Adventure", "Sci-Fi"],
}
TEMPLATES = {
    "content": "Similar to {ref}, which you liked.",
    "collab": "Popular with people who liked {ref}.",
    "preference": "Matches your love of {genres}.",
    "context": "Fits a {mood} night.",
    "quality": "Rated {avg:.1f}/5 by {n} viewers.",
    "discovery": "Something different that still fits your taste.",
}


def stage_of(profile):
    n = len(profile.get("liked", [])) + len(profile.get("disliked", []))
    return "cold" if n == 0 else "warming" if n < 20 else "established"


def mm(x):
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


@dataclass
class Prep:
    cand: np.ndarray      # item indices that may be recommended
    sig: dict             # signal name -> normalised scores over cand (only active signals)
    stage: str
    penalty: np.ndarray   # seen_recently penalty over cand
    liked: list           # item indices with positive weight
    L: list               # all interacted item indices
    w: np.ndarray         # weights matching L
    genres: list


class Model:
    def __init__(self, movies, ratings, tags, min_ratings=5):
        self.min_ratings = min_ratings
        self.movies = movies.reset_index(drop=True)
        self.ids = self.movies.movieId.values
        self.idx = {m: i for i, m in enumerate(self.ids)}
        n = len(self.movies)
        gl = self.movies.genres.str.replace("(no genres listed)", "", regex=False).str.split("|")
        self.glist = [[g for g in l if g] for l in gl]
        self.gnames = sorted({g for l in self.glist for g in l})
        self.gcol = {g: i for i, g in enumerate(self.gnames)}
        rows = [i for i, l in enumerate(self.glist) for _ in l]
        cols = [self.gcol[g] for l in self.glist for g in l]
        self.Gbin = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(self.gnames)))
        self.G = normalize(self.Gbin)
        self.primary = [l[0] if l else "" for l in self.glist]
        self.year = self.movies.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).values

        # TF-IDF over genre tokens + user tags
        gtxt = [" ".join(g.lower().replace("-", "") for g in l) for l in self.glist]
        tagtxt = tags.assign(tag=tags.tag.astype(str).str.lower()).groupby("movieId").tag.apply(" ".join)
        tagtxt = tagtxt.reindex(self.ids).fillna("").values
        self.T = normalize(TfidfVectorizer(sublinear_tf=True, token_pattern=r"[a-z0-9]+")
                           .fit_transform([f"{a} {a} {b}" for a, b in zip(gtxt, tagtxt)]))

        # Item statistics and item-item CF, built from the ratings passed in
        r = ratings[ratings.movieId.isin(self.idx)]
        cnt = r.groupby("movieId").size().reindex(self.ids).fillna(0).values
        mean = r.groupby("movieId").rating.mean().reindex(self.ids)
        gm = r.rating.mean()
        self.count = cnt
        self.avg = mean.values
        self.bayes = (cnt * mean.fillna(gm).values + 10 * gm) / (cnt + 10)
        self.logc = np.log1p(cnt)
        ui = pd.Categorical(r.userId).codes
        ii = r.movieId.map(self.idx).values
        centered = (r.rating - r.groupby("userId").rating.transform("mean")).values
        shape = (n, ui.max() + 1)
        self.X = normalize(sparse.csr_matrix((centered, (ii, ui)), shape=shape))
        self.B = sparse.csr_matrix((np.ones(len(ii)), (ii, ui)), shape=shape)

    # ---- signals -------------------------------------------------------
    def _cf_sims(self, L, cols=None):
        """Shrunk adjusted-cosine similarity of interacted items L to all items (or cols)."""
        S = (self.X[L] @ self.X.T).toarray()
        C = (self.B[L] @ self.B.T).toarray()
        S = S * C / (C + 10)  # shrink pairs with few common raters
        return S if cols is None else S[:, cols]

    def prepare(self, profile, context=None):
        context = context or {}
        liked = [self.idx[m] for m in profile.get("liked", []) if m in self.idx]
        disliked = [self.idx[m] for m in profile.get("disliked", []) if m in self.idx]
        L = liked + disliked
        w = np.array([EVENT_W["liked"]] * len(liked) + [EVENT_W["disliked"]] * len(disliked))
        n = len(self.ids)
        excl = np.zeros(n, bool)
        excl[L] = True
        excl[[self.idx[m] for m in profile.get("seen", []) if m in self.idx]] = True
        avoid = [self.gcol[g] for g in profile.get("avoid", []) if g in self.gcol]
        if avoid:  # stated avoids are hard filters
            excl |= np.asarray(self.Gbin[:, avoid].sum(axis=1)).ravel() > 0
        cand = np.flatnonzero((self.count >= self.min_ratings) & ~excl)

        raw = {}
        genres = [g for g in profile.get("genres", []) if g in self.gcol]
        if genres:
            p = np.zeros(len(self.gnames)); p[[self.gcol[g] for g in genres]] = 1
            raw["preference"] = self.G @ (p / np.linalg.norm(p))
        if L:
            raw["content"] = self.T @ (self.T[L].T @ w)
            raw["collab"] = w @ self._cf_sims(L)
        mood = (context.get("mood") or "").lower()
        if mood in MOODS:
            p = np.zeros(len(self.gnames)); p[[self.gcol[g] for g in MOODS[mood] if g in self.gcol]] = 1
            raw["context"] = self.G @ (p / np.linalg.norm(p))
        raw["quality"] = .5 * mm(self.bayes[cand]) + .5 * mm(self.logc[cand])  # already over cand
        sig = {k: mm(v if k == "quality" else v[cand]) for k, v in raw.items()}

        # discovery: unfamiliar genres x how well the film matches taste and quality
        usual = set(genres)
        if not usual and liked:
            cnt = {}
            for i in liked:
                for g in self.glist[i]: cnt[g] = cnt.get(g, 0) + 1
            usual = set(sorted(cnt, key=cnt.get, reverse=True)[:3])
        if usual:
            nov = np.array([1 - len(usual & set(self.glist[i])) / len(self.glist[i]) if self.glist[i] else 0.0
                            for i in cand])
            match = .5 * sig["quality"] + .5 * np.maximum(sig.get("preference", 0), sig.get("content", 0))
            sig["discovery"] = mm(nov * match)

        sig = {k: v for k, v in sig.items() if np.ptp(v) > 0}  # drop inactive / constant signals
        shown = context.get("shown", {})  # movieId -> times shown without a click
        pen = np.array([0.5 if shown.get(int(self.ids[i]), 0) >= 3 else 0.0 for i in cand])
        return Prep(cand, sig, stage_of(profile), pen, liked, L, w, genres)

    def weights(self, prep, dial="different", stage_w=None):
        w = dict((stage_w or STAGE_W)[prep.stage]); w["discovery"] = DIAL[dial][1]
        w = {k: v for k, v in w.items() if k in prep.sig and v > 0}
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    def combine(self, prep, weights):
        """Per-signal weighted contributions (len(cand) each) and total score."""
        contrib = {k: w * prep.sig[k] for k, w in weights.items() if k in prep.sig}
        return contrib, sum(contrib.values()) - prep.penalty

    # ---- diversity -----------------------------------------------------
    def mmr(self, prep, score, k, lam, pool=100):
        order = np.argsort(-score)[:pool]
        sim = (self.T[prep.cand[order]] @ self.T[prep.cand[order]].T).toarray()
        cap = max(1, k // 2)  # no genre above 50% of the row
        chosen, used, maxsim = [], {}, np.zeros(len(order))
        while len(chosen) < min(k, len(order)):
            val = lam * score[order] - (1 - lam) * maxsim
            val[chosen] = -np.inf
            ok = np.array([used.get(self.primary[prep.cand[o]], 0) < cap for o in order])
            if ok.any() and np.isfinite(val[ok]).any(): val[~ok] = -np.inf
            j = int(np.argmax(val))
            chosen.append(j)
            g = self.primary[prep.cand[order[j]]]
            used[g] = used.get(g, 0) + 1
            maxsim = np.maximum(maxsim, sim[j])
        return order[chosen]

    # ---- explanation ---------------------------------------------------
    def explain(self, prep, pos, contrib, profile, context):
        """Pick the 2 largest contributions (each >=10% of the score) and fill templates."""
        item = prep.cand[pos]
        parts = {k: v[pos] for k, v in contrib.items()}
        total = sum(parts.values()) or 1
        pos_liked = prep.liked
        text, reasons = [], []
        for k in sorted(parts, key=parts.get, reverse=True):
            if parts[k] / total < .10 or len(reasons) == 2: continue
            if k == "content" and pos_liked:
                ref = pos_liked[int(np.argmax((self.T[pos_liked] @ self.T[item].T).toarray().ravel()))]
                s = TEMPLATES[k].format(ref=self.movies.title[ref])
            elif k == "collab" and pos_liked:
                ref = pos_liked[int(np.argmax(self._cf_sims(pos_liked, [item]).ravel()))]
                s = TEMPLATES[k].format(ref=self.movies.title[ref])
            elif k == "preference":
                gs = [g for g in self.glist[item] if g in prep.genres][:2]
                s = TEMPLATES[k].format(genres=" and ".join(gs).lower()) if gs else None
            elif k == "context":
                s = TEMPLATES[k].format(mood=context["mood"].lower())
            elif k == "quality":
                s = TEMPLATES[k].format(avg=self.avg[item], n=int(self.count[item]))
            elif k == "discovery":
                s = TEMPLATES[k]
            else:
                s = None
            if s: text.append(s); reasons.append(k)
        return " ".join(text), reasons

    # ---- public --------------------------------------------------------
    def recommend(self, profile, context=None, k=10):
        context = context or {}
        dial = context.get("discovery", "different")
        prep = self.prepare(profile, context)
        contrib, score = self.combine(prep, self.weights(prep, dial))
        picks = self.mmr(prep, score, k, DIAL[dial][0])
        out = []
        for p in picks:
            item = prep.cand[p]
            text, reasons = self.explain(prep, p, contrib, profile, context)
            if not text:  # every shown reason must come from the breakdown
                top = max(contrib, key=lambda kk: contrib[kk][p])
                text, reasons = f"Ranked well on {top}.", [top]
            out.append({
                "movie_id": int(self.ids[item]), "title": self.movies.title[item],
                "score": float(score[p]),
                "breakdown": {kk: float(v[p]) for kk, v in contrib.items()},
                "explanation": text, "reasons": reasons, "stage": prep.stage,
                "genres": self.glist[item],
                "avg_rating": None if np.isnan(self.avg[item]) else float(self.avg[item]),
                "n_ratings": int(self.count[item]),
            })
        return out


@lru_cache(maxsize=1)
def get_model():
    return Model(pd.read_csv(DATA / "movies.csv"), pd.read_csv(DATA / "ratings.csv"),
                 pd.read_csv(DATA / "tags.csv"))


def recommend(profile: dict, context: dict, k: int = 10) -> list[dict]:
    """profile: {"genres": [...], "liked": [movie_ids], "disliked": [movie_ids]}
    context: {"mood": "funny"|None, "discovery": "comfort"|"different"|"surprise"}"""
    return get_model().recommend(profile, context, k)
