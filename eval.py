"""Offline evaluation: per-user time split (first 80% train, last 20% test).
Relevant = test items rated >= 4. Models are built from TRAIN ratings only (no leakage).
Run: python eval.py   -> prints tables, writes results/eval_results.csv
"""
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import recommender as R

K = 10
OUT = Path(__file__).parent / "results"


def split(ratings):
    r = ratings.sort_values(["userId", "timestamp", "movieId"])
    pos = r.groupby("userId").cumcount()
    n = r.groupby("userId").movieId.transform("size")
    train = pos < (n * 0.8).astype(int)
    return r[train], r[~train]


def ndcg(ranked, rel, k=K):
    dcg = sum(1 / math.log2(i + 2) for i, m in enumerate(ranked[:k]) if m in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg


def ild(model, ids):
    gs = [set(model.glist[model.idx[m]]) for m in ids]
    d = [1 - len(a & b) / max(1, len(a | b)) for i, a in enumerate(gs) for b in gs[i + 1:]]
    return float(np.mean(d)) if d else 0.0


def top_genres(model, items, n=3):
    c = Counter(g for i in items for g in model.glist[model.idx[i]])
    return [g for g, _ in c.most_common(n)]


def single(prep, key):
    return prep.sig.get(key, np.zeros(len(prep.cand)))


def methods(model, prep, count):
    """name -> list of recommended movieIds (top K)."""
    ids = lambda pos: [int(model.ids[prep.cand[p]]) for p in pos]
    top = lambda s: np.argsort(-s, kind="stable")[:K]
    _, hs = model.combine(prep, model.weights(prep, "different"))
    _, ss = model.combine(prep, model.weights(prep, "different", R.SPEC_W))
    return {
        "Popularity only": ids(top(count[prep.cand])),
        "Preference only": ids(top(single(prep, "preference"))),
        "Content only": ids(top(single(prep, "content"))),
        "Collaborative only": ids(top(single(prep, "collab"))),
        "Hybrid (spec weights)": ids(top(ss)),
        "Hybrid (tuned)": ids(top(hs)),
        "Hybrid (tuned) + MMR": ids(model.mmr(prep, hs, K, R.DIAL["different"][0])),
    }


def run(model, profiles, relevant, skip):
    rows, recs = [], {}
    for u, prof in profiles.items():
        prep = model.prepare(prof)
        for name, lst in methods(model, prep, model.count).items():
            if name in skip: continue
            rel = relevant[u]
            hits = sum(m in rel for m in lst)
            rows.append(dict(method=name, user=u, recall=hits / len(rel), ndcg=ndcg(lst, rel),
                             ild=ild(model, lst)))
            recs.setdefault(name, set()).update(lst)
    df = pd.DataFrame(rows)
    out = df.groupby("method")[["recall", "ndcg", "ild"]].mean()
    out["coverage"] = pd.Series({m: len(s) / (model.count >= model.min_ratings).sum() for m, s in recs.items()})
    order = [m for m in ["Popularity only", "Preference only", "Content only", "Collaborative only",
                         "Hybrid (spec weights)", "Hybrid (tuned)", "Hybrid (tuned) + MMR"] if m in out.index]
    out = out.loc[order]
    out.columns = [f"Recall@{K}", f"NDCG@{K}", "Intra-list diversity", "Catalog coverage"]
    out["n_users"] = df.user.nunique()
    return out, df


def paired(df, a, b, metric="ndcg"):
    """Paired t-test across users: is method a better than method b?"""
    from scipy import stats
    x = df[df.method == a].set_index("user")[metric]
    y = df[df.method == b].set_index("user")[metric]
    t, p = stats.ttest_rel(x, y.loc[x.index])
    return f"{a} vs {b}: NDCG diff {x.mean() - y.mean():+.4f} ({(x.mean() / y.mean() - 1) * 100:+.0f}%), p={p:.4f}"


def main():
    d = R.DATA
    movies, ratings, tags = pd.read_csv(d / "movies.csv"), pd.read_csv(d / "ratings.csv"), pd.read_csv(d / "tags.csv")
    train, test = split(ratings)
    model = R.Model(movies, train, tags)
    print(f"train {len(train):,} ratings, test {len(test):,}, users {ratings.userId.nunique()}")

    relevant = {u: set(g.movieId[g.rating >= 4]) for u, g in test.groupby("userId")}
    full, cold = {}, {}
    for u, g in train.groupby("userId"):
        if len(g) < 20 or not relevant.get(u): continue
        pos = g[g.rating >= 4]
        full[u] = dict(liked=list(pos.movieId), disliked=list(g.movieId[g.rating <= 2]),
                       genres=top_genres(model, pos.movieId), seen=list(g.movieId))
        if len(pos) >= 5:  # cold start: only what a quiz gives (5 favourites + genres)
            fav = list(pos.sort_values(["rating", "timestamp"], ascending=[False, True]).movieId[:5])
            cold[u] = dict(liked=fav, disliked=[], genres=top_genres(model, fav), seen=list(g.movieId))
    print(f"eval users: {len(full)} (established history), {len(cold)} (cold-start quiz)")

    res_full, df_full = run(model, full, relevant, skip=set())
    res_cold, df_cold = run(model, cold, relevant, skip={"Collaborative only"})
    OUT.mkdir(exist_ok=True)
    pd.concat({"full_history": res_full, "cold_start": res_cold}).to_csv(OUT / "eval_results.csv")
    for title, res in [("Full history (users with 20+ ratings)", res_full), ("Cold start (5 favourites + genres)", res_cold)]:
        print(f"\n{title}\n{res.round(4).to_string()}")
    tests = [paired(df_full, "Hybrid (tuned)", "Popularity only"), paired(df_full, "Hybrid (tuned)", "Collaborative only"),
             paired(df_full, "Hybrid (tuned)", "Content only"), paired(df_full, "Hybrid (tuned) + MMR", "Hybrid (tuned)"),
             paired(df_cold, "Hybrid (tuned)", "Popularity only"), paired(df_cold, "Hybrid (tuned)", "Content only")]
    print("\nPaired t-tests (full history first four, cold start last two)\n" + "\n".join(tests))
    (OUT / "significance.txt").write_text("\n".join(tests) + "\n")
    return res_full, res_cold


if __name__ == "__main__":
    main()
