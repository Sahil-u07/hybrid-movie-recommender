"""Tune stage weights on a VALIDATION split carved out of the training data (never the test set).
Run: python tune.py  -> prints weights to paste into recommender.STAGE_W
"""
import numpy as np
import pandas as pd

import eval as E
import recommender as R


def score(model, preps, relevant, stage, w4):
    sw = {stage: dict(preference=w4[0], content=w4[1], collab=w4[2], context=.15, quality=w4[3], discovery=.10)}
    tot = 0.0
    for u, prep in preps.items():
        _, s = model.combine(prep, model.weights(prep, "different", sw))
        top = np.argsort(-s, kind="stable")[:E.K]
        tot += E.ndcg([int(model.ids[prep.cand[p]]) for p in top], relevant[u])
    return tot / len(preps)


def search(model, preps, relevant, stage, n=300, seed=0):
    rng = np.random.default_rng(seed)
    cands = [np.array([R.SPEC_W[stage][k] for k in ("preference", "content", "collab", "quality")])]
    # spec: quality is a capped tiebreaker -> quality in [0.05, 0.15]; other signals keep a 0.05 floor; all sum to 0.75
    for _ in range(n):
        q = rng.uniform(0.05, 0.15)
        a, b, c = 0.05 + rng.dirichlet(np.ones(3)) * (0.75 - q - 0.15)
        cands.append(np.array([a, b, c, q]))
    res = sorted(((score(model, preps, relevant, stage, w), tuple(np.round(w, 3))) for w in cands), reverse=True)
    spec = next(r for r in res if r[1] == tuple(np.round(cands[0], 3)))
    return res[0], spec


def main():
    d = R.DATA
    movies, ratings, tags = pd.read_csv(d / "movies.csv"), pd.read_csv(d / "ratings.csv"), pd.read_csv(d / "tags.csv")
    train, _ = E.split(ratings)
    inner, val = E.split(train)  # validation = last 20% of each user's TRAIN history
    model = R.Model(movies, inner, tags)
    relevant = {u: set(g.movieId[g.rating >= 4]) for u, g in val.groupby("userId")}
    prof = {"established": {}, "warming": {}}
    for u, g in inner.groupby("userId"):
        if len(g) < 20 or not relevant.get(u): continue
        pos = g[g.rating >= 4]
        p = dict(liked=list(pos.movieId), disliked=list(g.movieId[g.rating <= 2]),
                 genres=E.top_genres(model, pos.movieId), seen=list(g.movieId))
        if R.stage_of(p) == "established": prof["established"][u] = p
        if len(pos) >= 5:
            fav = list(pos.sort_values(["rating", "timestamp"], ascending=[False, True]).movieId[:5])
            prof["warming"][u] = dict(liked=fav, disliked=[], genres=E.top_genres(model, fav), seen=list(g.movieId))
    for stage, profs in prof.items():
        preps = {u: model.prepare(p) for u, p in profs.items()}
        (best, w), (spec, _) = search(model, preps, relevant, stage)
        print(f"{stage}: {len(profs)} users | validation NDCG@10 spec={spec:.4f} tuned={best:.4f}")
        print(f"  preference={w[0]}, content={w[1]}, collab={w[2]}, quality={w[3]} (context .15, discovery .10 fixed)")


if __name__ == "__main__":
    main()
