import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import eval as E
import recommender as R

M = R.get_model()


def mid(title):
    return int(M.movies[M.movies.title == title].movieId.iloc[0])


SCIFI = dict(genres=["Sci-Fi", "Action"],
             liked=[mid(t) for t in ["Matrix, The (1999)", "Inception (2010)", "Blade Runner (1982)",
                                     "Interstellar (2014)", "Terminator, The (1984)"]],
             disliked=[])


def test_shape_and_exclusions():
    recs = R.recommend(SCIFI, {"discovery": "different"}, 10)
    assert len(recs) == 10 and len({r["movie_id"] for r in recs}) == 10
    assert not {r["movie_id"] for r in recs} & set(SCIFI["liked"])
    assert {"movie_id", "title", "score", "breakdown", "explanation"} <= set(recs[0])


def test_not_for_me_never_returns():
    first = R.recommend(SCIFI, {}, 10)[0]["movie_id"]
    again = R.recommend(dict(SCIFI, disliked=[first]), {}, 10)
    assert first not in {r["movie_id"] for r in again}


def test_explanations_only_cite_breakdown_signals():
    for ctx in [{}, {"mood": "funny"}, {"discovery": "surprise"}]:
        for r in R.recommend(SCIFI, ctx, 10):
            assert r["explanation"] and set(r["reasons"]) <= set(r["breakdown"])
            total = sum(r["breakdown"].values())
            assert all(r["breakdown"][k] / total >= .10 for k in r["reasons"] if "Ranked well" not in r["explanation"])


def test_dial_changes_results():
    a = {r["movie_id"] for r in R.recommend(SCIFI, {"discovery": "comfort"}, 10)}
    b = {r["movie_id"] for r in R.recommend(SCIFI, {"discovery": "surprise"}, 10)}
    assert a != b


def test_cold_start_without_any_input():
    recs = R.recommend(dict(genres=[], liked=[], disliked=[]), {}, 10)
    assert len(recs) == 10 and recs[0]["stage"] == "cold"


def test_avoid_is_hard_filter():
    recs = R.recommend(dict(genres=[], liked=[], disliked=[], avoid=["Horror"]), {}, 20)
    assert all("Horror" not in r["genres"] for r in recs)


def test_weights_sum_to_one_and_stages():
    for stage in R.STAGE_W:
        prep = M.prepare(dict(SCIFI, liked=SCIFI["liked"] * (1 if stage != "established" else 1)))
        prep.stage = stage
        assert sum(M.weights(prep).values()) == pytest.approx(1)
    assert R.stage_of(dict(liked=[], disliked=[])) == "cold"
    assert R.stage_of(dict(liked=[1] * 19, disliked=[])) == "warming"
    assert R.stage_of(dict(liked=[1] * 20, disliked=[])) == "established"


def test_seen_recently_penalty():
    top = R.recommend(SCIFI, {}, 1)[0]["movie_id"]
    prep = M.prepare(SCIFI, {"shown": {top: 3, 1: 2}})
    assert prep.penalty.sum() == 0.5 and prep.penalty[list(prep.cand).index(M.idx[top])] == 0.5


def test_ndcg_and_split():
    assert E.ndcg([1, 2, 3], {1, 2, 3}) == pytest.approx(1)
    assert E.ndcg([9, 8], {1}) == 0
    import pandas as pd
    r = pd.DataFrame(dict(userId=[1] * 10, movieId=range(10), rating=4.0, timestamp=range(10)))
    tr, te = E.split(r)
    assert len(tr) == 8 and te.timestamp.min() > tr.timestamp.max()


def test_ui_flow(monkeypatch):
    monkeypatch.setenv("ARTWORK", "off")   # no poster lookups from the test suite
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception
    [b for b in at.button if b.label == "Skip"][0].click().run()   # the quiz is skippable
    assert not at.exception
    assert any("Tonight's top pick" in m.value for m in at.markdown)   # billboard rendered
    cards = lambda: [m.value for m in at.markdown if "class='ctitle'" in m.value]
    before = cards()
    assert len(before) >= 10                                       # rows of cards, not one grid
    [b for b in at.button if b.label == "Not for me"][0].click().run()
    assert not at.exception and len(at.session_state.disliked) == 1
    assert cards() != before                                       # the row re-ranks at once


def test_pretty_title_moves_trailing_article():
    import app
    assert app.pretty("Matrix, The (1999)") == "The Matrix (1999)"
    assert app.pretty("Pulp Fiction (1994)") == "Pulp Fiction (1994)"
    assert app.pretty("Postino, Il (The Postman) (1994)") == "Postino, Il (The Postman) (1994)"


def test_three_scifi_favourites_get_mostly_scifi_but_not_only():
    p = dict(genres=["Sci-Fi"], disliked=[],
             liked=[mid(t) for t in ["Matrix, The (1999)", "Blade Runner (1982)", "Interstellar (2014)"]])
    n = sum("Sci-Fi" in r["genres"] for r in R.recommend(p, {"discovery": "different"}, 10))
    assert 5 <= n < 10


def test_fetch_art_builds_image_urls(monkeypatch):
    import io

    import app
    monkeypatch.setattr(app, "TMDB_IMG", "http://img")
    monkeypatch.setattr(app.urllib.request, "urlopen",
                        lambda url, timeout=0: io.BytesIO(b'{"poster_path":"/p.jpg","backdrop_path":"/b.jpg"}'))
    assert app._fetch_art(1, "k") == ("http://img/w342/p.jpg", "http://img/w1280/b.jpg")


def test_fetch_art_survives_a_film_with_no_images_and_a_dead_tmdb(monkeypatch):
    import io

    import app
    monkeypatch.setattr(app.urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(b"{}"))
    assert app._fetch_art(1, "k") == (None, None)

    def boom(*a, **k):
        raise OSError("tmdb down")
    monkeypatch.setattr(app.urllib.request, "urlopen", boom)
    assert app._fetch_art(1, "k") == (None, None)


def test_wiki_candidates_are_ordered_most_specific_first():
    import app
    assert app.wiki_candidates("The Matrix (1999)") == [
        "The Matrix (1999 film)", "The Matrix (film)", "The Matrix"]
    assert app.wiki_candidates("Seven (a.k.a. Se7en) (1995)")[0] == "Seven (1995 film)"
    assert app.wiki_candidates("No Year Here") == ["No Year Here (film)", "No Year Here"]


def test_fetch_wiki_prefers_the_specific_article_and_follows_redirects(monkeypatch):
    import io

    import app
    body = json.dumps({"query": {
        "redirects": [{"from": "The Matrix (film)", "to": "The Matrix"}],
        "pages": {
            "1": {"title": "The Matrix", "thumbnail": {"source": "http://i/matrix.jpg"}},
            "2": {"title": "Fargo (1996 film)", "thumbnail": {"source": "http://i/fargo.jpg"}},
            "3": {"title": "Gone (film)"},
        }}}).encode()
    monkeypatch.setattr(app.urllib.request, "urlopen", lambda req, timeout=0: io.BytesIO(body))
    got = app._fetch_wiki(["The Matrix (1999)", "Fargo (1996)", "Gone (2012)"])
    assert got["Fargo (1996)"] == "http://i/fargo.jpg"      # exact year article wins
    assert got["The Matrix (1999)"] == "http://i/matrix.jpg"  # reached through the redirect
    assert got["Gone (2012)"] is None                        # an article with no image


def test_fetch_wiki_survives_wikipedia_being_down(monkeypatch):
    import app

    def boom(*a, **k):
        raise OSError("wikipedia down")
    monkeypatch.setattr(app.urllib.request, "urlopen", boom)
    assert app._fetch_wiki(["The Matrix (1999)"]) == {"The Matrix (1999)": None}
