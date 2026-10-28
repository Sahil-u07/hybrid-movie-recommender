# Hybrid Movie Recommender

Mini project for Recommender Systems. Team: Sahil Lenka and Kanishk Shukla.

A Streamlit app that recommends films for a brand-new or returning user. A short skippable quiz replaces the cold start, every pick carries a one-line reason, and a discovery dial sets how adventurous the list is. Like and Not for me re-rank the page immediately.

The interface is a dark browse page: a billboard for tonight's top pick, then horizontally scrolling rows of poster cards. Each card shows why it is there, and "Why this?" opens the score breakdown that produced the pick. It works down to phone width. Posters come from TMDB when a key is set; without one each card gets a generated cover, so nothing is ever a broken image.

## Run it

Python 3.11 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The MovieLens data is already in `data/`, so nothing is downloaded.

**Film posters.** Out of the box the app pulls each film's poster from its English Wikipedia article, which needs no API key or signup. These are non-free images that Wikipedia hosts under fair use, so for anything beyond a course demo use TMDB instead: it licenses artwork for exactly this and is free for non-commercial use. Set a TMDB key as `TMDB_API_KEY` (an environment variable, or `.streamlit/secrets.toml` locally, or Manage app, Settings, Secrets on Streamlit Community Cloud) and the app switches to it, matching on the TMDB id in `links.csv` rather than searching by title, and adding a wide backdrop behind the top pick. A film with no image found anywhere falls back to a generated cover, so nothing is ever a broken image.

Artwork for a row is fetched in parallel and cached, including misses. `ARTWORK=off` skips the lookups entirely, which is how the tests stay offline.

Other commands:

```
python -m pytest -q tests     # 11 tests, including a headless UI test
python eval.py                # offline evaluation, about 30 seconds, writes results/
python tune.py                # re-tunes the stage weights on a validation split, about 1.5 minutes
```

## How it works

Six signals are scaled to 0 to 1 inside the candidate set and combined with weights that depend on how much the user has interacted (cold: nothing, warming: 1 to 19 likes or dislikes, established: 20 or more).

| Signal | Source |
|---|---|
| Preference | Genres picked in the quiz |
| Content | TF-IDF on genres and MovieLens tags, cosine similarity to liked films (disliked films count against) |
| Collaborative | Item-item adjusted cosine on ratings, shrunk for pairs with few common raters |
| Context | Mood chip mapped to genres |
| Quality | Bayesian average rating blended with log rating count, capped weight |
| Discovery | Unfamiliar genres times how well the film matches taste and quality |

Then an MMR re-rank picks each next film by its score minus its similarity to films already chosen. The dial sets lambda (0.85, 0.70 or 0.55) and the discovery weight (0.05, 0.10 or 0.20). No genre may take more than half of a row. The two largest signal contributions fill a reason template, and a signal below 10% of the score is never cited.

Public interface (`recommender.py`):

```python
recommend(profile, context, k=10) -> list[dict]
# profile: {"genres": [...], "liked": [movie_ids], "disliked": [movie_ids]}   optional: "avoid", "seen"
# context: {"mood": "funny" | None, "discovery": "comfort" | "different" | "surprise"}
# each result: movie_id, title, score, breakdown, explanation, reasons, genres, avg_rating, n_ratings
```

## Evaluation

Each user's ratings are split by time: the first 80% train, the last 20% are the test. A test film rated 4 or higher is a hit. Popularity, similarity and the collaborative model are built from training ratings only. Every film the user rated in training is hidden from every method.

Stage weights started from the spec's values. The tuned weights come from a random search on a validation slice cut from the training data, with quality capped at 0.15 and every signal kept at 0.05 or more. The test split was not used for tuning.

**Users with history (585 users with 20 or more ratings)**

| Method | Recall@10 | NDCG@10 | Genre diversity | Catalog coverage |
|---|---|---|---|---|
| Popularity only | 0.0455 | 0.0649 | 0.796 | 4% |
| Preference only | 0.0100 | 0.0137 | 0.114 | 21% |
| Content only | 0.0088 | 0.0122 | 0.194 | 30% |
| Collaborative only | 0.0631 | 0.0945 | 0.764 | 17% |
| Hybrid (spec weights) | 0.0311 | 0.0522 | 0.397 | 24% |
| **Hybrid (tuned)** | 0.0657 | 0.0956 | 0.724 | 13% |
| Hybrid (tuned) + MMR | 0.0569 | 0.0873 | 0.827 | 13% |

**Cold start (581 users; the model sees only their 5 best-rated training films and the top 3 genres of those)**

| Method | Recall@10 | NDCG@10 | Genre diversity | Catalog coverage |
|---|---|---|---|---|
| Popularity only | 0.0438 | 0.0643 | 0.795 | 4% |
| Preference only | 0.0076 | 0.0119 | 0.128 | 30% |
| Content only | 0.0051 | 0.0082 | 0.197 | 38% |
| Hybrid (spec weights) | 0.0213 | 0.0342 | 0.271 | 33% |
| **Hybrid (tuned)** | 0.0532 | 0.0817 | 0.694 | 18% |
| Hybrid (tuned) + MMR | 0.0491 | 0.0808 | 0.807 | 17% |

Paired t-tests over users on NDCG@10 (first four rows: full history, last two: cold start):

- Hybrid (tuned) vs Popularity only: NDCG diff +0.0307 (+47%), p=0.0000
- Hybrid (tuned) vs Collaborative only: NDCG diff +0.0012 (+1%), p=0.6290
- Hybrid (tuned) vs Content only: NDCG diff +0.0835 (+686%), p=0.0000
- Hybrid (tuned) + MMR vs Hybrid (tuned): NDCG diff -0.0083 (-9%), p=0.0014
- Hybrid (tuned) vs Popularity only: NDCG diff +0.0174 (+27%), p=0.0022
- Hybrid (tuned) vs Content only: NDCG diff +0.0736 (+899%), p=0.0000

What the numbers support: the tuned hybrid clearly beats popularity and content-only, in both settings. It ties collaborative-only on accuracy. MMR trades a small accuracy loss for more genre variety. Catalog coverage is 13%, below the 20% target in the project spec.

## Layout

```
app.py              Streamlit interface (quiz, home rows, Why this?, profile, results page)
recommender.py      Model: signals, stage weights, MMR, explanations
eval.py             Offline evaluation and paired tests
tune.py             Weight tuning on a validation split
tests/              pytest suite
data/               MovieLens ml-latest-small CSVs
results/            eval_results.csv, significance.txt
deck/               Presentation (pptx and pdf)
REPORT.md           Short written report
```

## Differences from the spec

- Data is the 2016 release of ml-latest-small (9,125 movies, 100,004 ratings, 671 users, 1,296 tags), not the 2018 release the spec describes. The file format is the same, so the code runs on either.
- The franchise cap (one film per franchise per row) is not implemented. The 50% genre cap per row is.
- The repeat penalty for films shown 3 or more times is implemented in the model and tested, but the app does not pass shown counts, because every Streamlit rerun would count as a showing.
- Runtime and language filters are left out because the data has neither.
- Next.js, FastAPI, Postgres, accounts, group night, conversational search and availability are future work.

## Data credit

MovieLens data from GroupLens Research, University of Minnesota. F. Maxwell Harper and Joseph A. Konstan, "The MovieLens Datasets: History and Context", ACM Transactions on Interactive Intelligent Systems 5(4), 2015. Check the GroupLens usage terms before redistributing the data. Posters and metadata, if enabled, come from TMDB, which is not endorsed or certified by this project.
