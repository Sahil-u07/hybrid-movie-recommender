# Hybrid Movie Recommender: project report

Sahil Lenka and Kanishk Shukla, Recommender Systems

## Problem

People spend more time scrolling a streaming catalog than watching. Common recommenders fail new users because there is no history, give no reason for a pick, and offer no control over how familiar or surprising the list is. We built a recommender that gives usable picks within a minute of signing up, explains each pick, and lets the user choose how adventurous it is.

## Approach

The recommender combines six signals: quiz preference, content similarity, item-item collaborative filtering, mood context, quality, and a discovery bonus. Each signal is min-max scaled inside the candidate set and combined with weights that depend on the profile stage. A film is a candidate if it has at least 5 ratings and the user has not seen it. Disliked films, seen films and avoided genres are removed first.

Content similarity uses TF-IDF over genre tokens and MovieLens tags. Collaborative similarity is adjusted cosine on user-mean-centred ratings, shrunk by n/(n+10) so pairs with few common raters count less. A liked film adds +1.0 and a disliked film -0.7 to the taste vector, the event weights from the spec.

After scoring, an MMR re-rank selects each next film by lambda times its score minus (1 - lambda) times its largest similarity to films already in the row, with at most half of a row from one primary genre. Lambda is 0.85, 0.70 or 0.55 for the three dial positions. The two largest signal contributions pick the reason shown on the card, so a reason never names a signal that did not affect the ranking.

## Cold start

With no quiz and no history the weights collapse to the quality signal alone, which returns well-rated, widely rated films. A quiz adds the preference signal, and picked favourites add content similarity and, once the user has any explicit feedback, collaborative filtering. The collaborative weight is zero for a user with no interactions and grows with the stage. Every like or Not for me moves the user from cold to warming to established and re-ranks the page.

## Evaluation

Setup: ml-latest-small, per-user time split (first 80% train, last 20% test), a hit is a test rating of 4 or higher, all models built from training ratings only. Metrics are Recall@10, NDCG@10, genre diversity inside the list, and catalog coverage.

The weights in the spec were untuned starting values. With them the hybrid scored below plain popularity. We tuned the warming and established weights by random search on a validation slice of the training data, capping quality at 0.15 (the spec calls it a tiebreaker) and keeping every signal at 0.05 or more. A first search without the cap pushed quality to 0.40 and returned generic hits to a user who picked five sci-fi films, so we capped it.

**Users with history (585 users)**

| Method | Recall@10 | NDCG@10 | Genre diversity | Catalog coverage |
|---|---|---|---|---|
| Popularity only | 0.0455 | 0.0649 | 0.796 | 4% |
| Preference only | 0.0100 | 0.0137 | 0.114 | 21% |
| Content only | 0.0088 | 0.0122 | 0.194 | 30% |
| Collaborative only | 0.0631 | 0.0945 | 0.764 | 17% |
| Hybrid (spec weights) | 0.0311 | 0.0522 | 0.397 | 24% |
| **Hybrid (tuned)** | 0.0657 | 0.0956 | 0.724 | 13% |
| Hybrid (tuned) + MMR | 0.0569 | 0.0873 | 0.827 | 13% |

**Cold start (581 users, 5 favourites plus their top 3 genres)**

| Method | Recall@10 | NDCG@10 | Genre diversity | Catalog coverage |
|---|---|---|---|---|
| Popularity only | 0.0438 | 0.0643 | 0.795 | 4% |
| Preference only | 0.0076 | 0.0119 | 0.128 | 30% |
| Content only | 0.0051 | 0.0082 | 0.197 | 38% |
| Hybrid (spec weights) | 0.0213 | 0.0342 | 0.271 | 33% |
| **Hybrid (tuned)** | 0.0532 | 0.0817 | 0.694 | 18% |
| Hybrid (tuned) + MMR | 0.0491 | 0.0808 | 0.807 | 17% |

Paired t-tests over users on NDCG@10:

- Hybrid (tuned) vs Popularity only: NDCG diff +0.0307 (+47%), p=0.0000
- Hybrid (tuned) vs Collaborative only: NDCG diff +0.0012 (+1%), p=0.6290
- Hybrid (tuned) vs Content only: NDCG diff +0.0835 (+686%), p=0.0000
- Hybrid (tuned) + MMR vs Hybrid (tuned): NDCG diff -0.0083 (-9%), p=0.0014
- Hybrid (tuned) vs Popularity only: NDCG diff +0.0174 (+27%), p=0.0022
- Hybrid (tuned) vs Content only: NDCG diff +0.0736 (+899%), p=0.0000

## Results in words

The tuned hybrid beats popularity by about 47% NDCG@10 with full history and about 27% in the cold-start test, and it beats content-only by a wide margin. It does not beat collaborative-only: the two are within 1% and the difference is not significant. The hybrid's advantage over collaborative filtering alone is functional, not numeric: it works with no history, it explains its picks, and it exposes a diversity control. MMR lowers NDCG@10 by about 9% and raises genre diversity by about 0.10, which is why diversity is its own stage and a user setting.

## Limitations

- Catalog coverage is 13%, below the 20% target in the spec. The untuned spec weights reach 24% coverage at much lower accuracy, so there is a real trade-off.
- The dataset is small and older than the one the spec names. Weights were tuned on one split of 671 users.
- Content similarity is weak on this data (content-only is far below the other methods), probably because most films have only genre tokens and few have tags.
- MovieLens small has no runtime or language fields, so those filters are not built.
- The app has not been tested with real users, so online metrics from the spec (pick rate, completion, negative feedback rate) are not measured.

## Future work

Move to the full stack in the spec (Next.js, FastAPI, Postgres with pgvector), train collaborative filtering on MovieLens 25M, add embeddings from film overviews, add the franchise cap, group movie night, conversational discovery and availability, and measure online metrics with real users.
