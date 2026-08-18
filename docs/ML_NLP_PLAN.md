# ML_NLP_PLAN.md — CineMind AI

Planning only — no models are trained or implemented yet. Each component lists Input / Output / Possible Algorithm / Dataset / Evaluation, plus the tier it belongs to.

---

## 1. Content-Based Recommendation — MVP
- **Input:** Movie metadata (genres, keywords, overview text, cast/crew if available from TMDB).
- **Output:** Ranked list of movies similar to a given movie or to a user's rated/liked movies.
- **Possible algorithm:** TF-IDF or simple feature vectors over genres/keywords + cosine similarity. Simple, fast, fully explainable — good MVP choice since it needs no training data beyond the movie catalog itself.
- **Dataset:** TMDB-sourced movie metadata (our own cached `movies`/`genres` tables).
- **Evaluation:** Qualitative review (do similar movies make sense?) plus basic precision-at-k against a small manually curated "these should be similar" test set.

## 2. Collaborative Filtering — V2
- **Input:** User-movie rating matrix (real ratings + MovieLens as bootstrap/training data).
- **Output:** Predicted rating / ranked recommendations per user.
- **Possible algorithm:** Matrix factorization (e.g., ALS or SVD via `scikit-surprise` or a NumPy implementation); scikit-learn based nearest-neighbor collaborative filtering is a simpler fallback.
- **Dataset:** MovieLens (ml-latest-small to start, given student-project scale) for pretraining/cold-start, blended with real in-app ratings as they accumulate.
- **Evaluation:** RMSE/MAE on held-out ratings; precision/recall@k for top-N recommendation quality.

## 3. Hybrid Recommendation — V2
- **Input:** Outputs of content-based and collaborative models, plus user context (recency, watch history).
- **Output:** Final blended, ranked recommendation list with per-item score and contributing method.
- **Possible algorithm:** Weighted linear blend of content-based and collaborative scores initially (simple, explainable); switching mechanism for cold-start users (new users get content-based only until enough ratings exist).
- **Dataset:** Same as components 1 and 2.
- **Evaluation:** Same top-N metrics as collaborative filtering, compared against each component alone to justify the hybrid approach.

## 4. Sentiment Analysis — V2
- **Input:** Review text.
- **Output:** Overall sentiment label (positive/negative/neutral) + confidence score.
- **Possible algorithm:** Fine-tuned or off-the-shelf transformer classifier (e.g., a pretrained sentiment model via Hugging Face `transformers`) or a simpler scikit-learn baseline (TF-IDF + logistic regression) trained on a labeled movie-review dataset if a lighter-weight approach is preferred for explainability/viva purposes.
- **Dataset:** IMDB review sentiment dataset (standard, well-known) for training/fine-tuning; applied to our own real review text at inference time.
- **Evaluation:** Accuracy/F1 on a held-out labeled test split.

## 5. Aspect-Based Sentiment Analysis — V2
- **Input:** Review text.
- **Output:** Sentiment broken down per aspect (e.g., acting, plot, pacing, visuals, ending).
- **Possible algorithm:** Aspect-term extraction (keyword/rule-based initially for interpretability) combined with sentiment scoring per extracted aspect span; a more advanced transformer-based ABSA model can be substituted later if time allows.
- **Dataset:** Reused review text; aspect keyword lists curated manually for the movie domain (no large labeled ABSA movie dataset readily available, so a hybrid rule + model approach is realistic for project scope).
- **Evaluation:** Manual spot-check accuracy on a sample of reviews; aspect-coverage rate (how often aspects are found at all).

## 6. Review Summarization — V2
- **Input:** All reviews for a given movie.
- **Output:** Short natural-language summary of what reviewers are saying.
- **Possible algorithm:** Extractive summarization (e.g., TextRank over review sentences) as a lightweight, explainable baseline; abstractive summarization via a pretrained transformer (e.g., a small `transformers` summarization model) as a stretch upgrade.
- **Dataset:** In-app review text (no external dataset needed; this runs on our own data at inference time).
- **Evaluation:** Manual qualitative review; optionally ROUGE score if a reference summary set is hand-created for a sample of movies.

## 7. Semantic Search / Embeddings — V2
- **Input:** Natural-language search query; movie overview/metadata text.
- **Output:** Ranked movies by semantic similarity to the query.
- **Possible algorithm:** Sentence-Transformers (e.g., `all-MiniLM-L6-v2`) to embed movie overviews and queries; cosine-similarity / nearest-neighbor search via `pgvector` in PostgreSQL.
- **Dataset:** TMDB movie overview text (already cached in our DB).
- **Evaluation:** Qualitative relevance checks on sample queries; optionally simple recall@k against a small manually labeled query→relevant-movie test set.

## 8. Explainable Recommendations — V2
- **Input:** Recommendation output (from hybrid model) + the signals that produced it (shared genres, similar highly-rated movies, collaborative neighbors).
- **Output:** Human-readable reason string(s) attached to each recommendation (stored in `recommendation_results.reason`).
- **Possible algorithm:** Not a separate ML model — a rule-based explanation layer that inspects which component (content/collaborative) contributed most and surfaces the top contributing features (e.g., "because you liked movies with themes: redemption, isolation").
- **Dataset:** N/A (derived from existing recommendation pipeline outputs).
- **Evaluation:** Qualitative — do explanations make sense to a human reading them; this is primarily a UX/trust feature rather than an accuracy metric.

## Evaluation Summary Table

| Component | Primary Metric |
|---|---|
| Content-based | Precision@k (manual test set) |
| Collaborative filtering | RMSE/MAE, Precision/Recall@k |
| Hybrid | Precision/Recall@k vs. individual components |
| Sentiment analysis | Accuracy/F1 |
| Aspect-based sentiment | Manual spot-check accuracy |
| Summarization | Qualitative / optional ROUGE |
| Semantic search | Qualitative relevance / Recall@k |
| Explainability | Qualitative (human review) |

## Notes
- MVP intentionally uses **no trained models** — only metadata similarity — to keep the first working version simple, fast to build, and easy to defend in a viva.
- All V2 models should be trainable/runnable on modest hardware (student laptop or a single cloud GPU/CPU instance at most) — large fine-tuning jobs are out of scope; prefer pretrained models used via inference or lightweight fine-tuning.
- Every model choice above should be revisited and confirmed (not silently swapped) at the point its milestone is actually reached, since library availability/ease-of-use may shift the specific choice within the same approach.
