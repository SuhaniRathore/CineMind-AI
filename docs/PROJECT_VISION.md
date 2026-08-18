# PROJECT_VISION.md — CineMind AI

## Problem Statement

Most movie-rating platforms (IMDb, Letterboxd clones, typical BTech "movie recommendation" projects) stop at:
star ratings, a review text box, and a basic "users who liked X also liked Y" recommender.

This leaves several real problems unsolved:

- Users struggle to describe *what they actually want to watch* ("something like Inception but less confusing") using star ratings or genre filters alone.
- Reviews are walls of unstructured text — nobody reads 200 reviews to figure out if the pacing is slow or the ending is worth it.
- Recommendations are black boxes — users don't know *why* a movie was suggested, so they don't trust it.
- Review quality varies wildly (spam, spoilers, bots, low-effort one-liners) and platforms rarely flag this.
- Mood and context ("I want something light after a long day") are ignored even though they drive real-world watch decisions more than genre does.

## Project Objective

Build a movie discovery platform where **natural language and AI-driven understanding**, not just filters and star ratings, drive how users find and evaluate movies — while keeping the system explainable, incrementally buildable, and viva/interview-defensible by a BTech student.

## Target Users

- **Primary:** Individual movie watchers looking for better discovery than "genre + rating sort."
- **Secondary:** Small groups deciding what to watch together (V3 feature).
- **Academic audience:** Evaluators/interviewers assessing the technical depth of the system (recommendation systems, NLP, system design).

## Core Value Proposition

1. **Understand movies deeply** — via NLP-driven analysis of reviews (sentiment, aspects, summaries, spoilers, credibility) instead of treating them as opaque text blobs.
2. **Understand users deeply** — via a "Movie DNA" preference profile built from ratings, reviews, and behavior, not just genre checkboxes.
3. **Explain recommendations** — every suggestion should be traceable to *why* ("recommended because you rated similar slow-burn thrillers highly, and this shares themes X, Y").
4. **Search movies the way people think** — natural language / semantic search ("movies about grief and found family") instead of rigid keyword filters.

## What Makes CineMind Different

| Ordinary movie-rating site | CineMind AI |
|---|---|
| Star ratings only | Ratings + aspect-based sentiment from review text |
| Keyword/genre search | Semantic, natural-language search via embeddings |
| Black-box "you might like" | Explainable recommendations with stated reasons |
| Reviews are raw text | AI-summarized, spoiler-flagged, credibility-scored reviews |
| Fixed collaborative filtering | Hybrid (content + collaborative) recommender with a mood dimension |
| One-size-fits-all | Per-user "Movie DNA" preference profile |

## Long-Term Vision

CineMind AI's long-term vision is a movie companion that understands taste the way a well-read friend would: it reads between the lines of reviews, remembers what you actually enjoyed (not just what you rated 5 stars), and can explain its reasoning in plain language. Later stages (V3) extend this toward mood-aware and group-aware recommendations and optional voice interaction, while always keeping the core recommendation and NLP pipeline transparent enough to explain in an academic setting.

This document is the north star for scope decisions — if a proposed feature doesn't serve discovery quality, explainability, or depth of movie/user understanding, it should be questioned before being added.
