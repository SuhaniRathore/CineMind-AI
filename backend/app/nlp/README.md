# CineMind AI — NLP Pipeline

## Overview
Milestone 12 introduces a lightweight NLP pipeline for analyzing
movie reviews. The pipeline runs independently of review creation
and stores derived analysis results in the database.

## Components

### 1. Sentiment Analysis (`sentiment.py`)
- Classifies review text as positive, negative, or neutral.
- Returns a sentiment label and confidence score.
- Uses TF-IDF and Logistic Regression with keyword-based handling.

### 2. Aspect-Based Sentiment (`aspect_sentiment.py`)
Analyzes sentiment for mentioned movie aspects:
- Acting
- Plot
- Pacing
- Visuals
- Ending

Uses keyword-based aspect detection and sentiment analysis.

### 3. Batch Processing (`batch.py`)
- Processes reviews that have not yet been analyzed.
- Runs sentiment and aspect analysis for each review.
- Stores results and processing timestamp in the database.
- Handles individual review failures without stopping the batch.

### 4. Admin Endpoint
`POST /api/v1/admin/nlp/run`

Triggers NLP batch processing. Requires admin authentication.

## Database Fields
NLP results are stored on each review:

| Field | Description |
|---|---|
| `sentiment_label` | Positive, negative, or neutral |
| `sentiment_score` | Sentiment confidence score |
| `aspect_sentiment` | Aspect-level sentiment results |
| `nlp_processed_at` | Last successful processing timestamp |

## Running Tests
From the `backend` directory:

```bash
python -m pytest tests/test_nlp_sentiment.py -v
python -m pytest tests/test_nlp_aspect.py -v
python -m pytest tests/test_nlp_batch.py -v