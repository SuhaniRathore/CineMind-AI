"""
Tests for NLP summarization batch pipeline and logic — Milestone 13.
"""

from unittest.mock import MagicMock, patch

from app.models.movie import Movie
from app.models.review import Review
from app.nlp.summarization import generate_review_summary
from app.nlp.summarization_batch import NLPSummarizationRunner


def test_generate_review_summary_empty():
    assert generate_review_summary([]) is None
    assert generate_review_summary(["  "]) is None


def test_generate_review_summary_short():
    reviews = ["This is great.", "I loved it."]
    summary = generate_review_summary(reviews, max_sentences=3)

    assert summary == "This is great. I loved it."


def test_generate_review_summary_long():
    reviews = [
        (
            "The movie was an absolute masterpiece. "
            "The acting was incredible and the plot was engaging from start to finish."
        ),
        (
            "A truly remarkable film. "
            "The visual effects were stunning, but the pacing felt a bit slow in the middle."
        ),
        (
            "I highly recommend this movie to anyone. "
            "It is a must-watch for fans of the genre. "
            "The director did a fantastic job."
        ),
    ]

    summary = generate_review_summary(reviews, max_sentences=3)

    assert summary is not None

    sentences = [s.strip() for s in summary.split(".") if s.strip()]

    assert len(sentences) <= 3
    assert len(summary) < len(" ".join(reviews))


def test_summarization_batch_runner():
    movie = MagicMock()
    movie.id = 1
    movie.review_summary = None
    movie.review_summary_updated_at = None

    review = MagicMock()
    review.body = "This is a great movie. I loved it."

    # Mock the pending-movies query.
    pending_mock = MagicMock()
    pending_mock.count.return_value = 1

    # The corrected runner first takes a snapshot of pending movie IDs.
    pending_mock.with_entities.return_value.order_by.return_value.all.return_value = [
        (1,)
    ]

    # Mock the actual batch query.
    batch_query = MagicMock()
    batch_query.filter.return_value.order_by.return_value.all.return_value = [
        movie
    ]

    # Mock the reviews query.
    reviews_query = MagicMock()
    reviews_query.filter.return_value.all.return_value = [review]

    db = MagicMock()

    query_count = {"movie": 0}

    def mock_query(*args, **kwargs):
        # Movie queries:
        # 1. Build pending_movies
        # 2. Fetch the actual batch
        if args and args[0] is Movie:
            query_count["movie"] += 1

            if query_count["movie"] == 1:
                pending_builder = MagicMock()
                pending_builder.join.return_value.filter.return_value = pending_mock
                return pending_builder

            return batch_query

        # Review query used by _process_movie().
        if args and args[0] is Review:
            return reviews_query

        # Query used to build the latest_reviews subquery.
        return MagicMock()

    db.query.side_effect = mock_query

    runner = NLPSummarizationRunner(db, MagicMock())

    result = runner.run(batch_size=50)

    assert result.processed == 1
    assert result.failed == 0
    assert result.total_seen == 1

    assert movie.review_summary is not None
    assert movie.review_summary_updated_at is not None

    db.commit.assert_called_once()

def test_summarization_batch_runner_multiple_batches():
    movies = []

    for movie_id in [1, 2, 3]:
        movie = MagicMock()
        movie.id = movie_id
        movie.review_summary = None
        movie.review_summary_updated_at = None
        movies.append(movie)

    reviews = []

    for movie_id in [1, 2, 3]:
        review = MagicMock()
        review.body = f"Review for movie {movie_id}. Great movie."
        reviews.append(review)

    pending_mock = MagicMock()
    pending_mock.count.return_value = 3
    pending_mock.with_entities.return_value.order_by.return_value.all.return_value = [
        (1,),
        (2,),
        (3,),
    ]

    batch_query = MagicMock()
    batch_query.filter.return_value.order_by.return_value.all.side_effect = [
        movies[:2],
        movies[2:],
    ]

    reviews_query = MagicMock()

    db = MagicMock()

    movie_query_count = {"count": 0}
    review_query_count = {"count": 0}

    def mock_query(*args, **kwargs):
        if args and args[0] is Movie:
            movie_query_count["count"] += 1

            # First Movie query builds pending_movies.
            if movie_query_count["count"] == 1:
                pending_builder = MagicMock()
                pending_builder.join.return_value.filter.return_value = pending_mock
                return pending_builder

            # Following Movie queries fetch each batch.
            return batch_query

        if args and args[0] is Review:
            review_query_count["count"] += 1

            review_builder = MagicMock()
            review_builder.filter.return_value.all.return_value = [
                reviews[review_query_count["count"] - 1]
            ]
            return review_builder

        return MagicMock()

    db.query.side_effect = mock_query

    runner = NLPSummarizationRunner(db, MagicMock())

    result = runner.run(batch_size=2)

    assert result.processed == 3
    assert result.failed == 0
    assert result.total_seen == 3

    for movie in movies:
        assert movie.review_summary is not None
        assert movie.review_summary_updated_at is not None

    assert db.commit.call_count == 2

def test_admin_nlp_summarize_endpoint():
    from app.api.routes.admin import run_nlp_summarization_batch

    db = MagicMock()
    settings = MagicMock()

    with patch(
        "app.nlp.summarization_batch.NLPSummarizationRunner"
    ) as MockRunner:
        mock_instance = MockRunner.return_value

        mock_result = MagicMock()
        mock_result.processed = 1
        mock_result.skipped = 0
        mock_result.failed = 0
        mock_result.total_seen = 1
        mock_result.errors = []

        mock_instance.run.return_value = mock_result

        response = run_nlp_summarization_batch(
            db=db,
            settings=settings,
            _auth=None,
        )

        assert response.processed == 1
        assert response.failed == 0
        assert response.total_seen == 1
        assert "complete" in response.message