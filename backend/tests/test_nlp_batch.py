from unittest.mock import MagicMock, call

from app.nlp.batch import NLPBatchRunner


def test_nlp_batch_processes_review():
    review = MagicMock()
    review.id = 1
    review.body = "The acting was amazing and the plot was excellent."
    review.nlp_processed_at = None

    pending_query = MagicMock()
    pending_query.count.return_value = 1

    first_batch_query = MagicMock()
    first_batch_query.filter.return_value.order_by.return_value.limit.return_value.offset.return_value.all.return_value = [
        review
    ]

    second_batch_query = MagicMock()
    second_batch_query.filter.return_value.order_by.return_value.limit.return_value.offset.return_value.all.return_value = []

    db = MagicMock()
    db.query.side_effect = [
        pending_query,
        first_batch_query,
        second_batch_query,
    ]

    settings = MagicMock()

    runner = NLPBatchRunner(db, settings)
    result = runner.run(batch_size=100)

    assert result.processed == 1
    assert result.skipped == 0
    assert result.failed == 0
    assert result.total_seen == 1
    assert result.errors == []

    assert review.sentiment_label == "positive"
    assert review.sentiment_score is not None
    assert review.aspect_sentiment is not None
    assert "acting" in review.aspect_sentiment
    assert "plot" in review.aspect_sentiment
    assert review.nlp_processed_at is not None

    db.commit.assert_called()


def test_nlp_batch_skips_empty_review():
    review = MagicMock()
    review.id = 2
    review.body = "   "
    review.nlp_processed_at = None

    pending_query = MagicMock()
    pending_query.count.return_value = 1

    first_batch_query = MagicMock()
    first_batch_query.filter.return_value.order_by.return_value.limit.return_value.offset.return_value.all.return_value = [
        review
    ]

    second_batch_query = MagicMock()
    second_batch_query.filter.return_value.order_by.return_value.limit.return_value.offset.return_value.all.return_value = []

    db = MagicMock()
    db.query.side_effect = [
        pending_query,
        first_batch_query,
        second_batch_query,
    ]

    settings = MagicMock()

    runner = NLPBatchRunner(db, settings)
    result = runner.run(batch_size=100)

    assert result.processed == 0
    assert result.skipped == 1
    assert result.failed == 0
    assert result.total_seen == 1
    assert result.errors == []

    assert review.nlp_processed_at is not None
    db.commit.assert_called()