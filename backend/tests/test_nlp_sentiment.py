from app.nlp.sentiment import analyze_sentiment


def test_positive_sentiment():
    result = analyze_sentiment(
        "This movie was absolutely amazing and enjoyable"
    )

    assert result["label"] == "positive"
    assert 0 <= result["score"] <= 1


def test_negative_sentiment():
    result = analyze_sentiment(
        "This movie was terrible and boring"
    )

    assert result["label"] == "negative"
    assert 0 <= result["score"] <= 1


def test_neutral_sentiment():
    result = analyze_sentiment(
        "The movie was okay"
    )

    assert result["label"] == "neutral"
    assert 0 <= result["score"] <= 1


def test_sentiment_result_structure():
    result = analyze_sentiment("The acting was fantastic")

    assert "label" in result
    assert "score" in result
    assert result["label"] in {"positive", "negative", "neutral"}