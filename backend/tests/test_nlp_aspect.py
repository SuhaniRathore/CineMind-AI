from app.nlp.aspect_sentiment import analyze_aspects


def test_acting_aspect():
    result = analyze_aspects(
        "The acting was amazing."
    )

    assert "acting" in result
    assert result["acting"]["label"] == "positive"
    assert 0 <= result["acting"]["score"] <= 1


def test_plot_aspect():
    result = analyze_aspects(
        "The plot was excellent."
    )

    assert "plot" in result
    assert result["plot"]["label"] == "positive"
    assert 0 <= result["plot"]["score"] <= 1


def test_multiple_aspects():
    result = analyze_aspects(
        "The acting was amazing. The plot was excellent."
    )

    assert "acting" in result
    assert "plot" in result


def test_no_aspect():
    result = analyze_aspects(
        "I watched this movie yesterday."
    )

    assert result == {}