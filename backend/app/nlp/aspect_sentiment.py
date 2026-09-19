from __future__ import annotations

import re

from app.nlp.sentiment import analyze_sentiment


ASPECT_KEYWORDS: dict[str, set[str]] = {
    "acting": {
        "acting", "actor", "actors", "actress", "actresses",
        "performance", "performances", "cast", "character", "characters",
    },
    "plot": {
        "plot", "story", "storyline", "script",
        "screenplay", "narrative", "writing",
    },
    "pacing": {
        "pace", "pacing", "slow", "fast", "length",
        "dragged", "dragging",
    },
    "visuals": {
        "visual", "visuals", "cinematography", "effects",
        "effect", "animation", "graphics", "scenery", "camera",
    },
    "ending": {
        "ending", "finale", "final", "conclusion", "climax",
    },
}


def _split_sentences(text: str) -> list[str]:
    """Split review text into simple sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]


def analyze_aspects(text: str) -> dict[str, dict[str, object]]:
    """Analyze sentiment for each movie-review aspect mentioned."""

    results: dict[str, dict[str, object]] = {}

    for sentence in _split_sentences(text):
        words = set(
            re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        )

        for aspect, keywords in ASPECT_KEYWORDS.items():
            if words & keywords:
                sentiment = analyze_sentiment(sentence)

                results[aspect] = {
                    "label": sentiment["label"],
                    "score": sentiment["score"],
                }

    return results