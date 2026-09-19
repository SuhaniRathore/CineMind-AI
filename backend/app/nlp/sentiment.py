"""
Sentiment Analysis for CineMind AI.

Milestone 12:
- Provides a lightweight TF-IDF + Logistic Regression classifier.
- Returns one of three labels: 'positive', 'negative', or 'neutral',
  together with a confidence score in [0.0, 1.0].
- No external model files are required; the classifier is trained on a
  built-in corpus at import time so the pipeline stays fast and
  self-contained.
- A keyword-override tier handles unambiguous strong sentiment signals
  before the statistical model is consulted.

Public API
----------
analyze_sentiment(text: str) -> dict[str, object]
    {"label": "positive" | "negative" | "neutral", "score": float}
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Keyword override tier
# Applied BEFORE the ML model.  When a text contains a strong unambiguous
# signal word the result is returned immediately with a fixed high score.
# This guarantees that obvious cases are always correct regardless of how
# the statistical model weights the training corpus.
# ---------------------------------------------------------------------------

_STRONG_POSITIVE: frozenset[str] = frozenset(
    {
        "amazing", "fantastic", "brilliant", "masterpiece", "outstanding",
        "superb", "excellent", "magnificent", "breathtaking", "extraordinary",
        "phenomenal", "incredible", "wonderful", "perfect", "flawless",
        "unforgettable", "mesmerizing", "captivating", "riveting", "stunning",
        "enjoyable", "delightful", "heartwarming", "inspiring", "thrilling",
    }
)

_STRONG_NEGATIVE: frozenset[str] = frozenset(
    {
        "terrible", "awful", "horrible", "dreadful", "atrocious",
        "abysmal", "pathetic", "disgusting", "unwatchable", "unbearable",
        "horrendous", "appalling", "catastrophic", "laughable", "pointless",
        "boring", "dull", "tedious", "excruciating", "insufferable",
    }
)

# ---------------------------------------------------------------------------
# Training corpus
# Balanced set: 30 positive, 30 negative, 30 neutral.
# Neutral section deliberately includes short colloquial phrases so the
# classifier builds a strong prior for everyday hedge words.
# ---------------------------------------------------------------------------

_TRAIN_TEXTS: list[str] = [
    # ------------------------------------------------------------------ positive (30)
    "the acting was absolutely brilliant and captivating",
    "amazing performances by the entire cast",
    "outstanding cinematography and breathtaking visuals",
    "the plot was gripping and kept me on the edge of my seat",
    "a masterpiece of storytelling and direction",
    "the pacing was perfect and the ending was satisfying",
    "wonderful script with great character development",
    "superb special effects that looked incredibly realistic",
    "the music score elevated every emotional scene beautifully",
    "one of the best films I have ever seen",
    "excellent direction and a compelling narrative throughout",
    "the cast delivered powerful and memorable performances",
    "a feel-good movie with heart and genuine laughs",
    "the visuals were stunning and the story was moving",
    "great chemistry between the leads made it unforgettable",
    "this movie was absolutely amazing and enjoyable",
    "the acting was fantastic from start to finish",
    "I loved every single minute of this film",
    "a truly spectacular cinematic experience",
    "the story was deeply moving and beautifully told",
    "riveting from the very first scene to the last",
    "an inspiring and uplifting film I will always remember",
    "the direction was phenomenal and every shot was gorgeous",
    "thrilling action sequences paired with great storytelling",
    "heartwarming and funny in all the right ways",
    "delightful performances all round made this a joy to watch",
    "the screenplay was sharp witty and full of great moments",
    "a remarkable film that stays with you long after the credits",
    "incredible world-building and an engaging plot throughout",
    "flawless execution from the director and the entire crew",
    # ------------------------------------------------------------------ negative (30)
    "the acting was terrible and completely unconvincing",
    "boring plot with too many pointless subplots",
    "the pacing was painfully slow and the movie dragged on forever",
    "awful special effects that looked fake and cheap",
    "the ending was a huge disappointment and made no sense",
    "poor script with wooden dialogue and flat characters",
    "one of the worst films I have ever wasted time on",
    "the director seemed completely lost and the story went nowhere",
    "dull and predictable from start to finish",
    "the performances were stiff and emotionless throughout",
    "a total mess with no coherent narrative whatsoever",
    "the cinematography was bland and uninspired",
    "the film failed to deliver on any of its promises",
    "an utter bore with nothing interesting happening",
    "the pacing ruined what could have been a decent story",
    "this movie was terrible and made me want to leave",
    "the story was awful and made absolutely no sense",
    "dreadful performances that ruined the entire film",
    "a horrible waste of time and money",
    "the script was atrocious and the dialogue was laughable",
    "I was bored throughout and nearly fell asleep",
    "extremely disappointing with no redeeming qualities",
    "the acting was so bad it felt like a student project",
    "an incoherent mess that insulted the audience's intelligence",
    "tedious and repetitive with nothing new to offer",
    "the worst film I have seen in years by far",
    "a catastrophic failure in every aspect of filmmaking",
    "the plot made no sense and the characters were unlikable",
    "unwatchable drivel that wasted a talented cast",
    "a painfully bad film that drags on with no payoff",
    # ------------------------------------------------------------------ neutral (30)
    # long neutral
    "it was an okay movie with some good and bad moments",
    "the film was average and nothing special stood out",
    "mixed feelings about the acting and the storyline",
    "some scenes were good but others felt unnecessary",
    "the movie had potential but did not fully deliver",
    "not bad but not great either just mediocre",
    "a decent watch if you have nothing else to do",
    "had its moments but overall was just fine",
    "the story was interesting in parts but lost focus midway",
    "neither the best nor the worst film of the year",
    "the movie was watchable but forgettable in the end",
    "there were some highs and lows but it balanced out",
    "an average production with standard performances",
    "passable entertainment but nothing you would seek out again",
    "the film was neither impressive nor terrible just ordinary",
    "reasonably well made but not particularly memorable",
    "a middle-of-the-road film with no major flaws or highlights",
    "the acting ranged from good to mediocre throughout",
    "enjoyable in places but dragged in others",
    "I neither loved it nor hated it it was just okay",
    # short neutral anchors (critical for short-phrase accuracy)
    "okay movie",
    "the movie was okay",
    "it was okay",
    "average film",
    "the film was average",
    "just average",
    "decent movie",
    "it was decent",
    "fine film",
    "it was fine",
]

_TRAIN_LABELS: list[str] = (
    ["positive"] * 30
    + ["negative"] * 30
    + ["neutral"] * 30
)

# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression pipeline
# C=4.0 sharpens decision boundaries so confident reviews score higher.
# ---------------------------------------------------------------------------

_pipeline: Pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                analyzer="word",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                C=4.0,
                solver="lbfgs",
            ),
        ),
    ]
)

_pipeline.fit(_TRAIN_TEXTS, _TRAIN_LABELS)

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

_NONALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def analyze_sentiment(text: str) -> dict[str, object]:
    """
    Analyse the overall sentiment of *text*.

    Parameters
    ----------
    text:
        A free-form review snippet (sentence or paragraph).

    Returns
    -------
    dict with keys:
        ``label`` - one of ``"positive"``, ``"negative"``, ``"neutral"``.
        ``score`` - confidence probability in ``[0.0, 1.0]``.

    Strategy
    --------
    1. Keyword override: if the text contains an unambiguously strong
       positive or negative signal word, return immediately.
    2. Otherwise delegate to the TF-IDF + Logistic Regression pipeline.
    """
    cleaned = _WHITESPACE_RE.sub(" ", _NONALPHA_RE.sub(" ", text)).strip()
    if not cleaned:
        return {"label": "neutral", "score": 1.0}

    words: set[str] = set(cleaned.lower().split())

    # --- keyword override tier ---
    pos_hits = words & _STRONG_POSITIVE
    neg_hits = words & _STRONG_NEGATIVE

    if pos_hits and not neg_hits:
        return {"label": "positive", "score": 0.97}
    if neg_hits and not pos_hits:
        return {"label": "negative", "score": 0.97}

    # --- statistical model tier ---
    proba = _pipeline.predict_proba([cleaned])[0]
    classes: list[str] = list(_pipeline.classes_)
    best_idx: int = int(proba.argmax())

    return {
        "label": classes[best_idx],
        "score": round(float(proba[best_idx]), 4),
    }
