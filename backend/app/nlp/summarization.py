"""
Review Summarization — Milestone 13.

Provides an extractive summarization function to generate a short
natural-language summary of what reviewers are saying. Uses a lightweight
word-frequency-based sentence scoring algorithm as the baseline.
"""

import re
from collections import Counter

def generate_review_summary(reviews_texts: list[str], max_sentences: int = 3) -> str | None:
    """
    Generate an extractive summary from a list of review bodies.

    Args:
        reviews_texts: List of review text bodies.
        max_sentences: Maximum number of sentences to include in the summary.

    Returns:
        A summarized string, or None if there is not enough text to summarize.
    """
    if not reviews_texts:
        return None

    # Combine all reviews into a single text
    combined_text = " ".join(t.strip() for t in reviews_texts if t.strip())
    if not combined_text:
        return None

    # Simple sentence splitting
    # Matches sentences ending with . ! or ? followed by a space or end of string.
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$')
    sentences = [s.strip() for s in sentence_pattern.split(combined_text) if s.strip()]

    if not sentences:
        return None

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Tokenize words to calculate frequencies (ignoring common stop words)
    words = re.findall(r'\b\w+\b', combined_text.lower())

    # A simple list of stop words to ignore during frequency counting
    stop_words = {
        'the', 'and', 'a', 'to', 'of', 'in', 'i', 'is', 'that', 'it', 'on', 'you',
        'this', 'for', 'but', 'with', 'are', 'have', 'be', 'at', 'or', 'as', 'was',
        'so', 'if', 'out', 'not', 'movie', 'film'
    }

    word_freq = Counter(w for w in words if w not in stop_words)

    # Avoid div by zero if word_freq is empty
    if not word_freq:
        return " ".join(sentences[:max_sentences])

    max_freq = max(word_freq.values())

    # Normalize frequencies
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    # Score sentences
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0.0
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        for word in sentence_words:
            if word in word_freq:
                score += word_freq[word]
        # Normalize by length to avoid bias towards extremely long sentences
        # Add small constant to avoid division by zero
        length_penalty = len(sentence_words) + 1
        sentence_scores[i] = score / length_penalty

    # Sort sentences by score in descending order
    ranked_sentences_indices = sorted(
        sentence_scores.keys(),
        key=lambda i: sentence_scores[i],
        reverse=True
    )

    # Take top `max_sentences`
    top_indices = sorted(ranked_sentences_indices[:max_sentences])

    # Reconstruct summary in original order
    summary_sentences = [sentences[i] for i in top_indices]

    return " ".join(summary_sentences)
