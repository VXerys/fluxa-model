"""Rule-based description extractor for Indonesian finance voice transcripts.

Extracts transaction descriptions by identifying context prepositions
(buat, untuk, karena, keur, kanggo) and capturing all following text as
the description context.

Examples:
    "kopi kenangan buat lembur"  ->  "buat lembur"
    "nasi goreng untuk makan siang"  ->  "untuk makan siang"
    "bayar listrik karena telat"  ->  "karena telat"
"""

from __future__ import annotations


# Context prepositions that introduce transaction purpose/reason.
_CONTEXT_PREPOSITIONS: frozenset[str] = frozenset({
    "buat", "untuk", "karena", "keur", "kanggo"
})


def extract_description(cleaned_text: str) -> str:
    """Extract transaction description starting from a context preposition.

    Args:
        cleaned_text: Text after date and amount tokens have been removed.

    Returns:
        Description string starting from the first context preposition
        (max 200 chars at word boundary). Empty string if no preposition found.
    """
    if not cleaned_text:
        return ""

    # Tokenize the cleaned text
    tokens = cleaned_text.lower().split()

    # Scan for the first context preposition
    for i, token in enumerate(tokens):
        if token in _CONTEXT_PREPOSITIONS:
            # Join the preposition and all following tokens
            description_candidate = " ".join(tokens[i:])

            # Truncate to 200 characters at the last complete word boundary
            if len(description_candidate) <= 200:
                return description_candidate

            # Find the last space before or at position 200
            truncated = description_candidate[:200]
            last_space = truncated.rfind(" ")

            if last_space > 0:
                return truncated[:last_space]
            else:
                # No space found, return up to 200 chars
                return truncated

    # No preposition found
    return ""
