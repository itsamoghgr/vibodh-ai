"""
Text Processing Utilities
"""

import re
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str, int]]:
    """
    Chunk text into smaller segments with overlap.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of (chunk_text, start_position) tuples
    """
    if not text or len(text) == 0:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending punctuation
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            last_break = max(last_period, last_newline)

            if last_break > start:
                end = last_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))

        start = end - overlap

    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)

    # Strip and return
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation).

    Args:
        text: Input text
        max_keywords: Maximum number of keywords

    Returns:
        List of keywords
    """
    if not text:
        return []

    # Simple word frequency approach
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    # Remove common stop words
    stop_words = {
        'that', 'this', 'with', 'from', 'have', 'been', 'were', 'said',
        'what', 'when', 'where', 'which', 'about', 'their', 'there',
        'these', 'those', 'would', 'could', 'should'
    }

    words = [w for w in words if w not in stop_words]

    # Count frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:max_keywords]]


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)].strip() + suffix
