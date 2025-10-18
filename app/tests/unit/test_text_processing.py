"""
Unit tests for text processing utilities
"""

import pytest
from app.utils.text_processing import chunk_text, clean_text, extract_keywords, truncate_text


def test_chunk_text():
    """Test text chunking"""
    text = "This is a test. " * 100  # Create long text
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 0
    assert all(isinstance(chunk, tuple) for chunk in chunks)
    assert all(len(chunk) == 2 for chunk in chunks)


def test_clean_text():
    """Test text cleaning"""
    dirty_text = "  This   is  a\n\n test   "
    clean = clean_text(dirty_text)

    assert clean == "This is a test"
    assert "  " not in clean


def test_extract_keywords():
    """Test keyword extraction"""
    text = "Python programming is great. Python is powerful and versatile."
    keywords = extract_keywords(text, max_keywords=5)

    assert len(keywords) <= 5
    assert "python" in keywords  # Should be lowercase


def test_truncate_text():
    """Test text truncation"""
    long_text = "A" * 1000
    truncated = truncate_text(long_text, max_length=100)

    assert len(truncated) <= 100
    assert truncated.endswith("...")
