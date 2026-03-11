"""Document chunking for embedding — token-approximate splitting with overlap.

Splits large documents into overlapping passages for vector embedding.
Character-based approximation: ~4 chars/token for English text.

Usage:
    from legion_koi.chunking import chunk_text

    chunks = chunk_text(long_document)
    # Returns ["chunk 1...", "chunk 2...", ...]
"""

import re

from .constants import CHUNK_CHARS, CHUNK_OVERLAP_CHARS, MIN_SPLIT_CHARS

# Paragraph/sentence boundary pattern
_SPLIT_RE = re.compile(r"\n{2,}|\n(?=#+\s)|(?<=[.!?])\s+")


def chunk_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping chunks at sentence/paragraph boundaries.

    Returns list of chunk strings. Short documents return a single chunk.
    Empty/whitespace-only text returns an empty list.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) < MIN_SPLIT_CHARS:
        return [text]

    # Split into segments at natural boundaries
    segments = _SPLIT_RE.split(text)
    segments = [s for s in segments if s and s.strip()]

    if not segments:
        return [text]

    chunks = []
    current = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # If adding this segment would exceed chunk size, finalize current chunk
        if current and len(current) + len(segment) + 1 > chunk_chars:
            chunks.append(current.strip())
            # Start new chunk with overlap from end of previous
            if overlap_chars > 0 and len(current) > overlap_chars:
                current = current[-overlap_chars:]
            else:
                current = ""

        if current:
            current += " " + segment
        else:
            current = segment

        # Handle segments that are individually larger than chunk_chars
        while len(current) > chunk_chars * 1.5:
            # Force-split at chunk_chars, preferring space boundaries
            split_at = chunk_chars
            space_pos = current.rfind(" ", chunk_chars // 2, chunk_chars + 100)
            if space_pos > 0:
                split_at = space_pos

            chunks.append(current[:split_at].strip())
            if overlap_chars > 0:
                overlap_start = max(0, split_at - overlap_chars)
                current = current[overlap_start:]
            else:
                current = current[split_at:]

    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())

    return chunks
