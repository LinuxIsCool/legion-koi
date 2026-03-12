"""Query router — classify queries to select optimal search strategy.

Rule-based classifier (no model call):
- KEYWORD: exact terms, short queries → FTS only (alpha=1.0)
- CONCEPTUAL: questions, long queries → semantic-heavy (alpha=0.5)
- TEMPORAL: date/time references → FTS with date filter
- HYBRID: default → convex combination with learned alpha
"""

from __future__ import annotations

import re
from enum import Enum


class QueryType(Enum):
    KEYWORD = "keyword"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


# Patterns for temporal queries
_TEMPORAL_PATTERNS = re.compile(
    r"\b("
    r"yesterday|today|last\s+week|this\s+week|last\s+month|this\s+month"
    r"|last\s+\d+\s+days?|recent|lately|since\s+\w+"
    r"|202\d[-/]\d{2}([-/]\d{2})?"
    r")\b",
    re.IGNORECASE,
)

# Question starters that indicate conceptual queries
_QUESTION_STARTERS = re.compile(
    r"^\s*(how|what|why|when|where|who|which|explain|describe|tell\s+me)\b",
    re.IGNORECASE,
)

# Quoted phrases indicate keyword intent
_QUOTED = re.compile(r'"[^"]+"|\'[^\']+\'')


def classify_query(query: str) -> QueryType:
    """Classify a query to select the best search strategy.

    Rules (applied in order):
    1. Has quotes or is <=2 words → KEYWORD
    2. Contains date/time references → TEMPORAL
    3. Is a question (how/what/why/when...) or >6 words → CONCEPTUAL
    4. Default → HYBRID
    """
    stripped = query.strip()
    if not stripped:
        return QueryType.HYBRID

    # Quoted phrases → user wants exact match
    if _QUOTED.search(stripped):
        return QueryType.KEYWORD

    # Very short queries → keyword intent
    words = stripped.split()
    if len(words) <= 2:
        return QueryType.KEYWORD

    # Temporal references
    if _TEMPORAL_PATTERNS.search(stripped):
        return QueryType.TEMPORAL

    # Questions → conceptual
    if _QUESTION_STARTERS.match(stripped):
        return QueryType.CONCEPTUAL

    # Long queries tend to be conceptual
    if len(words) > 6:
        return QueryType.CONCEPTUAL

    return QueryType.HYBRID
