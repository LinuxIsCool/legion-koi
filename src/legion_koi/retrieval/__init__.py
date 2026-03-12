"""Retrieval improvements — convex fusion, query routing."""

from .fusion import convex_combine
from .router import QueryType, classify_query

__all__ = ["convex_combine", "QueryType", "classify_query"]
