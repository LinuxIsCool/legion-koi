"""URL sanitization and hashing for browser history deduplication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .privacy_config import DomainSuppressionList, ParamPolicy


# Tracking parameters to strip from URLs
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "utm_creative_format", "utm_marketing_tactic",
    "fbclid", "gclid", "gclsrc", "dclid", "gbraid", "wbraid",
    "mc_cid", "mc_eid",
    "msclkid",
    "twclid",
    "igshid",
    "s_kwcid", "ef_id",
    "_ga", "_gl", "_hsenc", "_hsmi",
    "yclid",
    "ref", "ref_src", "ref_url",
    "si",  # Spotify/YouTube share tracking
}

# Parameters to preserve (carry semantic meaning)
KEEP_PARAMS = {
    "q", "query", "search", "s",  # search queries
    "v", "list", "t",  # video/playlist/time
    "page", "p", "offset",  # pagination
    "id", "ids",  # identifiers
    "tab", "view",  # navigation state
    "branch", "path", "file",  # code navigation
    "sort", "order", "filter",  # result ordering
    "lang", "locale",  # language
}

# Suppressed URL schemes (internal browser pages)
SUPPRESSED_SCHEMES = {"about", "chrome", "moz-extension", "resource", "data", "blob", "javascript"}

# Suppressed domains (auth/banking — Phase 1 hardcoded, Phase 2 configurable)
SUPPRESSED_DOMAINS = {
    "accounts.google.com",
    "login.microsoftonline.com",
    "login.live.com",
    "auth0.com",
    "signin.aws.amazon.com",
    "id.atlassian.com",
}


@dataclass
class SanitizeResult:
    """Extended sanitization result with privacy metadata."""
    url: str
    params_stripped: list[str] = field(default_factory=list)
    policy_applied: str = "default"


def sanitize_url(url: str) -> str:
    """Normalize a URL for deduplication.

    - Lowercase scheme and host
    - Strip tracking parameters
    - Remove fragments
    - Normalize trailing slashes on path-only URLs
    """
    if not url:
        return ""

    parsed = urlparse(url)

    # Suppress internal browser URLs early
    if parsed.scheme.lower() in SUPPRESSED_SCHEMES:
        return url  # Return as-is; caller checks is_suppressed separately

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip tracking params, keep semantic ones
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {}
        for key, values in params.items():
            key_lower = key.lower()
            if key_lower in TRACKING_PARAMS:
                continue
            # Keep all non-tracking params (Phase 2: restrict to KEEP_PARAMS only)
            filtered[key] = values
        query = urlencode(filtered, doseq=True)
    else:
        query = ""

    # Normalize path: strip trailing slash unless it's the root
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Drop fragment
    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def url_hash(url: str) -> str:
    """SHA-256 hash of sanitized URL, truncated to 12 hex chars."""
    clean = sanitize_url(url)
    return hashlib.sha256(clean.encode()).hexdigest()[:12]


def is_suppressed(url: str, domain: str, suppression_list: DomainSuppressionList | None = None) -> bool:
    """Check if a URL should be excluded from ingestion.

    Suppresses:
    - Internal browser schemes (about:, chrome:, moz-extension:, etc.)
    - Known auth/banking domains (hardcoded fallback or configurable suppression list)
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() in SUPPRESSED_SCHEMES:
        return True
    if suppression_list is not None:
        return suppression_list.is_suppressed(domain)
    if domain.lower() in SUPPRESSED_DOMAINS:
        return True
    return False


def sanitize_url_ext(url: str, domain: str, param_policy: ParamPolicy | None = None) -> SanitizeResult:
    """Extended sanitization with privacy tracking metadata.

    Same normalization as sanitize_url() but uses ParamPolicy for param filtering
    and returns metadata about what was stripped.
    """
    if not url:
        return SanitizeResult(url="")

    parsed = urlparse(url)

    # Suppress internal browser URLs early
    if parsed.scheme.lower() in SUPPRESSED_SCHEMES:
        return SanitizeResult(url=url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    params_stripped: list[str] = []
    policy_applied = "default"

    # Apply param policy
    if parsed.query and param_policy is not None:
        query, params_stripped, policy_applied = param_policy.resolve(domain, parsed.query)
    elif parsed.query:
        # Fallback: same logic as sanitize_url()
        params = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {}
        for key, values in params.items():
            key_lower = key.lower()
            if key_lower in TRACKING_PARAMS:
                params_stripped.append(key)
                continue
            filtered[key] = values
        query = urlencode(filtered, doseq=True)
    else:
        query = ""

    # Normalize path: strip trailing slash unless it's the root
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Drop fragment
    clean_url = urlunparse((scheme, netloc, path, parsed.params, query, ""))

    return SanitizeResult(
        url=clean_url,
        params_stripped=params_stripped,
        policy_applied=policy_applied,
    )
