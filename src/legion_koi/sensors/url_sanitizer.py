"""URL sanitization and hashing for browser history deduplication."""

import hashlib
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


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
            if key_lower in KEEP_PARAMS or key_lower not in TRACKING_PARAMS:
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


def is_suppressed(url: str, domain: str) -> bool:
    """Check if a URL should be excluded from ingestion.

    Suppresses:
    - Internal browser schemes (about:, chrome:, moz-extension:, etc.)
    - Known auth/banking domains
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() in SUPPRESSED_SCHEMES:
        return True
    if domain.lower() in SUPPRESSED_DOMAINS:
        return True
    return False
