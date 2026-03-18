"""Privacy configuration for browser history sensor — hot-reloadable domain suppression and param policies."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import yaml

log = logging.getLogger(__name__)

# Fallback defaults (mirror url_sanitizer.py hardcoded values)
_FALLBACK_SUPPRESSED_DOMAINS = {
    "accounts.google.com",
    "login.microsoftonline.com",
    "login.live.com",
    "auth0.com",
    "signin.aws.amazon.com",
    "id.atlassian.com",
}

_FALLBACK_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "utm_creative_format", "utm_marketing_tactic",
    "fbclid", "gclid", "gclsrc", "dclid", "gbraid", "wbraid",
    "mc_cid", "mc_eid", "msclkid", "twclid", "igshid", "s_kwcid", "ef_id",
    "_ga", "_gl", "_hsenc", "_hsmi", "yclid", "ref", "ref_src", "ref_url", "si",
}


class DomainSuppressionList:
    """Hot-reloadable domain suppression list using fnmatch glob patterns."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float = 0
        self._patterns: list[str] = []
        self._loaded = False

    def _reload_if_changed(self) -> None:
        try:
            st = self._path.stat()
            if st.st_mtime != self._mtime:
                self._mtime = st.st_mtime
                lines = self._path.read_text().splitlines()
                self._patterns = [
                    line.strip().lower()
                    for line in lines
                    if line.strip() and not line.strip().startswith("#")
                ]
                self._loaded = True
                log.info("privacy_config.suppression_loaded", extra={
                    "path": str(self._path), "pattern_count": len(self._patterns),
                })
        except FileNotFoundError:
            if not self._loaded:
                self._patterns = list(_FALLBACK_SUPPRESSED_DOMAINS)
                self._loaded = True
            log.warning("privacy_config.suppression_file_missing, using %s",
                        "fallback" if not self._loaded else "stale config",
                        extra={"path": str(self._path)})
        except Exception:
            if not self._loaded:
                self._patterns = list(_FALLBACK_SUPPRESSED_DOMAINS)
                self._loaded = True
            log.exception("privacy_config.suppression_reload_error, keeping %s",
                          "fallback" if not self._loaded else "stale config")

    def is_suppressed(self, domain: str) -> bool:
        """Check if a domain matches any suppression pattern."""
        self._reload_if_changed()
        domain_lower = domain.lower()
        for pattern in self._patterns:
            if fnmatch.fnmatch(domain_lower, pattern):
                return True
        return False

    @property
    def patterns(self) -> list[str]:
        self._reload_if_changed()
        return list(self._patterns)


class ParamPolicy:
    """Hot-reloadable per-domain query parameter keep/strip rules."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float = 0
        self._config: dict = {}
        self._loaded = False

    def _reload_if_changed(self) -> None:
        try:
            st = self._path.stat()
            if st.st_mtime != self._mtime:
                self._mtime = st.st_mtime
                with open(self._path) as f:
                    self._config = yaml.safe_load(f) or {}
                self._loaded = True
                domain_count = len(self._config.get("domains", {}))
                log.info("privacy_config.param_policy_loaded", extra={
                    "path": str(self._path), "domain_rules": domain_count,
                })
        except FileNotFoundError:
            if not self._loaded:
                self._config = {}
                self._loaded = True
            log.warning("privacy_config.param_policy_file_missing, using %s",
                        "fallback" if not self._loaded else "stale config",
                        extra={"path": str(self._path)})
        except Exception:
            if not self._loaded:
                self._config = {}
                self._loaded = True
            log.exception("privacy_config.param_policy_reload_error, keeping %s",
                          "fallback" if not self._loaded else "stale config")

    def _find_domain_rule(self, domain: str) -> tuple[dict | None, str]:
        """Find the best matching domain rule. Exact match first, then glob."""
        domains = self._config.get("domains", {})
        domain_lower = domain.lower()

        # Exact match
        if domain_lower in domains:
            return domains[domain_lower], domain_lower

        # Glob match (first match wins)
        for pattern, rule in domains.items():
            if fnmatch.fnmatch(domain_lower, pattern):
                return rule, pattern

        return None, "default"

    def resolve(self, domain: str, query_string: str) -> tuple[str, list[str], str]:
        """Apply param policy to a query string.

        Returns (filtered_query_string, stripped_keys, policy_name).
        """
        self._reload_if_changed()

        if not query_string:
            return "", [], "default"

        params = parse_qs(query_string, keep_blank_values=False)
        if not params:
            return "", [], "default"

        defaults = self._config.get("defaults", {})
        default_strip = {s.lower() for s in defaults.get("strip", _FALLBACK_TRACKING_PARAMS)}

        rule, policy_name = self._find_domain_rule(domain)

        filtered = {}
        stripped = []

        if rule is None:
            # Default mode: strip tracking params, keep everything else
            for key, values in params.items():
                if key.lower() in default_strip:
                    stripped.append(key)
                else:
                    filtered[key] = values
        else:
            mode = rule.get("mode", "strip")
            if mode == "keep":
                # Only keep explicitly listed params
                keep_set = {k.lower() for k in rule.get("keep", [])}
                for key, values in params.items():
                    if key.lower() in keep_set:
                        filtered[key] = values
                    else:
                        stripped.append(key)
            elif mode == "strip_only":
                # Strip defaults but always preserve domain's keep list
                keep_set = {k.lower() for k in rule.get("keep", [])}
                for key, values in params.items():
                    key_lower = key.lower()
                    if key_lower in keep_set:
                        filtered[key] = values
                    elif key_lower in default_strip:
                        stripped.append(key)
                    else:
                        filtered[key] = values
            else:
                # "strip" mode (same as default)
                strip_set = {s.lower() for s in rule.get("strip", default_strip)}
                for key, values in params.items():
                    if key.lower() in strip_set:
                        stripped.append(key)
                    else:
                        filtered[key] = values

        filtered_qs = urlencode(filtered, doseq=True) if filtered else ""
        return filtered_qs, stripped, policy_name


class PrivacyConfig:
    """Container for all privacy configuration components."""

    def __init__(self, suppression_path: Path, param_policy_path: Path) -> None:
        self._suppression = DomainSuppressionList(suppression_path)
        self._params = ParamPolicy(param_policy_path)

    @property
    def suppression(self) -> DomainSuppressionList:
        return self._suppression

    @property
    def params(self) -> ParamPolicy:
        return self._params
