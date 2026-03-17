"""Firefox profile discovery — parses profiles.ini to find places.sqlite databases."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.stdlib.get_logger()


@dataclass
class FirefoxProfile:
    """A discovered Firefox profile with its places.sqlite path."""

    name: str
    path: Path
    machine_name: str
    _slug: str = field(init=False, repr=False)

    def __post_init__(self):
        self._slug = f"{self.machine_name}-{self.name}"

    @property
    def slug(self) -> str:
        """Profile slug: '{machine_name}-{profile_name}'."""
        return self._slug

    @property
    def places_path(self) -> Path:
        """Path to places.sqlite within this profile."""
        return self.path / "places.sqlite"


def discover_profiles(
    firefox_dir: Path,
    machine_name: str,
) -> list[FirefoxProfile]:
    """Discover Firefox profiles from profiles.ini.

    Args:
        firefox_dir: Firefox config directory (e.g., ~/.config/mozilla/firefox/)
        machine_name: Machine identifier for slug generation (e.g., 'legion')

    Returns:
        List of profiles that have a places.sqlite file.
    """
    ini_path = firefox_dir / "profiles.ini"
    if not ini_path.exists():
        log.warning("firefox.no_profiles_ini", path=str(ini_path))
        return []

    config = configparser.ConfigParser()
    config.read(ini_path)

    profiles: list[FirefoxProfile] = []
    for section in config.sections():
        if not section.startswith("Profile"):
            continue

        name = config.get(section, "Name", fallback=None)
        path_str = config.get(section, "Path", fallback=None)
        is_relative = config.getboolean(section, "IsRelative", fallback=True)

        if not name or not path_str:
            continue

        if is_relative:
            profile_path = firefox_dir / path_str
        else:
            profile_path = Path(path_str)

        if not profile_path.exists():
            log.debug("firefox.profile_dir_missing", name=name, path=str(profile_path))
            continue

        profile = FirefoxProfile(
            name=name,
            path=profile_path,
            machine_name=machine_name,
        )

        if profile.places_path.exists():
            profiles.append(profile)
            log.info(
                "firefox.profile_found",
                slug=profile.slug,
                places=str(profile.places_path),
            )
        else:
            log.debug("firefox.no_places", slug=profile.slug, path=str(profile_path))

    return profiles
