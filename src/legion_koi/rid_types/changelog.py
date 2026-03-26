from rid_lib.core import ORN


class LegionChangelog(ORN):
    namespace = "legion.claude-changelog"

    def __init__(self, repo: str, version: str):
        self.repo = repo        # e.g. "anthropics/claude-code"
        self.version = version  # e.g. "2.1.83"

    @property
    def reference(self) -> str:
        return f"{self.repo}/{self.version}"

    @classmethod
    def from_reference(cls, reference: str):
        # Split on last "/" to separate version from repo path
        idx = reference.rfind("/")
        if idx <= 0:
            raise ValueError(
                f"LegionChangelog reference must be 'owner/repo/version', "
                f"got: {reference!r}"
            )
        return cls(repo=reference[:idx], version=reference[idx + 1:])
