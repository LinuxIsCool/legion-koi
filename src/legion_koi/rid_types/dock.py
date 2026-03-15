from rid_lib.core import ORN


class LegionDock(ORN):
    namespace = "legion.claude-dock"

    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo

    @property
    def reference(self) -> str:
        return f"{self.owner}/{self.repo}"

    @classmethod
    def from_reference(cls, reference: str):
        parts = reference.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return cls(*parts)
        raise ValueError(
            "LegionDock reference must be '<owner>/<repo>' "
            f"(e.g. 'astral-sh/uv'), got: {reference!r}"
        )
