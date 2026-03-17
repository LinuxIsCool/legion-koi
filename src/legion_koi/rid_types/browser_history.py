from rid_lib.core import ORN


class LegionBrowserHistory(ORN):
    namespace = "legion.claude-browser-history"

    def __init__(self, profile: str, entry_id: str):
        self.profile = profile
        self.entry_id = entry_id

    @property
    def reference(self) -> str:
        return f"{self.profile}/{self.entry_id}"

    @classmethod
    def from_reference(cls, reference: str):
        parts = reference.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return cls(*parts)
        raise ValueError(
            "LegionBrowserHistory reference must be '<profile>/<entry_id>', "
            f"got: {reference!r}"
        )
