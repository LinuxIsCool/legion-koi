from rid_lib.core import ORN


class LegionJournal(ORN):
    namespace = "legion.claude-journal"

    def __init__(self, date: str, slug: str):
        self.date = date
        self.slug = slug

    @property
    def reference(self) -> str:
        return f"{self.date}/{self.slug}"

    @classmethod
    def from_reference(cls, reference: str):
        parts = reference.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return cls(*parts)
        raise ValueError(
            "LegionJournal reference must be '<date>/<slug>' "
            f"(e.g. '2026-03-10/1225-the-plugin-koi-insight'), got: {reference!r}"
        )
