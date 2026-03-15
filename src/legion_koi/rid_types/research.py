from rid_lib.core import ORN


class LegionResearch(ORN):
    namespace = "legion.claude-research"

    def __init__(self, slug: str):
        self.slug = slug

    @property
    def reference(self) -> str:
        return self.slug

    @classmethod
    def from_reference(cls, reference: str):
        if not reference:
            raise ValueError(
                f"LegionResearch reference must be non-empty, got: {reference!r}"
            )
        return cls(slug=reference)
