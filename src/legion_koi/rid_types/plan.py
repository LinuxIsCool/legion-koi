from rid_lib.core import ORN


class LegionPlan(ORN):
    namespace = "legion.claude-plan"

    def __init__(self, slug: str):
        self.slug = slug

    @property
    def reference(self) -> str:
        return self.slug

    @classmethod
    def from_reference(cls, reference: str):
        if not reference:
            raise ValueError(
                "LegionPlan reference must be a non-empty slug, "
                f"got: {reference!r}"
            )
        return cls(slug=reference)
