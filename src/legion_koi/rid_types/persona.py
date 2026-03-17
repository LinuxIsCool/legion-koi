from rid_lib.core import ORN


class LegionPersona(ORN):
    """RID type for persona data items.

    ORN pattern: orn:legion.claude-persona:{slug}:{item_type}:{identifier}
    Item types: observation, belief, fact, relationship, style-sample, decision
    """

    namespace = "legion.claude-persona"

    def __init__(self, slug: str, item_type: str, identifier: str):
        self.slug = slug
        self.item_type = item_type
        self.identifier = identifier

    @property
    def reference(self) -> str:
        return f"{self.slug}:{self.item_type}:{self.identifier}"

    @classmethod
    def from_reference(cls, reference: str):
        parts = reference.split(":", 2)
        if len(parts) == 3 and all(parts):
            return cls(*parts)
        raise ValueError(
            "LegionPersona reference must be '<slug>:<item_type>:<identifier>' "
            f"(e.g. 'darren:belief:knowledge-sovereignty'), got: {reference!r}"
        )
