from rid_lib.core import ORN


class LegionVenture(ORN):
    namespace = "legion.venture"

    def __init__(self, stage: str, id: str):
        self.stage = stage
        self.id = id

    @property
    def reference(self) -> str:
        return f"{self.stage}/{self.id}"

    @classmethod
    def from_reference(cls, reference: str):
        parts = reference.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return cls(*parts)
        raise ValueError(
            "LegionVenture reference must be '<stage>/<id>' "
            f"(e.g. 'active/oral-history-ontology'), got: {reference!r}"
        )
