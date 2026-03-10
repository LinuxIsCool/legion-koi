from rid_lib.core import ORN


class LegionRecording(ORN):
    namespace = "legion.claude-recording"

    def __init__(self, source: str, identifier: str):
        self.source = source
        self.identifier = identifier

    @property
    def reference(self) -> str:
        return f"{self.source}/{self.identifier}"

    @classmethod
    def from_reference(cls, reference: str):
        parts = reference.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return cls(*parts)
        raise ValueError(
            "LegionRecording reference must be '<source>/<identifier>' "
            f"(e.g. 'otter/2026-03-10-standup'), got: {reference!r}"
        )
