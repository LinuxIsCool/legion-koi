from rid_lib.core import ORN


class LegionTranscript(ORN):
    namespace = "legion.claude-transcript"

    def __init__(self, identifier: str):
        self.identifier = identifier

    @property
    def reference(self) -> str:
        return self.identifier

    @classmethod
    def from_reference(cls, reference: str):
        if not reference:
            raise ValueError(
                "LegionTranscript reference must be a non-empty string "
                f"(e.g. '2026-03-12-regen-standup'), got: {reference!r}"
            )
        return cls(identifier=reference)
