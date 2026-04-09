from rid_lib.core import ORN


class LegionVoiceEvent(ORN):
    namespace = "legion.claude-voice"

    def __init__(self, event_ref: str):
        self.event_ref = event_ref

    @property
    def reference(self) -> str:
        return self.event_ref

    @classmethod
    def from_reference(cls, reference: str):
        if reference:
            return cls(reference)
        raise ValueError("LegionVoiceEvent reference must be a non-empty string")
