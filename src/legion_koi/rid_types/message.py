from rid_lib.core import ORN


class LegionMessage(ORN):
    namespace = "legion.claude-message"

    def __init__(self, message_id: str):
        self.message_id = message_id

    @property
    def reference(self) -> str:
        return self.message_id

    @classmethod
    def from_reference(cls, reference: str):
        if reference:
            return cls(reference)
        raise ValueError("LegionMessage reference must be a non-empty message ID")
