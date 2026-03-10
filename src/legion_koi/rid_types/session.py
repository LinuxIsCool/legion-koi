from rid_lib.core import ORN


class LegionSession(ORN):
    namespace = "legion.claude-session"

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def reference(self) -> str:
        return self.session_id

    @classmethod
    def from_reference(cls, reference: str):
        if reference:
            return cls(reference)
        raise ValueError("LegionSession reference must be a non-empty session ID")
