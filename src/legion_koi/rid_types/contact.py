from rid_lib.core import ORN


class LegionContact(ORN):
    namespace = "legion.claude-contact"

    def __init__(self, identity_id: str):
        self.identity_id = identity_id

    @property
    def reference(self) -> str:
        return self.identity_id

    @classmethod
    def from_reference(cls, reference: str):
        if reference:
            return cls(reference)
        raise ValueError("LegionContact reference must be a non-empty identity ID")
