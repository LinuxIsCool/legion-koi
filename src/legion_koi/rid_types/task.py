from rid_lib.core import ORN


class LegionTask(ORN):
    namespace = "legion.claude-task"

    def __init__(self, task_id: str):
        self.task_id = task_id

    @property
    def reference(self) -> str:
        return self.task_id

    @classmethod
    def from_reference(cls, reference: str):
        if reference:
            return cls(reference)
        raise ValueError("LegionTask reference must be a non-empty task ID")
