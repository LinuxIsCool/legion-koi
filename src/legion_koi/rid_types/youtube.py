from rid_lib.core import ORN


class LegionYoutube(ORN):
    namespace = "legion.claude-youtube"

    def __init__(self, channel: str, video_id: str):
        self.channel = channel
        self.video_id = video_id

    @property
    def reference(self) -> str:
        return f"{self.channel}/{self.video_id}"

    @classmethod
    def from_reference(cls, reference: str):
        if "/" not in reference:
            raise ValueError(
                f"LegionYoutube reference must be 'channel/video_id', "
                f"got: {reference!r}"
            )
        channel, video_id = reference.split("/", 1)
        return cls(channel=channel, video_id=video_id)
