import asyncio
import time
import numpy as np
import cv2
from aiortc import VideoStreamTrack
from av import VideoFrame

class VideoTrack(VideoStreamTrack):
    """Video track for WebRTC, receives frames via update_frame()."""

    def __init__(self, name="rgb"):
        super().__init__()
        self._latest_frame = None
        self.name = name

    def update_frame(self, frame: np.ndarray):
        self._latest_frame = frame

    async def recv(self):
        # Wait until next frame timestamp
        pts, time_base = await self.next_timestamp()

        if self._latest_frame is None:
            # If no frame is available, return a black frame
            self._latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        video_frame = VideoFrame.from_ndarray(self._latest_frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
