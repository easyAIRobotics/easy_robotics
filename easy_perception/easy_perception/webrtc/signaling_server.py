import json
import numpy as np
from urllib.parse import urlparse, parse_qs
from aiortc import RTCPeerConnection, RTCSessionDescription
import websockets

from easy_perception.webrtc.webrtc_streamer import VideoTrack

class WebRTCSignalingServer:
    """
    Handles multiple camera tracks:
    """
    def __init__(self, tracks: list[str]):
        self.track_names = tracks
        self.tracks = {}

    def update_frame(self, camera: str, frame: np.ndarray):
        if camera in self.tracks:
            self.tracks[camera].update_frame(frame)

    async def handler(self, websocket):
        try:
            # Parse camera query
            path = websocket.request.path
            query = urlparse(path).query
            params = parse_qs(query)
            camera = params.get("camera", [None])[0]

            if camera not in self.track_names:
                print(f"Unknown camera requested: {camera}", flush=True)
                await websocket.send(json.dumps({"error": f"Unknown camera '{camera}'"}))
                return

            pc = RTCPeerConnection()
            self.tracks[camera] = VideoTrack(camera)
            # Add the correct track
            pc.addTrack(self.tracks[camera])

            @pc.on("icecandidate")
            async def on_icecandidate(candidate):
                if candidate:
                    await websocket.send(json.dumps({"type": "ice", "candidate": candidate.toJSON()}))

            async for message in websocket:
                data = json.loads(message)

                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await websocket.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))

                elif data["type"] == "ice" and "candidate" in data:
                    await pc.addIceCandidate(data["candidate"])
                    
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket disconnected: {e}")
        except Exception as e:
            print(f"Handler error: {e}")
