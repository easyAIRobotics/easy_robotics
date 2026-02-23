import asyncio
import json
from typing import Set
import websockets

class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=9001):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handler(self, ws):
        self.clients.add(ws)
        try:
            async for msg in ws:
                pass  # inbound handled by modules later
        finally:
            self.clients.remove(ws)

    async def start(self):
        return await websockets.serve(self.handler, self.host, self.port)

    async def broadcast(self, data: dict):
        if not self.clients:
            return
        msg = json.dumps(data)
        await asyncio.gather(*(c.send(msg) for c in self.clients))
