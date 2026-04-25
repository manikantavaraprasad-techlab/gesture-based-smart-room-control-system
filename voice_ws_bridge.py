# voice_ws_bridge.py
# WebSocket voice bridge — sends actions to browser visualizer
# Now supports dynamic honorific (sir/madam) from face gender detection

import json
import time
import queue
import threading
import asyncio

try:
    import websockets
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

WS_PORT = 9600


class WebVoiceAssistant:
    """
    Sends voice actions to the HTML visualizer via WebSocket.
    The browser handles TTS locally for zero latency and perfect spectrum sync.

    Now supports dynamic honorific based on detected gender.
    """

    def __init__(self, enabled=True, debug=False, port=WS_PORT):
        self.enabled = enabled
        self.debug = debug
        self.port = port
        self.q: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.clients: set = set()
        self._lock = threading.Lock()

        self.last_spoken_text = ""
        self.last_spoken_time = 0.0

        # Dynamic honorific — updated by face gender detector
        self._honorific = "sir"
        self._honorific_lock = threading.Lock()

        if self.enabled:
            self._thread = threading.Thread(target=self._run_server, daemon=True)
            self._thread.start()
            time.sleep(0.3)

    @property
    def honorific(self) -> str:
        with self._honorific_lock:
            return self._honorific

    @honorific.setter
    def honorific(self, value: str):
        with self._honorific_lock:
            self._honorific = value.lower().strip()

    def _run_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _handler(self, ws):
        """Handle a single WebSocket client connection."""
        with self._lock:
            self.clients.add(ws)
        if self.debug:
            print(f"[WS] Client connected ({len(self.clients)} total)")
        try:
            async for _ in ws:
                pass
        finally:
            with self._lock:
                self.clients.discard(ws)
            if self.debug:
                print(f"[WS] Client disconnected ({len(self.clients)} total)")

    async def _serve(self):
        server = await websockets.serve(self._handler, "0.0.0.0", self.port)
        if self.debug:
            print(f"[WS] Voice server running on ws://localhost:{self.port}")

        while not self.stop_flag.is_set():
            try:
                msg = self.q.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue

            if msg is None:
                break

            # Snapshot clients to avoid iteration-mutation race
            with self._lock:
                clients_snapshot = set(self.clients)

            if clients_snapshot:
                payload = json.dumps(msg)
                dead = set()
                for client in clients_snapshot:
                    try:
                        await client.send(payload)
                    except Exception:
                        dead.add(client)

                if dead:
                    with self._lock:
                        self.clients -= dead

        server.close()
        await server.wait_closed()

    def clear_pending(self):
        """Drain all pending messages from the queue."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def say(self, text: str, clear_old=False, action_text=None):
        """
        Send a voice message to the browser visualizer.

        Args:
            text: Speech text (will have honorific appended)
            clear_old: If True, drain queue before adding
            action_text: Display text for the HUD (defaults to text)
        """
        if not self.enabled or not text:
            return

        t = time.time()
        if text == self.last_spoken_text and (t - self.last_spoken_time) < 1.5:
            return

        if clear_old:
            self.clear_pending()
        else:
            # Cap queue depth
            while self.q.qsize() >= 3:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

        self.last_spoken_text = text
        self.last_spoken_time = t

        msg = {
            "action": action_text or text,
            "speech": text,
            "honorific": self.honorific,
            "timestamp": t,
        }

        if self.debug:
            print(f"[VOICE->WS] {text} (honorific={self.honorific})")

        self.q.put(msg)

    def shutdown(self):
        """Cleanly shut down the WebSocket server."""
        self.stop_flag.set()
        self.q.put(None)
        if hasattr(self, '_thread'):
            self._thread.join(timeout=3.0)
