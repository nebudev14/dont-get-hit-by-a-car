# server/webapp.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
import asyncio, cv2, time, os

app = FastAPI()
PCS = set()

# Tunables (env overrides): smaller frames + lower JPEG quality = less latency over the internet
OUT_MAX_W = int(os.environ.get("OUT_MAX_W", "640"))
OUT_MAX_H = int(os.environ.get("OUT_MAX_H", "360"))
JPG_QUALITY = int(os.environ.get("JPG_QUALITY", "70"))

class FrameHub:
    def __init__(self):
        self.subscribers: set[asyncio.Queue[bytes]] = set()
        self.lock = asyncio.Lock()
        self.latest_jpg: bytes | None = None
        self.frames_total = 0
        self.last_pub_ts = 0.0

    async def subscribe(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
        async with self.lock:
            self.subscribers.add(q)
            if self.latest_jpg:
                try: q.put_nowait(self.latest_jpg)
                except asyncio.QueueFull: pass
        return q

    async def unsubscribe(self, q: asyncio.Queue[bytes]):
        async with self.lock:
            self.subscribers.discard(q)

    async def publish(self, jpg: bytes):
        self.latest_jpg = jpg
        self.frames_total += 1
        self.last_pub_ts = time.time()
        async with self.lock:
            for q in list(self.subscribers):
                try: q.put_nowait(jpg)
                except asyncio.QueueFull: pass

HUB = FrameHub()

class OfferPayload(BaseModel):
    sdp: str
    type: str

def _resize_keep_ar(img, maxw, maxh):
    h, w = img.shape[:2]
    scale = min(maxw / w, maxh / h)
    if scale >= 1.0:  # already small
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

@app.post("/camera/stream")
async def camera_stream(body: OfferPayload):
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    PCS.add(pc)

    @pc.on("iceconnectionstatechange")
    def on_ice():
        print("[ICE]", pc.iceConnectionState)

    @pc.on("track")
    def on_track(track):
        print("[RTC] on_track:", track.kind)
        if track.kind != "video":
            return

        async def reader():
            frames = 0
            try:
                while True:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    img = _resize_keep_ar(img, OUT_MAX_W, OUT_MAX_H)
                    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
                    if ok:
                        await HUB.publish(enc.tobytes())
                    frames += 1
                    if frames % 30 == 0:
                        print(f"[RTC] frames received: {frames} (broadcast total: {HUB.frames_total})")
            except Exception as e:
                print("[RTC] reader ended:", repr(e))

        asyncio.create_task(reader())

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print("[RTC] Answer created.")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

BOUNDARY = "frameboundary"

@app.get("/mjpeg")
async def mjpeg():
    q = await HUB.subscribe()
    print("[MJPEG] client connected")
    async def gen():
        yield f"--{BOUNDARY}\r\n".encode("ascii")
        try:
            while True:
                try:
                    jpg = await asyncio.wait_for(q.get(), timeout=0.8)
                except asyncio.TimeoutError:
                    jpg = HUB.latest_jpg
                    if jpg is None:
                        yield b""
                        continue
                part = (
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii") + jpg + b"\r\n" + f"--{BOUNDARY}\r\n".encode("ascii")
                yield part
        finally:
            print("[MJPEG] client disconnected")
            await HUB.unsubscribe(q)
    headers = {
        "Age": "0",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), headers=headers)

# NEW: low-latency WebSocket transport sending raw JPEG frames
@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    q = await HUB.subscribe()
    print("[WS] client connected")
    try:
        while True:
            jpg = await q.get()
            # send as binary message
            await ws.send_bytes(jpg)
    except WebSocketDisconnect:
        pass
    finally:
        await HUB.unsubscribe(q)
        print("[WS] client disconnected")

@app.get("/latest.jpg")
async def latest_jpg():
    if HUB.latest_jpg is None:
        return Response(content=b"", media_type="image/jpeg", status_code=404)
    return Response(content=HUB.latest_jpg, media_type="image/jpeg")

@app.get("/stats")
async def stats():
    return {
        "frames_total": HUB.frames_total,
        "has_latest": HUB.latest_jpg is not None,
        "last_pub_age_s": (time.time() - HUB.last_pub_ts) if HUB.last_pub_ts else None,
        "subscribers": len(HUB.subscribers),
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<body style="font-family:system-ui;margin:0;padding:1rem">
  <h3>Phone Camera → Server</h3>
  <video id="v" autoplay playsinline muted style="width:100%;max-width:480px;border:1px solid #ccc;border-radius:12px"></video>
  <div style="margin-top:12px;display:flex;gap:8px;margin-top:12px">
    <button id="start">Start</button>
    <button id="stop">Stop</button>
  </div>
  <p>Viewers: use <code>/ws</code> (WebSocket) for lowest latency, or <code>/mjpeg</code>. Stats at <code>/stats</code>.</p>
  <pre id="log" style="font-size:12px;white-space:pre-wrap"></pre>
  <script>
    let pc, stream;
    const log = (...a) => (document.getElementById('log').textContent += a.join(' ') + "\\n");
    async function waitIceGathering(pc) {
      if (pc.iceGatheringState === "complete") return;
      await new Promise(resolve => {
        function check(){ if (pc.iceGatheringState === "complete"){ pc.removeEventListener("icegatheringstatechange", check); resolve(); } }
        pc.addEventListener("icegatheringstatechange", check);
      });
    }
    async function start() {
      pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
      pc.oniceconnectionstatechange = () => log("ICE:", pc.iceConnectionState);
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:15} },
          audio: false
        });
      } catch (e) { log("getUserMedia failed:", e); return; }
      document.getElementById('v').srcObject = stream;
      stream.getTracks().forEach(t => pc.addTrack(t, stream));
      const offer = await pc.createOffer(); await pc.setLocalDescription(offer); await waitIceGathering(pc);
      const res = await fetch("/camera/stream", { method:"POST", headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }) });
      const ans = await res.json(); await pc.setRemoteDescription(new RTCSessionDescription(ans));
      log("Started. Use /ws or /mjpeg from viewers.");
    }
    function stop(){ if(pc){pc.close(); pc=null;} if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;} log("Stopped."); }
    document.getElementById('start').onclick = start; document.getElementById('stop').onclick = stop;
  </script>
</body>
"""
