# server/maybe.py  ← minimal correct header

from fastapi import FastAPI

app = FastAPI()                              # define app FIRST



# --- your imports/routes below ---
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from fastapi.responses import HTMLResponse

class OfferPayload(BaseModel):
    sdp: str
    type: str

@app.post("/camera/stream")
async def camera_stream(body: OfferPayload):
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return
        import asyncio
        async def reader():
            while True:
                frame = await track.recv()
                _ = frame.to_ndarray(format="bgr24")  # handle the frame here
        asyncio.create_task(reader())

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.get("/", response_class=HTMLResponse)
def sender_page():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<body style="font-family:system-ui;margin:0;padding:1rem">
  <video id="v" autoplay playsinline muted style="width:100%;max-width:480px;border:1px solid #ccc;border-radius:12px"></video>
  <div style="margin-top:12px;display:flex;gap:8px">
    <button id="start">Start</button>
    <button id="stop">Stop</button>
  </div>
  <pre id="log" style="font-size:12px;white-space:pre-wrap"></pre>
  <script>
    let pc, stream;
    const log = (...a) => (document.getElementById('log').textContent += a.join(' ') + "\\n");

    async function start() {
      pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]}); // add TURN for internet later
      pc.oniceconnectionstatechange = () => log("ICE:", pc.iceConnectionState);

      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width:{ ideal:1280 }, height:{ ideal:720 }, frameRate:{ ideal:15 } },
        audio: false
      });
      document.getElementById('v').srcObject = stream;
      stream.getTracks().forEach(t => pc.addTrack(t, stream));

      const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
      const res = await fetch("/camera/stream", {
        method:"POST", headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
      });
      const ans = await res.json();
      await pc.setRemoteDescription(new RTCSessionDescription(ans));
      log("Started.");
    }

    function stop() {
      if (pc) { pc.close(); pc = null; }
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      log("Stopped.");
    }

    document.getElementById('start').onclick = start;
    document.getElementById('stop').onclick = stop;
  </script>
</body>
"""