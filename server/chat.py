from fastapi import FastAPI, Request, Response, status
from pydantic import BaseModel
from typing import Optional
import asyncio, threading, queue

import numpy as np
import cv2

from aiortc import (
    RTCPeerConnection, RTCSessionDescription,
    RTCConfiguration, RTCIceServer
)

app = FastAPI()

# -----------------------------
# 0) OPTIONAL: device gate (your original middleware)
# -----------------------------
@app.middleware("http")
async def checkphone(request: Request, call_next):
    platform = request.headers.get("sec-ch-ua-platform")   # e.g., "Android"
    model    = request.headers.get("sec-ch-ua-model")      # e.g., "Pixel 8"

    is_pixel = (
        platform and "android" in platform.lower()
        and model and "pixel" in model.lower()
    )

    if not is_pixel:
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    request.state.is_pixel = True
    return await call_next(request)

# -----------------------------
# 1) Inference worker plumbing
# -----------------------------
FrameQ: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=5)

def segmentation_worker():
    """
    Load your model ONCE here, then process frames forever.
    Replace the pseudo-code with your model.
    """
    # Example:
    # import torch
    # model = torch.jit.load("seg_model.ts").eval()

    while True:
        img = FrameQ.get()           # np.ndarray, shape (H, W, 3), BGR8
        if img is None:
            break  # allow graceful shutdown if you ever enqueue None

        # --- TODO: run your segmentation here ---
        # inp = cv2.resize(img, (512, 512))
        # tens = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).float()/255.0
        # with torch.no_grad():
        #     mask = model(tens)
        # ----------------------------------------

        # For now, show we received frames:
        # print("Frame:", img.shape)

# Start worker thread
threading.Thread(target=segmentation_worker, daemon=True).start()

# -----------------------------
# 2) WebRTC signaling payload
# -----------------------------
class OfferPayload(BaseModel):
    sdp: str
    type: str  # "offer"
    # Optional: if you want to pass a token, room id, etc.
    token: Optional[str] = None

# -----------------------------
# 3) WebRTC ICE / TURN config
# -----------------------------
TURN_DOMAIN = "turn.yourdomain.com"   # <-- change me if using TURN
TURN_USER   = "webrtcuser"            # <-- change me
TURN_PASS   = "a-very-strong-password"  # <-- change me

ICE_CONFIG = RTCConfiguration(iceServers=[
    # You can test locally without TURN; keep at least a STUN or empty list
    RTCIceServer(urls=f"stun:{TURN_DOMAIN}:3478"),
    RTCIceServer(urls=f"turn:{TURN_DOMAIN}:3478", username=TURN_USER, credential=TURN_PASS),
])

# -----------------------------
# 4) WebRTC receiver helper
# -----------------------------
def handle_video_track(pc: RTCPeerConnection, track):
    if track.kind != "video":
        return

    async def reader():
        while True:
            frame = await track.recv()                     # aiortc VideoFrame
            img = frame.to_ndarray(format="bgr24")         # HxWx3, uint8
            try:
                FrameQ.put_nowait(img)
            except queue.Full:
                # Drop frames if back-pressured; keeps latency low
                pass

    # Schedule reader loop without blocking
    asyncio.create_task(reader())

# -----------------------------
# 5) Signaling endpoint (POST /camera/stream)
# -----------------------------
@app.post("/camera/stream", status_code=status.HTTP_200_OK)
async def camera_stream(body: OfferPayload, request: Request):
    """
    Receives SDP 'offer' from the phone, answers with SDP 'answer'.
    After ICE connects, @pc.on('track') fires and we start getting frames.
    """
    # (Optional) gate by token here if you want:
    # if body.token != "expected":
    #     return Response(status_code=status.HTTP_401_UNAUTHORIZED)

    pc = RTCPeerConnection(configuration=ICE_CONFIG)

    @pc.on("track")
    def _on_track(track):
        handle_video_track(pc, track)

    # 1) Apply the remote description (the browser's offer)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))

    # 2) Create our answer and set as local description
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # 3) Return SDP answer to browser
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
