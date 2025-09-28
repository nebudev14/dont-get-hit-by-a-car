#!/usr/bin/env python3
import argparse, sys, time, asyncio, math, json
import cv2, numpy as np
import websockets
from ultralytics import YOLO

# ----------------------------
# Shared state & signaling
# ----------------------------
changed = asyncio.Event()
latest = {"risk": "SAFE", "direction": "CENTER"}  # updated by detection, read by sender()

# ----------------------------
# Detection Config
# ----------------------------
MODEL_DIR   = "yolov8n_openvino_model"
IMG_SIZE    = 640
CONF        = 0.5
IOU         = 0.5
WATCH       = {0, 2, 3, 5, 7}   # person, car, motorcycle, bus, truck

CENTER_BAND = 0.35              # within 35% of image center laterally
REL_CAUTION = 0.28
REL_DANGER  = 0.35
DIST_CAUTION_M = 10.0           # (not used in this heuristic but kept for future)
DIST_DANGER_M  = 6.0
REAL_H = {0:1.70, 2:1.40, 3:1.30, 5:3.00, 7:3.00}

# If a class's box height shrinks a lot, treat as passing object and ignore
SHRINK_RATIO = 0.5              # ignore if h < last_h * 0.5 ( >50% shrink)
last_heights = {}               # class_id -> last box height

# ----------------------------
# Load YOLO model
# ----------------------------
print("[detection] Loading YOLO model...")
model = YOLO(MODEL_DIR)

RISK_LEVEL = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
RISK_COLOR = {
    "SAFE":    (0, 200, 0),
    "CAUTION": (0, 165, 255),
    "DANGER":  (0, 0, 255),
}

def _direction_from_cx(cx_norm: float) -> str:
    if cx_norm < 2.0/5.0:
        return "LEFT"
    elif cx_norm < 3.0/5.0:
        return "CENTER"
    else:
        return "RIGHT"

def _maybe_update_latest(risk: str, direction: str) -> None:
    """Update global latest & set event if changed."""
    global latest
    if latest.get("risk") != risk or latest.get("direction") != direction:
        latest["risk"] = risk
        latest["direction"] = direction
        changed.set()

def process_detection(frame):
    """
    Process frame with YOLO detection, draw overlays, and update (latest, changed)
    with the WORST (max) risk seen in this frame and its direction.
    """
    H, W = frame.shape[:2]
    y0 = H // 3                     # focus on lower 2/3
    roi = frame[y0:H, :]

    t0 = time.time()
    res = model.predict(roi, imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False)[0]

    # Track worst risk in this frame
    worst_level = RISK_LEVEL["SAFE"]
    worst_risk = "SAFE"
    worst_direction = "CENTER"
    alert = False

    for b in res.boxes:
        cls = int(b.cls[0])
        if cls not in WATCH:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        y1 += y0; y2 += y0
        w, h = x2 - x1, y2 - y1

        if h <= 0 or w <= 0:
            continue

        # Ignore very small boxes (too far)
        if h < H * 0.05:
            continue

        # Shrink check (passing)
        last_h = last_heights.get(cls)
        if last_h is not None and h < last_h * SHRINK_RATIO:
            continue
        last_heights[cls] = h

        # Normalized center X and closeness proxy
        cx = (x1 + x2) * 0.5 / float(W)
        rel_closeness = h / float(H)
        direction = _direction_from_cx(cx)

        # Risk logic
        if rel_closeness >= REL_DANGER and abs(cx - 0.5) <= CENTER_BAND:
            risk = "DANGER"; color = RISK_COLOR[risk]; alert = True
        elif rel_closeness >= REL_CAUTION and abs(cx - 0.5) <= CENTER_BAND:
            risk = "CAUTION"; color = RISK_COLOR[risk]
        else:
            risk = "SAFE";    color = RISK_COLOR[risk]

        # Keep worst-of-frame
        lvl = RISK_LEVEL[risk]
        if lvl > worst_level:
            worst_level = lvl
            worst_risk = risk
            worst_direction = direction

        # Draw translucent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Draw bbox + label
        cls_name = model.names[cls] if hasattr(model, "names") else str(cls)
        label = f"{cls_name} {risk} {direction}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # FPS
    fps = 1.0 / max(time.time() - t0, 1e-6)
    cv2.putText(frame, f"{fps:0.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if alert:
        cv2.putText(frame, "APPROACH WARNING!", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    _maybe_update_latest(worst_risk, worst_direction)

    return frame

def try_videocap(url, title, enable_detection=True):
    print(f"[viewer] trying VideoCapture on {url}")
    print(f"[detection] YOLO detection {'enabled' if enable_detection else 'disabled'}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[viewer] VideoCapture could not open stream.", file=sys.stderr)
        return False

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    last, shown = time.time(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            if enable_detection:
                frame = process_detection(frame)

            cv2.imshow(title, frame); shown += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            if time.time() - last > 2:
                print(f"[viewer] frames shown (VideoCapture): {shown}")
                shown, last = 0, time.time()
    finally:
        cap.release()
        try:
            cv2.destroyWindow(title)
        except Exception:
            pass
    return True

async def run_ws_camera(url, title, enable_detection=True):
    """
    Receive JPEG frames over a WebSocket (camera publisher),
    display them, and run detection.
    """
    print(f"[viewer] connecting camera WebSocket {url}")
    print(f"[detection] YOLO detection {'enabled' if enable_detection else 'disabled'}")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    shown, last = 0, time.time()

    async with websockets.connect(url, max_size=None) as ws:
        while True:
            jpg = await ws.recv()  # bytes
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            if enable_detection:
                img = process_detection(img)

            cv2.imshow(title, img); shown += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            if time.time() - last > 2:
                print(f"[viewer] frames shown (WebSocket): {shown}")
                shown, last = 0, time.time()

    try:
        cv2.destroyWindow(title)
    except Exception:
        pass

async def sender(out_ws_url: str):
    """
    Connects to the Express WebSocket server (same port as HTTP, no path),
    and pushes {"ts","risk","direction"} whenever detection updates.
    """
    backoff = 0.5
    prev = (None, None)
    MIN_INTERVAL = 0.15
    last_sent_ts = 0.0

    while True:
        try:
            print(f"[sender] connecting to {out_ws_url} ...")
            async with websockets.connect(out_ws_url) as ws:
                print("[sender] connected")
                backoff = 0.5

                while True:
                    await changed.wait()
                    changed.clear()

                    risk = latest.get("risk", "SAFE")
                    direction = latest.get("direction", "CENTER")
                    now = time.time()

                    if (risk, direction) == prev and (now - last_sent_ts) < MIN_INTERVAL:
                        continue

                    payload = {"ts": now, "risk": risk, "direction": direction}
                    await ws.send(json.dumps(payload))
                    print("[sender] sent:", payload)

                    prev = (risk, direction)
                    last_sent_ts = now

        except (ConnectionRefusedError, OSError) as e:
            print(f"[sender] WS connect/send failed: {e}. Retrying in {backoff:.2f}s ...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
        except websockets.ConnectionClosed:
            print("[sender] connection closed by server. Reconnecting...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)

async def orchestrate_ws_camera(cam_ws_url: str, title: str, enable_detection: bool, out_ws_url: str):
    viewer_task = asyncio.create_task(run_ws_camera(cam_ws_url, title, enable_detection))
    tx_task = asyncio.create_task(sender(out_ws_url))
    done, pending = await asyncio.wait({viewer_task, tx_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()

async def orchestrate_videocap(url: str, title: str, enable_detection: bool, out_ws_url: str):
    viewer_task = asyncio.create_task(asyncio.to_thread(try_videocap, url, title, enable_detection))
    tx_task = asyncio.create_task(sender(out_ws_url))
    done, pending = await asyncio.wait({viewer_task, tx_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Camera URL. If ws(s)://, treated as WS JPEG stream; otherwise cv2.VideoCapture.")
    ap.add_argument("--title", default="Camera Stream")
    ap.add_argument("--no-detection", action="store_true", help="Disable YOLO detection")
    ap.add_argument("--out-ws", default="ws://127.0.0.1:3000",
                    help="WebSocket endpoint of the Express server to send risk/direction (default: ws://127.0.0.1:3000)")
    args = ap.parse_args()

    enable_detection = not args.no_detection
    url = args.url

    if url.startswith("ws://") or url.startswith("wss://"):
        asyncio.run(orchestrate_ws_camera(url, args.title, enable_detection, args.out_ws))
    else:
        ok = True
        try:
            asyncio.run(orchestrate_videocap(url, args.title, enable_detection, args.out_ws))
        except Exception as e:
            ok = False
            print(f"[viewer] error: {e}", file=sys.stderr)
        if not ok:
            print("[viewer] fallback to parser disabled on purpose; prefer WS for low latency.", file=sys.stderr)

if __name__ == "__main__":
    main()
