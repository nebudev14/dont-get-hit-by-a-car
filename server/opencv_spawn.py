import argparse, sys, time, asyncio, math
import cv2, numpy as np
import requests
import websockets
from ultralytics import YOLO

# ----------------------------
# Detection Config
# ----------------------------
MODEL_DIR   = "yolov8n_openvino_model"
IMG_SIZE    = 640
CONF        = 0.5
IOU         = 0.5
WATCH       = {0,2,3,5,7}   # person, car, motorcycle, bus, truck

CENTER_BAND = 0.35
REL_CAUTION = 0.28
REL_DANGER  = 0.35
DIST_CAUTION_M = 10.0
DIST_DANGER_M  = 6.0

REAL_H = {0:1.70, 2:1.40, 3:1.30, 5:3.00, 7:3.00}

# ----------------------------
# Load YOLO model
# ----------------------------
print("[detection] Loading YOLO model...")
model = YOLO(MODEL_DIR)

# Track last heights per class
last_heights = {}
SHRINK_RATIO = 0.2   # if height shrinks >50% vs last frame, ignore box

def process_detection(frame):
    """Process frame with YOLO detection and return annotated frame"""
    H, W = frame.shape[:2]
    y0 = H // 3
    roi = frame[y0:H, :]
    
    # Run YOLO detection
    t0 = time.time()
    res = model.predict(roi, imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False)[0]
    
    alert = False
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls not in WATCH:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        y1 += y0; y2 += y0
        w, h = x2 - x1, y2 - y1

        # Ignore very small boxes (too far)
        if h < H * 0.05:
            continue

        # Rapid shrink check (passing)
        last_h = last_heights.get(cls, None)
        if last_h is not None and h < last_h * SHRINK_RATIO:
            continue

        # Update last seen height for this class
        last_heights[cls] = h

        cx   = (x1 + x2) * 0.5 / W
        rel_closeness = h / float(H)

        # Risk logic (simplified)
        if rel_closeness >= REL_DANGER and abs(cx-0.5) <= CENTER_BAND:
            risk, color = "DANGER", (0, 0, 255); alert = True
        elif rel_closeness >= REL_CAUTION and abs(cx-0.5) <= CENTER_BAND:
            risk, color = "CAUTION", (0, 165, 255)
        else:
            risk, color = "SAFE", (0, 200, 0)

        # Draw translucent overlay for the box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cls_name = model.names[cls] if hasattr(model, "names") else str(cls)
        cv2.putText(frame, f"{cls_name} {risk}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # FPS
    fps = 1.0 / max(time.time() - t0, 1e-6)
    cv2.putText(frame, f"{fps:0.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if alert:
        cv2.putText(frame, "APPROACH WARNING!", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    
    return frame

def try_videocap(url, title, enable_detection=True):
    print(f"[viewer] trying VideoCapture on {url}")
    if enable_detection:
        print("[detection] YOLO detection enabled")
    else:
        print("[detection] YOLO detection disabled")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[viewer] VideoCapture could not open stream.", file=sys.stderr)
        return False
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    last, shown = time.time(), 0
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            # if not ok:
            #     time.sleep(0.02); continue
            
            frame_count += 1
            # Process every 3rd frame for detection (like in original)
            if enable_detection:
                frame = process_detection(frame)
            
            cv2.imshow(title, frame); shown += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            if time.time() - last > 2:
                print(f"[viewer] frames shown (VideoCapture): {shown}")
                shown, last = 0, time.time()
    finally:
        cap.release(); 
        try: cv2.destroyWindow(title)
        except: pass
    return True

async def run_ws(url, title, enable_detection=True):
    print(f"[viewer] connecting WebSocket {url}")
    if enable_detection:
        print("[detection] YOLO detection enabled")
    else:
        print("[detection] YOLO detection disabled")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    shown, last = 0, time.time()
    frame_count = 0
    async with websockets.connect(url, max_size=None) as ws:
        while True:
            jpg = await ws.recv()  # bytes
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            frame_count += 1
            # Process every 3rd frame for detection (like in original)
            if enable_detection:
                img = process_detection(img)
            
            cv2.imshow(title, img); shown += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            if time.time() - last > 2:
                print(f"[viewer] frames shown (WebSocket): {shown}")
                shown, last = 0, time.time()
    try: cv2.destroyWindow(title)
    except: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--title", default="WebRTC Camera Stream")
    ap.add_argument("--no-detection", action="store_true", help="Disable YOLO detection")
    args = ap.parse_args()

    if args.url.startswith("ws://") or args.url.startswith("wss://"):
        asyncio.run(run_ws(args.url, args.title, enable_detection=not args.no_detection))
    else:
        ok = try_videocap(args.url, args.title, enable_detection=not args.no_detection)
        if not ok:
            print("[viewer] fallback to parser disabled on purpose; prefer WS for low latency.", file=sys.stderr)

if __name__ == "__main__":
    main()
