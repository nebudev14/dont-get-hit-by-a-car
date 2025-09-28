import time, math, cv2
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------
MODEL_DIR   = "yolov8n_openvino_model"
IMG_SIZE    = 640
CONF        = 0.5
IOU         = 0.5
WATCH       = {0,2,3,5,7}   # person, car, motorcycle, bus, truck

CENTER_BAND = 0.35
REL_CAUTION = 0.10
REL_DANGER  = 0.20  # acceptable

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO(MODEL_DIR)

# Track last heights per class
last_heights = {}
SHRINK_RATIO = 0.2   # if height shrinks >50% vs last frame, ignore box

cap = cv2.VideoCapture("rearen2.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:   # only process every 3rd frame
        continue

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

        cx = (x1 + x2) * 0.5 / W
        rel_closeness = h / float(H)

        # ----------------------------
        # Direction (3 vertical zones)
        # ----------------------------
        if cx < 2/5:
            direction = "LEFT"
        elif cx < 3/5:
            direction = "CENTER"
        else:
            direction = "RIGHT"

        # Risk logic
        if rel_closeness >= REL_DANGER and abs(cx - 0.5) <= CENTER_BAND:
            risk, color = "DANGER", (0, 0, 255); alert = True
        elif rel_closeness >= REL_CAUTION and abs(cx - 0.5) <= CENTER_BAND:
            risk, color = "CAUTION", (0, 165, 255)
        else:
            risk, color = "SAFE", (0, 200, 0)

        # Draw translucent overlay for the box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Draw bounding box and label (with direction)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cls_name = model.names[cls] if hasattr(model, "names") else str(cls)
        cv2.putText(frame, f"{cls_name} {risk} {direction}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # FPS
    fps = 1.0 / max(time.time() - t0, 1e-6)
    cv2.putText(frame, f"{fps:0.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if alert:
        cv2.putText(frame, "APPROACH WARNING!", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    scale = 0.5
    disp = cv2.resize(frame, (int(W * scale), int(H * scale)))
    cv2.imshow("rear-guard", disp)

    # Press q to exit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
