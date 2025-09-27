import time, math, cv2
from ultralytics import YOLO

MODEL_DIR   = "yolov8n_openvino_model"
IMG_SIZE    = 416
CONF        = 0.30
IOU         = 0.5
WATCH       = {0,2,3,5,7}

CENTER_BAND = 0.35     
PIX_CLOSE   = 0.35     
GROWTH_MIN  = 1.10     
WIN         = 5

VFOV_DEG   = None
FY_PIXELS  = None

REAL_H = {
    0: 1.70,
    2: 1.40,
    3: 1.30,
    5: 3.00,
    7: 3.00,
}

REL_CAUTION = 0.28
REL_DANGER  = 0.35
DIST_CAUTION_M = 10.0
DIST_DANGER_M  = 6.0

model = YOLO(MODEL_DIR)
frame = cv2.imread("test.jpg")
if frame is None:
    raise FileNotFoundError("test.jpg not found or could not be loaded.")
H, W = frame.shape[:2]

def focal_from_vfov(vfov_deg: float, Hpix: int) -> float:
    return (Hpix * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)

def est_distance_m(h_pix: float, cls_id: int, Hpix: int) -> float | None:
    H_real = REAL_H.get(cls_id)
    if H_real is None or h_pix <= 1:
        return None
    fy = FY_PIXELS if FY_PIXELS else (focal_from_vfov(VFOV_DEG, Hpix) if VFOV_DEG else None)
    if fy is None:
        return None
    return (fy * H_real) / h_pix

y0  = H // 3
roi = frame[y0:H, :]

t0  = time.time()
res = model.predict(roi, imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False)[0]

alert = False
for b in res.boxes:
    cls = int(b.cls[0])
    if cls not in WATCH:
        continue

    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
    y1 += y0; y2 += y0
    w, h = x2 - x1, y2 - y1
    cx   = (x1 + x2) * 0.5 / W

    rel_closeness = h / float(H)
    dist_m        = est_distance_m(h, cls, H)
    centered      = abs(cx - 0.5) <= CENTER_BAND

    if dist_m is not None:
        if dist_m <= DIST_DANGER_M and centered:
            risk, color = "DANGER", (0, 0, 255)
            alert = True
        elif dist_m <= DIST_CAUTION_M and centered:
            risk, color = "CAUTION", (0, 165, 255)
        else:
            risk, color = "SAFE", (0, 200, 0)
        label_extra = f"{dist_m:0.1f}m"
    else:
        if rel_closeness >= REL_DANGER and centered:
            risk, color = "DANGER", (0, 0, 255)
            alert = True
        elif rel_closeness >= REL_CAUTION and centered:
            risk, color = "CAUTION", (0, 165, 255)
        else:
            risk, color = "SAFE", (0, 200, 0)
        label_extra = f"{100*rel_closeness:0.0f}% H"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cls_name = model.names[cls] if hasattr(model, "names") else str(cls)
    cv2.putText(frame, f"{cls_name} {risk} {label_extra}",
                (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

fps = 1.0 / max(time.time() - t0, 1e-6)
cv2.putText(frame, f"{fps:0.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

if alert:
    cv2.putText(frame, "APPROACH WARNING!", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

cv2.imshow("rear-guard", frame)
cv2.waitKey(10000)
cv2.destroyAllWindows()
