import cv2
import torch
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI


# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----------------------------
# 1. Load YOLOv8 for detection
# ----------------------------
yolo = YOLO("yolov8n.pt")  # pretrained COCO weights

# ----------------------------
# 2. Load SAM2 for segmentation
# ----------------------------
checkpoint = "sam2_hiera_small.pth"   # path to your SAM2 weights
model_cfg  = "sam2_hiera_s.yaml"      # path to matching config
sam2_model = build_sam2(model_cfg, checkpoint)
predictor  = SAM2ImagePredictor(sam2_model)

# ----------------------------
# 3. Run on an image
# ----------------------------
image_path = "test.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run YOLO detection
results = yolo(image_rgb)[0]

# Feed image to SAM2
predictor.set_image(image_rgb)

for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
    x1, y1, x2, y2 = map(int, box.tolist())
    label = results.names[int(cls)]

    if label in ["person", "car"]:  # only process cars & people
        # Run SAM2 segmentation using YOLO’s box
        masks = predictor.predict(box=np.array([x1, y1, x2, y2]))

        # Convert boolean mask → overlay
        mask = masks[0]  # take first mask
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        image[mask] = cv2.addWeighted(image, 0.7, np.full_like(image, color), 0.3, 0)[mask]

        # Draw YOLO bounding box + label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ----------------------------
# 4. Show result
# ----------------------------
cv2.imshow("YOLO + SAM2", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
