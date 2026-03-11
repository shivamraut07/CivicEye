from ultralytics import YOLO

# load model
model = YOLO("models/yolov8n.pt")

VEHICLE_CLASSES = ["car", "bicycle", "motorcycle", "bus", "truck"]
CONFIDENCE_THRESHOLD = 0.5


def detect_vehicles(frame):

    results = model(frame, device="cpu")

    vehicles = []

    for result in results:
        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in VEHICLE_CLASSES:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                vehicles.append({
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

    return vehicles