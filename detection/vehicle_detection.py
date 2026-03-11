from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
model.to("cuda")

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle"]

def detect_and_track(frame):

    results = model.track(frame, persist=True, device=0)

    vehicles = []

    for r in results:
        boxes = r.boxes

        if boxes.id is None:
            continue

        for box, track_id in zip(boxes.xyxy, boxes.id):

            cls_id = int(boxes.cls[0])
            label = model.names[cls_id]

            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)

            vehicles.append({
                "id": int(track_id),
                "label": label,
                "bbox": (x1, y1, x2, y2)
            })

    return vehicles