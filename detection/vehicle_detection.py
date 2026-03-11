from ultralytics import YOLO

# load YOLO model
model = YOLO("yolov8n.pt")

# vehicle classes we care about
VEHICLE_CLASSES = ["car", "bicycle", "motorcycle", "bus", "truck"]

def detect_vehicles(frame):

    results = model(frame)

    vehicles = []

    for result in results:
        boxes = result.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in VEHICLE_CLASSES:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                vehicle_data = {
                    "label": label,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0])
                }

                vehicles.append(vehicle_data)

    return vehicles