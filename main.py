import cv2
from detection.vehicle_detection import detect_vehicles

# open laptop webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    vehicles = detect_vehicles(frame)

    for vehicle in vehicles:

        x1, y1, x2, y2 = vehicle["bbox"]
        label = vehicle["label"]
        conf = vehicle["confidence"]

        text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("CivicEye - Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()