import cv2
from detection.vehicle_detection import detect_and_track

#SOURCE = "webcam"
SOURCE = "video"
#SOURCE = "phone"

cap = cv2.VideoCapture("videos/traffic.mp4")
if SOURCE == "webcam":
    cap = cv2.VideoCapture(0)

elif SOURCE == "video":
    cap = cv2.VideoCapture("videos/traffic.mp4")

elif SOURCE == "phone":
    cap = cv2.VideoCapture("http://10.245.75.220:8080/video")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    vehicles = detect_and_track(frame)

    for v in vehicles:

        x1, y1, x2, y2 = v["bbox"]
        label = v["label"]
        track_id = v["id"]

        text = f"{label} ID:{track_id}"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,text,(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("CivicEye Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()