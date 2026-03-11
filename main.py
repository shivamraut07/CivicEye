import cv2

from detection.vehicle_detection import detect_and_track
from vision.violation_engine import check_speed_violation, log_violation


# ==============================
# Input Source Configuration
# ==============================

SOURCES = {
    "webcam": 0,
    "video": "videos/traffic.mp4",
    "phone": "http://10.245.75.220:8080/video"
}

SOURCE = "video"


def get_video_source():

    source = SOURCES.get(SOURCE)

    if source is None:
        raise ValueError("Invalid SOURCE selected")

    return cv2.VideoCapture(source)


# ==============================
# Visualization
# ==============================

def draw_vehicle(frame, vehicle):

    x1, y1, x2, y2 = vehicle["bbox"]
    label = vehicle["label"]
    track_id = vehicle["id"]

    text = f"{label} ID:{track_id}"

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(
        frame,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )


# ==============================
# Main Pipeline
# ==============================

def main():

    cap = get_video_source()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        vehicles = detect_and_track(frame)

        for v in vehicles:

            # Draw bounding box
            draw_vehicle(frame, v)

            # Violation detection
            if check_speed_violation(v):
                log_violation(v, "speeding", frame)

        cv2.imshow("CivicEye Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()