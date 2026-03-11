import cv2
import time
import os

violations = []

EVIDENCE_FOLDER = "evidence"

os.makedirs(EVIDENCE_FOLDER, exist_ok=True)


def check_speed_violation(vehicle):
    # placeholder logic for now
    return False


def log_violation(vehicle, violation_type, frame):

    timestamp = int(time.time())

    filename = f"{violation_type}_vehicle{vehicle['id']}_{timestamp}.jpg"

    filepath = os.path.join(EVIDENCE_FOLDER, filename)

    cv2.imwrite(filepath, frame)

    event = {
        "vehicle_id": vehicle["id"],
        "type": violation_type,
        "image": filepath
    }

    violations.append(event)

    print("Violation saved:", event)