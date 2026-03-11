violations = []

def check_speed_violation(vehicle):

    # placeholder logic
    if vehicle["label"] == "car":
        return False

    return False


def log_violation(vehicle, violation_type):

    event = {
        "vehicle_id": vehicle["id"],
        "type": violation_type
    }

    violations.append(event)

    print("Violation detected:", event)