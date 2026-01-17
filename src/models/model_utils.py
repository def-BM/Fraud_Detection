import numpy as np

def build_behavioral_features(
        amount: float,
        txn_hour: int,
        velocity: int,
        location_risk: str,
        device_type: str
):
    """
    Converts raw user inputs into normalized behavioral signals
    """

    location_map = {"Low": 0.2, "Medium": 0.5, "High": 0.9}
    device_map = {"Known Device": 0.1, "New Device": 0.8}

    features = {
        "amount": amount,
        "txn_hour": txn_hour,
        "velocity": velocity,
        "location_risk": location_map.get(location_risk, 0.5),
        "device_type": device_map.get(device_type, 0.5)
    }

    return features