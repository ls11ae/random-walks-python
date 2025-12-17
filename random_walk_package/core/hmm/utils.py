import json
from collections import Counter

import numpy as np


def to_json(density, num_directions=8, name="test"):
    print(len(density))
    data = {"name": name,
            "d": num_directions,
            "s": int((len(density[0]) - 1) / 2),
            "tm": [list(densit.flatten()) for densit in density]
            }
    with open("noncor.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)


def rotate_vector(a, b, c):
    # Define the vectors
    d = np.array(a) - np.array(b)
    e = np.array(c) - np.array(b)

    # Compute the angle to rotate
    # Angle between d and the horizontal axis
    angle_d = np.arctan2(d[1], d[0])

    theta = -angle_d

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotate vector e
    e_rotated = np.dot(rotation_matrix, e)
    return e_rotated


def angle_diff(a, b):
    d = b - a
    return (d + np.pi) % (2 * np.pi) - np.pi


def generate_angles(num_directions: int) -> list[float]:
    step = 360.0 / num_directions
    return [i * step for i in range(num_directions)]


def detect_typical_interval(entries):
    diffs = []
    for i in range(1, len(entries)):
        delta = ((entries[i][2] - entries[i - 1][2]).total_seconds() + 0.5) // 60
        diffs.append(delta)
    if len(diffs) == 0:
        return None
    # Häufigster Zeitabstand
    mode = Counter(diffs).most_common(1)[0][0]
    return mode


def calculate_durations(animal_trajectories):
    durations = []
    for bettong_id, entries in animal_trajectories.items():
        # Calculate time differences between consecutive entries
        for i in range(1, len(entries)):
            time_diff = entries[i][2] - entries[i - 1][2]
            durations.append(time_diff.total_seconds() // 60)  # Convert to minutes
    return durations
