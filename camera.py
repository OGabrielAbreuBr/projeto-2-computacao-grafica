import math

import numpy as np

from config import SCENE_LIMIT
from matrices import look_at, normalize


class Camera:
    def __init__(self) -> None:
        self.position = np.array([0.0, 3.5, 28.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = -5.0

        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.update_vectors()

    def update_vectors(self) -> None:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        front = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.sin(yaw) * math.cos(pitch),
        ], dtype=np.float32)

        self.front = normalize(front)
        self.right = normalize(np.cross(
            self.front,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ))
        self.up = normalize(np.cross(self.right, self.front))

    def get_view(self) -> np.ndarray:
        return look_at(self.position, self.position + self.front, self.up)

    def move(self, direction: str, amount: float) -> None:
        flat_front = normalize(np.array([
            self.front[0],
            0.0,
            self.front[2],
        ], dtype=np.float32))

        flat_right = normalize(np.array([
            self.right[0],
            0.0,
            self.right[2],
        ], dtype=np.float32))

        if direction == "forward":
            self.position += flat_front * amount
        elif direction == "backward":
            self.position -= flat_front * amount
        elif direction == "left":
            self.position -= flat_right * amount
        elif direction == "right":
            self.position += flat_right * amount

        self.position[0] = np.clip(self.position[0], -SCENE_LIMIT, SCENE_LIMIT)
        self.position[2] = np.clip(self.position[2], -SCENE_LIMIT, SCENE_LIMIT)
        self.position[1] = np.clip(self.position[1], 1.0, 20.0)