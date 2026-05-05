import math
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)

    if n < 1e-8:
        return v

    return v / n


def perspective(fovy_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_degrees) / 2.0)

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0

    return m


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)

    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f

    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)

    return m


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale_matrix(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotate_x(angle_degrees: float) -> np.ndarray:
    a = math.radians(angle_degrees)
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def rotate_y(angle_degrees: float) -> np.ndarray:
    a = math.radians(angle_degrees)
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def rotate_z(angle_degrees: float) -> np.ndarray:
    a = math.radians(angle_degrees)
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def model_matrix(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> np.ndarray:
    return (
        translate(position[0], position[1], position[2])
        @ rotate_y(rotation[1])
        @ rotate_x(rotation[0])
        @ rotate_z(rotation[2])
        @ scale_matrix(scale[0], scale[1], scale[2])
    )