import glfw
import numpy as np

from config import CAMERA_SPEED, MOUSE_SENSITIVITY
from camera import Camera
from scene import SceneControls


class InputState:
    def __init__(self) -> None:
        self.first_mouse = True
        self.last_x = 0.0
        self.last_y = 0.0
        self.wireframe = False


def mouse_callback(window, xpos: float, ypos: float, camera: Camera, state: InputState) -> None:
    if state.first_mouse:
        state.last_x = xpos
        state.last_y = ypos
        state.first_mouse = False

    x_offset = xpos - state.last_x
    y_offset = state.last_y - ypos

    state.last_x = xpos
    state.last_y = ypos

    camera.yaw += x_offset * MOUSE_SENSITIVITY
    camera.pitch += y_offset * MOUSE_SENSITIVITY

    camera.pitch = float(np.clip(camera.pitch, -89.0, 89.0))
    camera.update_vectors()


def process_input(
    window,
    delta: float,
    camera: Camera,
    scene_controls: SceneControls,
) -> None:
    amount = CAMERA_SPEED * delta

    # -------------------------------------------------------------------------
    # Movimento da câmera
    # -------------------------------------------------------------------------
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera.move("forward", amount)

    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera.move("backward", amount)

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera.move("left", amount)

    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera.move("right", amount)

    # -------------------------------------------------------------------------
    # 1) ROTAÇÃO: pessoa
    # Teclas Q/E giram a pessoa.
    # -------------------------------------------------------------------------
    rotating_object = scene_controls.rotating_object

    if rotating_object is not None:
        rx, ry, rz = rotating_object.rotation

        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            ry += 90.0 * delta

        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            ry -= 90.0 * delta

        rotating_object.rotation = (rx, ry, rz)

    # -------------------------------------------------------------------------
    # 2) ESCALA: sofá
    # Teclas Z/X aumentam e diminuem o sofá.
    # -------------------------------------------------------------------------
    scaling_object = scene_controls.scaling_object

    if scaling_object is not None:
        sx, sy, sz = scaling_object.scale

        if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
            factor = 1.0 + 0.8 * delta
            scaling_object.scale = (
                sx * factor,
                sy * factor,
                sz * factor,
            )

        if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
            factor = 1.0 - 0.8 * delta
            scaling_object.scale = (
                max(0.05, sx * factor),
                max(0.05, sy * factor),
                max(0.05, sz * factor),
            )

    # -------------------------------------------------------------------------
    # 3) TRANSLAÇÃO: carro
    # Teclas J/L movem o carro para os lados.
    # Teclas I/K movem o carro para frente/trás.
    # -------------------------------------------------------------------------
    translating_object = scene_controls.translating_object

    if translating_object is not None:
        x, y, z = translating_object.position

        move_speed = 5.0 * delta

        if glfw.get_key(window, glfw.KEY_J) == glfw.PRESS:
            x -= move_speed

        if glfw.get_key(window, glfw.KEY_L) == glfw.PRESS:
            x += move_speed

        if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
            z -= move_speed

        if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS:
            z += move_speed

        x = float(np.clip(x, -25.0, 25.0))
        z = float(np.clip(z, -25.0, 30.0))

        translating_object.position = (x, y, z)