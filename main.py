import glfw
from OpenGL.GL import *

from config import WIDTH, HEIGHT, TITLE, SKYBOX_DIR
from shaders import (
    VERTEX_SHADER,
    FRAGMENT_SHADER,
    SKYBOX_VERTEX_SHADER,
    SKYBOX_FRAGMENT_SHADER,
    create_program,
    set_int,
    set_mat4,
)
from matrices import perspective
from camera import Camera
from skybox import Skybox
from scene import build_scene
from controls import InputState, mouse_callback, process_input


def main() -> None:
    if not glfw.init():
        raise RuntimeError("Erro ao inicializar GLFW.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)

    if not window:
        glfw.terminate()
        raise RuntimeError("Erro ao criar janela GLFW.")

    glfw.make_context_current(window)

    print("Contexto atual:", glfw.get_current_context())
    print("Versão OpenGL:", glGetString(GL_VERSION))

    camera = Camera()
    input_state = InputState()
    input_state.last_x = WIDTH / 2
    input_state.last_y = HEIGHT / 2

    def on_mouse(window, xpos, ypos):
        mouse_callback(window, xpos, ypos, camera, input_state)

    def on_key(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        if key == glfw.KEY_P and action == glfw.PRESS:
            input_state.wireframe = not input_state.wireframe

            if input_state.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                print("[INFO] Wireframe ligado")
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                print("[INFO] Wireframe desligado")

    glfw.set_cursor_pos_callback(window, on_mouse)
    glfw.set_key_callback(window, on_key)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glViewport(0, 0, WIDTH, HEIGHT)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)

    program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    skybox_program = create_program(SKYBOX_VERTEX_SHADER, SKYBOX_FRAGMENT_SHADER)

    glUseProgram(program)
    set_int(program, "texture1", 0)

    skybox = Skybox(SKYBOX_DIR)

    scene_objects, scene_controls = build_scene()

    print()
    print("Controles:")
    print("W/A/S/D  -> mover câmera")
    print("Mouse    -> olhar ao redor")
    print("P        -> mostrar/ocultar wireframe")
    print("Q/E      -> rotacionar pessoa")
    print("Z/X      -> aumentar/diminuir sofá")
    print("J/L      -> mover carro esquerda/direita")
    print("I/K      -> mover carro frente/trás")
    print("ESC      -> sair")
    print()

    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta = float(current_time - last_time)
        last_time = current_time

        process_input(window, delta, camera, scene_controls)

        glClearColor(0.55, 0.70, 0.90, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = camera.get_view()
        projection = perspective(60.0, WIDTH / HEIGHT, 0.1, 300.0)

        glUseProgram(program)
        set_mat4(program, "view", view)
        set_mat4(program, "projection", projection)

        for obj in scene_objects:
            obj.draw(program)

        # Skybox sempre preenchida, mesmo quando o wireframe está ligado.
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        skybox.draw(skybox_program, view, projection)

        if input_state.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()