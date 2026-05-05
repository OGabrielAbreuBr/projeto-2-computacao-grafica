import numpy as np
from OpenGL.GL import *


VERTEX_SHADER = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""


FRAGMENT_SHADER = """
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture1;
uniform int useTexture;
uniform vec3 tint;

void main()
{
    if (useTexture == 1)
        FragColor = texture(texture1, TexCoord) * vec4(tint, 1.0);
    else
        FragColor = vec4(tint, 1.0);
}
"""


SKYBOX_VERTEX_SHADER = """
#version 330 core

layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    TexCoords = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
"""


SKYBOX_FRAGMENT_SHADER = """
#version 330 core

in vec3 TexCoords;
out vec4 FragColor;

uniform samplerCube skybox;

void main()
{
    FragColor = texture(skybox, TexCoords);
}
"""


def compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(error)

    return shader


def create_program(vertex_source: str, fragment_source: str) -> int:
    vertex = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment = compile_shader(fragment_source, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(error)

    glDeleteShader(vertex)
    glDeleteShader(fragment)

    return program


def set_int(program: int, name: str, value: int) -> None:
    glUniform1i(glGetUniformLocation(program, name), value)


def set_vec3(program: int, name: str, value: tuple[float, float, float]) -> None:
    glUniform3f(glGetUniformLocation(program, name), value[0], value[1], value[2])


def set_mat4(program: int, name: str, matrix: np.ndarray) -> None:
    glUniformMatrix4fv(
        glGetUniformLocation(program, name),
        1,
        GL_TRUE,
        matrix.astype(np.float32),
    )