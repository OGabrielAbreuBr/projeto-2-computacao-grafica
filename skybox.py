import ctypes
from pathlib import Path

import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_2_0 import (
    glVertexAttribPointer as raw_glVertexAttribPointer,
)

from shaders import set_int, set_mat4


SKYBOX_VERTICES = [
    -1.0,  1.0, -1.0,
    -1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,

    -1.0, -1.0,  1.0,
    -1.0, -1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,

     1.0, -1.0, -1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0, -1.0,
     1.0, -1.0, -1.0,

    -1.0, -1.0,  1.0,
    -1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
    -1.0, -1.0,  1.0,

    -1.0,  1.0, -1.0,
     1.0,  1.0, -1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,

    -1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0,  1.0,
]


def load_cubemap(faces: list[Path]) -> int:
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    for i, face in enumerate(faces):
        if not face.exists():
            raise FileNotFoundError(f"Face da skybox não encontrada: {face}")

        img = Image.open(face).convert("RGB")
        data = np.array(img, dtype=np.uint8)

        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0,
            GL_RGB,
            img.width,
            img.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            data,
        )

        print(f"[OK] Face skybox carregada: {face.name}")

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return texture_id


class Skybox:
    def __init__(self, folder: Path) -> None:
        faces = [
            folder / "px.png",
            folder / "nx.png",
            folder / "py.png",
            folder / "ny.png",
            folder / "pz.png",
            folder / "nz.png",
        ]

        self.texture_id = load_cubemap(faces)

        vertices = np.array(SKYBOX_VERTICES, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        raw_glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            GL_FALSE,
            3 * 4,
            ctypes.c_void_p(0),
        )
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self, program: int, view: np.ndarray, projection: np.ndarray) -> None:
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_FALSE)

        glUseProgram(program)

        view_without_translation = view.copy()
        view_without_translation[0:3, 3] = 0.0

        set_mat4(program, "view", view_without_translation)
        set_mat4(program, "projection", projection)
        set_int(program, "skybox", 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)