from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_2_0 import (
    glVertexAttribPointer as raw_glVertexAttribPointer,
)

from shaders import set_int, set_vec3
from textures import load_texture


@dataclass
class Material:
    texture_path: Optional[Path] = None
    diffuse: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Mesh:
    vertices: list[float]
    texture_path: Optional[Path] = None
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)

    vao: int = 0
    vbo: int = 0
    count: int = 0
    texture_id: int = 0

    def upload(self) -> None:
        arr = np.array(self.vertices, dtype=np.float32)

        self.count = len(arr) // 5
        self.texture_id = load_texture(self.texture_path)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)

        stride = 5 * 4

        raw_glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(0),
        )
        glEnableVertexAttribArray(0)

        raw_glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(3 * 4),
        )
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self, program: int) -> None:
        set_vec3(program, "tint", self.tint)

        if self.texture_id != 0:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            set_int(program, "texture1", 0)
            set_int(program, "useTexture", 1)
        else:
            set_int(program, "useTexture", 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.count)
        glBindVertexArray(0)


@dataclass
class OBJModel:
    meshes: list[Mesh]

    def draw(self, program: int) -> None:
        for mesh in self.meshes:
            mesh.draw(program)


def create_plane(
    width: float,
    depth: float,
    uv_repeat: float,
    texture_path: Optional[Path],
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Mesh:
    w = width / 2.0
    d = depth / 2.0

    vertices = [
        -w, 0.0, -d, 0.0, 0.0,
         w, 0.0, -d, uv_repeat, 0.0,
         w, 0.0,  d, uv_repeat, uv_repeat,

        -w, 0.0, -d, 0.0, 0.0,
         w, 0.0,  d, uv_repeat, uv_repeat,
        -w, 0.0,  d, 0.0, uv_repeat,
    ]

    mesh = Mesh(
        vertices=vertices,
        texture_path=texture_path,
        tint=tint,
    )
    mesh.upload()
    return mesh