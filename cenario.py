from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glfw
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_2_0 import (
    glVertexAttribPointer as raw_glVertexAttribPointer,
)


# =============================================================================
# Configurações
# =============================================================================

WIDTH = 1280
HEIGHT = 720
TITLE = "Projeto 2 - Cottage com Skybox"

ROOT = Path(__file__).resolve().parent

HOUSE_DIR = ROOT / "casa"
HOUSE_OBJ = HOUSE_DIR / "Cottage_FREE.obj"

MODELS_DIR = ROOT / "modelos"
SKYBOX_DIR = ROOT / "skybox"

TEXTURES_DIR = ROOT / "texturas"
GRASS_TEXTURE = TEXTURES_DIR / "grass.jpg"
HOUSE_FLOOR_TEXTURE = TEXTURES_DIR / "wood.jpg"


CAMERA_SPEED = 5.0
MOUSE_SENSITIVITY = 0.01
SCENE_LIMIT = 80.0


# =============================================================================
# Shaders dos objetos
# =============================================================================

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


# =============================================================================
# Shaders da skybox
# =============================================================================

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


# =============================================================================
# Compilação de shaders
# =============================================================================

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


def set_vec3(program: int, name: str, value: Tuple[float, float, float]) -> None:
    glUniform3f(glGetUniformLocation(program, name), value[0], value[1], value[2])


def set_mat4(program: int, name: str, matrix: np.ndarray) -> None:
    glUniformMatrix4fv(
        glGetUniformLocation(program, name),
        1,
        GL_TRUE,
        matrix.astype(np.float32),
    )


# =============================================================================
# Matrizes
# =============================================================================

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
    position: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    scale: Tuple[float, float, float],
) -> np.ndarray:
    return (
        translate(position[0], position[1], position[2])
        @ rotate_y(rotation[1])
        @ rotate_x(rotation[0])
        @ rotate_z(rotation[2])
        @ scale_matrix(scale[0], scale[1], scale[2])
    )


# =============================================================================
# Texturas
# =============================================================================

texture_cache: Dict[Path, int] = {}


def load_texture(path: Optional[Path]) -> int:
    if path is None:
        return 0

    path = path.resolve()

    if path in texture_cache:
        return texture_cache[path]

    if not path.exists():
        print(f"[AVISO] Textura não encontrada: {path}")
        return 0

    try:
        img = Image.open(path).convert("RGBA")
    except Exception as e:
        print(f"[AVISO] Não consegui carregar textura {path}: {e}")
        return 0

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    data = np.array(img, dtype=np.uint8)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        img.width,
        img.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data,
    )

    glGenerateMipmap(GL_TEXTURE_2D)

    texture_cache[path] = tex
    return tex


def resolve_texture_path(base_dir: Path, tex_name: str) -> Optional[Path]:
    tex_name = tex_name.replace("\\", "/").strip()

    direct = base_dir / tex_name

    if direct.exists():
        return direct

    wanted = Path(tex_name).name.lower()

    # Procura em toda a pasta do modelo.
    for candidate in base_dir.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate

    # Procura também a partir da raiz do projeto.
    for candidate in ROOT.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate

    return None


# =============================================================================
# Cubemap da skybox
# =============================================================================

def load_cubemap(faces: List[Path]) -> int:
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


# =============================================================================
# Malhas e materiais
# =============================================================================

@dataclass
class Material:
    texture_path: Optional[Path] = None
    diffuse: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Mesh:
    vertices: List[float]
    texture_path: Optional[Path] = None
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)

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

def create_plane(width: float, depth: float, uv_repeat: float, texture_path: Optional[Path], tint=(1.0, 1.0, 1.0)) -> Mesh:
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


@dataclass
class OBJModel:
    meshes: List[Mesh]

    def draw(self, program: int) -> None:
        for mesh in self.meshes:
            mesh.draw(program)


# =============================================================================
# Parser MTL / OBJ
# =============================================================================

def parse_mtl(mtl_path: Path) -> Dict[str, Material]:
    materials: Dict[str, Material] = {}
    current_name: Optional[str] = None

    if not mtl_path.exists():
        print(f"[AVISO] MTL não encontrado: {mtl_path}")
        return materials

    print(f"[OK] Lendo MTL: {mtl_path}")

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw in file:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "newmtl" and len(parts) >= 2:
                current_name = " ".join(parts[1:])
                materials[current_name] = Material()

            elif cmd == "Kd" and current_name is not None and len(parts) >= 4:
                try:
                    r = float(parts[1])
                    g = float(parts[2])
                    b = float(parts[3])
                    materials[current_name].diffuse = (r, g, b)
                except ValueError:
                    pass

            elif cmd == "map_Kd" and current_name is not None and len(parts) >= 2:
                tokens = parts[1:]
                cleaned: List[str] = []
                i = 0

                while i < len(tokens):
                    token = tokens[i]

                    if token.startswith("-"):
                        if token in ["-s", "-o", "-t"]:
                            i += 4
                        elif token in ["-bm", "-boost"]:
                            i += 2
                        else:
                            i += 1
                    else:
                        cleaned.append(token)
                        i += 1

                if cleaned:
                    tex_name = " ".join(cleaned)
                    tex_path = resolve_texture_path(mtl_path.parent, tex_name)

                    if tex_path is None:
                        print(f"[AVISO] Textura do material não encontrada: {tex_name}")
                    else:
                        print(f"[OK] Textura encontrada: {tex_path}")

                    materials[current_name].texture_path = tex_path

    return materials


def parse_obj_index(text: str, size: int) -> int:
    idx = int(text)

    if idx < 0:
        return size + idx

    return idx - 1


def load_obj(obj_path: Path) -> OBJModel:
    if not obj_path.exists():
        raise FileNotFoundError(f"Arquivo OBJ não encontrado: {obj_path}")

    print(f"[OK] Carregando OBJ: {obj_path}")

    positions: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []

    materials: Dict[str, Material] = {}

    current_vertices: List[float] = []
    current_material = Material()

    submeshes_data: List[Tuple[List[float], Material]] = []

    def finish_submesh() -> None:
        nonlocal current_vertices

        if current_vertices:
            submeshes_data.append((current_vertices, current_material))
            current_vertices = []

    def add_vertex(token: str) -> None:
        parts = token.split("/")

        v_index = parse_obj_index(parts[0], len(positions))
        pos = positions[v_index]

        if len(parts) >= 2 and parts[1] != "":
            vt_index = parse_obj_index(parts[1], len(texcoords))
            uv = texcoords[vt_index]
        else:
            uv = (0.0, 0.0)

        current_vertices.extend([
            pos[0],
            pos[1],
            pos[2],
            uv[0],
            uv[1],
        ])

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw in file:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "mtllib" and len(parts) >= 2:
                mtl_name = " ".join(parts[1:])
                mtl_path = obj_path.parent / mtl_name
                materials.update(parse_mtl(mtl_path))

            elif cmd == "usemtl" and len(parts) >= 2:
                finish_submesh()

                mat_name = " ".join(parts[1:])
                current_material = materials.get(mat_name, Material())

            elif cmd == "v" and len(parts) >= 4:
                positions.append((
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                ))

            elif cmd == "vt" and len(parts) >= 3:
                texcoords.append((
                    float(parts[1]),
                    float(parts[2]),
                ))

            elif cmd == "f" and len(parts) >= 4:
                face = parts[1:]

                for i in range(1, len(face) - 1):
                    add_vertex(face[0])
                    add_vertex(face[i])
                    add_vertex(face[i + 1])

    finish_submesh()

    meshes: List[Mesh] = []

    for vertices, material in submeshes_data:
        mesh = Mesh(
            vertices=vertices,
            texture_path=material.texture_path,
            tint=material.diffuse,
        )
        mesh.upload()
        meshes.append(mesh)

    if not meshes:
        raise RuntimeError(f"O OBJ não gerou nenhuma malha válida: {obj_path}")

    print(f"[OK] OBJ carregado com {len(meshes)} submalhas.")
    return OBJModel(meshes)


# =============================================================================
# Bounds e ajuste automático
# =============================================================================

def get_obj_bounds(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    positions: List[List[float]] = []

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw in file:
            line = raw.strip()

            if line.startswith("v "):
                parts = line.split()

                if len(parts) >= 4:
                    positions.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ])

    if not positions:
        raise RuntimeError(f"Não encontrei vértices no OBJ: {obj_path}")

    arr = np.array(positions, dtype=np.float32)

    return arr.min(axis=0), arr.max(axis=0)


def fit_obj_on_ground(
    obj_path: Path,
    target_width: float,
    target_center: Tuple[float, float, float],
    rotation_y_degrees: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    min_v, max_v = get_obj_bounds(obj_path)

    size = max_v - min_v
    center = (min_v + max_v) / 2.0

    maior_lado_horizontal = max(float(size[0]), float(size[2]))

    if maior_lado_horizontal <= 0.0001:
        factor = 1.0
    else:
        factor = target_width / maior_lado_horizontal

    angle = math.radians(rotation_y_degrees)
    c = math.cos(angle)
    s = math.sin(angle)

    center_scaled = center * factor

    rotated_center_x = c * center_scaled[0] + s * center_scaled[2]
    rotated_center_z = -s * center_scaled[0] + c * center_scaled[2]

    tx = target_center[0] - rotated_center_x
    ty = target_center[1] - float(min_v[1]) * factor
    tz = target_center[2] - rotated_center_z

    print(f"[INFO] {obj_path.name}: scale={factor}, position={(tx, ty, tz)}")

    return (tx, ty, tz), (factor, factor, factor)


def fit_small_obj_on_ground(
    obj_path: Path,
    target_size: float,
    target_center: Tuple[float, float, float],
    rotation_y_degrees: float = 0.0,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    min_v, max_v = get_obj_bounds(obj_path)

    size = max_v - min_v
    center = (min_v + max_v) / 2.0

    maior_lado = max(float(size[0]), float(size[1]), float(size[2]))

    if maior_lado <= 0.0001:
        factor = 1.0
    else:
        factor = target_size / maior_lado

    angle = math.radians(rotation_y_degrees)
    c = math.cos(angle)
    s = math.sin(angle)

    center_scaled = center * factor

    rotated_center_x = c * center_scaled[0] + s * center_scaled[2]
    rotated_center_z = -s * center_scaled[0] + c * center_scaled[2]

    tx = target_center[0] - rotated_center_x
    ty = target_center[1] - float(min_v[1]) * factor
    tz = target_center[2] - rotated_center_z

    print(f"[INFO] {obj_path.name}: scale={factor}, position={(tx, ty, tz)}")

    return (tx, ty, tz), (factor, factor, factor)


# =============================================================================
# Objetos da cena
# =============================================================================

@dataclass
class SceneObject:
    name: str
    mesh: object
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]

    def draw(self, program: int) -> None:
        m = model_matrix(self.position, self.rotation, self.scale)
        set_mat4(program, "model", m)
        self.mesh.draw(program)


def add_model_object(
    objects: List[SceneObject],
    name: str,
    filename: str,
    target_size: float,
    target_center: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    optional: bool = True,
) -> Optional[SceneObject]:
    obj_path = MODELS_DIR / filename

    if not obj_path.exists():
        msg = f"[AVISO] Modelo não encontrado: {obj_path}"

        if optional:
            print(msg)
            return None

        raise FileNotFoundError(msg)

    model = load_obj(obj_path)

    position, scale = fit_small_obj_on_ground(
        obj_path=obj_path,
        target_size=target_size,
        target_center=target_center,
        rotation_y_degrees=rotation[1],
    )

    scene_object = SceneObject(
        name=name,
        mesh=model,
        position=position,
        rotation=rotation,
        scale=scale,
    )

    objects.append(scene_object)

    return scene_object


# =============================================================================
# Câmera
# =============================================================================

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


camera = Camera()

first_mouse = True
last_x = WIDTH / 2
last_y = HEIGHT / 2

wireframe = False

# Objetos controlados por teclado para cumprir o requisito:
# rotação, escala e translação em modelos diferentes.
rotating_object: Optional[SceneObject] = None
scaling_object: Optional[SceneObject] = None
translating_object: Optional[SceneObject] = None


def mouse_callback(window, xpos: float, ypos: float) -> None:
    global first_mouse, last_x, last_y

    if first_mouse:
        last_x = xpos
        last_y = ypos
        first_mouse = False

    x_offset = xpos - last_x
    y_offset = last_y - ypos

    last_x = xpos
    last_y = ypos

    camera.yaw += x_offset * MOUSE_SENSITIVITY
    camera.pitch += y_offset * MOUSE_SENSITIVITY

    camera.pitch = float(np.clip(camera.pitch, -89.0, 89.0))
    camera.update_vectors()


def key_callback(window, key, scancode, action, mods) -> None:
    global wireframe

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_P and action == glfw.PRESS:
        wireframe = not wireframe

        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            print("[INFO] Wireframe ligado")
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            print("[INFO] Wireframe desligado")


def process_input(window, delta: float) -> None:
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
    # 1) ROTAÇÃO: TV
    # Teclas Q/E giram a TV.
    # -------------------------------------------------------------------------
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


# =============================================================================
# Montagem da cena
# =============================================================================

def build_scene() -> List[SceneObject]:
    global rotating_object, scaling_object, translating_object
    
    objects: List[SceneObject] = []

    if not HOUSE_OBJ.exists():
        raise FileNotFoundError(
            f"Não encontrei a casa: {HOUSE_OBJ}\n\n"
            "Coloque os arquivos assim:\n"
            "  casa/Cottage_FREE.obj\n"
            "  casa/Cottage_FREE.mtl\n"
            "  casa/Cottage_Clean/texturas...\n"
        )

    house_model = load_obj(HOUSE_OBJ)

    house_rotation_y = 0.0

    house_position, house_scale = fit_obj_on_ground(
        HOUSE_OBJ,
        target_width=26.0,
        target_center=(0.0, 0.0, 0.0),
        rotation_y_degrees=house_rotation_y,
    )

    objects.append(SceneObject(
        name="cottage",
        mesh=house_model,
        position=house_position,
        rotation=(0.0, house_rotation_y, 0.0),
        scale=house_scale,
    ))

    # -------------------------------------------------------------------------
    # Piso interno da casa
    # -------------------------------------------------------------------------
    house_floor_mesh = create_plane(
        width=20.0,
        depth=18.65,
        uv_repeat=8.0,
        texture_path=HOUSE_FLOOR_TEXTURE,
    )

    objects.append(SceneObject(
    name="piso_interno_casa",
    mesh=house_floor_mesh,
    position=(0.0, 0.08, -1.9),
    rotation=(0.0, house_rotation_y, 0.0),
    scale=(1.0, 1.0, 1.0),
    )) 

     # Piso parte menor (a “perna” do L)
    house_floor_mesh_2 = create_plane(
        width=8.0,
        depth=4.0,
        uv_repeat=4.0,
        texture_path=HOUSE_FLOOR_TEXTURE,
    )

    objects.append(SceneObject(
        name="piso_parte_2",
        mesh=house_floor_mesh_2,
        position=(-6.0, 0.08, 9.4),
        rotation=(0.0, house_rotation_y, 0.0),
        scale=(1.0, 1.0, 1.0),
    ))

    # -------------------------------------------------------------------------
    # Chão de grama
    # -------------------------------------------------------------------------
    grass_mesh = create_plane(
        width=120.0,
        depth=120.0,
        uv_repeat=20.0,
        texture_path=GRASS_TEXTURE,
    )

    objects.append(SceneObject(
        name="grama",
        mesh=grass_mesh,
        position=(0.0, -0.02, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    ))

    # -------------------------------------------------------------------------
    # Objetos internos
    # Ajuste target_center conforme o interior da cottage.
    # -------------------------------------------------------------------------

    scaling_object = add_model_object(
        objects=objects,
        name="sofa_interno",
        filename="sofa/HSM0012.obj",
        target_size=5,
        target_center=(-3.0, -1.0, -1.0),
        rotation=(-90.0, 90.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="mesa_interna",
        filename="mesa/mesa.obj",
        target_size=2.2,
        target_center=(-1.4, 0.10, -2.6),
        rotation=(0.0, 0.0, 0.0),
        optional=True,
    )

    rotating_object = add_model_object(
        objects=objects,
        name="retro_tv_interna",
        filename="retro_tv/retro_tv.obj",
        target_size=1.7,
        target_center=(2.0, 0.10, -2.1),
        rotation=(0.0, -90.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="pessoa_interna",
        filename="pessoa/human.obj",
        target_size=2.2,
        target_center=(5.0, 0.10, -2.0),
        rotation=(0.0, 180.0, 0.0),
        optional=True,
    )

    translating_object = add_model_object(
        objects=objects,
        name="carro",
        filename="carro/Generic_Old_Car.obj",
        target_size=10.2,
        target_center=(-4.4, 0.10, 20.6),
        rotation=(0.0, 0.0, 0.0),
        optional=True,
    )

    rotating_object = add_model_object(
        objects=objects,
        name="Galinha",
        filename="galinha/Chicken_Quad.obj",
        target_size=1.7,
        target_center=(5.0, 0.10, -2.1),
        rotation=(0.0, -90.0, 0.0),
        optional=True,
    )

    rotating_object = add_model_object(
        objects=objects,
        name="Vaca",
        filename="vaca/Cow_Low_Poly.obj",
        target_size=1.7,
        target_center=(2.0, 0.10, -2.1),
        rotation=(0.0, -90.0, 0.0),
        optional=True,
    )

    return objects


# =============================================================================
# Programa principal
# =============================================================================

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

    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glViewport(0, 0, WIDTH, HEIGHT)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)

    program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    skybox_program = create_program(SKYBOX_VERTEX_SHADER, SKYBOX_FRAGMENT_SHADER)

    glUseProgram(program)
    set_int(program, "texture1", 0)

    skybox = Skybox(SKYBOX_DIR)

    print()
    print("Controles:")
    print("W/A/S/D  -> mover câmera")
    print("Mouse    -> olhar ao redor")
    print("P        -> mostrar/ocultar wireframe")
    print("Q/E      -> rotacionar TV")
    print("Z/X      -> aumentar/diminuir sofá")
    print("J/L      -> mover carro esquerda/direita")
    print("I/K      -> mover carro frente/trás")
    print("ESC      -> sair")
    print()

    scene_objects = build_scene()

    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta = float(current_time - last_time)
        last_time = current_time

        process_input(window, delta)

        glClearColor(0.55, 0.70, 0.90, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = camera.get_view()
        projection = perspective(60.0, WIDTH / HEIGHT, 0.1, 300.0)

        glUseProgram(program)
        set_mat4(program, "view", view)
        set_mat4(program, "projection", projection)

        for obj in scene_objects:
            obj.draw(program)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        skybox.draw(skybox_program, view, projection)

        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()