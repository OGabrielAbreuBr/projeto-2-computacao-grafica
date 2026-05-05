import math
from pathlib import Path
from typing import Optional

import numpy as np

from config import ROOT
from mesh import Material, Mesh, OBJModel
from textures import resolve_texture_path


def parse_mtl(mtl_path: Path) -> dict[str, Material]:
    materials: dict[str, Material] = {}
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
                cleaned: list[str] = []
                i = 0

                # Remove opções do Blender, por exemplo:
                # map_Kd -s 3 3 3 textures/arquivo.jpg
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

    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []

    materials: dict[str, Material] = {}

    current_vertices: list[float] = []
    current_material = Material()

    submeshes_data: list[tuple[list[float], Material]] = []

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

                # Triangula faces com mais de 3 vértices.
                for i in range(1, len(face) - 1):
                    add_vertex(face[0])
                    add_vertex(face[i])
                    add_vertex(face[i + 1])

    finish_submesh()

    meshes: list[Mesh] = []

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


def get_obj_bounds(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []

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
    target_center: tuple[float, float, float],
    rotation_y_degrees: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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
    target_center: tuple[float, float, float],
    rotation_y_degrees: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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