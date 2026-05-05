from dataclasses import dataclass
from typing import Optional

from config import (
    HOUSE_FLOOR_TEXTURE,
    HOUSE_OBJ,
    MODELS_DIR,
    GRASS_TEXTURE,
)
from matrices import model_matrix
from shaders import set_mat4
from mesh import create_plane
from obj_loader import load_obj, fit_obj_on_ground, fit_small_obj_on_ground


@dataclass
class SceneObject:
    name: str
    mesh: object
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]

    def draw(self, program: int) -> None:
        m = model_matrix(self.position, self.rotation, self.scale)
        set_mat4(program, "model", m)
        self.mesh.draw(program)


@dataclass
class SceneControls:
    rotating_object: Optional[SceneObject] = None
    scaling_object: Optional[SceneObject] = None
    translating_object: Optional[SceneObject] = None


def add_model_object(
    objects: list[SceneObject],
    name: str,
    filename: str,
    target_size: float,
    target_center: tuple[float, float, float],
    rotation: tuple[float, float, float],
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


def build_scene() -> tuple[list[SceneObject], SceneControls]:
    objects: list[SceneObject] = []
    controls = SceneControls()

    if not HOUSE_OBJ.exists():
        raise FileNotFoundError(
            f"Não encontrei a casa: {HOUSE_OBJ}\n\n"
            "Coloque os arquivos assim:\n"
            "  casa/Cottage_FREE.obj\n"
            "  casa/Cottage_FREE.mtl\n"
            "  casa/Cottage_Clean/texturas...\n"
        )

    # -------------------------------------------------------------------------
    # Casa
    # -------------------------------------------------------------------------
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

    # Piso da parte menor da casa.
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
    # Chão externo de grama
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
    # -------------------------------------------------------------------------

    # ESCALA: sofá com Z/X.
    controls.scaling_object = add_model_object(
        objects=objects,
        name="sofa_interno",
        filename="sofa/HSM0012.obj",
        target_size=5.0,
        target_center=(5.0, -1.0, -0.5),
        rotation=(-90.0, 90.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="mesa_interna",
        filename="mesa/mesa.obj",
        target_size=2.2,
        target_center=(8.0, 0.10, -0.5),
        rotation=(0.0, 0.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="mesa_cozinha_interna",
        filename="mesa_cozinha/SET1.obj",
        target_size=5.5,
        target_center=(-3.0, 0.10, 0.4),
        rotation=(0.0, 0.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="retro_tv_interna",
        filename="retro_tv/retro_tv.obj",
        target_size=1.7,
        target_center=(8.0, 1.3, -0.5),
        rotation=(0.0, -90.0, 0.0),
        optional=True,
    )

    # ROTAÇÃO: pessoa com Q/E.
    controls.rotating_object = add_model_object(
        objects=objects,
        name="pessoa_interna",
        filename="pessoa/human.obj",
        target_size=4.0,
        target_center=(-5.0, 0.10, -2.0),
        rotation=(0.0, 180.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="lareira",
        filename="lareira/13110_Fireplace_v2_l3.obj",
        target_size=5.0,
        target_center=(-7.9, -2.0, 0.0),
        rotation=(270.0, 90.0, 0.0),
        optional=True,
    )

    # -------------------------------------------------------------------------
    # Objetos externos
    # -------------------------------------------------------------------------

    # TRANSLAÇÃO: carro com I/J/K/L.
    controls.translating_object = add_model_object(
        objects=objects,
        name="carro",
        filename="carro/Generic_Old_Car.obj",
        target_size=10.2,
        target_center=(4.0, 0.10, 16.6),
        rotation=(0.0, 180.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="galinha",
        filename="galinha/Chicken_Quad.obj",
        target_size=1.7,
        target_center=(-10.0, 0.10, 20.0),
        rotation=(0.0, -90.0, 0.0),
        optional=True,
    )

    add_model_object(
        objects=objects,
        name="vaca",
        filename="vaca/Cow_Low_Poly.obj",
        target_size=4.0,
        target_center=(-5.0, 0.10, 15.0),
        rotation=(0.0, -90.0, 0.0),
        optional=True,
    )

    return objects, controls