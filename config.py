from pathlib import Path

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

# A grama tem 120x120, então vai de -60 até 60.
# Deixo 58 para a câmera não sair do chão.
SCENE_LIMIT = 58.0