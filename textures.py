from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from OpenGL.GL import *

from config import ROOT


texture_cache: dict[Path, int] = {}


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

    # Procura dentro da pasta do modelo.
    for candidate in base_dir.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate

    # Procura também a partir da raiz do projeto.
    for candidate in ROOT.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate

    return None