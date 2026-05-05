# Projeto 2

Projeto desenvolvido para a disciplina de Computação Gráfica.

O objetivo é construir uma cena 3D usando modelos `.obj`, texturas, câmera navegável e transformações geométricas com teclado. A cena representa uma casa com ambiente interno e externo, contendo móveis, pessoa, animais, carro, árvores, piso interno, grama externa e skybox.

---

## Ideia do projeto

A cena representa uma casa em um ambiente rural.

O ambiente interno contém objetos como sofá, mesas, televisão, pessoa e lareira.  
O ambiente externo contém carro, galinha, vaca, grama e céu com textura usando skybox.

A proposta é permitir que o usuário explore livremente a cena em primeira pessoa, usando câmera com mouse e teclado. Além disso, alguns objetos podem ser manipulados por transformações geométricas:

- rotação em uma pessoa;
- escala em um sofá;
- translação em um carro.

O projeto usa matrizes `Model`, `View` e `Projection`, shaders modernos em OpenGL e carregamento de modelos Wavefront `.obj` com materiais `.mtl` e texturas.

---

## Estrutura esperada do projeto

A estrutura dos arquivos deve ficar parecida com esta:

```text
seu_projeto/
  main.py
  config.py
  shaders.py
  matrices.py
  textures.py
  mesh.py
  obj_loader.py
  skybox.py
  camera.py
  scene.py
  controls.py

  casa/
    Cottage_FREE.obj
    Cottage_FREE.mtl
    ...

  modelos/
    sofa/
    mesa/
    mesa_cozinha/
    retro_tv/
    pessoa/
    lareira/
    carro/
    galinha/
    vaca/

  skybox/
    px.png
    nx.png
    py.png
    ny.png
    pz.png
    nz.png

  texturas/
    grass.jpg
    wood.jpg
```

---

## Dependências

Para rodar o projeto, é necessário ter Python instalado e as seguintes bibliotecas:

```bash
pip install -r requirements.txt
```

Bibliotecas usadas:

- `glfw` para criação da janela e captura de entrada;
- `PyOpenGL` para chamadas OpenGL;
- `numpy` para operações com matrizes e vetores;
- `Pillow` para carregar imagens de textura.

---

## Como rodar

Abra o terminal na pasta principal do projeto, onde está o arquivo `main.py`.

Depois execute:

```bash
python3 main.py
```

Uma janela será aberta mostrando a cena 3D.

---

## Comandos disponíveis

### Movimento da câmera

| Tecla / Mouse | Ação |
|---|---|
| `W` | mover para frente |
| `S` | mover para trás |
| `A` | mover para a esquerda |
| `D` | mover para a direita |
| Mouse | olhar ao redor |
| `ESC` | fechar o programa |

---

### Transformações em objetos

| Tecla | Ação |
|---|---|
| `Q` | rotacionar a pessoa para um lado |
| `E` | rotacionar a pessoa para o outro lado |
| `Z` | aumentar a escala do sofá |
| `X` | diminuir a escala do sofá |
| `J` | mover o carro para a esquerda |
| `L` | mover o carro para a direita |
| `I` | mover o carro para frente |
| `K` | mover o carro para trás |

---

### Visualização da malha

| Tecla | Ação |
|---|---|
| `P` | alternar entre visualização normal e wireframe |

Ao pressionar `P`, os objetos passam a ser exibidos em modo de malha poligonal. Pressionando novamente, a cena volta ao modo preenchido.

---

## Modelos usados na cena

### Ambiente interno

- sofá;
- mesa;
- mesa de cozinha;
- televisão;
- pessoa;
- lareira.

### Ambiente externo

- carro;
- galinha;
- vaca;
- árvores;
- grama;
- skybox.

---

## Texturas

As texturas são carregadas principalmente a partir dos arquivos `.mtl`, usando linhas como:

```mtl
map_Kd caminho/da/textura.jpg
```

Cada modelo `.obj` deve apontar para seu respectivo arquivo `.mtl` com uma linha semelhante a:

```obj
mtllib nome_do_material.mtl
```
---

## Skybox

O céu é feito com uma skybox usando seis imagens:

```text
skybox/
  px.png
  nx.png
  py.png
  ny.png
  pz.png
  nz.png
```

Essas imagens representam os seis lados do cubo usado como ambiente externo.

---