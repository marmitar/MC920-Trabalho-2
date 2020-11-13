"""
Varredura seguindo uma espiral retangular.
Não funciona corretamente.
"""
from tipos import Image, ErrorDist
from typing import Tuple
import numpy as np

from .direcao import ErrorDistDir, Dir, deslocamento
from .nb import jit


@jit("void(uint8[:,::1], float32[:,::1], float32[:,::1], uint8, UniTuple(int32, 4))")
def aplica_em_pixel(res: Image, img: Image, dist: ErrorDist, d: int, pos: Tuple[int, int, int, int]) -> None:
    """
    Aplicação da redução de níveis de cinza e distribuição dos erros.
    Função interna.

    Parâmetros
    ----------
    res: np.ndarray
        Imagem resultante
    img: np.ndarray
        Imagem original, onde serão distribuídos os erros.
    dist: np.ndarray
        Distribuição de erros naquela direção.
    d: int
        Direção, de acordo com a enum ``Dir``.
    pos: tuple
        Informações de dimensões da imagem e ponto atual.
    """
    # dimensões da img e ponto atual
    H, W, y, x = pos

    # dim. da distribuição
    tH, tW = dist.shape
    # deslocamento da máscara para essa direção
    dH, dW = deslocamento(d, tH, tW)

    # meios tons simples
    intensidade = img[y, x]
    if intensidade < 128.0:
        res[y, x] = 0
        valor = 0.0
    else:
        res[y, x] = 1
        valor = 255.0

    erro = intensidade - valor
    # carregamento do erro seguindo aquela direção
    for i in range(tH):
        yi = y + i - dH
        for j in range(tW):
            xj = x + j - dW
            # cuidado com acesso out-of-bounds
            if 0 <= yi < H and 0 <= xj < W:
                img[yi, xj] += dist[i, j] * erro


@jit("uint8[:,::1](uint8[:,::1], UniTuple(float32[:,::1], 4))")
def varredura_espiral(img: Image, dists: ErrorDistDir) -> Image:
    """
    Varredura pela seguindo uma espiral retangular.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz 2D com `uint8` em ordem row-major, representando a imagem.
    dists: tuple
        Quatro (4) matrizes 2D com `float32` em ordem row-major, com as
        distribuições de erros para cada direção de aplicação.

    Retorno
    -------
    out: np.ndarray
        Imagem resultante. Matriz 2D com `uint8` em ordem row-major.
    """
    # dimensões da imagem
    H, W = img.shape
    # imagem em ponto flutuante
    img = img.astype(np.float32)
    # imagem resultante
    res = np.empty((H, W), dtype=np.uint8)

    # cada espiral, de fora para dentro
    for s in range((1 + min(H, W)) // 2):
        # linha superior
        y = s
        d = Dir.direita.value
        for x in range(s, W - s):
            aplica_em_pixel(res, img, dists[d], d, (H, W, y, x))
        # linha lateral direita
        x = W - 1 - s
        d = Dir.baixo.value
        for y in range(s + 1, H - s):
            aplica_em_pixel(res, img, dists[d], d, (H, W, y, x))
        # linha inferior
        y = H - 1 - s
        d = Dir.esquerda.value
        for x in range(W - 1 - s, s, -1):
            aplica_em_pixel(res, img, dists[d], d, (H, W, y, x - 1))
        # linha lateral esquerda
        x = s
        d = Dir.cima.value
        for y in range(H - 1 - s, s + 1, -1):
            aplica_em_pixel(res, img, dists[d], d, (H, W, y - 1, x))

    return res
