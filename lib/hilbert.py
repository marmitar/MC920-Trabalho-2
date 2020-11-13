"""
Varredura seguindo curvas de Hilbert.
"""
from tipos import ErrorDist, Image
from typing import Tuple, Optional, Union, overload

from enum import IntEnum, unique
from .direcao import ErrorDistDir, Dir, deslocamento
from .nb import jit
import numpy as np



# # # # # # # # # # #
# Índices da Curvas #

@jit("UniTuple(uint32, 2)(uint32, uint32)")
def hilbert_prox_ind(ordem: int, idx: int) -> Tuple[int, int]:
    """
    Cálculo do i-ésimo ponto em uma curva de Hilbert.

    Parâmetros
    ----------
    ordem: int
        Ordem ou profundidade da curva.
    idx: int
        Posição do ponto na linha.

    Retorno
    -------
    x, y: int
        Índices do `idx`-ésimo ponto.
    """
    # baseado em:
    # http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve

    x = int(idx % 4 > 1)
    y = int((idx + 1) % 4 > 1)

    idx = idx // 4

    for i in range(1, ordem):
        n = 1 << i

        if idx % 4 == 0:
            x, y = y, x
        elif idx % 4 == 1:
            y += n
        elif idx % 4 == 2:
            x += n
            y += n
        else:# idx % 4 == 3
            x, y = 2*n-1-y, n-1-x

        idx = idx // 4

    return x, y


@jit("uint32(uint32, uint32, uint32, uint32)")
def direcao(x: int, ox: int, y: int, oy: int) -> int:
    """
    Direção da curva de Hilbert, a partir do ponto atual e do anterior.

    Parâmetros
    ----------
    x, y: int
        Ponto atual.
    ox, oy: int
        Ponto anterior.

    Retorno
    -------
    dir: int
        Direção, de acordo com a enum ``Dir``.
    """
    if x > ox:
        return Dir.direita
    elif x < ox:
        return Dir.esquerda
    elif y < oy:
        return Dir.cima
    else:
        return Dir.baixo


@jit("uint32(uint32)")
def log2(num: int) -> int:
    """
    Log base 2 inteiro, aproximado para a próxima potência de 2.
    Basicamente `ceil(log2(num))`, funcionando como número de bits
    necessários para representar `num`.

    Parâmetros
    ----------
    num: int
        Entrada.

    Retorno
    -------
    out: int
        Log base 2 de `num`.
    """
    if num == 0:
        return 0

    i = 0
    num -= 1
    while num > 0:
        num //= 2
        i += 1
    return i

@jit("uint32[:,::1](uint32, uint32)")
def hilbert_indices(H: int, W: int) -> np.ndarray:
    """
    Monta um vetor com os índices ordenados pela curva de
    Hilbert. Além disso, as direções (veja ``Dir``) da curva
    também estão no vetor.

    Parâmetros
    ----------
    H, W: int
        Dimensões da imagem.

    Retorno
    -------
    idx: np.ndarray
        Vetor com os índices e a direção.
    """
    idx = np.empty((H * W, 3), dtype=np.uint32)

    # ordem da menor curva de Hilbert que encobre a imagem
    logN = log2(max(H, W))
    # tamanho real da curva
    N = 1 << logN

    j = 0
    # índices anteriores, para cálculo da direção
    oy, ox = hilbert_prox_ind(logN, 0)
    # checa todos os índices da curva
    for i in range(N * N):
        x, y = hilbert_prox_ind(logN, i)
        # marca apenas os que estão dentro da imagem
        if y < H and x < W:
            idx[j] = y, x, direcao(x, ox, y, oy)
            oy, ox = y, x
            j += 1

    return idx


# # # # # # #
# Varredura #

@jit("uint8[:,::1](uint8[:,::1], UniTuple(float32[:,::1], 4), optional(uint32[:,::1]))")
def varredura_hilbert(img: Image, dists: ErrorDistDir, idx: Optional[np.ndarray]=None) -> Image:
    """
    Varredura pela seguindo as curvas de Hilbert.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz 2D com `uint8` em ordem row-major, representando a imagem.
    dists: tuple
        Quatro (4) matrizes 2D com `float32` em ordem row-major, com as
        distribuições de erros para cada direção de aplicação.
    idx: np.ndarray, optional
        Índices da curva de Hilbert.

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

    # cálculo dos índices, se necessário
    if idx is None:
        idx = hilbert_indices(H, W)

    # percorre seguindo os índices
    for y, x, d in idx:
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
        tH, tW = dists[d].shape
        dH, dW = deslocamento(d, tH, tW)
        for i in range(tH):
            yi = y + i - dH
            for j in range(tW):
                xj = x + j - dW
                # cuidado com acesso out-of-bounds
                if 0 <= yi < H and 0 <= xj < W:
                    img[yi, xj] += dists[d][i, j] * erro
    return res
