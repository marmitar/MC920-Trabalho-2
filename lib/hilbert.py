from tipos import ErrorDist, Image
from typing import Tuple, Optional, Union, overload

from enum import IntEnum, unique
from .nb import jit
import numpy as np



@jit("uint32(uint32)")
def log2(num: int) -> int:
    if num == 0:
        return 0

    i = 0
    num -= 1
    while num > 0:
        num //= 2
        i += 1
    return i

# http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve
@jit("UniTuple(uint32, 2)(uint32, uint32)")
def hilbert_prox_ind(logN: int, idx: int) -> Tuple[int, int]:
    x = int(idx % 4 > 1)
    y = int((idx + 1) % 4 > 1)

    idx = idx // 4

    for i in range(1, logN):
        n = 1 << i

        if idx % 4 == 0:
            old_y = y
            y = x
            x = old_y
        elif idx % 4 == 1:
            y += n
        elif idx % 4 == 2:
            x += n
            y += n
        else:# idx % 4 == 3
            old_x = x
            x = n + n - 1 - y
            y = n - 1 - old_x

        idx = idx // 4

    return x, y



@unique
class Dir(IntEnum):
    direita = 0
    esquerda = 1
    cima = 2
    baixo = 3



@jit("uint32(uint32, uint32)")
def direcao(dx: int, dy: int):
    if dx > 0:
        return Dir.direita
    elif dx < 0:
        return Dir.esquerda
    elif dy < 0:
        return Dir.cima
    else:
        return Dir.baixo

@jit("uint32[:,::1](uint32, uint32)")
def hilbert_indices(H: int, W: int) -> np.ndarray:
    idx = np.empty((H * W, 3), dtype=np.uint32)

    logN = log2(max(H, W))
    N = 1 << logN
    j = 0

    ox, oy = hilbert_prox_ind(logN, 0)
    for i in range(N * N):
        x, y = hilbert_prox_ind(logN, i)
        if x < H and y < W:
            idx[j] = x, y, direcao(x - ox, y - oy)
            ox, oy = x, y
            j += 1

    return idx

ErrorDistDir = Tuple[ErrorDist, ErrorDist, ErrorDist, ErrorDist]

@jit("UniTuple(float32[:,::1], 4)(float32[:,::1])")
def dist_direcoes(dist: ErrorDist) -> ErrorDistDir:
    direita = dist
    esquerda = np.copy(np.flip(dist))
    cima = np.copy(dist[:,::-1].T)
    baixo = np.copy(dist[::-1].T)
    return direita, esquerda, cima, baixo



@overload
def varredura_hilbert(img: Image, dist: ErrorDist) -> Image:
    ...
@overload
def varredura_hilbert(img: Image, dist: None, idx: np.ndarray, dists: ErrorDistDir) -> Image:
    ...
@jit([
    "uint8[:,::1](uint8[:,::1], optional(float32[:,::1]), uint32[:,::1], UniTuple(float32[:,::1], 4))",
    "uint8[:,::1](uint8[:,::1], float32[:,::1], optional(uint32[:,::1]), optional(UniTuple(float32[:,::1], 4)))"
])
def varredura_hilbert(img: Image, dist: Optional[ErrorDist]=None, idx: Optional[np.ndarray]=None, dists: Optional[ErrorDistDir]=None) -> Image:
    H, W = img.shape

    if idx is None:
        idx = hilbert_indices(H, W)
    if dists is None:
        dists = dist_direcoes(dist)

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    for x, y, d in idx:
        intensidade = img[x, y]
        if intensidade < 128.0:
            res[x, y] = 0
            valor = 0.0
        else:
            res[x, y] = 1
            valor = 255.0

        erro = intensidade - valor

        tH, tW = dists[d].shape
        dH, dW = (tH - 1)//2, (tW - 1)//2
        for i in range(tH):
            xi = x + i - dH
            for j in range(tW):
                yj = y + j - dW
                if 0 <= xi < H and 0 <= yj < W:
                    img[xi, yj] += dists[d][i, j] * erro
    return res
