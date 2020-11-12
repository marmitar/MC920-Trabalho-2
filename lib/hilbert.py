from tipos import ErrorDist, Image
from typing import Tuple, Optional, Union, overload

from enum import IntEnum, unique
from .direcao import ErrorDistDir, Dir
from .nb import jit
import numpy as np



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


@jit("uint32(uint32, uint32, uint32, uint32)")
def direcao(x: int, ox: int, y: int, oy: int) -> int:
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
    idx = np.empty((H * W, 3), dtype=np.uint32)

    logN = log2(max(H, W))
    N = 1 << logN
    j = 0

    oy, ox = hilbert_prox_ind(logN, 0)
    for i in range(N * N):
        y, x = hilbert_prox_ind(logN, i)
        if y < H and x < W:
            idx[j] = y, x, direcao(x, ox, y, oy)
            oy, ox = y, x
            j += 1

    return idx


@jit("uint8[:,::1](uint8[:,::1], UniTuple(float32[:,::1], 4), optional(uint32[:,::1]))")
def varredura_hilbert(img: Image, dists: ErrorDistDir, idx: Optional[np.ndarray]=None) -> Image:
    H, W = img.shape

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    if idx is None:
        idx = hilbert_indices(H, W)

    for y, x, d in idx:
        intensidade = img[y, x]
        if intensidade < 128.0:
            res[y, x] = 0
            valor = 0.0
        else:
            res[y, x] = 1
            valor = 255.0

        erro = intensidade - valor

        tH, tW = dists[d].shape
        dW = (tW - 1) //  2
        for i in range(tH):
            yi = y + i
            for j in range(tW):
                xj = x + j - dW
                if 0 <= yi < H and 0 <= xj < W:
                    img[yi, xj] += dists[d][i, j] * erro
    return res
