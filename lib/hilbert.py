from tipos import ErrorDist, Image
from typing import Tuple, Optional
from .nb import jit, cfunc
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

@jit("uint32[:,::1](uint32, uint32)")
def hilbert_indices(H: int, W: int) -> np.ndarray:
    idx = np.empty((H * W, 2), dtype=np.uint32)

    logN = log2(max(H, W))
    N = 1 << logN
    j = 0
    for i in range(N * N):
        x, y = hilbert_prox_ind(logN, i)
        if x < H and y < W:
            idx[j, 0] = x
            idx[j, 1] = y
            j += 1

    return idx



@jit("uint8[:,::1](uint8[:,::1], float32[:,::1], optional(uint32[:,::1]))")
def varredura_hilbert(img: Image, dist: ErrorDist, idx: Optional[np.ndarray] = None) -> Image:
    H, W = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)
    if idx is None:
        idx = hilbert_indices(H, W)

    for x, y in idx:
        intensidade = img[x, y]
        if intensidade < 128.0:
            res[x, y] = 0
            valor = 0.0
        else:
            res[x, y] = 1
            valor = 255.0

        erro = intensidade - valor
        for i in range(tH):
            xi = x + i - dH
            for j in range(tW):
                yj = y + j - dW
                if 0 <= xi < H and 0 <= yj < W:
                    img[xi, yj] += dist[i, j] * erro
    return res
