from tipos import Image, ErrorDist
from typing import Tuple
import numpy as np

from .direcao import ErrorDistDir, Dir
from .nb import jit


@jit("void(uint8[:,::1], float32[:,::1], float32[:,::1], uint32, uint32, uint32, uint32)")
def aplica_varredura(res: Image, img: Image, dist: ErrorDist, H: int, W: int, y: int, x: int) -> None:
    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    intensidade = img[y, x]
    if intensidade < 128.0:
        res[y, x] = 0
        valor = 0.0
    else:
        res[y, x] = 1
        valor = 255.0

    erro = intensidade - valor

    for i in range(tH):
        yi = y + i - dH
        for j in range(tW):
            xj = x + j - dW
            if 0 <= yi < H and 0 <= xj < W:
                img[yi, xj] += dist[i, j] * erro


@jit("uint8[:,::1](uint8[:,::1], UniTuple(float32[:,::1], 4))")
def varredura_espiral(img: Image, dists: ErrorDistDir) -> Image:
    H, W = img.shape

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    for s in range((1 + min(H, W)) // 2):

        y = s
        d = Dir.direita.value
        for x in range(s, W - s):
            aplica_varredura(res, img, dists[d], H, W, y, x)

        x = W - 1 - s
        d = Dir.baixo.value
        for y in range(s + 1, H - s):
            aplica_varredura(res, img, dists[d], H, W, y, x)

        y = H - 1 - s
        d = Dir.esquerda.value
        for x in range(W - 1 - s, s, -1):
            aplica_varredura(res, img, dists[d], H, W, y, x - 1)

        x = s
        d = Dir.cima.value
        for y in range(H - 1 - s, s + 1, -1):
            aplica_varredura(res, img, dists[d], H, W, y - 1, x)

    return res
