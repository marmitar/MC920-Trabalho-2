"""
Varreduras horizontais: undirecional ou alternada.
"""
from tipos import Image, ErrorDist
import numpy as np
from .nb import jit



@jit("uint8[:,::1](uint8[:,::1], float32[:,::1])")
def varredura_unidirecional(img: Image, dist: ErrorDist) -> Image:
    H, W = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
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
    return res


@jit("uint8[:,::1](uint8[:,::1], float32[:,::1])")
def varredura_alternada(img: Image, dist: ErrorDist) -> Image:
    H, W = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    for ym in range((H + 1)//2):
        y = 2 * ym

        for x in range(W):
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

        y += 1
        if y == H:
            break

        for xm in range(W):
            x = W - 1 - xm
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
                for jm in range(tW):
                    j = tW - 1 - jm
                    xj = x + j - dW
                    if 0 <= yi < H and 0 <= xj < W:
                        img[yi, xj] += dist[i, j] * erro
    return res