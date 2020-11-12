from tipos import Image, ErrorDist
from .nb import jit
import numpy as np



@jit()
def varredura_unidirecional(img: Image, dist: ErrorDist) -> Image:
    H, W = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    for x in range(H):
        for y in range(W):
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


@jit()
def varredura_alternada(img: Image, dist: ErrorDist) -> Image:
    H, W = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.empty((H, W), dtype=np.uint8)

    for xm in range((H + 1)//2):
        x = 2 * xm

        for y in range(W):
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

        x += 1
        if x == H:
            break

        for ym in range(W):
            y = W - 1 - ym
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
                for jm in range(tW):
                    j = tW - j
                    yj = y + j - dW
                    if 0 <= xi < H and 0 <= yj < W:
                        img[xi, yj] += dist[i, j] * erro
    return res