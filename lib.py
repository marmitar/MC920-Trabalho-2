from tipos import ErrorDist, Image
from typing import Tuple
from numba import jit, float32, uint8, uint32, void, prange
import numpy as np


jit_opt = {
    'nopython': True, 'nogil': True, 'cache': True, 'fastmath': True,
    'parallel': True, 'error_model': 'numpy'
}

@jit(uint8[:,:,::1](uint8[:,:,::1], float32[:,::1]), **jit_opt)
def varredura_unidirecional(img: Image, dist: ErrorDist) -> Image:
    H, W, _ = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.zeros((H, W, 3), dtype=np.uint8)

    for c in prange(3):
        for x in range(H):
            for y in range(W):
                intensidade = img[x, y, c]
                if intensidade < 128.0:
                    # res[x, y, c] = 0
                    valor = 0.0
                else:
                    res[x, y, c] = 1
                    valor = 255.0

                erro = intensidade - valor
                for i in range(tH):
                    xi = x + i - dH
                    for j in range(tW):
                        yj = y + j - dW
                        if 0 <= xi < H and 0 <= yj < W:
                            img[xi, yj, c] += dist[i, j] * erro
    return res


@jit(uint8[:,:,::1](uint8[:,:,::1], float32[:,::1]), **jit_opt)
def varredura_alternada(img: Image, dist: ErrorDist) -> Image:
    H, W, _ = img.shape

    tH, tW = dist.shape
    dH, dW = (tH - 1)//2, (tW - 1)//2

    img = img.astype(np.float32)
    res = np.zeros((H, W, 3), dtype=np.uint8)

    for c in prange(3):
        for xm in range((H + 1)//2):
            x = 2 * xm

            for y in range(W):
                intensidade = img[x, y, c]
                if intensidade < 128.0:
                    # res[x, y, c] = 0
                    valor = 0.0
                else:
                    res[x, y, c] = 1
                    valor = 255.0

                erro = intensidade - valor
                for i in range(tH):
                    xi = x + i - dH
                    for j in range(tW):
                        yj = y + j - dW
                        if 0 <= xi < H and 0 <= yj < W:
                            img[xi, yj, c] += dist[i, j] * erro

            x += 1
            if x == H:
                break

            for ym in range(W):
                y = W - 1 - ym
                intensidade = img[x, y, c]
                if intensidade < 128.0:
                    # res[x, y, c] = 0
                    valor = 0.0
                else:
                    res[x, y, c] = 1
                    valor = 255.0

                erro = intensidade - valor
                for i in range(tH):
                    xi = x + i - dH
                    for jm in range(tW):
                        j = tW - j
                        yj = y + j - dW
                        if 0 <= xi < H and 0 <= yj < W:
                            img[xi, yj, c] += dist[i, j] * erro
    return res


def meios_tons(img: Image, dist: ErrorDist, varredura='unidirecional') -> Image:
    if varredura == 'unidirecional':
        return varredura_unidirecional(img, dist)
    elif varredura == 'alternada':
        return varredura_alternada(img, dist)
    else:
        raise ValueError(f'tipo de varredura desconhecida: "{varredura}"')
