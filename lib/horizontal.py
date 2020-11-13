"""
Varreduras horizontais: undirecional ou alternada.
"""
from tipos import Image, ErrorDist
import numpy as np
from .nb import jit


# # # # # # # # #
# Unidirecional #

@jit("uint8[:,::1](uint8[:,::1], float32[:,::1])")
def varredura_unidirecional(img: Image, dist: ErrorDist) -> Image:
    """
    Varredura unidirecional pela imagem, reduzindo os níveis de cinza
    e redistribuindo os erros em relação a imagem original.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz 2D com `uint8` em ordem row-major, representando a imagem.
    dist: np.ndarray
        Matriz 2D com `float32` em ordem row-major, representando a
        distribuição de erros que deve ser feita.

    Retorno
    -------
    out: np.ndarray
        Imagem resultante. Matriz 2D com `uint8` em ordem row-major.
    """
    # dimensões da imagem
    H, W = img.shape
    # dimensões da distribuição de erros
    tH, tW = dist.shape
    # deslocamento em `x` do início da dist.
    dW = (tW - 1) // 2

    # imagem em ponto flutuante, para não arredondar erros
    img = img.astype(np.float32)
    # imagem resultante
    res = np.empty((H, W), dtype=np.uint8)

    # aplicação em cada pixel
    for y in range(H):
        for x in range(W):
            # meio ton simples
            intensidade = img[y, x]
            if intensidade < 128.0:
                res[y, x] = 0
                valor = 0.0
            else:
                res[y, x] = 1
                valor = 255.0

            # carregamento do erro
            erro = intensidade - valor
            for i in range(tH):
                yi = y + i
                for j in range(tW):
                    xj = x + j - dW
                    # cuidado com acesso out-of-bounds
                    if 0 <= yi < H and 0 <= xj < W:
                        img[yi, xj] += dist[i, j] * erro
    return res


# # # # # # #
# Alternada #

@jit("uint8[:,::1](uint8[:,::1], float32[:,::1])")
def varredura_alternada(img: Image, dist: ErrorDist) -> Image:
    # dimensões da imagem
    H, W = img.shape
    # dimensões da distribuição de erros
    tH, tW = dist.shape
    # deslocamento da máscara de dist. erros
    dW = (tW - 1) // 2

    # imagem em ponto flutuante
    img = img.astype(np.float32)
    # imagem resultante
    res = np.empty((H, W), dtype=np.uint8)

    for ym in range((H + 1)//2):
        # aplicação nas linhas pares
        y = 2 * ym
        for x in range(W):
            # semelhante ao unidirecional
            intensidade = img[y, x]
            if intensidade < 128.0:
                res[y, x] = 0
                valor = 0.0
            else:
                res[y, x] = 1
                valor = 255.0

            # carregamento do erro
            erro = intensidade - valor
            for i in range(tH):
                yi = y + i
                for j in range(tW):
                    xj = x + j - dW
                    if 0 <= yi < H and 0 <= xj < W:
                        img[yi, xj] += dist[i, j] * erro

        if y + 1 == H:
            break
        # aplicação nas linhas ímpares
        y += 1
        for xm in range(W):
            # ordem invertida na linha
            x = W - 1 - xm

            intensidade = img[y, x]
            if intensidade < 128.0:
                res[y, x] = 0
                valor = 0.0
            else:
                res[y, x] = 1
                valor = 255.0

            # carregamento do erro
            erro = intensidade - valor
            for i in range(tH):
                yi = y + i
                for j in range(tW):
                    # aplicação invertida da máscara de erros
                    xj = x + dW - j
                    if 0 <= yi < H and 0 <= xj < W:
                        img[yi, xj] += dist[i, j] * erro
    return res