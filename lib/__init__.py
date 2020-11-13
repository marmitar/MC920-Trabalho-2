"""
Operação de pontilhado e modos de varredura.
"""
from typing import Optional
from enum import IntEnum, unique
from tipos import Image, ErrorDist
import numpy as np

from .nb import jit, prange, USANDO_NUMBA
from .horizontal import varredura_unidirecional, varredura_alternada
from .direcao import err_dist_direcoes
from .hilbert import varredura_hilbert, hilbert_indices
from .espiral import varredura_espiral


# # # # # # # # # # # #
# Modos de Varredura  #

@unique
class Varredura(IntEnum):
    """
    Modos de varredura.

    Variantes
    ---------
    unidirecional
        Aplica a distribuição dos erros da esquerda para a direita,
        linha a linha.
    alternada
        Aplica a distribução linha a linha, alternando a direção
        tanto de aplicação quanto da distribuição.
    hilbert
        Aplica a distribuição seguindo uma curva de Hilbert.
    espiral
        Tentativa de aplicar os erros em varredura espiral.
    """
    unidirecional   = 0
    alternada       = 1
    hilbert         = 2
    espiral         = 3

    def __call__(self, img: Image, dist: ErrorDist) -> Image:
        """
        Chama a função de varredura com as transformações necessárias.
        """
        if self == Varredura.unidirecional:
            return varredura_unidirecional(img, dist)
        elif self == Varredura.alternada:
            return varredura_alternada(img, dist)
        elif self == Varredura.hilbert:
            return varredura_hilbert(img, err_dist_direcoes(dist), None)
        elif self == Varredura.espiral:
            return varredura_espiral(img, err_dist_direcoes(dist))

    def __str__(self) -> str:
        """
        Nome que aparece na linha de comando.
        """
        return self.name


# # # # # # # # # # # # # # #
# Aplicação dos meios-tons  #

def meios_tons(img: Image, dist: Optional[ErrorDist]=None, varredura=Varredura) -> Image:
    """
    Aplicação da técnica de meios-tons seguindo uma distribuição e um padrão
    de varredura. Quando a distribuição não é dada, faz a conversão de escalas
    de cinza simples.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz de 2 (em escalas de cinza) ou 3 (com canais RGB) dimensões que
        representa a imagem.
    dist: np.ndarray, opcional
        Matriz com as distribuições de erro à serem aplicadas. Assume largura
        ímpar, iniciando a aplicação na posição intermediária da primeira linha.
    varredura: Varredura, opcional
        Ordem de aplicação na imagem.

    Retorno
    -------
    out: np.ndarray
        Imagem resultante do pontilhado.
    """
    if dist is None:
        # sem distribuição de erro, conversão simples
        return np.where(img < 128, 0, 1).astype(np.uint8)
    elif img.ndim == 3:
        # aplicação em imagens RGB
        return meios_tons_colorida(img, dist, varredura.value)
    else:
        # imagens em escala de cinza
        return varredura(img, dist)


@jit("uint8[:,:,::1](uint8[:,:,::1], float32[:,::1], uint8)", parallel=True)
def meios_tons_colorida(img: Image, dist: ErrorDist, varredura: int) -> Image:
    """
    Especialização para imagens RGB com o Numba. Cada canal de cor é resolvido
    em paralelo.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz 3D com `uint8` em ordem row-major.
    dist: np.ndarray
        Matriz 2D com `float32` em ordem row-major.
    varredura: int
        Ordem de aplicação na imagem, de acordo com a classe ``Varredura``.

    Retorno
    -------
    out: np.ndarray
        Matriz 3D com `uint8` em ordem row-major.
    """

    H, W, _ = img.shape
    # imagem resultante
    res = np.empty((H, W, 3), dtype=np.uint8)

    # índices da curva de Hilbert
    if varredura == Varredura.hilbert:
        idx = hilbert_indices(H, W)
    # distribuição rotacionada em 0, 90, 180 e 270 graus
    if varredura >= Varredura.hilbert:
        dists = err_dist_direcoes(dist)

    # aplica os canais em paralelo
    for ch in prange(3):
        # copia para manter o row-major
        subimg = np.copy(img[..., ch])

        # aplicação da varredura em canal único
        if varredura == Varredura.unidirecional:
            ans = varredura_unidirecional(subimg, dist)
        elif varredura == Varredura.alternada:
            ans = varredura_alternada(subimg, dist)
        elif varredura == Varredura.hilbert:
            ans = varredura_hilbert(subimg, dists, idx)
        else:
            ans = varredura_espiral(subimg, dists)
        # escreve o resultado no canal correto
        res[..., ch] = ans

    return res
