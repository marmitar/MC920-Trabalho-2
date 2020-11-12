"""
Direções locais de aplicação das distribuições e
rotações das matrizes das distribuições de erros.
"""
from enum import IntEnum, unique
from typing import Tuple
import numpy as np
from tipos import ErrorDist
from .nb import jit


@unique
class Dir(IntEnum):
    """
    Representação das direções locais ao longo da aplicação
    do pontilhado.

    Variantes
    ---------
    direita
        Direção +x.
    esquerda
        Direção -x.
    cima
        Direção -y.
    baixo
        Direção +y.
    """
    direita = 0
    esquerda = 1
    cima = 2
    baixo = 3


# tupla com as matrizes de erros rotacionadas
# as posições seguem as direções acima
ErrorDistDir = Tuple[ErrorDist, ErrorDist, ErrorDist, ErrorDist]

@jit("UniTuple(float32[:,::1], 4)(float32[:,::1])")
def err_dist_direcoes(dist: ErrorDist) -> ErrorDistDir:
    """
    Rotação da matriz das distribuições de erros para aplicação
    em direções de aplicação diferentes, descritas em ``Dir``.

    Parâmetros
    ----------
    dist: np.ndarray
        Matriz 2D com as pesos da distribuição de erro.

    Retorno
    -------
    out: tuple
        Tupla com as quatro rotações da matriz, seguindo as direções.
    """
    # sem rotação
    direita = dist
    # rotação de 90
    baixo = np.copy(dist[::-1].T)
    # 180 graus
    esquerda = np.copy(dist[::-1,::-1])
    # 270 graus
    cima = np.copy(dist[:,::-1].T)

    # posições de acordo com `Dir`
    return direita, esquerda, cima, baixo
