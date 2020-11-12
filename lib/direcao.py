from enum import IntEnum, unique
from typing import Tuple
import numpy as np
from tipos import ErrorDist
from .nb import jit


@unique
class Dir(IntEnum):
    direita = 0
    esquerda = 1
    cima = 2
    baixo = 3


ErrorDistDir = Tuple[ErrorDist, ErrorDist, ErrorDist, ErrorDist]

@jit("UniTuple(float32[:,::1], 4)(float32[:,::1])")
def err_dist_direcoes(dist: ErrorDist) -> ErrorDistDir:
    direita = dist
    esquerda = np.copy(np.flip(dist))
    cima = np.copy(dist[:,::-1].T)
    baixo = np.copy(dist[::-1].T)

    return direita, esquerda, cima, baixo
