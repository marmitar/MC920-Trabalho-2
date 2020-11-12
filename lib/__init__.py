from tipos import Image, ErrorDist
from typing import Callable, Optional
from enum import IntEnum, unique

import numpy as np
from .nb import jit, prange, USANDO_NUMBA
from .horizontal import varredura_unidirecional, varredura_alternada
from .hilbert import varredura_hilbert, hilbert_indices, dist_direcoes

from numba import jitclass


@unique
class Varredura(IntEnum):
    unidirecional   = 0
    alternada       = 1
    hilbert         = 2

    def __call__(self, img: Image, dist: ErrorDist) -> Image:
        if self == unidirecional:
            return varredura_unidirecional(img, dist)
        elif self == alternada:
            return varredura_alternada(img, dist)
        elif self == hilbert:
            return varredura_hilbert(img, dist)

    def __str__(self) -> str:
        """
        Nome que aparece na linha de comando.
        """
        return self.name


def meios_tons(img: Image, dist: ErrorDist, varredura=Varredura) -> Image:
    if img.ndim == 3:
        return meios_tons_cor(img, dist, varredura.value)
    else:
        return varredura(img, dist)


@jit("uint8[:,:,::1](uint8[:,:,::1], float32[:,::1], uint8)", parallel=True)
def meios_tons_cor(img: Image, dist: ErrorDist, varredura: Varredura) -> Image:

    H, W, _ = img.shape
    res = np.empty((H, W, 3), dtype=np.uint8)

    if varredura == Varredura.hilbert:
        idx = hilbert_indices(H, W)
        dists = dist_direcoes(dist)

    for ch in prange(3):
        subimg = np.copy(img[..., ch])

        if varredura == Varredura.unidirecional:
            ans = varredura_unidirecional(subimg, dist)
        elif varredura == Varredura.alternada:
            ans = varredura_alternada(subimg, dist)
        else:
            ans = varredura_hilbert(subimg, None, idx, dists)

        res[..., ch] = ans
    return res

