from tipos import Image, ErrorDist
from typing import Callable, Optional
from enum import IntEnum, unique

import numpy as np
from .nb import jit, prange, SIG_VARREDURA
from .horizontal import varredura_unidirecional, varredura_alternada
from .hilbert import varredura_hilbert, hilbert_indices


@unique
class Varredura(IntEnum):
    unidirecional   = 0
    alternada       = 1
    hilbert         = 2

    @property
    def function(self) -> Callable[[Image, ErrorDist], Image]:
        if self == unidirecional:
            return varredura_unidirecional
        elif self == alternada:
            return varredura_alternada
        elif self == hilbert:
            return varredura_hilbert

    def __call__(self, img: Image, dist: ErrorDist) -> Image:
        return self.function(img, dist)


def meios_tons(img: Image, dist: ErrorDist, varredura=Varredura) -> Image:
    if img.ndim == 3:
        return meios_tons_cor(img, dist, varredura.value)
    else:
        return varredura(img, dist)


@jit("uint8[:,:,::1](uint8[:,:,::1], float32[:,::1], uint8)", parallel=True)
def meios_tons_cor(img: Image, dist: ErrorDist, varredura: int) -> Image:

    H, W, _ = img.shape
    res = np.empty((H, W, 3), dtype=np.uint8)
    if varredura == 2:
        idx = hilbert_indices(H, W)

    for ch in prange(3):
        subimg = np.copy(img[..., ch])

        if varredura == 0:
            ans = varredura_unidirecional(subimg, dist)
        elif varredura == 1:
            ans = varredura_alternada(subimg, dist)
        else:
            ans = varredura_hilbert(subimg, dist, idx)

        res[..., ch] = ans
    return res

