from tipos import Image, ErrorDist
from typing import Callable, Optional
from enum import IntEnum, unique

import numpy as np
from .nb import jit, prange, SIG_VARREDURA
from .horizontal import varredura_unidirecional, varredura_alternada


@unique
class Varredura(IntEnum):
    unidirecional   = 0
    alternada       = 1


def meios_tons(img: Image, dist: ErrorDist, varredura='unidirecional') -> Image:
    if img.ndim == 3:
        try:
            return meios_tons_cor(img, dist, Varredura[varredura].value)
        except KeyError:
            raise ValueError(f'tipo de varredura desconhecida: "{varredura}"')


    if varredura == 'unidirecional':
        return varredura_unidirecional(img, dist)
    elif varredura == 'alternada':
        return varredura_alternada(img, dist)
    # elif varredura == 'hilbert':
    #     return varredura_hilbert(img, dist)
    else:
        raise ValueError(f'tipo de varredura desconhecida: "{varredura}"')


@jit("uint8[:,:,::1](uint8[:,:,::1], float32[:,::1], uint8)", parallel=True)
def meios_tons_cor(img: Image, dist: ErrorDist, varredura: int) -> Image:

    H, W, _ = img.shape
    res = np.empty((H, W, 3), dtype=np.uint8)

    for ch in prange(3):
        subimg = np.copy(img[..., ch])

        if varredura == 0:
            ans = varredura_unidirecional(subimg, dist)
        else:
            ans = varredura_alternada(subimg, dist)

        res[..., ch] = ans
    return res

