from typing import Dict, List
from tipos import ErrorDist
import numpy as np


ERR_DIST: Dict[str, ErrorDist] = {}

def distribuicao(nome: str, total: int, data: List[List[float]]) -> None:
    assert total == np.sum(data)

    data = [[x / total for x in row] for row in data]
    dist = np.asarray(data, dtype=np.float32, order='C')

    ERR_DIST[nome] = dist
    for nome in nome.split('_'):
        ERR_DIST[nome] = dist



distribuicao('FLOYD_STEINBERG', 16, [
    [0, 0, 7],
    [3, 5, 1]
])

distribuicao('STEVENSON_ARCE', 200, [
    [ 0,  0,  0,  0,  0, 32,  0],
    [12,  0, 26,  0, 30,  0, 16],
    [ 0, 12,  0, 26,  0, 12,  0],
    [ 5,  0, 12,  0, 12,  0,  5]
])

distribuicao('BURKES', 32, [
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2]
])

distribuicao('SIERRA', 32, [
    [0, 0, 0, 5, 3],
    [2, 4, 5, 4, 2],
    [0, 2, 3, 2, 0]
])

distribuicao('STUCKI', 42, [
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1]
])

distribuicao('JARVIS_JUDICE_NINKE', 48, [
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
])
