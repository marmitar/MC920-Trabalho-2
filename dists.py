"""
Definição das ditribuições de erro para o trabalho 2.
"""
from typing import Dict, List
from tipos import ErrorDist
from typing import Optional
import numpy as np


# # # # # # # # # #
# Acesso por nome #

# Dicinário para acesso por nome.
ERR_DIST: Dict[str, Optional[ErrorDist]] = {
    'NONE': None
}


def distribuicao(nome: str, total: int, data: List[List[int]]) -> None:
    """
    Insere a distribuição no dicionário, usando o nome em conjunto dos seus
    idealizadores (ex. 'FLOYD_STEINBERG') e o nome separado de cada um (
    'FLOYD' e 'STEINBERG').

    Parâmetros
    ----------
    nome: str
        Nome da distribuição de erros.
    total: int
        Soma total da distribuição.
    data: list
        Os pesos da vizinhança na distribuição de erros.
    """
    # evitando erros simples na distribuição
    assert total == np.sum(data)

    # monta a matriz da distribuição, como float32 para o Numba
    data = [[x / total for x in row] for row in data]
    dist = np.asarray(data, dtype=np.float32, order='C')

    # insere a dstribuição em cada um dos seus nomes
    ERR_DIST[nome] = dist
    for nome in nome.split('_'):
        ERR_DIST[nome] = dist


# # # # # # # # # # # # # # #
# Distribuições do trabalho #

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
