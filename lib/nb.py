"""
Tratamento da biblioteca Numba, em caso de não ser encontrada.
"""
# funcionamento correto da biblioteca
try:
    from numba import jit as nb_jit, prange

    USANDO_NUMBA = True

# biblioteca inexistente, cria funções com mesma API, mas que não
# fazem nada no código, apenas continuam a execução e Python puro
except ImportError:
    def nb_jit(*args, **kwargs):
        return lambda func: func

    prange = range
    USANDO_NUMBA = False



def jit(signature: str, *, parallel=False, locals={}):
    """
    Wrapper da ``numba.jit``, com opções padrões diferentes.
    """
    return nb_jit(signature, locals=locals, parallel=parallel, error_model='numpy',
                  nopython=True, nogil=True, cache=True, fastmath=True)
