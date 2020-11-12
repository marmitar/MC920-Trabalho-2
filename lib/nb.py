try:
    from numba import jit as nb_jit, prange

    USANDO_NUMBA = True

except ImportError:
    def nb_jit(*args, **kwargs):
        return lambda func: func

    prange = range
    USANDO_NUMBA = False


def jit(signature: str, *, parallel=False, locals={}):
    return nb_jit(signature, locals=locals, parallel=parallel,
        nopython=True, nogil=True, cache=True, fastmath=True, error_model='numpy')
