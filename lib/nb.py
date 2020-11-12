try:
    from numba import jit as nb_jit, prange

    USANDO_NUMBA = True

except ImportError:
    def nb_jit(*args, **kwargs):
        return lambda func: func

    prange = range

    USANDO_NUMBA = False


SIG_VARREDURA = "uint8[:,::1](uint8[:,::1], float32[:,::1])"
def jit(signature=SIG_VARREDURA, *, parallel=False, locals={}):
    return nb_jit(signature, locals=locals, parallel=parallel, debug=True,
        nopython=True, nogil=True, cache=True, fastmath=True, error_model='numpy')
