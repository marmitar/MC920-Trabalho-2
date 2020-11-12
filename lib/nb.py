try:
    from numba import jit as nb_jit, prange, cfunc as  nb_cfunc

except ImportError:
    class GhostType:
        def __getattribute__(self, key):
            return self
        def __getitem__(self, key):
            return self
        def __call__(self, *args):
            return self

    def nb_jit(*args, **kwargs):
        return lambda func: func

    prange = range
    nb_cfunc = nb_jit


SIG_VARREDURA = "uint8[:,::1](uint8[:,::1], float32[:,::1])"
def jit(signature=SIG_VARREDURA, *, parallel=False, locals={}):
    return nb_jit(signature, locals=locals, parallel=parallel, debug=True,
        nopython=True, nogil=True, cache=True, fastmath=True, error_model='numpy')


def cfunc(signature: str, *, locals={}):
    return nb_cfunc(signature, locals=locals, cache=True, nopython=True)
