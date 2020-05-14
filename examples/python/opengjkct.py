import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
gjklib = ctypes.CDLL("./opengjk-ctypes.so")

gjklib.csFunction.argtypes = [ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                           ctypes.c_int,
                           ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
gjklib.csFunction.restype = ctypes.c_double


def gjk(bd1, bd2):
    nbd1 = 1
    nbd2 = 1
    if len(bd1.shape) == 2:
        nbd1 = bd1.shape[0]
    if len(bd2.shape) == 2:
        nbd2 = bd2.shape[0]  
    
    q1 = bd1.transpose().copy()
    q2 = bd2.transpose().copy()
    r = gjklib.csFunction(nbd1, q1, nbd2, q2)
    return r
