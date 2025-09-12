#!/usr/bin/env python3

# (C) 2008 Norbert Nemec
# This file is part of the CASINO distribution.
# Permission is given to use the script along with the CASINO program and modify
# it for personal use.

import sys
try:
    import numpy as np
except ImportError:
    print('This program requires the numpy library, which could not be found.')
    sys.exit()

pi = np.pi

F2P_bool={".false.":False,".true.":True}
P2F_bool={False:".false.",True:".true."}

All = slice(None,None,None)

def integral(fx,x):
    assert x.shape == (fx.shape[:1])
    return (0.5*(fx[1:,...] + fx[:-1,...])*(x[1:]-x[:-1])[(All,) + (None,)*(len(fx.shape)-1)]).sum(axis=0)

def cyl_integral(f,phi,rho,z):
    return integral(integral(integral(rho[(None,All,) + (None,)*(len(f.shape)-2)]*f,phi),rho),z)
