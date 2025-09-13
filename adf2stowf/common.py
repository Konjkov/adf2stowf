#!/usr/bin/env python3.9

# (C) 2008 Norbert Nemec
# This file is part of the CASINO distribution.
# Permission is given to use the script along with the CASINO program and modify
# it for personal use.

import sys
try:
    import numpy
except ImportError:
    print('This program requires the numpy library, which could not be found.')
    sys.exit()

pi = numpy.pi

F2P_bool={".false.":False,".true.":True}
P2F_bool={False:".false.",True:".true."}

def mapunion(a,b):
    res = a.copy()
    res.update(b)
    return res

import inspect

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

All = slice(None,None,None)

def integral(fx,x):
    assert x.shape == (fx.shape[:1])
    return (0.5*(fx[1:,...] + fx[:-1,...])*(x[1:]-x[:-1])[(All,) + (None,)*(len(fx.shape)-1)]).sum(axis=0)

def cyl_integral(f,phi,rho,z):
    return integral(integral(integral(rho[(None,All,) + (None,)*(len(f.shape)-2)]*f,phi),rho),z)

def weave_inline(support_code,code,dict,defs=[]):
    try:
        import weave
    except ImportError:
        print('This program requires the weave library, which could not be')
        print('found.')
        sys.exit()
    weave.inline(
        headers = ["<cstdlib>","<cmath>"],
        support_code = "\n".join(["#define "+d for d in defs] + [support_code]),
        code = "\n".join(["// #define "+d for d in defs] + [code]),
        arg_names = dict.keys(),
        local_dict = dict,
        type_converters = weave.converters.blitz,
        compiler = 'gcc',
        extra_compile_args = ["-Wno-all"],
        verbose = 1,
    )
