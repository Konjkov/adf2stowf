#!/usr/bin/env python3.9

# (C) 2008 Norbert Nemec
# This file is part of the CASINO distribution.
# Permission is given to use the script along with the CASINO program and modify
# it for personal use.

import re
import sys
try:
    import numpy
except ImportError:
    print('This program requires the numpy library, which could not be found.')
    sys.exit()

def floatX(x):
    x = re.sub("E*-","E-",x)
    if x[0] == "E":
        x = x[1:]
    return float(x)

def splitN(s,N):
    return [ s[i:i+N].strip() for i in range(0,len(s),N) ]

def intX(s):
    if s=="**********":
        return -2**31
    return int(s)

def adfread(infname="TAPE21.asc",outfile=None):

    try:
        lines = open(infname,encoding='latin-1').readlines()
    except(FileNotFoundError,IsADirectoryError):
        print('File '+infname+' could not be read.')
        sys.exit()

    data = {}

    try:
        i = 0
        lastgroup = ""

        while i<len(lines):
            group = lines[i].strip() ; i += 1
            key = lines[i].strip() ; i += 1
            len1,len2,typ = [ int(s) for s in lines[i][:-1].split() ] ; i += 1
            # assert len1 == len2
            # len1: reserved length, len2: used length
            if typ == 1: # integer
                value = []
                while len(value) < len2:
                    value += [ intX(s) for s in splitN(lines[i][:-1],12) ] ; i += 1
                assert len(value) == len2
                value = numpy.array(value,int)
            elif typ == 2: # float
                value = []
                while len(value) < len2:
                    value += [ floatX(s) for s in splitN(lines[i][:-1],28) ] ; i += 1
                assert len(value) == len2
                value = numpy.array(value,float)
            elif typ == 3: # string
                value = ""
                while len(value) < len2:
                    value += lines[i][:-1] ; i += 1
                assert len(value) == len2
                value = [ value[160*j:160*(j+1)].strip() for j in range((len(value)+159)//160) ]
            elif typ == 4: # bool
                value = []
                while len(value) < len2:
                    value += [ {"T": True, "F": False}[s] for s in lines[i][:-1] ] ; i += 1
                assert len(value) == len2
            else:
                raise ValueError("type 1..4 expected")

            if len1 == 0:
                i += 1

            if outfile is not None:
                if group != lastgroup:
                    outfile.write("\n"+group+"\n")
                    lastgroup = group
                vallen = len(value)
                valstr = str(value)
                if '\n' in valstr:
                    outfile.write("  "+key+" = {"+str(vallen)+"}\n\t"
                                  +re.sub('\n','\n\t',valstr)+"\n")
                else:
                    outfile.write("  "+key+" = "+valstr+"\n")

            if group not in data:
                data[group] = {}

            assert key not in data[group]
            data[group][key] = value

    except:
        for x in range(max(0,i-3),i):
            print("   |"+lines[x][:-1])
        print(">>>>"+lines[i][:-1])
        for x in range(i+1,min(i+4,len(lines))):
            print("   |"+lines[x][:-1])
        raise

    return data


if __name__=="__main__":
    ascfname = sys.argv[1]
    assert ascfname[-4:] == ".asc"
#    numpy.set_printoptions(threshold=10)
    data = adfread(infname=ascfname,outfile=open(ascfname[:-4]+".out","w",encoding='latin-1'))
