import numpy as np
import sympy as sp

#-------------------------------------------------------------#
# SUGARING POLYNOMIALS FUNCTIONS
#-------------------------------------------------------------#

from internal import *
from built_ins import *
from head import *

def issugargreater(sugar1,sugar2,str):
    """
    Input:
    sugar1: sugar of polynomial in G
    sugar2: sugar of polynomial in G
    str: string

    Output:
    flag, q: flag True if sugar1 greater than or equal to sugar2, q bool to say when they are exactly equal
    """
    if sugar1==sugar2:
        q = True
    else:
        q = False
    if sugar1 >= sugar2:
        return True, q
    else:
        return False, q

def sugardegree(p,X):
    """
    Input:
    p: SymPy expression or polynomial
    X: list of vars SymPy symbols

    OutputL
    d: degree of p
    """
    return degree(p,set(X))

def sugarmult(i,sugar,mterm,X):
    """
    Input:
    i: index int of polynomial G[i]
    sugar: sugar of G
    mterm: SymPy polynomial term / monomial
    X: list of vars of Sympy symbols

    Output:
    surgar of G[i]*mterm
    """
    return sugar[i]+sugardegree(mterm,X)

def sugaradd(sugar1,sugar2):
    """
    Input:
    sugar1: sugar of polynomial in G
    sugar2: sugar of polynomial in G

    Output:
    sugar of sum of polynomials
    """
    return max(sugar1,sugar2)

def sugarSpol(IJ,lcmht,X,HT,sugar):
    """
    Input:
    IJ: [i,j] index pair
    etc.

    Output:
    sugar of S-polynomial S_ij
    """
    s1 = sugarmult(IJ[0],sugar,divide(lcmht[*IJ],HT[IJ[0]],'q')[1],X)
    s2 = sugarmult(IJ[1],sugar,divide(lcmht[*IJ],HT[IJ[1]],'q')[1],X)
    return sugaradd(s1,s2)

def normal_strategy(IJ,KL,LCMHT,bugfix,to):
    """
    Input:
    e.t.c

    Output:
    True if IJ and KL meet normal selection strategy, False if not
    """
    lcmIJ = LCMHT[*bugfix[*IJ]]
    lcmKL = LCMHT[*bugfix[*KL]]
    if lcmIJ == lcmKL:
        if IJ[1]<= KL[1]:
            return True
        else:
            return False
    if isgreaterorder2(lcmIJ, lcmKL, to):
        return False
    else:
        return True

def sugar(IJ,KL,LCMHT,sugarIJ,bugfix,to):
    """
    Input:
    e.t.c

    Output:
    If sugar of IJ equal to sugar KL, check normal selection criteria, else returns True if sugar1 < sugar2 and False otherwise
    """
    s1 = sugarIJ[*bugfix[*IJ]]
    s2 = sugarIJ[*bugfix[*KL]]
    flag, q = issugargreater(s1,s2,'q')
    if q: 
        return normal_strategy(IJ,KL,LCMHT,bugfix,to)
    else:
        return not(flag)

