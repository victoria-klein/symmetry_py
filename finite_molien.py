import numpy as np
import sympy as sp

#-------------------------------------------------------------#
# MOLIEN SERIES FOR FINITE GROUPS
#-------------------------------------------------------------#

def molien(g,lam):
    s = molien_(g,lam)
    s = poincaresimp(s)
    return(s)

def molien_(g,lam):
    s = 0
    i_d = sp.eye(g['allelements']['_r0'].shape[0])
    for elem in g['allelements'].values():
        s = s+1/(i_d-lam*elem).det()
        # s = sp.factor(s)
    s = s/order(g)
    return(s)

def order(g):
    return len(g['allelements'])

def poincaresimp(res,lam = None):
    r = sp.factor(res)
    nu, de = sp.fraction(r)
    n1 = 1
    k = 20
    if lam == None:
        lam = res.free_symbols.pop()
    while k>0:
        nnew, r = sp.div(de,(1-lam**k), domain='QQ')
        if r == 0:
            n1 = n1*(1-(lam**k))
            de = nnew
        else:
            k -= 1
    return(nu/(de*n1))
