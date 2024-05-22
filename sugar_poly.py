import numpy as np
import sympy as sp
import time as time

from internal import *
from built_ins import *
from head import *
from head_poly import *
from sugar import *
import settings

import inspect

#-------------------------------------------------------------#
# SUGARING POLYNOMIALS IN POLY DOMAIN
#-------------------------------------------------------------#

def sugardegree_poly(t, X):
    """
    Input:
    t: dict of terms in polynomial
    X: list of vars SymPy symbols

    Output:
    degree of sugar of t
    """
    return max([degree(elem,X) for elem in indices(t)])
    
def sp_poly(IJ,G,HT,HC,LCMHT,SugarIJ,bugfix,s):
    """
    Input:
    e.t.c

    Output:
    p: difference of polynomialas G[I] and G[J]
    s: S-polynomial of G[I] and G[J]
    LCMHT, SugarIJ, G: updated
    """
    i = IJ[0]
    j = IJ[1]
    c1 = HC[i]
    c2 = HC[j]
    t1 = HT[i]
    t2 = HT[j]
    L = LCMHT[*bugfix[i,j]]
    cm2, cm1 = gcd(c1,c2,'cm2','cm1')
    u1 = L/t1
    u2 = L/t2
    G[i] = ctmul(cm1,u1,G[i])
    G[j] = ctmul(cm2,u2,G[j])
    p = sub_internal(G[i],G[j]) #sub_internal(ctmul(cm1,u1,G[i]),ctmul(cm2,u2,G[j]))         
    s = SugarIJ[*bugfix[i,j]]

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    ls = [var_name for var_name, var_val in callers_local_vars if var_val is LCMHT]
    if len(ls) == 1:
        str_s = '_'.join([str(elem) for elem in bugfix[i,j]])
        settings.add([sp.Symbol(ls[0]+'_'+str(str_s))])
        LCMHT[*bugfix[i,j]] = settings.return_globals()[ls[0]+'_'+str(str_s)]
    else:
        raise('sp_poly: Multiple instances for LCMHT')
    
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    ls = [var_name for var_name, var_val in callers_local_vars if var_val is SugarIJ]
    if len(ls) == 1:
        str_s = '_'.join([str(elem) for elem in bugfix[i,j]])
        settings.add([sp.Symbol(ls[0]+'_'+str(str_s))])
        SugarIJ[*bugfix[i,j]] = settings.return_globals()[ls[0]+'_'+str(str_s)]
    else:
        print(ls)
        raise('sp_poly: Multiple instances for SugarIJ')

    return p, s, LCMHT, SugarIJ, G

def hred_poly(p,s,F,HT,HC,sugar,X,to,sugarres):
    """
    Input:
    p: SymPy polynomial or expression
    s: sugar of f
    F: list of pols
    HT: leading terms of G
    HC: coefficients of lts of G
    sugar: dict of sugar of pols. G
    X: list of vars SymPy symbols
    to: dict of term order
    sugarres: sugar of reduced pol. -> output

    Output:
    rp: dict of reduced head of polynomial p
    sugarres: sugar of reduced polynomial p
    """

    n = nops(F)
    rp = p
    pt = head_poly(rp,X,to)
    rp = gcont(rp,pt)[1]
    pc = rp[pt]
    sres = s
    oldpt = pt
    reds = []
    for j in range(n):
        if len(indices(rp)) != 0:
            bl, u = divide(pt,HT[j],'u')
            if bl:
                reds = reds + [HT[j]]
                smult = sugarmult(j,sugar,u,X)
                sres = sugaradd(smult,sres)
                m1, m2 = gcd(HC[j],pc,'m1','m2')
                rp = cmul(m1,rp)
                F[j] = ctmul(-m2, u, F[j])
                rp = addto(rp, F[j], 1)
                # print(rp)
                if len(indices(rp)) != 0:
                    pt = head_poly(rp,X,to)
                    rp = gcont(rp,pt)[1]
                    pc = rp[pt]
                else:
                    pt = 0
                j = 0
    sugarres = sres
    if len(reds) != 0:
        print('hred_poly: ',oldpt,' head-reduced to ',pt,' w.r.t. ',reds)
    return rp, sugarres

def sred_poly(p,s,F,HT,HC,sugar,X,to,contin,scale,cont,su):
    """
    Input:
    p: SymPy polynomial or expression in dict representation
    s: sugar of f
    F: list of pols in dict representation
    HT: leading terms of G
    HC: coefficients of lts of G
    sugar: dict of sugar of pols. G
    X: list of vars SymPy symbols
    to: dict of term order
    contin: int
    scale: string for output
    cont: string for output
    su: string for output
    
    Output:
    rp: dict of reduced polynomial p
    scale:
    cont:
    su: sugar of reduced pol. -> output
    """
    n = len(F)
    ascale = 1
    acont = 1
    rp = p
    pt = head_poly(rp,X,to)
    pc = rp[pt]
    sres = s
    reds = []
    for j in range(n):
        while len(indices(rp)) != 0:
            bl, u = divide(pt,HT[j],'u')
            if bl:
                reds.append(HT[j])
                smult = sugarmult(j,sugar,u,X)
                sres = sugaradd(smult,sres)
                m1, m2 = gcd(HC[j],pc,'m1','m2')
                rp = cmul(m1,rp)
                F[j] = ctmul(-m2, u, F[j])
                rp = addto(rp, F[j], 1)
                ascale = m1*ascale
                if len(indices(rp)) != 0:
                    pt = head_poly(rp,X,to)
                    junk, rp = gcont(rp,pt)
                    acont = acont*junk
                    pc = rp[pt]
                else:
                    acont = 0
                j = 0
    su = sres
    if reds == []:
        print('sred_poly: Reductions made w.r.t. ',reds,'.')
    scale = ascale
    cont = acont*contin
    if nops(reds) > 0: 
        return [rp, True], scale, cont, su
    else:
        return [rp, False], scale, cont, su

def reduce_poly(f,s,G,X,to,HT,HC,sugar,lowest,sugarres):
    """
    Input:
    f: dict of SymPy polynomial or expression
    s: sugar of f
    G: list of pols in dict representation
    X - list of vars SymPy symbols
    HT: leading terms of G
    HC: coefficients of lts of G
    sugar: dict of sugar of pols. G
    lowest: smallest leading term of G in to
    sugarres: string for output

    Output:
    k: rescaled polynomial f in dict representation
    sugarres: sugar of reduced poly = S-polynomial of f
    """
    if len(indices(f)) == 0:
        return f, {}
    if nops(G) == 0:
        temp = f
        ht = head_poly(temp,X,to)
        temp = gcont(temp,ht)[1]
        if sign(temp[ht]) == -1:
             temp = cmul(-1,temp)
        sugarres = s
        return temp, sugarres
    stt = time.time()

    # First top reductions
    temp, sres = hred_poly(f,s,G,HT,HC,sugar,X,to,'sres')
    sugarres = sres
    if len(temp.keys()) == 0:
        st = time.time()
        print('reduce_poly: Time ',st,', elapsed in reduce ',st-stt,'.')
        return temp, sugarres
    
    # Reduction of lower monomials
    ht = head_poly(temp,X,to)
    k = {}
    k[ht] = temp[ht]
    hk = ht
    del temp[ht] # temp[ht] = evaln(temp[ht])
    rest = op(temp)
    contin = 1
    while len(rest) != 0:
        temp, scale, tcont, sres = sred_poly(rest,sres,G,HT,HC,sugar,X,to,contin,'scale','tcont','sres')
        temp2 = temp
        temp = op(1,temp2)
        if op(2,temp2):
            ck = gcont #function defined above
            scale = ck*scale
            scale, tcont = gcd(scale,tcont,'scale','tcont')
        if len(indices(temp)) == 0:
            break
        ht = head_poly(temp,X,to)
        k = cmul(scale,k)
        if not(isgreaterorder2(head_poly(temp,X,to),lowest,to)):
            temp = cmul(tcont,temp)
            k = addto(k,temp,1)
            break
        k[ht] = expand(tcont*temp[ht])
        del temp[ht] # temp[ht] = evaln(temp[ht])
        rest = op(temp)
        contin = tcont
    sugarres = sres
    st = time.time()
    print('reduce: Time ',st,', elapsed in reduce ',st-stt,'.')
    if sign(k[hk]) == -1:
        k = cmul(-1,k)
    return k, sugarres