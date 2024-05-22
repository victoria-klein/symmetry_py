import numpy as np
import sympy as sp

from internal import *
from built_ins import *
from head_inte import *
from sugar import *

import time as time
import inspect

#-------------------------------------------------------------#
# SUGARING POLYNOMIALS IN INTE DOMAIN
#-------------------------------------------------------------#

def sp_inte(IJ,G,HT,HC,LCMHT,SugarIJ,bugfix,s):
    """
    Input:
    e.t.c

    Output:
    p: difference of polynomialas G[I] and G[J]
    s: S-polynomial of G[I] and G[J]
    LCMHT, SugarIJ: updated
    """
    i = IJ[0]
    j = IJ[1]
    c1 = HC[i]
    c2 = HC[j]
    t1 = HT[i]
    t2 = HT[j]
    L = LCMHT[*bugfix[i,j]]
    h = igcd(c1,c2)
    _, cm1 = divide(c2,h,'cm1',list(c2.free_symbols))
    _, cm2 = divide(c1,h,'cm2',list(c1.free_symbols))
    u1 = L/t1
    u2 = L/t2
    p = expand(u1*cm1*G[i] - u2*cm2*G[j])
    s = SugarIJ[*bugfix[i,j]]

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    ls = [var_name for var_name, var_val in callers_local_vars if var_val is LCMHT]
    if len(ls) == 1:
        str_s = '_'.join([str(elem) for elem in bugfix[i,j]])
        settings.add([sp.Symbol(ls[0]+'_'+str(str_s))])
        LCMHT[*bugfix[i,j]] = settings.return_globals()[ls[0]+'_'+str(str_s)]
    else:
        raise('sp_inte: Multiple instances for LCMHT')
    
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    ls = [var_name for var_name, var_val in callers_local_vars if var_val is SugarIJ]
    if len(ls) == 1:
        str_s = '_'.join([str(elem) for elem in bugfix[i,j]])
        settings.add([sp.Symbol(ls[0]+'_'+str(str_s))])
        SugarIJ[*bugfix[i,j]] = settings.return_globals()[ls[0]+'_'+str(str_s)]
    else:
        print(ls)
        raise('sp_inte: Multiple instances for SugarIJ')
    
    return p, s, LCMHT, SugarIJ

def hred_inte(p,s,F,HT,HC,sugar,X,to,sugarres):
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
    rp: reduced head of polynomial p
    sugarres: sugar of reduced polynomial p
    """
    n = nops(F)
    _, rp = divide(p,icontent(p),'rp')
 #   pc := moregroebner[hcoeff](rp,X,to);
 #   pt := moregroebner[hterm](rp,X,to) ;
    # print('mgb rp: ',rp)
    pc = head_inte(rp,X,to)
    # print('mgb pc: ',pc, 'mgb check pc: ',rp.as_poly(*X).LM(order='lex').as_expr())
    pt = pc[1]
    pc = pc[0]
    sres = s
    oldpt = pt
    reds = []
    for j in range(n):
        bl, u = divide(pt,HT[j],'u')
        if bl:
            reds = reds + [HT[j]]
            smult = sugarmult(j,sugar,u,X)
            sres = sugaradd(smult,sres)
            h = univar(HT[j])
            if h != False and (to['ordername']=='plex' or to['ordername']=='tdeg' or to['ordername']=='gradlex'):
            # small modification august 95 KG
                u = list(indets(HT[j]))[0]
                rp = prem(rp,F[j],u)
            else:
                h = igcd(HC[j],pc)
                _, m1 = divide(HC[j],h,'m1')
                _, m2 = divide(pc,h,'m2')
                h = (m1*rp)
                rp = expand((-m2)*u*F[j])
                rp = rp + h
            if rp == 0:
                pt = 0
                break
            _,  rp = divide(rp,icontent(rp),'rp')
        #    pc := `moregroebner/src/inte/hcoeff`(rp,X,to); 
        #    pt := `moregroebner/src/inte/hterm`(rp,X,to);
            pc = head_inte(rp,X,to)
            pt = pc[1]
            pc = pc[0]
            j = 0
    sugarres = sres
    if reds != []:
        print('hred_inte: ',oldpt,' top-reduced to ',pt,' w.r.t. ',reds)
    return rp, sugarres

def sred_inte(p,s,F,HT,HC,sugar,X,to,contin,scale,cont,su):
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
    pc, pt = head_inte(rp,X,to)
    sres = s
    reds = []
    for j in range(n):
        bl, u = divide(pt,HT[j],'u')
        if bl:
            reds.append(HT[j])
            smult = sugarmult(j,sugar,u,X)
            sres = sugaradd(smult,sres)
            h = univar(HT[j])
            if h != False and (to['ordername']=='plex' or to['ordername']=='tdeg' or to['ordername']=='gradlex'):
            # small modification august 95 KG
                u = list(indets(HT[j]))[0]
                rp, junk = prem(rp,F[j],u,'junk')
                ascale = junk*ascale
            else:
                h = igcd(HC[j],pc)
                _, m1 = divide(HC[j],h,'m1')
                _, m2 = divide(pc,h,'m2')
                h = (m1*rp)
                rp = expand((-m2)*u*F[j])
                rp = rp + h
                ascale = m1*ascale
            junk = icontent(rp)
            acont = acont*junk
            if rp == 0:
                break
            _, rp = divide(rp,junk,'rp')
          #  pc := moregroebner[hcoeff](rp,X,termorder); 
          #  pt := moregroebner[hterm](rp,X,termorder);
            pc, pt = head_inte(rp,X,to)
            j = 0
    su = sres
    if reds == []:
        print('sred_inte: Reductions made w.r.t. ',reds,'.')
    scale = ascale
    cont = acont*contin
    return rp, scale, cont, su


def reduce_inte(f,s,G,X,to,HT,HC,sugar,lowest,sugarres):
    """
    Input:
    f: SymPy polynomial or expression
    s: sugar of f
    G: list of pols
    X - list of vars SymPy symbols
    HT: leading terms of G
    HC: coefficients of lts of G
    sugar: dict of sugar of pols. G
    lowest: smallest leading term of G in to
    sugarres: sugar of reduced pol. -> output

    Output:
    -k or k: rescaled polynomial f
    sres: S-polynomial of f
    """
    if f == 0:
        return 0, s
    if len(G) == 0:
        temp = f/icontent(f)
        sugarres = s
        if hcoeff_inte(temp,X,to) < 0:
            return -temp, sugarres
        else:
            return temp, sugarres
    stt  = time.time()
    # First top reductions
    # print('mgb hred: ',f)
    temp, sres = hred_inte(f,s,G,HT,HC,sugar,X,to,'sres')
    # print('mgb hred: ',temp)
    sugarres = sres
    # Reduction of lower monomials
    h = expand(convert(list(head_inte(temp,X,to)),'*'))
    k = h
    rest = temp - h
    contin = 1
    while rest != 0:
        if not(isgreaterorder2(hterm_inte(rest,X,to),lowest,to)):
            k = k + contin*rest
            break
        temp, scale, tcont, sres  = sred_inte(rest,sres,G,HT,HC,sugar,X,to,contin,'scale','tcont','sres')
        if temp != rest:
            ck = icontent(k)
            _, k = divide(k,ck,'k')
            scale = ck*scale
            junk = igcd(scale,tcont)
            _, tcont = divide(tcont,junk,'tcont')
            _, scale = divide(scale,junk,'scale')
        h = expand( convert(list(head_inte(temp,X,to)),'*') )
        k = (scale*k) + (tcont*h)
        rest = temp - h
        contin = tcont
    sugarres = sres
    st = time.time()
 #   if k<>0 then k:=k/icontent(k);fi;
    print('reduce_inte: Time ',st,', elapsed in reduce ',st-stt,'.')
    if hcoeff_inte(k,X,to) < 0:
        return -k, sugarres
    else:
        return k, sugarres