import numpy as np
import sympy as sp

from internal import *
from built_ins import *
from head import *
import settings

#-------------------------------------------------------------#
# HEAD FOR POLY DOMAIN
#-------------------------------------------------------------#


def head2_poly(monomset,X,to):
    """
    Input:
    monomset: list of SymPy polynomial terms / monomials
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    mset: set of head term of monomset w.r.t term order
    """
    # monomset - {a X^i, ...} a integer
    mset = []

    if len(monomset) == 1:
        return monomset

    if to['ordername']=='gradlex': 
        vs = eval(to['vs'])
        temp = monomset[0]
        mset = set([temp])
        d = degree(temp,set(vs))
        for i in range(1,len(monomset)):
            temp = monomset[i]
            tempdeg = degree(temp,set(vs))
            if tempdeg > d: 
                d = tempdeg
                mset = set([temp])
            elif tempdeg == d:
                mset = union(mset,set([temp]))
        if nops(mset)>1:
            mset = head2_poly(mset,X,{'ordername': 'plex','vs': vs})
        return mset

    elif to['ordername']=='tdeg': 
        vs = eval(to['vs'])
        temp = monomset[0]
        mset = set([temp])
        d = degree(temp,set(vs))
        for i in range(1,len(monomset)):
            temp = monomset[i]
            tempdeg = degree(temp,set(vs))
            if tempdeg > d: 
                d = tempdeg
                mset = set([temp])
            elif tempdeg == d:
                mset = union(mset,set([temp]))
        if nops(mset)>1:
            ma = matrix(len(vs),len(vs),0)
            for j in range(len(vs)-1):
                ma[j,len(vs)-1-j] = -1
            mset = head2_poly(mset,X,{'ordername': 'mat','vs': vs,'mat': ma})
        return mset

    elif to['ordername']=='plex': 
        vs = to['vs']
        p = convert(monomset,'+')
    #    p:=expand(p);
    #    p:=collect(p,vs); 
        coe, term = lcoeff(p,vs,'term')
        p = expand(coe*term)
        if is_type(p,'+'):
            mset = set(op(p))
        else:
            mset = set([p])
        return mset

    elif to['ordername']=='mat': 
        vs = to['vs']
        m = to['mat']
        mset = monomset
        grading = {}
        for i in range(rowdim(m)):
            for j in range(len(vs)):
                grading[str(vs[j])] = m[i,j]
            temp = list(mset)[0]
            mset2 = set([temp])
            d =  degree_internal(temp,vs,grading)
            # print('mset2d',mset2,d)
            for j in range(1,nops(mset)):
                temp = list(mset)[j]
                tempdeg = degree_internal(temp,vs,grading)
                # print('tempd',temp,tempdeg)
                if tempdeg > d: 
                    d = tempdeg
                    mset2 = set([temp])
                elif tempdeg == d:
                    mset2 = union(mset2,set([temp]))
            mset = mset2
            if nops(mset)==1:
                break
        if 'order1' in to.keys(): 
            mset = head2_poly(mset,X,to['order1'])
        return mset

    elif to['ordername']=='blocked': 
        mset = head2_poly(monomset,X,to['order1'])
        if nops(mset)>1:
            mset = head2_poly(mset,X,to['order2'])
        else:
            raise('head2_poly: Unknown term order ',to, '.')    
        return mset

def thterm_poly(s,X,to):
    """
    Input:
    s: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    ht: head term in X of s
    """
    newset = head2_poly(s,X,to)
    if nops(newset)>1:
       newset = head2_poly(newset,X,{'vs': X,'ordername': 'plex'})
    if nops(newset)>1:
         raise('thterm_poly: Non-uniqueness.')
    hc, ht = lcoeff(list(newset)[0],X,'ht')
    return ht

def head_poly(t,X,to):
    """
    Input:
    t: dict of SymPy polynomial or expression
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    ht: total degree head term in X of s
    """
    return thterm_poly(list(set(indices(t))),X,to)