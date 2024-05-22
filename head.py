import numpy as np
import sympy as sp

from internal import *
from built_ins import *
import settings

#-------------------------------------------------------------#
# HEAD
#-------------------------------------------------------------#

def head3(p,X,to):
    """
    Input:
    p: SymPy polynomial or expression
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    s: Leading monomial of p for term order mat or blocked
    """

    if to['ordername']=='gradlex': 
        vs = to['vs']
        s = expand(p)
        t = 0
        d = degree(s,set(vs))
        if is_type(s,'+'):
            for i in range(nops(s)):
                temp = op(i+1,s)
                # print(s,i,temp)
                if degree(temp,set(vs)) == d:
                    t = t + temp
        else:
            return s
        if is_type(t,'+'):
            return head3(t,X,{'ordername':'plex','vs':vs})
        else:
            return t
        
    elif to['ordername']=='tdeg': 
        vs = to['vs']
        s = expand(p)
        t = 0
        d = degree(s,set(vs))
        if is_type(s,'+'):
            for i in range(nops(s)):
                temp = op(i+1,s)
                if degree(temp,set(vs)) == d:
                    t = t + temp
        else:
            return s
        if is_type(t,'+'):
            n = len(vs)
            Y = [vs[n-1-i] for i in range(n)]
            hc, ht = tcoeff(t,Y,'ht')
            t = expand(hc*ht)
            return t
        else:
            return t

    elif to['ordername']=='plex': 
        vs = to['vs']
        s = expand(p)
        hc, ht = lcoeff(s,vs,'ht')
        t = expand(hc*ht)
        return t

    elif to['ordername']=='mat': 
        vs = to['vs']
        m = to['mat']
        s = expand(p)
        grading = {}
        for i in range(rowdim(m)):
            for j in range(len(vs)):
                grading[str(vs[j])] = m[i,j]
            if not(is_type(s,'+')):
                break
            args = s.args
            temp = args[0]
            t = temp
            d = degree_internal(temp,vs,grading)
            for j in range(1,len(args)):
                temp = args[j]
                tempdeg = degree_internal(temp,vs,grading)
                if tempdeg > d: 
                    d = tempdeg
                    t = temp
                elif tempdeg == d:
                    t = t + temp
            s = expand(t)
            if not(is_type(s,'+')):
                break
        if 'order1' in to.keys(): 
            s = head3(s,X,to['order1'])
        return s

    elif to['ordername']=='blocked': 
        s = head3(p,X,to['order1'])
        if is_type(s,'+'):
            s = head3(s,X,to['order2'])
        return s

    else:
        raise('head3: Unknown term order.')

def isgreaterorder2(m1,m2,to):
    """
    Input:
    m1: SymPy polynomial term
    m2: SymPy polynomial term
    to: dict of term order

    Output:
    bool: False if m2 is higher degree than m1 with respect to term ordering, True if not
    """
    if head3(m1+m2,to['vs'],to) == m2:
        return False
    else:
        return True