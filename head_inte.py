import numpy as np
import sympy as sp

from internal import *
from built_ins import *
from head import *
import settings

#-------------------------------------------------------------#
# HEAD FOR INTE DOMAIN
#-------------------------------------------------------------#



def head_inte(p,X,to):
    """
    Input:
    p: SymPy polynomial or expression
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    hc: leading coefficient
    ht: leading monomial
    """

    p = sp.sympify(p)

    if to['ordername']=='tdeg':
        return p.as_poly(*X).LC(order='grevlex'),p.as_poly(*X).LM(order='grevlex').as_expr()
    elif to['ordername']=='gradlex':
        return p.as_poly(*X).LC(order='grlex'),p.as_poly(*X).LM(order='grlex').as_expr()
    if to['ordername']=='plex':
        return p.as_poly(*X).LC(order='lex'),p.as_poly(*X).LM(order='lex').as_expr()
    elif to['ordername']=='mat' or to['ordername']=='blocked':
        s = head3(p,X,to)
        if is_type(s,'+'):
            s = head3(s,X,{'vs':X,'ordername':'plex'})
        if is_type(s,'+'):
            raise('head_inte: Nonuniqueness.')
        hc,ht = lcoeff(s,X,'ht')
        return hc,ht
    else:
        raise('head_inte: Unknown term order.')

def hterm_inte(p,X,to):
    """
    Input:
    p: SymPy polynomial or expression
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    Leading monomial
    """ 
    return head_inte(p,X,to)[1]

def hcoeff_inte(p,X,to):
    """
    Input:
    p: SymPy polynomial or expression
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    Coefficient of leading monomial
    """
    return head_inte(p,X,to)[0]