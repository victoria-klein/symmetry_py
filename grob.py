import numpy as np
import sympy as sp
import time as time

from internal import *
from built_ins import *
from grob_helpers import *
from grob_inte import *
from grob_poly import *
from grob_inte_sympy import grob_inte_sympy

#-------------------------------------------------------------#
# FUNCTIONS FOR GRADGROEBNER
#-------------------------------------------------------------#
from sympy.printing.aesaracode import aesara_function

def aesera_sp_grob(F, X, order):
    func = list(sp.groebner(F, *X, order).args[0])
    return aesera_function(func)

def grob(F, X, to=None, *adds):
    """
    Input:
    F: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    to: dict of term order ['lex'',grlex','grevlex'] ['plex'',tdeg','gradlex','mat','blocked']
    adds: additional gradings

    Output:
    G: reduced Grobner basis
    """
    if isinstance(F, list) and isinstance(X,list):
        if len(adds) == 0:
            # if to['ordername'] in set(['plex','gradlex','tdeg','mat','blocked']):
            if to['ordername'] in set(['plex','gradlex','tdeg']):
                print('grob: Calculating Grobner basis using SymPy.')
                if to['ordername'] == 'plex':
                    # G = list(sp.groebner(F, *X, order='lex',method='f5b').args[0])
                    G = list(grob_inte_sympy(F, *X, order='lex').args[0])
                elif to['ordername'] == 'gradlex':
                    # G = list(sp.groebner(F, *X, order='grlex',method='f5b').args[0])
                    G = list(grob_inte_sympy(F, *X, order='grlex').args[0])
                elif to['ordername'] == 'tdeg':
                    # G = list(sp.groebner(F, *X, order='grevlex',method='f5b').args[0])
                    G = list(grob_inte_sympy(F, *X, order='grevlex').args[0])
                # # Checking:
                # G_internal = grob_internal_inte(F,X,to)
                # if set(G_internal) != set(G):
                #     print('grob: Mismatch in Grobner basis internally ',G_internal,' and SymPy ',G,' for list ',F,X,to)
                #     raise('grob: Mismatch.')
            elif to['ordername'] in set(['mat','blocked']):
                G = grob_internal_inte(F,X,to)
            else:
                raise('grob: Unknown term order.')
        else:
            # if len(grads) == 1 and all(grads[0][str(elem)] == 1 for elem in X):
            #     return grob(F, X, to)
            if to['ordername'] in set(['plex','gradlex','tdeg','mat','blocked']):
                # raise('grob: Using built-in GB function.')
                G = grob_internal_inte(F,X,to,*adds)
            else:
                raise('grob: Unknown term order.')
    else:
        raise('grob: Grobner basis can only be called on list of polynomials and list of generators.')
    print('grob: Grobner basis calculated.')
    return G

def gradgroebner(*args):
    """
    Input:
    polys: list or set of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    to: dict of term order
    grads2: dict of grading {'minint': int, 'maxint': int, 'elem': non-neg int for every elem in vs}

    Output:
    answer: Grobner basis with respect to term order and additional gradings in grads2
    """
    
    nargs = len(args)

    if nargs < 3:
        raise('gradgroebner: Too few arguments for gradgroebner.')
    elif nargs > 3:
        grads2 = args[3:]
    polys, X, to = args[0], args[1], args[2]

    if not(all(isinstance(elem, sp.Symbol) for elem in X)):
        raise('gradgroebner: Bad variable list X for gradgroebner, should be SymPy symbols.')
    # if not(istermorder(to)):
    #     raise('The third argument for gradgroebner should be a dict including the term order.')
    if not(set(X) == set(to['vs'])):
        raise('gradgroebner: The variables in the list X in gradgroebner and those in the termorder should be the same.')
    if not(all(is_type(elem,'polynom',X) for elem in polys)):
        raise('gradgroebner: First argument of gradgroebner must be list of polynomials over ',X,'.')

    F = minus(set(list(map(expand, polys))),{0})
    F = [elem.as_content_primitive(X)[1] for elem in list(F)]

    # F  = [elem.subs([(Catalan,sp.Catalan),(Pi,sp.Pi),(E,sp.EulerGamma),(gamma,sp.Gamma)]) for elem in F]

    Y = set()
    for z in F:
        Y = union(Y,indets(z))
    if not(minus(Y,set(X)) in [set(),{},sp.EmptySet]):
        dom = 'poly'
    else:
        dom = 'inte'
    if len(minus(set(X),Y))>0:
        raise('gradgroebner: Polynomials do not depend on all variables in X.')
    print('gradgroebner: Checked polynomials.')

    if nargs > 3:
        for g2 in grads2:
            if not(type(g2) == dict):
                raise('gradgroebner: Third argument of gradgroebner must be dict.')
            if 'minint' not in g2.keys():
                g2['minint']=0
            if 'maxint' not in g2.keys():
                g2['maxint']=sp.oo
            if not(all(is_type(g2[str(elem)],'integer') for elem in X)):
                raise('gradgroebner: Weights in grading for gradgroebner have to be integers for all variables')
            if any(is_type(g2[str(elem)],'negative') for elem in X):
                raise('gradgroebner: Weights in grading for gradgroebner have to be non-negative.')
        print('gradgroebner: Checked grading.')
        for z in F:
            for g2 in grads2:
                if not(ishomogeneous(z,X,g2)):
                    raise('gradgroebner: Non-homogeneous polynomial w.r.t. grading found.')
                
    adds = list(args[3:])[:]
    tt = to.copy()

    bl = arerationalpols(F,X)
    if not(bl):
        print('gradgroebner: Not rational.')
        F,v,p,g = convert_internal(F,list(Y),'v','p','g')
        F = F + p
        tt = mktermorder(X+v,'blocked',to,mktermorder(v,'plex'))
        X = X + v
        for i in range(len(adds)):
            for va in v:
                adds[i][va] = 0

    if dom == 'inte':
        answer = grob(F,X,tt,*adds)
    else:
        answer = grob_internal_poly(F,X,tt,*adds)

    if not(bl):
        answer = convertback(answer,X,v,p,g)

    # answer = answer.subs([sp.Eq(Catalan,sp.Catalan),sp.Eq(Pi,sp.Pi),sp.Eq(E,sp.EulerGamma),sp.Eq(gamma,sp.Gamma)])
    return answer

def homgroebner(*args):
    """
    Input:
    polys,to,grads2,weights,H

    Output:
    Groebner basis given a Hilbert series
    """
    nargs = len(args)
    print('homgroebner: Multigraded Hilbert series driven.')
    if nargs < 5:
        raise('homgroebner: Too few arguments.')
    if nargs > 5:
        raise('homgroebner: Too many arguments.')
    polys,to,grads2,weights,H = args[0], args[1], args[2], args[3], args[4]
#   establish term ordering 
    if not(isinstance(to, dict)): 
        raise('homgroebner: Second argument should be a table including the termorder')
    X = to['vs']; 
# check polynomials
    if not(all(is_type(elem,'polynom',X) for elem in polys)): 
        raise('homgroebner: Input must be polynomials over',X)
#   first, expand the polynomials and remove any zeros
    F = list(minus(set([expand(elem) for elem in polys]),{0}))
#   if there are more indeterminates than variables, the coeff. domain
#   is "poly"
#   (before testing, remove the polynomial content and front-end some constants)
    F = [icontent(elem,X,'str')[1] for elem in F]
    # F := subs({Catalan=CATALAN,Pi=PI,E=EULER,gamma=GAMMA}, F);
    Y = set()
    for z in F:
        Y = union(Y,indets(z))
    if len(minus(Y,set(X))) != 0:
        dom  = 'poly'
    else:
        dom = 'inte'
# fuer testzwecke
# dom := `poly`;
    if len(minus(set(X),Y))>0:
        raise('homgroebner: Polynomials do not depend on all variables.')
    print('homgroebner: Polynomials checked.')
# check grading
    if not(isinstance(grads2,(list,set))): 
        raise('homgroebner: Third argument has to be a set of gradings.')
    if isinstance(grads2, set):
        grads2 = list(grads2)
    for grading2 in grads2:
        if not isinstance(grading2, dict): 
            raise('homgroebner: This argument should be a table containing a grading.')
        if not('minint' in grading2.keys()):
            grading2['minint'] = 0
        if not('maxint' in grading2.keys()):
            grading2['maxint'] = sp.oo
        if not(all(is_type(grading2[str(elem)],'integer') for elem in X)):
            raise('homgroebner: Weights in grading have to be integer for all variables.')
        if any(is_type(grading2[str(elem)],'negative') for elem in X):
            raise('homgroebner: Weights in secondary grading have to be nonnegative.')
    print('Gradingset checked.')
    # check whether polynomials are homogeneous wrt gradings in gradingset
    for z in F: 
        for grading2 in grads2:
            # userinfo(4,moregroebner,`help`);
            if not(ishomogeneous(z,X,grading2)):
                raise('homgroebner: Non-homogeneous polynomial found wrt grading.')
    # userinfo(4,moregroebner,`help help`);
    # check weight system
    if not(isinstance(weights,list)):
        raise('homgroebner: Fourth argument has to be a list.')
    for grading2 in grads2:
        if not(isinstance(grading2,dict)): 
            raise('homgroebner: Fourth argument should be a list of tables.')
        if not(all(is_type(grading2[str(elem)],'integer') for elem in X)):
            raise('homgroebner: Weights in grading have to be integer for all variables.')
        if any(is_type(grading2[str(elem)],'negative') for elem in X):
            raise('homgroebner: Weights in grading have to be nonnegative.')
    for elem in X:
        flag = False
        for grading2 in weights:
            if grading2[str(elem)]>0:
                flag = True
        if not(flag):
            raise('homgroebner: No positive weight for variable ',elem,'.')
    print('homgroebner: Gradinglist checked.')
# check whether polynomials are homogeneous wrt gradings in weight system
    for z in F:
        for grading2 in weights:
            if not(ishomogeneous(z,X,grading2)):
                raise('homgroebner: Non-homogeneous polynomial found wrt grading in gradinglist.')
# check zvars
    zvars = []
    for grading2 in weights:
        if '_Hseriesvar' in grading2.keys(): 
            if is_type(grading2['_Hseriesvar'],'name'):
                zvars.append(grading2['_Hseriesvar'])
            else:
                raise('homgroebner: _Hseriesvar should be a SymPy symbol.')
        else:
            raise('homgroebner: Each grading in fourth argument needs an entry _Hseriesvar.')
    if len(weights) != len(zvars): 
        raise('homgroebner: Variables in _Hseriesvar in gradinglist have to be different.')
    # check Hilbert series
    if len(minus(indets(H),set(zvars))) > 0 or len(indets(H)) != len(zvars):
        raise('homgroebner: Something wrong with fifth argument.')
# check for roots, complex numbers, etc.
    dom2 = arerationalpols(F,X)
    if dom2: 
        if dom == 'inte':
            answer = integrob(F,X,to,grads2,weights,H,zvars)
        else:
            answer = polygrob(F,X,to,grads2,weights,H,zvars)
    else:
        F, v, p, g = convert(F,Y,'v','p','g')
        tt = mktermorder(v,'plex')
        p2 = grob(p,v,tt)
        F = F + p2
        tt = mktermorder(X + v,'blocked',to,tt)
        Xa = X + v
# copy the grading and the weight system
        grads3 = []
        for grad in grads2:
            T = SparseDict()
            keys = grads2.keys()
            for key in keys:
                T[key] = grad[keys[0]]
            grads3.append(T)
        weights2 = []
        for weight in weights:
            T = SparseDict()
            keys = weights.keys()
            for key in keys:
                T[key] = weight[keys[0]]
            weights2.append(T)
        if dom == 'inte':
            answer = interadgrob(F,Xa,tt,grads3,weights2,H,zvars,v)
        else:
            answer = polyradgrob(F,Xa,tt,grads3,weights2,H,zvars,v)
        answer  = convertback(answer,X,v,p,g)
    # answer := subs({CATALAN=Catalan,PI=Pi,EULER=E,GAMMA=gamma}, answer);
    return answer

def mkGB(pols,tt,gradings,weights,hts):
    """
    Input:
    pols: list of SymPy polynomials or expressions.
    tt: term order
    gradings: list of gradings
    weights: weighting for Hilbert series
    hts: list of SymPy monomial head terms

    Output:
    gb: Grobner basis driven by Hilbert series
    """
    X = tt['vs']
    dom = arerationalpols(pols,X)
    # case too many variables
    X1 = indets(pols)
    if len(minus(set(X),X1)) != 0:
        gbhts = []
        for monom in set(hts):
            if len(minus(indets(monom),X1)) == 0:
                gbhts.append(monom)
        X = X1
        termord = restricttorder(tt,X)
    else:
        termord = tt
        gbhts = hts
    print('mkGB: Compute Groebner bases with Hilbert series driven Buchberger algorithm')
    hp = hilbertseries(gbhts,X,weights)
    print('mkGB: Use Hilbert series of ideal',hp,'.')
    gb = homgroebner(pols,termord,gradings,weights,hp)
    print('mkGB: Groebner bases finished')
    return gb