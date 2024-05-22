from built_ins import *
from internal import *
from grob import *
import settings
import inspect
from grob_inte_sympy_F4 import GrobRadicalEqual
"""
ismemberofradical -imp test
isradicalequal -imp test
normalform
nf_inte
nf_poly
"""

#-------------------------------------------------------------#
# INTERNAL FUNCTIONS FOR INVARIANTS AND EQUIVARIANTS
#-------------------------------------------------------------#
# def new_ismemberofradical(Q,p,_Z,Q_polys,flag,opt,ring):
#     """
#     Input:
#     p: SymPy expression
#     Q: list of SymPy expressions
#     vs: list of SymPy symbols variables
#     gb_Q: dict of previous grobner basis calculations

#     Output:
#     True if [1] is Groebner basis for Q with 1-_Z*p, False if not i.e. True if p already in Q
#     """
#     if Q == []:
#         return False 
    
#     return 

def ismemberofradical(p,Q,vs):
    """
    Input:
    p: SymPy expression
    Q: list of SymPy expressions
    vs: list of SymPy symbols variables

    Output:
    True if [1] is Groebner basis for Q with 1-_Z*p, False if not i.e. True if p already in Q
    """
    # print('ismemberofradical: Started.')
    if Q == []:
        return False
    vars2 = union(indets(Q),indets(p))
    _Z = sp.Symbol('_Z')
    if not(member(_Z,vars2)):
        vars2 = list(vars2)+[_Z]
    else:
        raise('ismemberofradical: Implement new slack variable.')
    tt = mktermorder(vars2, 'tdeg')
    gb = gradgroebner(Q+[1-_Z*p],vars2,tt)
    if gb == [1]:
        # print('ismemberofradical: Finished.')
        return True
    else:
        # print('ismemberofradical: Finished.')
        return False

def isradicalequal(vs, Q):
    """
    Input:
    Q: list of SymPy expressions
    vs: list of SymPy symbols variables

    Output:
    True if radical of vs and Q are equal, False if otherwise
    """
    # print('ismemberofradical: Started.')
    if len(Q)<1:
        return False
    n = len(vs)
    k = 1
    flag = True
    while flag and k < n:
        ch = [(vs[i],0) for i in range(k-1)]+[(vs[k-1],1)] #i:='i';
        polys = [subs(ch,elem) for elem in Q]
        vs2 = list(indets(polys))
        if len(vs2) > 0:
            tt = mktermorder(vs2, 'tdeg')
            gb = gradgroebner(polys, vs2, tt)
            if not(gb == [1]):
                flag = False
            # flag = GrobRadicalEqual(vs2,polys).__call__()
        else:
            flag = False
        k += 1
    # print('ismemberofradical: Finished.')
    return flag

def normalform(*args):
    """
    Input:
    f: SymPy expression
    polys: list of SymPy expressions
    X: list of SymPy symbols variables
    to: term order

    Output:
    ansewer: normal form of f w.r.t polys
    """
    nargs = len(args)
    if nargs < 4:
        raise('normalform: Too few arguments.')
    elif nargs > 4:
        raise('normalform: Too many arguments.')
    f,polys,X,to = args[0],args[1],args[2],args[3]
    if not(is_type(X,'list')):
        raise('normalform: Variables X must be a list.')
    if not(all(elem.__class__ == sp.Symbol for elem in X)):
        raise('normalform: Third argument should be variable list.')
    # if not(istermorder(to)):
    #     raise('normalform: Fourth argument should be termorder.')
    if not(X == to['vs']):
        raise('normalform: Variables X should and those in termorder must be the same.')
    if not(is_type(polys,['list','set'])):
        raise('normalform: Second argument has to be a lisst or set of polynomials.')
    if not(is_type(f,'polynom',X)):
        raise('normalform: First argument must be a polynomial in X.')
    if not(all(is_type(elem,'polynom',X) for elem in polys)):
        raise('normalform: Polys input must be polynomials over X.')
    
    fb = expand(f)
    if fb == 0:
        return 0
    else:
        fcont, fb = icontent(fb,X,'fcont')
        # fb, fcont = primpart(fb,X,'fcont')
        # fb = fb.subs([(Catalan,sp.Catalan),(Pi,sp.Pi),(E,sp.EulerGamma),(gamma,sp.Gamma)])
    nzbasis = []
    Y = indets(fb)
    for p in polys: #polys.subs([(Catalan,sp.Catalan),(Pi,sp.Pi),(E,sp.EulerGamma),(gamma,sp.Gamma)])
        ep = expand(p)
        if ep != 0:
            nzbasis.append(p)
            Y = union(Y,indets(ep))
    if len(nzbasis) == 0:
        # fb = fb.subs([(Catalan,sp.Catalan),(Pi,sp.Pi),(E,sp.EulerGamma),(gamma,sp.Gamma)])
        return fcont*sp.collect(fb,X,)
    if not(minus(Y,set(X)) in (sp.EmptySet,set())):
        dom = 'poly'
    else:
        dom = 'inte'

    dom2 = arerationalpols(nzbasis,X)
    if dom2:
        dom2 = arerationalpols([fb],X)
    if dom2:
        if dom == 'inte':
            answer = fcont*nf_inte(fb,nzbasis,X,to) #Maple original: expand([op(nzbasis)])
        else:
            G = [ptotab(elem,X) for elem in nzbasis]
            fb = ptotab(fb,X)
            fb = nf_poly(fb,G,X,tt)
            fb = rmul(fcont,fb)
            answer = tabtop(fb,X,to)
    else:
        G = nzbasis #Maple original: expand([op(nzbasis)])
        vv, pp, gg, G = convert_internal([fb] + G,Y,'vv','pp','gg')
        fb = G[0]
    # test whether leading coefficients depend on roots
        HCs = [hcoeff_inte(elem,to['vs'],to) for elem in G[1:]]
        if nops(indets(HCs)) > 0: 
            raise('normalform: Case of leading coefficients depending on roots is not implemented yet.') 
    #   start division algorithm
        G = G[1:]+[pp]
        tt = mktermorder(X+vv,'blocked',to,mktermorder(vv,'plex'))
        Xx = X + vv
        if dom == 'inte': 
            answer = fcont*nf_inte(fb,G,Xx,tt)
        else:
            G = [ptotab(elem) for elem in G+pp]
            fb = ptotab(fb,X)
            fb = nf_poly(fb,G,Xx,tt)
            fb = rmul(fcont,fb)
            answer = tabtop(fb,Xx,tt)
        answer = convertback1(answer,Xx,vv,pp,gg)
    # answer = answer.subs([(Catalan,sp.Catalan),(Pi,sp.Pi),(E,sp.EulerGamma),(gamma,sp.Gamma)])
    return answer

def nf_inte(f,G,X,to):
    """
    Input:
    f: SymPy polynomial or expression
    G: list of SymPy polynomials or expressions
    X: list of SymPy symbols variables
    to: term order

    Output:
    k: normal form of f
    """
    if f == 0:
        return 0
    if len(G) == 0:
        return f
    stt = time.time()
    print('nf_inte: beginning')
    HT = {j:hterm_inte(G[j],X,to) for j in range(len(G))}
    HC = {j:hcoeff_inte(G[j],X,to) for j in range(len(G))}
    Sugar = {j:sugardegree(G[j],X) for j in range(len(G))}
    lowest=HT[0]
    print('nf_inte: for loop')
    for j in range(1,len(G)):
       if isgreaterorder2(lowest,HT[j],to):
           lowest=HT[j]
    print('nf_inte: icontent')
    oldcont = icontent(f)
    print('nf_inte: divide')
    _, rest = divide(f,oldcont,'rest')
    k = 0
    print('nf_inte: degree')
    sres = degree(f,set(X))
    while rest != 0: 
        if not(isgreaterorder2(hterm_inte(rest,X,to),lowest,to)):
            k = k + oldcont*rest
            break
        print('nf_inte: sred_inte')
        temp, scale, newcont, sres = sred_inte(rest,sres,G,HT,HC,Sugar,X,to,oldcont,'scale','newcont','sres')
        h = expand(convert(list(head_inte(temp,X,to)),'*'))
        oldcont = newcont/scale
        k = k + (oldcont*h)
        rest = temp - h
    st = time.time()
    print('nf_inte: Time',st,', elapsed ',st-stt)
    return k

def nf_poly(f,G,X,to):
    """
    Input:
    f: dict representation of SymPy polynomial or expression
    G: list of dict representation of SymPy polynomials or expressions
    X: list of SymPy symbols variables
    to: term order

    Output:
    k: normal form of f
    """
    if len(indices(f)) == 0:
        return f
    if len(G) == 0:
        return f
    stt = time.time()
    HT = {j:head_poly(G[j],X,to) for j in range(len(G))}
    HC = {G[j][HT[j]] for j in range(len(G))}
    Sugar = {j:sugardegree(G[j],X) for j in range(len(G))}
    lowest = HT[0]
    for j in range(1,len(G)+1):
        if isgreaterorder2(lowest,HT[j],to):
           lowest = HT[j]
    k = SparseDict()
    temp = SparseDict()
    rest = SparseDict()
    for key,value in f:
        rest[key] = value
    ht = head_poly(rest,X,to) 
    oldcont, rest = gcont(rest,ht)
    sres = degree(ht[1],set(X))
    while len(indices(rest)) != 0:
        if not(isgreaterorder2(ht,lowest,to)):
            oldcont = rmul(oldcont,rest)
            k = addto(k,rest,1)
            break
        temp2, scale, newcont, sres = sred_poly(rest,sres,G,HT,HC,Sugar,X,to,oldcont,'scale','newcont','sres')
        for key,value in temp2[0]:
            temp[key] = value
        if len(indices(temp)) == 0:
            break
        ht = head_poly(temp,X,to)
        oldcont = normal(newcont/scale)
        k[ht] = normal(oldcont*temp[ht])
        del temp[ht]
        for key,value in temp[0]:
            rest[key] = value
        ht = head_poly(rest,X,to)
    st = time.time()
    print('nf: Time: ',st,', elapsed ',st-stt)
    return k