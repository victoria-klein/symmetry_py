import numpy as np
import sympy as sp
from built_ins import *
import settings
import inspect

"""
tabtop - tested
addto - tested
rmul - tested
cmul - tested
ptotab - tested
sub_internal - tested
ctmul - tested
deletelist - tested
univar - tested
degree_internal - tested
sort_internal - tested
boolf_mod - tested
termlcm - tested
isinbounds - tested
merge_internal - tested
gcont - tested
occurence - tested
occurence2 - tested
convert_internal - tested
convertback - tested
convertback1 - tested
ishomogeneous - tested
ishomogeneous2 - tested
arerationalpols - tested
mktermorder - tested
nextmonom
degmultofsecondaries
linalgindependent
"""

#-------------------------------------------------------------#
# INTERNAL FUNCTIONS
#-------------------------------------------------------------#

def tabtop(t,X,to=None):
    """
    Input:
    t: dict representation of SymPy polynomial
    X: list of vars SymPy symbols
    to: dict of term order

    Output:
    p: SymPy polynomial of t
    """
    inds = list(map(op,indices(t)))
    p = 0
 #   inds:=`moregroebner/src/internal/sort`(inds,'`moregroebner/src/internal/isgreaterorder2`',op(thisorder));
    for z in inds:
        p = p + t[z]*z
    return p

def addto(a,b,m): #
    """
    Input:
    a: (list, set, sp.Matrix, np.ndarray or dict) of SymPy polynomials
    b: (list, set, sp.Matrix, np.ndarray or dict) of SymPy polynomials
    m: integer,float,sp.Float,sp.Integer,sp.Number

    Output:
    a: a plus m*b
    """
    inds = indices(b)
    for x in inds:
        if assigned(a,x):
            a[x] = a[x] + m*b[x]
            if a[x] == 0:
                del a[x]
                # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
                # ls = [var_name for var_name, var_val in callers_local_vars if var_val is a]
                # if len(ls) == 1:
                #     settings.add([sp.Symbol(ls[0]+'_'+str(x))])
                #     a[x] = settings.return_globals()[ls[0]+'_'+str(x)]
                # else:
                #     raise('Multiple instances for var_val')
        else:
            a[x] = m*b[x]
    return a

def rmul(r,p):
    """
    Input:
    r: SymPy polynomial or expression
    p: dict representation of SymPy polynomial or expression

    Output:
    p: polynomial p rescaled by coefficient r
    """
    inds = indices(p)
    for x in inds:
        p[x] = normal(r*p[x])
    return p

def cmul(c,p):
    """
    Input:
    c: SymPy polynomial or expression
    p: list of or individual SymPy polynomial or expression

    Output:
    polynomial p rescaled by coefficient c
    """
    if isinstance(p,(list,dict,set)):
        inds = indices(p)
    else:
        return expand(c*p)
    for x in inds:
        p[x] = expand(c*p[x])
    return p

def ptotab(p,X):
    """
    Input:
    p: SymPy expression or polynomial p
    X: list of vars Sympy symbols

    Output:
    t: dict of, for each type of term (monomial) in p, sum of leading coefficients
    """
    t = SparseDict()
    s = expand(p)
    if s == 0:
        return op(t)
    if is_type(s,'+'):
        for vv in s.args:
            sco = lcoeff(vv,X)
            ste = vv/sco
            # print(sco,ste,vv)
            if assigned(t,ste):
                t[ste] = t[ste] + sco
            else:
                t[ste] = sco
    else:
        sco = lcoeff(s,X)
        t[s/sco] = sco
    return t

def sub_internal(a,b):
    """
    Input:
    a: (list, set, sp.Matrix or dict) of SymPy polynomials
    b: (list, set, sp.Matrix or dict) of SymPy polynomials

    Output:
    t: a minus b
    """
    inds = indices(b)
    p = a.copy()
    for x in inds:
        if assigned(p,x):
            p[x] = p[x] - b[x]
            if p[x] == 0:
                del p[x]
                # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
                # ls = [var_name for var_name, var_val in callers_local_vars if var_val is a]
                # if len(ls) == 1:
                #     settings.add([sp.Symbol(ls[0]+'_'+str(x))])
                #     a[x] = settings.return_globals()[ls[0]+'_'+str(x)]
                # else:
                #     print(ls)
                #     raise('Multiple instances for var_val')
        else:
            p[x] = -b[x]
    return p

def ctmul(c,t,p):
    """
    Input:
    c:  SymPy polynomial coefficient term
    t: SymPy polynomial monmial term
    p: dict of SymPy polynomial or expression

    Output:
    r: dict representation polynomial p multiplied by c*t
    """
    r = SparseDict()
    inds = indices(p)
    for x in inds:
        r[x*t] = expand(p[x]*c)
    return r

def deletelist(P,J):
    """
    Input:
    P: list of objects
    J: set of indices

    Output:
    newP: P but without elements at indices
    """
    Icap = sorted(list(J))
    newP = P
    r = 0
    n = len(P)
    for i in range(len(Icap)):
        newP = newP[:Icap[i]-r] + newP[Icap[i]+1-r:n-r]
        r += 1
    return newP

def univar(htermj):
    """
    Input:
    htermj: SymPy polynomial term / monomial

    Output:
    True if univariate, otherwise False
    """
    vs = indets(htermj)
    if nops(vs) == 1:
        return True
    else:
        return False

def degree_internal(term,X,grading):
    """
    Input:
    term: SymPy polynomial term
    X: list of vars SymPy symbols
    grading: dict of gradings for each var in X

    Output:
    dsum: sum of degree(term,var)*grading(var) of term for every variable in X
    """
    if len(X) == 0:
        dsum = 0
    else:
        dsum = convert([degree(term,elem)*grading[str(elem)] for elem in X],'+')
    return dsum

def vecdegree(v,vs):
    """
    Input:
    v: list of terms
    vs: list of vars SymPy symbols
    """
    assert(type(v) == list)
    return max([degree(elem,vars) for elem in v])

def sort_internal(ll, boolf, *args):
    """
    Input:
    ll: list of objects
    boolf: python function f(*args) that returns True or False
    *args: arguments to boolf

    Output:
    ll1: list of objects sorted by boolf
    """
    # ll is a list of objects
    # boolf is a function of several arguments
    #   the first two arguments are objects from the list
    #   the rest of the arguments receive addargs as values
    # boolf is true if the first argument is smaller or equal than the second
    # boolf=lessorequalp
    n = len(ll)
    if n<2:
        return ll
    v = ll[n-1]
    ll1 = ll
    i = 0
    j = n-1
    while i<j:
        while boolf(ll1[i],v,*args) and i<n-1:
            i += 1
        while boolf(v,ll1[j],*args) and j>0:
            j -= 1
        # print('i',i,'j',j)
        if i<j:
            # print("i<j",ll1[:i],[ll1[j]],ll1[i+1:j],[ll1[i]],ll1[j+1:])
            ll1 = ll1[:i] + [ll1[j]] + ll1[i+1:j] + [ll1[i]] + ll1[j+1:]
            # print("ll1",ll1)
        else: # i = j
            if i != n-1:
                # print("i=j",ll1[:i],[v],ll1[i+1:n-1],[ll1[i]])
                ll1 = ll1[:i] + [v] + ll1[i+1:n-1] + [ll1[i]]
                # print("ll1",ll1)
    ll2 = sort_internal(ll1[:i],boolf,*args)
    ll3 = sort_internal(ll1[i+1:],boolf,*args)
    ll1 = ll2+[ll1[i]]+ll3
    return ll1

def boolf_mod(v,w,*args):
    if len(args) == 1:
        H = args[0]
        if H[v] <= H[w]:
            return True
        else:
            return False
    else:
        if degree(v) <= degree(w):
            return True
        else:
            return False

def termlcm(a,b,X):
    """
    Input:
    a: SymPy polynomial term / monomial
    b: SymPy polynomial term / monomial
    X: list of vars SymPy symbols

    Output:
    h: lowest common multiple of a and b
    """
    h = [x**max(degree(a,x),degree(b,x)) for x in X]
    return convert(h,'*')

def isinbounds(*args):
    """
    Input:
    monom: SymPy polynomial term / monomial
    X: list of vars SymPy symbols
    gradings: gradings on X

    Output:
    flag: True if degree of monom is within max and min of all gradings
    """
    if len(args) == 2:
        return True
    monom = args[0]
    X = args[1]
    gradings = list(args[2:])
    if () in gradings:
        gradings.remove(())
    # print(gradings)
    flag = True
    for grad in gradings:
        deg = degree_internal(monom,X,grad)
        if deg < grad['minint'] or deg > grad['maxint']:
            flag = False
            break
    return flag

def merge_internal(*args):
    """
    Input:
    ll1: list
    ll2: list
    boolf: function f(*addargs) that returns bool
    *addargs: arguments of boolf

    Output:
    llres: ll1 merged with ll2 based on boolf
    """
    ll1 = args[0]
    ll2 = args[1]
    boolf = args[2]
    addargs = args[3:]
    n1 = len(ll1)
    n2 = len(ll2)
    i = 0
    j = 0 
    llres = []
    while i<=n1-1 and j<=n2-1:
        # print(i,j)
        if boolf(ll1[i],ll2[j],*addargs):
            # print(llres,ll[i])
            llres.append(ll1[i])
            # print(llres)
            i += 1
        else:
            # print(llres,ll2[j])
            llres.append(ll2[j])
            # print(llres)
            j += 1
    # print(llres)
    if j<n2-1:
        llres = llres + ll2[j:]
    elif j == n2-1:
        llres.append(ll2[j])
    elif i<n1-1:
        llres = llres + ll1[i:]
    elif i == n1-1:
        llres.append(ll1[i])
    return llres

def gcont(t,hi):
    """
    Input:
    t: dict of SymPy polynomial terms
    hi: SymPy term / monomial that is key in t

    Output:
    g1 * g: g1 is gcd of terms in t and g is gcd of t[hi] and terms in t
    t: updated
    """
    inds = indices(t)
    n = nops(inds)
    if n == 0:
        return 0, t
    elif n == 1:
        g = t[hi]
        t[hi] = 1
        return g, t
    g1 = 0
    for j in range(n):
        if g1 != 1:
            g1 = igcd(g1,icontent(t[inds[j]]))
#   BG FIX - 29/04/92 by SRC:  line below used to say "g <> 1"
    
    if g1 != 1:
        for j in range(n):
            _, t[inds[j]] = divide(t[inds[j]],g1,'evaln(t[inds[j]])')
    
    _, g = divide(t[hi],icontent(t[hi]),'g')
    for j in range(n):
        if g != 1:
            if inds[j] == hi:
                continue
            if not(divide(t[inds[j]],g)):
                g = gcd(g,t[inds[j]])
    if g != 1:
        for j in range(n):
            _, t[inds[j]] = divide(t[inds[j]],g,'evaln(t[inds[j]])')
    return g1 * g, t

def occurence(ll,X):
    """
    Input:
    ll: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols

    Output:
    rs: list of any occurances of RootOf in polynomials in ll
    """
    rs = set()
    for p in ll:
        if not(is_type(p,'rationalpolynom',X)):
            ls = list(map(occurence2,set(coeffs(p,set(X)))))
            rs = union(rs,*ls) 
    return list(rs)

def occurence2(ee):
    """
    Input:
    ee: SymPy polynomial or expression

    Output:
    set: set of RootOf if any, or empty set
    """
    if is_type(ee,'RootOf'):
        return set([ee])
    elif is_type(ee,'*'):
        return union(*list(map(occurence2,convert(ee,'set'))))
    elif is_type(ee,'+'):
        return union(*list(map(occurence2,convert(ee,'set'))))
    # elif isinstance(ee,sp.Pow):
    #     return union(*list(map(occurence2,convert(ee,'set'))))
    elif is_type(ee,'rational'):
        return set()
    elif is_type(ee,'constant'):
        return set()
    else:
        raise('occurence2: Unknown domain.')
    
def convert_internal(ll,X,*args):
    """
    Input:
    ll: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    *args: [algvars,algpols,alggls]

    Ouput:
    polys: list of original polynomials ll with each RootOf substituted with new variable _wi
    algvars: list of variables _w1,...,_wn (one for each RootOf found in ll)
    algpols: list of polys p1,...,pn where pi substituted with _wi and pi is polynomial corresponding to i'th RootOf
    alggls:  list of equations _wi = i'th RootOf
    """
    polys = [convert(elem,'RootOf') for elem in ll]
    rs = occurence(polys,X)
    newvars = [sp.Symbol('_w'+str(i)) for i in range(len(rs))]
    algvars = newvars
    algpols = [rs[i].expr.subs(list(rs[i].expr.free_symbols)[0],newvars[i]) for i in range(len(rs))]
    gls = [(rs[i],newvars[i]) for i in range(len(rs))]
    polys = [p.subs(gls) for p in polys]
    alggls = [(newvars[i],rs[i]) for i in range(len(rs))]

    if len(args) == 0:
        return polys
    elif len(args) == 1:
        return polys, algvars
    elif len(args) == 2:
        return polys, algvars, algpols
    else:
        return polys, algvars, algpols, alggls
    
def convertback(ll,X,algvars,algpols,alggls):
    """
    Input:
    ll: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    *args: [algvars,algpols,alggls] see Output of convert_internal

    Output:
    p: list of original SymPy polynomials without RootOf instances in other variables
    """
    polys = []
    new_alggls = [(elem[0],convert(elem[1],'radical')) for elem in alggls]
    for elem in ll:
        if not(minus(indets(elem),set(algvars)) in [{},sp.EmptySet,set()]) or elem==1:
            polys.append(subs(new_alggls,elem))
    return polys

def convertback1(p,X,algvars,algpols,alggls):
    """
    Input:
    p: SymPy polynomial p
    X: list of SymPy symbols variables
    algvars: list of new SymPy symbols variables
    algpols: list of SymPy new polynomials corresponding to new variables
    alggls: list of SymPy equations for each new variable and new polynomials

    Output:
    p: SymPy polynomial p as a radical
    """
    return convert(subs(p,alggls),'radical')
    
def ishomogeneous(p,X,grading):
    """
    Input:
    p: SymPy polynomial or expression
    X: list of vars SymPy symbols
    grading: grading on X

    Output:
    bl: True if p homogeneous polynomial in X w.r.t grading
    """
    if not(is_type(p,'polynom',X)): 
        raise('ishomogeneous: p must be polynomial in X.')
    if not(is_type(grading,'table')): 
        raise('ishomogeneous: Third argument should be a dict containing grading.')
    if not(all(is_type(grading[str(elem)],'integer') for elem in X)):
        raise('ishomogeneous: Weights in grading have to be integer for all variables.')
    if not(all(not(is_type(grading[str(elem)],'negative')) for elem in X)):
        raise('ishomogeneous: Weights in grading have to be nonnegative.')
    return ishomogeneous2(p,X,grading)

def ishomogeneous2(p,X,grading):
    """
    Input:
    p: SymPy polynomial or expression
    X: list of vars SymPy symbols
    grading: grading on X

    Output:
    flag: True if p homogeneous polynomial in X w.r.t grading
    """
    s = expand(p)
    lc, term = lcoeff(s,X,'term')
    s = s-lc*term
    deg = degree_internal(term,X,grading)
    flag = True
    while s != 0 and flag:
        lc, term = lcoeff(s,X,'term')
        if not(deg == degree_internal(term,X,grading)): 
            flag = False
        s = expand(s-lc*term)
    return flag

def arerationalpols(ll, X):
    """
    Input:
    ll: list of SymPy expressions or polynomials
    X: list of SymPy symbols variables

    Output:
    True if all elements of ll are rational polynomials in X, False if not
    """
    flag = True
    for elem in ll:
        conv = convert(elem,'RootOf')
        if not(is_type(conv,'rationalpolynom',X)):
            print('arerationalpols: Polynomial ',conv,'of type ',conv.__class__,' is not rational in ',X,'.')
            flag = False
            break
    return flag

def mktermorder(vs, oname, *args):
    """
    Input:
    vs: list of vars SymPy symbols
    oname: name of term order
    *args: additional matrices M SymPy matrix or term orderings

    Output:
    to: dict of term order

    plex,tdeg,gradlex: {'vs': vs, 'ordername':oname}
    mat: : {'vs': vs, 'ordername':oname, 'mat': SymPy matrix}
    blocked: {'vs': vs, 'ordername':oname, 'order1': term order dict, 'order2': term order dict}
    """
    names = ['plex','tdeg','gradlex']
    nargs = len(args)

    if oname in names:
        if nargs != 0:
            raise('mktermorder: Too many arguments for plex, tdeg or gradlex.')
        else:
            res = {'vs': vs, 'ordername': oname}
            return res

    elif oname == 'mat':
        if nargs == 0:
            raise('mktermorder: Order matrix necessitates a valid matrix with no. cols equal no. vars.')
        elif nargs == 1:
            M = args[0]
            if rank(M) != len(vs):
                raise('mktermorder: If matrix rank is not equal to no. of variables, give a term order.')
        elif nargs == 2:
            M = args[0]
            to = args[1]
        else:
            raise('mktermorder: Too many arguments.')
        
        if  M.shape[1] != len(vs):
            raise('mktermorder: Wrong dimensions of order matrix.')
        for k in range(M.shape[1]):
            for l in range(M.shape[0]):
                if not(is_type(M[l,k],'integer')) and not(M[l,k]>=0):
                    raise('mktermorder: Matrix entries should be positive integers.')
        res = {'vs': vs, 'ordername': oname, 'mat': M}
        return res

        if nargs == 2:
            # if not(moregrobner.istermorder(to)):
            #     raise('Fourth argument not valid term order')
            if equal(vs, to['vs']):
                res['order1'] = to
                return op(res)
            else:
                raise('mktermorder: Variables are not equal.')

    elif oname == 'blocked':
        if nargs != 2: #or not(moregrobner.istermorder(args[0])) or not(moregrobner.istermorder(args[1]))
            raise('mktermorder: Blocked needs two term orderings.')
        xvars = args[0]['vs'] + args[1]['vs']
        if len(vs) != len(xvars):
            raise('mktermorder: Wrong no. of variables.')
        for k in range(len(vs)):
            if vs[k] != xvars[k]:
                raise('mktermorder: Bad variable list.')
        res = {'vs': vs, 'ordername': oname, 'order1': args[0], 'order2': args[1]}
        return res
    
    else:
        raise('mktermorder: Second argument has to be the name of a term order.')
    
def nextmonom(m,vs,d):
    """
    Input:
    m: SymPy monomial as expression
    vs: list of vars SymPy symbols
    d: int representing degree

    Output:
    vs[0]^d
    """
    if m is 0:
        return vs[0]**d
    if len(vs) == 1:
        return False
    else:
        # print('m',m)
        n1 = degree(m, vs[0])
        x1 = vs[0]
        res = m/x1**n1
        if isinstance(res, sp.core.numbers.One):
            # m1 = nextmonom(res,vs[1:],d-n1) #expand
            return vs[0]**(n1-1)*vs[1]**(d-n1+1)
        else:
            m1 = nextmonom(res,vs[1:],d-n1) #expand
        if m1 is not False:
            return vs[0]**n1*m1 #expand
        else:
            if n1 is 0:
                return False
            else:
                return vs[0]**(n1-1)*vs[1]**(d-n1+1) #expand
# def nextmonom(m,vs,d):
#     """
#     Input:
#     m: SymPy monomial as expression
#     vs: list of vars SymPy symbols
#     d: int representing degree

#     Output:
#     vs[0]^d
#     """
#     if m is 0:
#         return sp.poly(vs[0]**d,vs[0])
#     if len(vs) == 1:
#         return False
#     else:
#         # print('m',m)
#         n1 = degree(m, vs[0])
#         x1 = vs[0]
#         res = m/x1**n1
#         if isinstance(res, sp.core.numbers.One):
#             # m1 = nextmonom(res,vs[1:],d-n1) #expand
#             return sp.poly(vs[0]**(n1-1)*vs[1]**(d-n1+1), vs[0], vs[1]) #expand
#         else:
#             m1 = nextmonom(sp.poly_from_expr(res)[0],vs[1:],d-n1) #expand
#         if m1 is not False:
#             return sp.poly_from_expr(vs[0]**n1*m1)[0] #expand
#         else:
#             if n1 is 0:
#                 return False
#             else:
#                 return sp.poly(vs[0]**(n1-1)*vs[1]**(d-n1+1), vs[0], vs[1]) #expand

def degmultofsecondaries(M, prims, vs):
    """
    Input:
    M: Molien series of g
    prims: list of primary invariants
    vs: list of SymPy symbols variables

    Output:
    pol: polynomial with degrees and multiplicities of generators of free module
    """
    var = list(indets(M))[0]
    pol = M
    for q in prims:
        pol = pol*(1-var**degree(q,set(vs)))
    print(pol, len(prims))
    pol = expand(normal(expand(pol)))
    # print(pol)
    if not(is_type(pol, 'polynom',var)):
        raise('degmultofsecondaries: Pol not polynomial in var.')
    print('degmultofsecondaries: Pol with degs and multiplicities of generators of free module ', pol)
    return pol

def degmultofsecondariesMNISTD4(M):
    """
    Input:
    M: Molien series of g
    prims: list of primary invariants
    vs: list of SymPy symbols variables

    Output:
    pol: polynomial with degrees and multiplicities of generators of free module
    """
    var = list(indets(M))[0]
    pol = M
    degs = [1 for i in range(105)]+[2 for i in range(91+91+105)]+[4 for i in range(392)]
    for deg in degs:
        pol = pol*(1-var**deg)
    print('degmultofsecondaries: Expanding poly ',pol,'.')
    pol = sp.poly((1+var)**679 * (1+var**2)**392,var)#expand(normal(expand((1+var)**679 * (1+var**2)**392)))
    # if not(is_type(pol, 'polynom',var)):
    #     raise('degmultofsecondaries: Pol not polynomial in var.')
    print('degmultofsecondaries: Pol with degs and multiplicities of generators of free module ', pol)
    return pol

def degmultofsecondariessmallMNISTD4(M):
    """
    Input:
    M: Molien series of g
    prims: list of primary invariants
    vs: list of SymPy symbols variables

    Output:
    pol: polynomial with degrees and multiplicities of generators of free module
    """
    var = list(indets(M))[0]
    pol = M
    degs = [1 for i in range(10)]+[2 for i in range(6+6+10)]+[4 for i in range(32)]
    for deg in degs:
        pol = pol*(1-var**deg)
    print('degmultofsecondaries: Expanding poly ',pol,'.')
    pol = sp.poly((1+var)**(64-16) * (1+var**2)**32,var)#expand(normal(expand((1+var)**679 * (1+var**2)**392)))
    # if not(is_type(pol, 'polynom',var)):
    #     raise('degmultofsecondaries: Pol not polynomial in var.')
    print('degmultofsecondaries: Pol with degs and multiplicities of generators of free module ', pol)
    return pol

def degmultofsecondaries64D4(M):
    """
    Input:
    M: Molien series of g
    prims: list of primary invariants
    vs: list of SymPy symbols variables

    Output:
    pol: polynomial with degrees and multiplicities of generators of free module
    """
    var = list(indets(M))[0]
    pol = M
    degs = [1 for i in range(528)]+[2 for i in range(496+496+528)]+[4 for i in range(2048)]
    for deg in degs:
        pol = pol*(1-var**deg)
    print('degmultofsecondaries: Expanding poly ',pol,'.')
    pol = sp.poly((1+var)**3568 * (1+var**2)**2048,var)#expand(normal(expand((1+var)**679 * (1+var**2)**392)))
    # if not(is_type(pol, 'polynom',var)):
    #     raise('degmultofsecondaries: Pol not polynomial in var.')
    print('degmultofsecondaries: Pol with degs and multiplicities of generators of free module ', pol)
    return pol

def linalgindependent(g,qs,vs):
    """
    Input:
    g: SymPy polynomial or expression
    qs: list of SymPy polynomials or expressions
    vs: list of SymPy symbols variables

    Output:
    True if g depends linearly on polynomial qs, False if not
    """
    d = degree(g,set(vs))
    qds = []
    for i in range(len(qs)):
        if degree(qs[i],set(vs)) == d:
            qds.append(qs[i])
    if len(qds) == 0:
        return True
    _a, _b = sp.symbols('_a,_b')
    # print('list',[_a*i*qds[i-1] for i in range(1,len(qds)+1)])
    # print('_oplist',op([_a*i*qds[i-1] for i in range(1,len(qds)+1)]))
    pol = _b*g+convert([_a*i*qds[i-1] for i in range(1,len(qds)+1)],'+') #_a.i
    eqs = [expand(pol)]
    for i in range(len(vs)):
        neweqs = []
        for pol in eqs:
            neweqs = (neweqs)+ coeffs(sp.collect(pol,vs[i]),vs[i])
        eqs = neweqs
    res = sp.solve(eqs,[_b]+[_a*i for i in range(1,len(qds)+1)],dict=True) #_a.i
    # print('res',res)
    for sol in res:
        if _b in sol.keys() and sol[_b] == 0:
            return True
    return False

def ishilbertseries(m):
    """
    Input:
    m: Molien series

    Output:
    True if Molien series (series in one variable), False otherwise
    """
    return len(indets(m)) == 1
    
def seriesmindeg(h,m,svar,c=None):
    """
    Input:
    h: Hilbert series
    m: Hilbert series
    svar: SymPy symbol variable
    c: str

    Output:
    d: minimal degree where two Hilbert series differ
    """
    d = 0
    j = 1
    ser = normal(m-h)
    while d == 0:
        tc, d = tcoeff(series(ser,(svar,0),10*j).removeO(),svar,'d')
        j += 1
        d = degree(d,svar)
    c = tc
    if c == None:
        return d
    else:
        return d, c

def ismonomlist(monoms,vstr=None):
    """
    Input:
    monom: SymPy monomial
    vstr: string

    Output:
    True if monomial in v, False otherwise
    """
    vs = set()
    for m in monoms:
        vs = union(set(vs),indets(m))
    v = list(vs)
    flag = True
    for m in monoms:
        if is_type(m,'monomialinteger',v) and lcoeff(m)==1:
            flag = flag
        else:
            flag = False
    if vstr == None:
        return flag
    else:
        return flag, v
    
def hilbi_(mm,vs,gd,svar):
    """
    Input:
    mm: list or set of SymPy monomials
    vs: list of original SymPy symbols variables   
    gd: list of gradings
    svar: SymPy symbol variable

    Output:
    h: pre-Hilbert series
    """
    if len(mm) == 0:
        h = 1
    else:
        f = 1
        for k in range(len(gd)):
            f = f*svar[k]**degree_internal(mm[0],vs,gd[k])
        h = 1-f
    #     h:=1-svar^`moregroebner/src/internal/degree`(mm[1],vars,gd); 
    for i in range(1,len(mm)):
        J = set() # ideal quotient (mm_1,...,mm_(i-1)): mm_i
        for j in range(0,i-1+1):
            c = lcm(mm[i],mm[j])
            J = union(J,{c/mm[i]})
        J = list(J)
        Jh = set()
        for j in range(len(J)):  # reduce Monoms
            flag = False
            for k in range(len(J)):
                if j!=k and lcm(J[k],J[j]) == J[j]:
                    flag = True
                    break# J[j]= a* J[k]
            if not(flag):
                Jh = union(Jh,{J[j]})
        J = Jh  # after Reduction if Basis of ideal J
        Jlin = intersect(J,set(vs)) # linear monomials
        J1 = minus(J,Jlin) 
        hJ1 = hilbi_(list(J1),vs,gd,svar)
        for v in Jlin:
            ws = [gd[k][str(v)] for k in range(len(gd))]
            hJ1 = hJ1*(1-mono(svar,ws))
    #        hJ1:=hJ1*(1-svar^gd[v])
        f = 1
        for k in range(len(gd)):
            f = f*svar[k]**degree_internal(mm[i],vs,gd[k])
        h = h-hJ1*f
    #     h:=h-hJ1*svar^`moregroebner/src/internal/degree`(mm[i],vars,gd);
    if h!=1: 
        print('hilbi_: Returns numerator of Hilbert series ',expand(h),'.')
    return h
    
def mono(vs,es):
    """
    Input:
    vs: list of original SymPy symbols variables
    es: list of weights

    Output:
    monomial of variables in vs with weights in es
    """
    return convert(set([vs[i]**es[i] for i in range(len(es))]),'*')
    
def HP(mm,vs,vars2,gd,svar):
    """
    Input:
    mm: list or set of SymPy monomials
    vs: list of original SymPy symbols variables
    vars2: list of SymPy symbols variables of all mm
    gd: list of gradings
    svar: SymPy symbol variable

    Output:
    h: Hilbert series
    """
    h = expand(hilbi_(mm,vars2,gd,svar))
    for v in vs:
        ws = [gd[j][str(v)] for j in range(len(gd))]
        h = normal(h*1/(1-mono(svar,ws)))
    return h


def hilbi(*args):
    """
    Input:
    monomlistorset: list or set of SymPy monomials
    vs: list of SymPy symbols variables
    grading: grading
    zvar: SymPy symbol variable
    
    Output:
    h: Hilbert series
    """
    print('hilbi: Computation of Hilbert series for quotient over monomial ideal.')
    print('hilbi: Algorithm by Bayer, Stillman is used.')
    nargs = len(args)
    if nargs < 2:
        raise('hilbi: Too few arguments.')
    if nargs > 4:
        raise('hilbi: Too many arguments.')
    monomlistorset, vs, grading = args[0], args[1], args[2]
    if not(isinstance(monomlistorset,(list,set))): 
        raise('hilbi: First argument has to be a list or set.')
    bl, vars2 = ismonomlist(monomlistorset,'vars2')
    if not(bl): 
       raise('hilbi: First argument must be list or set of monomials.')
    # check second argument
    if not(isinstance(vs,(list,set))): 
        raise('hilbi: Second argument has to be a list or set.')
#    flag:=true;
#    for v in vars2 do 
#       if not(member(v, vars)) then flag:=false; fi;
#    od;
#    if not(flag) then ERROR(`monomials should only depend on given variables`) fi;
    if len(minus(set(vars2),set(vs))) > 0:
       raise('hilbi: Monomials should only depend on given variables.') 
    # reduction of monomials to minimal generating set
    if isinstance(monomlistorset, set):
        mm = list(monomlistorset)
    else:
        mm = monomlistorset
    MM = set()
    for j in range(len(mm)):
        flag = False
        for k in range(len(mm)):
            if j!=k and lcm(mm[k],mm[j])==mm[j]:
                flag = True # J[j]= a* J[k]
        if not(flag):
            MM = union(MM, {mm[j]})
    mm = list(MM)
    if nargs>3:
        zvar = args[3]
        if is_type(zvar,'name'):
            svar = [zvar]
        elif is_type(zvar,'listname'):
            if len(intersect(set(vs),set(zvar))) == 0:
                svar = zvar
            else:
                raise('hilbi: Names in fourth argument badly chosen.')
        else:
            raise('hilbi: Fourth argument has to be a name or list of names.')
    if nargs >2:
        if isinstance(grading,dict): 
            gd = grading.copy()
            for v in vs:
                if not(is_type(gd[str(v)],'integer')):
                    raise('hilbi: Grading should contain a weight for variable ',v,'.') 
                if not(is_type(gd[str(v)],'positive')):
                    raise('hilbi: Grading should contain a positive weight for variable ',v,'.') 
            gd = [gd]
            if nargs==3: 
                if '_Hseriesvar' in gd[0].keys():
                    svar = [gd[0]['_Hseriesvar']]
                else:
                    svar = ['lam']
        elif is_type(grading,'listtable'): 
            gd = grading
            for v in vs: 
                flag = False
                for tt in gd:
                    if not(is_type(tt[str(v)],'integer')):
                        raise('hilbi: Grading should contain a weight for variable ',v,'.')
                    if tt[str(v)]>0:
                        flag = True
                if not(flag):
                    raise('hilbi: Nonzero weight required for variable ',v,'.')
            if nargs==3:
                svar = []
                for j in range(len(gd)):
                    if '_Hseriesvar' in gd[j].keys():
                        svar = svar + [gd[j]['_Hseriesvar']]
                    else:
                        svar = svar + ['lam'+str(j)]
            elif len(gd) != len(svar):
                raise('hilbi: Nr of gradings has to equal nr of interminates.')
        elif is_type(grading,'name'): 
            svar = [grading]
            if nargs == 4:
                raise('hilbi: Arguments in wrong ordering.') 
            else:
                print('hilbi: No user defined grading - use natural grading.')
                gd = {str(vs[j]):1 for j in range(len(vs))}
        elif is_type(grading,'listname') and len(intersect(vs,set(grading))) == 0:
            svar = grading
            if nargs == 4:
                raise('hilbi: Arguments in wrong ordering.') 
            else:
                if len(svar)==1:
                    print('hilbi: No user defined grading - use natural grading.')
                    gd = {str(vs[j]):1 for j in range(len(vs))}
                else:
                    raise('hilbi: Gradings missing.')
        else:
            raise('hilbi: third argument has to be a grading (list or a name (list).')
    else:
        print('hilbi: No user defined grading - use natural grading.')
        gd = {str(vs[j]):1 for j in range(len(vs))}
        # gd = [table([map(v->v=1,vars)])]
        svar = ['lam']
    h = HP(mm,vs,vars2,gd,svar)
    return h
    
def hilbertseries(*args):
    return hilbi(*args)
    
def restrict_(tt,X):
    """
    Input:
    tt: term order
    X: list of SymPy symbols

    Output:
    term order to restricted to Xse
    """
    newt = {}
    vs = []
    oldvs = tt['vs']
    for elem in oldvs:
        if member(elem,X):
            vs.append(elem)
    if len(vs) == 0:
        return False
    else:
        newt['vs'] = vs
    if tt['ordername'] == 'plex': 
        newt['ordername'] = 'plex'
    elif tt['ordername'] == 'tdeg':
        newt['ordername'] = 'tdeg'
    elif tt['ordername'] == 'gradlex':
        newt['ordername'] = 'gradlex'
    elif tt['ordername'] == 'mat':
        mold = tt['mat']
        n = rowdim(mold)
        m = coldim(mold)
        mae = matrix(n,nops(vs))
        k = 0
        for j in range(m):
            if member(oldvs[j],X):
                k += 1
                for nu in range(n):
                    mae[nu,k] = mold[nu,j]
        newt['mat'] = mae
        if 'order1' in tt.keys():
                t1 = restrict_(tt['order1'],X)
                if isinstance(t1,dict):
                    newt['order1'] = t1
        newt['ordername'] = 'mat'
    elif tt['ordername'] == 'blocked':
        t1 = restrict_(tt['order1'],X)
        t2 = restrict_(tt['order2'],X)
        if isinstance(t1,dict) and isinstance(t2,dict):
            newt['ordername'] = 'blocked'
            newt['order1'] = t1
            newt['order2'] = t2
        else:
            if isinstance(t1,dict):
                newt = t1
            else:
                newt = t2
    return newt

    
def restricttorder(*args):
    """
    Input:
    to: term order
    Xse: list of SymPy symbols

    Output:
    term order to restricted to Xse
    """
    nargs = len(args)
    if nargs < 2:
        raise('restricorder: Too few arguments.')
    to, Xse = args[0],args[1]
    if not(isinstance(to, dict)): 
        raise('restrictorder: First argument should be a dict termorder.')
    X = to['vs']
    if not(is_type(Xse,'setname') or is_type(Xse,'listname')): 
       raise('restrictorder: Second argument should be set/list of SymPy symbols.')
    if len(minus(set(Xse),set(X))) != 0:
        raise('restrictorder: Second argument is not a subset of variables in the termorder.')
    return restrict_(to,Xse)


