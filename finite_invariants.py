import numpy as np
import sympy as sp
import time as time

from internal import *
from built_ins import *
from grob import *
from finite_molien import *
from finite_groups import *
from head import *
from variants import *
import settings
from grob_inte_sympy_F4 import GrobMemberRadical

"""
proj2inv2 -imp test
inv_ofthisdegree -imp test
inv_nextcandidate -imp test
primcand2Q -imp test
Q2primesec -imp test
primaryseconds -imp test
secondaries
CMBasis_ -tested
CMBasis -tested
"""

#-------------------------------------------------------------#
# INVARIANTS FOR FINITE GROUPS
#-------------------------------------------------------------#

def proj2inv2(m,vs,g):
    """
    Inputs:
    m: SymPy monomial
    vs: list of SymPy symbols variables
    g: dict representation of group
    """
    s = 0
    v = sp.Matrix(vs)
    mats = list(g['allelements'].values())
    for ma1 in mats:
        v2 = convert(ma1 * v, 'list')
        ll = [(v[i],v2[i][0]) for i in range(len(v2))]
        p = subs(ll,m)
        s = expand(p+s)
    s = s/len(mats)
    return s

def inv_ofthisdegree(g,vs,d,m_in,m_out=None):
    """
    Input:
    g: dict representation group
    vs: list of vars SymPy symbols
    d: int representing degree
    m_in: SymPy monomial
    m_out: string or None

    Output:
    inv: invariant of m_in
    m_out: updated
    """
    print('inv_ofthisdegree: Started.')
    inv = 0
    m = m_in
    while (inv is 0 or isinstance(inv, sp.core.numbers.Zero)) and m is not False:
        m = nextmonom(m,vs,d)
        if m is not False:
            inv = proj2inv2(m,vs,g)
        else:
            inv = False
    mout = m
    if m_out is None:
        print('inv_ofthisdegree: Finished.')
        return inv
    else:
        print('inv_ofthisdegree: Finished.')
        return inv, mout

def inv_nextcandidate(g,vs,Q,Qds,M,var,m_in,m_out=None):
    """
    Input:
    g: dict representation group
    vs: list of vars SymPy symbols
    Q: list of SymPy expressions
    Qds: list of total degree of each q in Q w.r.t vs
    M: Degree 20 Molien series of G with 1 and Big O removed
    var: SymPy variable of Molien series
    m_in: int
    m_out: string or None

    Output:
    q: False or SymPy monomial
    m_out: updated
    """
    print('inv_nextcandidate: Started 1st half.')
    m = m_in
    if m_in == 0:
        p = M.copy()
        if p == 0:
            raise('inv_nextcandidate: Unexpected case.')
        _, d = tcoeff(p, var, 'd')
        print(d,var)
        d = degree(d, var)
    else:
        p = M.copy()
        if p == 0:
            raise('inv_nextcandidate: Unexpected case.')
        d = degree(m_in,set(vs))
        c = coeff(p, var, d)
        print('inv_nextcandidate: 1st half for loop.')
        for i in range(len(Q)):
            if Qds[i] == d:
                c = c-1
        if c <= 0:
            print('inv_nextcandidate: 1st half if loop.')
            for i in range(1,d+1):
                p = p - coeff(p, var**i)*var**i
            _, d = tcoeff(p, var, 'd')
            d = degree(d, var)
            m = 0
    print('inv_nextcandidate: Started 2nd half.')
    if m_out is not None:
        q, m_out = inv_ofthisdegree(g,vs,d,m,m_out)
    else:
        q = inv_ofthisdegree(g,vs,d,m,m_out)
    while q is False:
        for i in range(1,d+1):
            p = p - coeff(p, var**i)*var**i
        if p is 0:
            raise('inv_nextcandidate: Unexpected case.')
        _, d = tcoeff(p, var, 'd')
        d = degree(d,var)
        m = 0
        if m_out is not None:
            q, m_out = inv_ofthisdegree(g,vs,d,m,m_out)
        else:
            q = inv_ofthisdegree(g,vs,d,m,m_out)
    q = sp.sympify(q)
    q = q / lcoeff(q,vs)
    if m_out is None:
        print('inv_nextcandidate: Finished.')
        return q
    else:
        print('inv_nextcandidate: Finished.')
        return q, m_out

from sympy.polys.rings import PolyRing

def primcand2Q(g, vs, M, invs):
    """
    Input:
    g: dict representation of group
    vs: list of SymPy symbols variables
    M: Molien series of g
    invs: list of already known invaraints

    Output:
    Q: list of invariants updated
    """
    varl = list(sp.Expr(M).free_symbols)
    if len(varl) > 1:
        raise('primcand2Q: Molien series has more than one variable.')
    else:
        var = varl[0]
    M = series(M,(var,0),20).removeO()-1
    n = len(vs)
    Q = []
    Qds = [] # Total degree of each q in Q w.r.t vs
    # for i in range(len(invs)):
    #     if isradicalequal(vs, Q):
    #         return Q
    #     if not(ismemberofradical(invs[i], Q, vs)):
    #         print('primcand2Q: Use suggestion by user as primary candidate ', invs[i],'.')
    Q = invs.copy()
    print('primcand2Q: Calculating degrees of previous invs.')
    Qds = [degree(inv, set(vs)) for inv in invs]
    m = 0
    # _Z = sp.Symbol('_Z')
    # print('primcand2Q: Setting up previous invs.')
    # gmr = GrobMemberRadical(vs,_Z,Q)
    while len(Q) < len(vs) or not(isradicalequal(vs, Q)):
        update = False
        while not(update):
            # print('primcand2Q: Started inv_nextcandidate.')
            p, m = inv_nextcandidate(g, vs, Q, Qds, M, var, m, 'str')
            # print('primcand2Q: Finished inv_nextcandidate.')
            if not(ismemberofradical(p,Q,vs)): #not(gmr(p)):
                print('primcand2Q: Candidate for primaries found ', p,'.')
                Q.append(p)
                f = open("D4MNISTinvariants.txt", "a")
                f.write(str(p)+"\n")
                f.close()
                Qds.append(degree(p, set(vs)))
                update = True
    print('primcand2Q: Output is ',Q,'.')
    return Q

def Q2primesec(Qq,vs,seccanditates=None):
    """
    Input:
    Qq: list of invariants
    vs: list of SymPy symbols variables
    seccanditates: string

    Output:
    Q: list of primary invariants
    seccanditates: list of secondary invariants
    """
    print('Q2primesec: There are ',len(Qq),' candidates for primaries.')
    Q = Qq
    sec = []
    n = len(vs)
    while nops(Q) > n:
        for i in [len(Q)-2-i for i in range(len(Q)+2)]:
            if isradicalequal(vs,Q[:i]+Q[i+1:]):
                sec.append(Q[i])
                print('Q2primsec: Drop candidate ', Q[i])
                Q = Q[:i]+Q[i+1:]
                i = -2
                break
        if i == 0 and len(Q) > n:
            raise('Q2primesec: Noether normalization/impl. of alg. for system of homogenous parameters necessary.')
    print('Q2primsec: Primaries found: ', Q)
    print('Q2primsec: Secondaries found: ', sec)
    if seccanditates == None:
        return Q
    else:
        return Q, sec
    
def primaryseconds(g, vs, M, invs, S=None):
    """
    Input:
    g: dict representation of group
    vs: list of SymPy symbols variables
    M: Molien series of g
    invs: list of apriori invariants
    S: string

    Output:
    prims: primary invariants
    S: secondary invariants
    """
    Q = primcand2Q(g, vs, M, invs)
    prims, Ss = Q2primesec(Q, vs, S)
    if S == None:
        return prims
    else:
        return prims, Ss

def secondaries(g,vs,prims,sec,M,k):
    """
    Input:
    g: dict representation of group
    vs: list of SymPy symbols variables
    prims: list of primary invariants
    sec: list of secondary invariants
    M: Molien series of g
    k: int (specifying degree up to which to calculate secondary invariants)

    Output:
    sec: scondary invariants
    """
    varl = list(sp.Expr(M).free_symbols)
    if len(varl) > 1:
        raise('primcand2Q: Molien series has more than one variable.')
    else:
        var = varl[0]
    cd, q = tcoeff(M,var,'q')
    d = degree(q,var)
    j = 0
    if d > k:
        return sec
    grading = {str(vs[i]): 1 for i in range(len(vs))}
    grading['minint'] = 0
    grading['maxint'] = min(degree(M,var),k)
    tt = mktermorder(vs,'tdeg')
    # print(prims,vs,tt,grading)
    gb = gradgroebner(prims,vs,tt,grading)
    # print('Check',gb,gradgroebner(prims,vs,tt))
    normalfsec = [normalform(elem,gb,vs,tt) for elem in sec]
    while M != 0 and d <= k:
        cd, q = tcoeff(M,var,'q')
        d = degree(q,var)
        if d > k:
            break
        m = 0
        while cd > 0:
            j += 1
            q, m = inv_ofthisdegree(g,vs,d,m,'str')
            print('secondaries: Secondary candidate ',q,'.')
            if q == False:
                raise('secondaries: Something wrong.')
            normalfq = normalform(q,gb,vs,tt)
            print('secondaries: Normal form of secondary candidate ',normalfq,'.')
            if normalfq != 0 and linalgindependent(normalfq,normalfsec,vs):
                M = M - var**d
                cd = cd - 1
                q = q/lcoeff(q,vs)
                print('secondaries: Secondary found = candidate',j,q,'.')
                sec.append(q)
                normalfsec.append(normalfq)
    if M != 0:
        print('secondaries: Warning, some secondary invariants are still missing, Molien left ',M,'.')
    return sec

def CMBasis_(g, vs, prims, k, prim_only):
    """
    Input:
    g: dict representation of group
    vs: list of SymPy symbols variables
    prims: list of apriori invariants
    k: int (specifying degree up to which to calculate secondary invariants)

    Output:
    dictionary of primary and secondary invariants
    """
    # first = 0
    var = sp.Symbol('lambda')
    M = molien_(g, var)
    print('CMBasis_: Molien series is ', M)

    sec = []
    if prims == []:
        prims, sec = primaryseconds(g,vs ,M ,[],'S')
    if prim_only == True:
        return {'primary_invs': prims}
    M = degmultofsecondaries(M, prims, vs)
    M -= 1
    if len(sec)>0:
        print('CMBasis_: Reuse info from computation of primaries.')
    for q in sec:
        print('CMBasis_: Use invariant of degree ', degree(q,set(vs)), q)
        M = M - var**degree(q,set(vs))
    M = expand(M)
    if len(sec) > 0:
        print('CMBasis_: Remaining polynomial encoding degrees of secondaries ',M)
    if M != 0:
        sec = secondaries(g, vs, prims, sec, M, k)
    return({'primary_invs': prims, 'secondary_invs': [1]+sec})

def CMBasis(*args):
    """
    Input:
    g: dict representation of group
    vs: list of SymPy symbols variables
    prims: list of apriori invariants
    k: int (specifying degree up to which to calculate secondary invariants)

    Output:
    dictionary of primary and secondary invariants
    """
    if len(args) > 4:
        raise('CMBasis: Too many arguments.')
    if len(args) <2:
        raise('CMBasis: Too few arguments.')
    g = args[0]
    var = args[1]
    if not(type(g) == dict):
        raise('CMBasis: First argument should be dict representation of group.')
    n = dimension(g)
    if len(var) !=  n:
        raise('CMBasis: Number of vars should be equal to the dimension of the representation.')
    k = sp.oo
    prims = []
    if len(args) == 3:
        if is_type(args[2],'integer'):
            k = args[2]
        elif type(args[2]) == list:
            prims = args[2]
        else:
            raise('CMBasis: Third argument must be integer k or list of primary invariants.')
    elif len(args) == 4:
        prims = args[2]
        k = args[3]
    if len(args) == 5:
        prim_only = args[4]
    else:
        prim_only = True
    return(CMBasis_(g, var, prims, k, prim_only))



def isinvequations(linvsgl,vs):
    """
    Input:
    linvsgl: list or set of invariants as SymPy expressions or polynomials
    vs: list of SymPy symbols variables

    Output:
    True if invariant equations in vs, False otherwise
    """
    if not(isinstance(linvsgl,list)):
      raise('isinvequations: First argument should be a list.')
    if not(isinstance(vs, list)) and not(all(elem.__class__ == sp.Symbol for elem in vs)):
        raise('areinvscomplete: Second argument must be list of SymPy variables.')
    if len(linvsgl)==2 and (isinstance(linvsgl[0],list) or isinstance(linvsgl[1],list)): 
        if isinstance(linvsgl[0],list) and isinstance(linvsgl[1],list):
            glse = linvsgl[0] + linvsgl[1]
        else:
            raise('isinvequations: First argument can be list or list of 2 lists.')
    else:
        glse = linvsgl
    if not(all(isinstance(elem, sp.Expr) for elem in glse)):
        raise('isinvequations: Invariants in list (or 2 lists) must be SymPy expressions.')
    if not(all(is_type(elem,'polynom',vs) for elem in glse)):
        raise('isinvequations: Invariants in list (or 2 lists) must be polynomials in variable list vs.')
#    invvars = extractinvvars(glse)
#    if nops({op(invvars)} minus {op(vars)}) <> nops(invvars) then
#       ERROR(`names of invariants incorrect`);
    grading = {str(vs[j]):1 for j in range(len(vs))}
    for j in range(len(glse)):
        if not(ishomogeneous(glse[j],vs,grading)):
            raise('isinvequations: Invariants in list (or 2 lists) should be homogeneous.')
    return True

def mkinvvars(linvsgl):
    """
    Input:
    linvsgl: list or set of invariants as SymPy expressions or polynomials

    Output:
    ll: list of new SymPy symbols for each invariant in linvsgl
    """
    if len(linvsgl)==2 and isinstance(linvsgl[0],list) and isinstance(linvsgl[1],list): 
        glse = linvsgl[0] + linvsgl[1]
    else:
        glse = linvsgl
    return [sp.Symbol('_inv_'+str(i)) for i in range(len(glse))]

def extractinvvars(linvsgl):
    """
    Input:
    linvsgl: list or set of equations of invariants SymPy expressions or polynomials

    Output:
    ll: lhs for each equation in linvsgl
    """
    if len(linvsgl)==2 and isinstance(linvsgl[0],list) and isinstance(linvsgl[1],list): 
        glse = linvsgl[0] + linvsgl[1]
    else:
        glse = linvsgl
    return [lhs(elem) for elem in glse]

def extractinvdegs(linvsgl,vs):
    """
    Input:
    linvsgl: list of SymPy equations of invariants
    vs: list of SymPy symbols variables
    
    Output:
    ll: degree of invariant of each equation in linvsgl
    """
    if len(linvsgl)==2 and isinstance(linvsgl[0],list) and isinstance(linvsgl[1],list): 
        glse = linvsgl[0] + linvsgl[1]
    else:
        glse = linvsgl
    return [degree(rhs(elem),set(vs)) for elem in glse]

def mkinvpols(linvgls):
    """
    Input:
    linvsgl: list or set of equations of invariants SymPy expressions or polynomials

    Output:
    ll: list of SymPy equations of invariants
    """
    if len(linvgls) == 2 and isinstance(linvgls[1],list):
        secondsp = True
    else:   
        secondsp = False
    if secondsp:
        glse = linvgls[0] + linvgls[1]
    else:
        glse = linvgls
    return [lhs(elem)-rhs(elem) for elem in glse]
    
def mkinvtermorder(linvgls,vs):
    """
    Input:
    linvsgl: list of SymPy equations of invariants
    vs: list of SymPy symbols variables

    Output:
    tt: term order for vs and ivars in linvsgl
    """
    if len(linvgls) == 2 and isinstance(linvgls[0],list):
        secondsp = True
    else:
        secondsp = False
    ivars = extractinvvars(linvgls)
    idegs = extractinvdegs(linvgls,vs)
    X = ivars + vs
    o1 = mktermorder(ivars,'tdeg')
    o2 = mktermorder(vs,'tdeg')
    o3 = mktermorder(X,'blocked',o1,o2)
    if secondsp:
        mae = matrix(3,len(X),0)
        for j in range(len(vs)):
            mae[0,j+len(ivars)] = 1
        for j in range(len(linvgls[1])):
            mae[1,j+len(linvgls[0])] = 1
        for j in range(len(ivars)):
            mae[2,j] = idegs[j]
    else:
        mae = matrix(2,len(X),0)
        for j in range(len(vs)):
            mae[0,j+len(ivars)] = 1
        for j in range(len(ivars)):
            mae[1,j] = idegs[j]
    tt = mktermorder(X,'mat',mae,o3)
    return tt
    
def mkdeggradinginv(linvgls,vs,d):
    """
    Input:
    linvsgl: list of SymPy equations of invariants
    vs: list of SymPy symbols variables
    d: maximum degree of grading

    Output:
    grad: grading for vs and ivars in linvsgl with maxint d
    """
    grad = {str(elem):1 for elem in vs}
    ivars = extractinvvars(linvgls)
    idegs = extractinvdegs(linvgls,vs)
    grad = grad | {str(ivars[k]):idegs[k] for k in range(len(ivars))}
    grad['minint'] = 0
    grad['maxint'] = d
    grad['_Hseriesvar'] = sp.Symbol('_hdi')
    return grad
    
def mkhtsinv(linvsgl):
    """
    Input:
    linvsgl: list of SymPy equations of invariants

    Output:
    
    """
    if len(linvsgl) == 2 and isinstance(linvsgl[0],list):
        secondsp = True
    else:
        secondsp = False
    if secondsp:
        glse = linvsgl[0] + linvsgl[1]
    else:
        glse = linvsgl
    return [lhs(elem) for elem in glse]
    
def gb2gbinv(gb,vs,ivars):
    """
    Input:
    gb: Grobner basis
    vs: list of SymPy symbols variables
    ivars: list of SymPy symbols variables for invariant equations

    Output:
    ll: elements of gb that are polynomials in ivars only
    """
    ll = []
    for k in range(len(gb)):
        if len(minus(indets(gb[k]),set(ivars))) == 0:
            ll.append(gb[k])
    return ll

def areinvscomplete(*args):
    """
    Input:
    linvsgl: list or set of invariants as SymPy expressions or polynomials
    vs: list of SymPy symbols variables
    m: Molien series of group g
    k: degree up to which invariant ring generated

    Output:
    True if invariant ring is generated up to degree k by linvsgl, False if not
    """
    nargs = len(args)
    if nargs<3 or nargs>4:
        raise('areinvscomplete: Only 3 to 4 arguments allowed.')
    linvsgl, vs, m = args[0], args[1], args[2]
    if not(isinstance(vs, list)) and not(all(elem.__class__ == sp.Symbol for elem in vs)):
        raise('areinvscomplete: Second argument must be list of SymPy variables.')
    isinvequations(linvsgl,vs)
    if not(ishilbertseries(m)):
        raise('areinvscomplete: Third argument must be valid Molien series.')
    if nargs==4:
        k = args[3]
        if is_type(k,'integer') and k>0:
            d = k
        else:
            raise('areinvscomplete: Fourth argument should be a positive integer.')
    else:    
        d = sp.oo
    return areinvscomplete_(linvsgl,vs,m,d)
    

def areinvscomplete_(linvsgl,vs,m,k):
    """
    Input:
    linvsgl: list or set of invariants as SymPy expressions or polynomials
    vs: list of SymPy symbols variables
    m: Molien series of group g
    k: degree up to which invariant ring generated

    Output:
    True if invariant ring is generated up to degree k by linvsgl, False if not
    """

    ivars = mkinvvars(linvsgl)
    linvsgl = [(ivars[i],linvsgl[i]) for i in range(len(linvsgl))]
    pols = mkinvpols(linvsgl)
    tt = mkinvtermorder(linvsgl,vs)
    graddeg = mkdeggradinginv(linvsgl,vs,k)
    hts = mkhtsinv(linvsgl)
    gb = mkGB(pols,tt,[graddeg],[graddeg],hts)
    relations = gb2gbinv(gb,vs,ivars)
    hts = [leadm(elem, tt) for elem in relations]
    svar = indets(m)
    if len(svar) > 1:
        raise('areinvscomplete_: Molien series in more than one variable not a Molien series.')
    svar = list(svar)[0]
    graddeg['_Hseriesvar'] = svar

    # compute Hilbert series of Ring generated by elements in linvgls
    # which fullfil relations

    print('areinvscomplete_: Compute Hilbert series by algorithm by Bayer and Stillman.')
    h = hilbertseries(hts,ivars,graddeg)
    print('areinvscomplete_: Hilbert series is',poincaresimp(h,svar))
    print('areinvscomplete_: Hilbert series equals ',series(h,(svar,0),10),'.')
    if normal(m-h) == 0: 
        print('areinvscomplete_: Invariant ring is completely generated')
        return True
    else: 
        ss, ts  = seriesmindeg(h,m,svar,'ts')
        print('areinvscomplete_: Invariant ring is generated up to degree ',ss-1,'.')
        print('areinvscomplete_: At degree',ss,' missing dimension ',ts,'.')
        if ss<=k:
            return ss
        else:
            return True
        
def primaries2(g,vs,invsuser):
    """
    Input:
    g: dict representation of group
    vs: list of SymPy symbols variables
    invsuser: list of invariants to use a priori

    Output:
    Q: primary invariants
    """
    lamb = sp.Symbol('_lamb')
    M = molien_(g,lamb) 
    print('primaries2: Molien series is ',M,'.')
    Q, S = primaryseconds(g,vs,M,invsuser,'S')
    return Q

def isinv(q,g,vs):
    """
    Input:
    q: SymPy polynomial or expression
    g: dict representation of group
    vs: list of SymPy symbols variables

    Output:
    True if q invariant w.r.t g, False otherwise
    """
    qinv = proj2inv2(q,vs,g)
    if expand(qinv-q)==0:
        return True
    else:
        return False


def areinvpols(lps,g,vs):
    """
    Input:
    lps: list of invariants
    g: dict representation of group
    vs: list of SymPy symbols variables

    Output:
    True if invariant w.r.t g polynomials, False otherwise
    """
    if not(isinstance(lps,(list,set))) or not(all(is_type(elem,'polynom',vs) for elem in lps)):
        raise('areinvpols: List or set of pols. expected.')
    if not(all(isinv(elem,g,vs) for elem in lps)):
        raise('areinvpols: Polynomials have to be invariant.')
    return True

def areprimaries(*args):
    """
    Input:
    lps: list of invariants
    g: dict representation of group
    vs: list of SymPy symbols variables

    Output:
    True if primary invariants, False otherwise
    """
    lps,g,vs = args[0], args[1], args[2]
    # if nops(lps) != nops(vs):   
    #     raise('areprimaries: Number of primaries should equal the Krull dimension.')
    areinvpols(lps,g,vs)
    if not(isradicalequal(vs,args[2])):
        raise('areprimaries: Polynomials form not a set of parameters.') 
    return True

# def inverse_grid_number(n, number):
#     if 1 <= number <= n**2:
#         row_index = (number - 1) // n + 1
#         column_index = (number - 1) % n + 1
#         return row_index, column_index
#     else:
#         raise ValueError("Number must be between 1 and n^2 inclusive.")

# def grid_number(n, a, b):
#     if 1 <= a <= n and 1 <= b <= n:
#         return (a - 1) * n + b
#     else:
#         raise ValueError("Row and column indices must be between 1 and n inclusive.")

# def reflection_grid(n, coordinates):
#     a, b = coordinates
#     reflected_b = n - b + 1
#     return a, reflected_b

# def rotation_grid(n, coordinates):
#     a, b = coordinates
#     rotated_a = n - b + 1
#     rotated_b = a
#     return rotated_a, rotated_b

# def rotate(n, number):
#     (a,b) = inverse_grid_number(n, number)
#     (new_a, new_b) = rotation_grid(n, (a,b))
#     return grid_number(n, new_a, new_b)

# def reflect(n, number):
#     (a,b) = inverse_grid_number(n, number)
#     (new_a, new_b) = reflection_grid(n, (a,b))
#     return grid_number(n, new_a, new_b)

# def generate_rotation_matrix(n):
#     # Define the size of the matrix
#     matrix_size = n ** 2

#     # Initialize a matrix with zeros
#     rotation_matrix = sp.zeros(matrix_size, matrix_size)

#     # Set 1 at the specified positions for each column
#     for m in range(1, matrix_size + 1):
#         rotated_position = rotate(n, m)
#         rotation_matrix[rotated_position - 1, m - 1] = 1  # Adjust for 0-based indexing

#     return rotation_matrix

# def generate_reflection_matrix(n):
#     # Define the size of the matrix
#     matrix_size = n ** 2

#     # Initialize a matrix with zeros
#     reflection_matrix = sp.zeros(matrix_size, matrix_size)

#     # Set 1 at the specified positions for each column
#     for m in range(1, matrix_size + 1):
#         reflected_position = reflect(n, m)
#         reflection_matrix[reflected_position - 1, m - 1] = 1  # Adjust for 0-based indexing

#     return reflection_matrix

# def generate_d4_matrices(n):
#     '''Outputs n^2 by n^2 matrices'''
#     # Get rotation and reflection matrices
#     R = generate_rotation_matrix(n)
#     S = generate_reflection_matrix(n)

#     # Calculate R^2, R^3, SR, SR^2, SR^3
#     R2 = R@R
#     R3 = R2@R
#     SR = S@R
#     SR2 = S@R2
#     SR3 = S@R3

#     # Generate D4 matrices
#     D4_matrices = [sp.eye(n**2), R, R2, R3, S, SR, SR2, SR3]

#     return D4_matrices

# D4_matrices = generate_d4_matrices(28)

# g = {}
# g['oname'] = 'dihedral4'
# g['allelements']={}
# g['allelements']['_r0'] = D4_matrices[0]
# g['allelements']['_r1'] = D4_matrices[1]
# g['allelements']['_r2'] = D4_matrices[2]
# g['allelements']['_r3'] = D4_matrices[3]
# g['allelements']['_s_r0'] = D4_matrices[4]
# g['allelements']['_s_r1'] = D4_matrices[5]
# g['allelements']['_s_r2'] = D4_matrices[6]
# g['allelements']['_s_r3'] = D4_matrices[7]
# g['generators']={}
# g['generators']['_s'] = g['allelements']['_s_r0']
# g['generators']['_r'] = g['allelements']['_r0']


# vs = list(sp.symbols('x1:785'))
# sp.var('x1:785')
# k = 4
# invs = [x1 + x28 + x757 + x784, x2 + x27 + x29 + x56 + x729 + x756 + x758 + x783, x26 + x3 + x57 + x701 + x728 + x759 + x782 + x84, x112 + x25 + x4 + x673 + x700 + x760 + x781 + x85, x113 + x140 + x24 + x5 + x645 + x672 + x761 + x780, x141 + x168 + x23 + x6 + x617 + x644 + x762 + x779, x169 + x196 + x22 + x589 + x616 + x7 + x763 + x778, x197 + x21 + x224 + x561 + x588 + x764 + x777 + x8, x20 + x225 + x252 + x533 + x560 + x765 + x776 + x9, x10 + x19 + x253 + x280 + x505 + x532 + x766 + x775, x11 + x18 + x281 + x308 + x477 + x504 + x767 + x774, x12 + x17 + x309 + x336 + x449 + x476 + x768 + x773, x13 + x16 + x337 + x364 + x421 + x448 + x769 + x772, x14 + x15 + x365 + x392 + x393 + x420 + x770 + x771, x30 + x55 + x730 + x755, x31 + x54 + x58 + x702 + x727 + x731 + x754 + x83, x111 + x32 + x53 + x674 + x699 + x732 + x753 + x86, x114 + x139 + x33 + x52 + x646 + x671 + x733 + x752, x142 + x167 + x34 + x51 + x618 + x643 + x734 + x751, x170 + x195 + x35 + x50 + x590 + x615 + x735 + x750, x198 + x223 + x36 + x49 + x562 + x587 + x736 + x749, x226 + x251 + x37 + x48 + x534 + x559 + x737 + x748, x254 + x279 + x38 + x47 + x506 + x531 + x738 + x747, x282 + x307 + x39 + x46 + x478 + x503 + x739 + x746, x310 + x335 + x40 + x45 + x450 + x475 + x740 + x745, x338 + x363 + x41 + x422 + x44 + x447 + x741 + x744, x366 + x391 + x394 + x419 + x42 + x43 + x742 + x743, x59 + x703 + x726 + x82, x110 + x60 + x675 + x698 + x704 + x725 + x81 + x87, x115 + x138 + x61 + x647 + x670 + x705 + x724 + x80, x143 + x166 + x619 + x62 + x642 + x706 + x723 + x79, x171 + x194 + x591 + x614 + x63 + x707 + x722 + x78, x199 + x222 + x563 + x586 + x64 + x708 + x721 + x77, x227 + x250 + x535 + x558 + x65 + x709 + x720 + x76, x255 + x278 + x507 + x530 + x66 + x710 + x719 + x75, x283 + x306 + x479 + x502 + x67 + x711 + x718 + x74, x311 + x334 + x451 + x474 + x68 + x712 + x717 + x73, x339 + x362 + x423 + x446 + x69 + x713 + x716 + x72, x367 + x390 + x395 + x418 + x70 + x71 + x714 + x715, x109 + x676 + x697 + x88, x108 + x116 + x137 + x648 + x669 + x677 + x696 + x89, x107 + x144 + x165 + x620 + x641 + x678 + x695 + x90, x106 + x172 + x193 + x592 + x613 + x679 + x694 + x91, x105 + x200 + x221 + x564 + x585 + x680 + x693 + x92, x104 + x228 + x249 + x536 + x557 + x681 + x692 + x93, x103 + x256 + x277 + x508 + x529 + x682 + x691 + x94, x102 + x284 + x305 + x480 + x501 + x683 + x690 + x95, x101 + x312 + x333 + x452 + x473 + x684 + x689 + x96, x100 + x340 + x361 + x424 + x445 + x685 + x688 + x97, x368 + x389 + x396 + x417 + x686 + x687 + x98 + x99, x117 + x136 + x649 + x668, x118 + x135 + x145 + x164 + x621 + x640 + x650 + x667, x119 + x134 + x173 + x192 + x593 + x612 + x651 + x666, x120 + x133 + x201 + x220 + x565 + x584 + x652 + x665, x121 + x132 + x229 + x248 + x537 + x556 + x653 + x664, x122 + x131 + x257 + x276 + x509 + x528 + x654 + x663, x123 + x130 + x285 + x304 + x481 + x500 + x655 + x662, x124 + x129 + x313 + x332 + x453 + x472 + x656 + x661, x125 + x128 + x341 + x360 + x425 + x444 + x657 + x660, x126 + x127 + x369 + x388 + x397 + x416 + x658 + x659, x146 + x163 + x622 + x639, x147 + x162 + x174 + x191 + x594 + x611 + x623 + x638, x148 + x161 + x202 + x219 + x566 + x583 + x624 + x637, x149 + x160 + x230 + x247 + x538 + x555 + x625 + x636, x150 + x159 + x258 + x275 + x510 + x527 + x626 + x635, x151 + x158 + x286 + x303 + x482 + x499 + x627 + x634, x152 + x157 + x314 + x331 + x454 + x471 + x628 + x633, x153 + x156 + x342 + x359 + x426 + x443 + x629 + x632, x154 + x155 + x370 + x387 + x398 + x415 + x630 + x631, x175 + x190 + x595 + x610, x176 + x189 + x203 + x218 + x567 + x582 + x596 + x609, x177 + x188 + x231 + x246 + x539 + x554 + x597 + x608, x178 + x187 + x259 + x274 + x511 + x526 + x598 + x607, x179 + x186 + x287 + x302 + x483 + x498 + x599 + x606, x180 + x185 + x315 + x330 + x455 + x470 + x600 + x605, x181 + x184 + x343 + x358 + x427 + x442 + x601 + x604, x182 + x183 + x371 + x386 + x399 + x414 + x602 + x603, x204 + x217 + x568 + x581, x205 + x216 + x232 + x245 + x540 + x553 + x569 + x580, x206 + x215 + x260 + x273 + x512 + x525 + x570 + x579, x207 + x214 + x288 + x301 + x484 + x497 + x571 + x578, x208 + x213 + x316 + x329 + x456 + x469 + x572 + x577, x209 + x212 + x344 + x357 + x428 + x441 + x573 + x576, x210 + x211 + x372 + x385 + x400 + x413 + x574 + x575, x233 + x244 + x541 + x552, x234 + x243 + x261 + x272 + x513 + x524 + x542 + x551, x235 + x242 + x289 + x300 + x485 + x496 + x543 + x550, x236 + x241 + x317 + x328 + x457 + x468 + x544 + x549, x237 + x240 + x345 + x356 + x429 + x440 + x545 + x548, x238 + x239 + x373 + x384 + x401 + x412 + x546 + x547, x262 + x271 + x514 + x523, x263 + x270 + x290 + x299 + x486 + x495 + x515 + x522, x264 + x269 + x318 + x327 + x458 + x467 + x516 + x521, x265 + x268 + x346 + x355 + x430 + x439 + x517 + x520, x266 + x267 + x374 + x383 + x402 + x411 + x518 + x519, x291 + x298 + x487 + x494, x292 + x297 + x319 + x326 + x459 + x466 + x488 + x493, x293 + x296 + x347 + x354 + x431 + x438 + x489 + x492, x294 + x295 + x375 + x382 + x403 + x410 + x490 + x491, x320 + x325 + x460 + x465, x321 + x324 + x348 + x353 + x432 + x437 + x461 + x464, x322 + x323 + x376 + x381 + x404 + x409 + x462 + x463, x349 + x352 + x433 + x436, x350 + x351 + x377 + x380 + x405 + x408 + x434 + x435, x378 + x379 + x406 + x407, x1**2 + x28**2 + x757**2 + x784**2, x1*x2 + x1*x29 + x27*x28 + x28*x56 + x729*x757 + x756*x784 + x757*x758 + x783*x784, x1*x3 + x1*x57 + x26*x28 + x28*x84 + x701*x757 + x728*x784 + x757*x759 + x782*x784]
# var = sp.Symbol('lambda')
# M = molien_(g, var)
# print(M)
# ans = primcand2Q(g, vs, M, invs)
# print(ans)


# vs = list(sp.symbols('x1:9'))
# k = 4
# var = sp.Symbol('lambda')
# g = dihedral(4,[1,2,3,4,5,5])
# M = molien_(g, var)
# print(M)
# ans = primcand2Q(g, vs, M,[])
# print(ans)


# import scipy
 
# def create_elements(num_initial_points):
#     I = np.eye(60,60)
#     blocks = [I]
#     for i in range(59):
#         I = np.roll(I,-1,0)
#         blocks.append(I)
#     elements = {f'_r{i}': sp.Matrix(scipy.linalg.block_diag(*[blocks[i]]*num_initial_points)) for i in range(60)}
#     return elements


# representation = create_elements(1)
# g = {"generators": {"_r": sp.eye(representation['_r0'].shape[0])}, "allelements": representation}
# g['oname'] = 'icosahedral'
# g['invelements'] = {f'_r{i}': g['allelements'][f'_r{i}'] for i in range(60)}
 

# vs = list(sp.symbols('x1:61'))
# sp.var('x1:61')
# gtheta = g.copy()
# var = sp.Symbol('lambda')
# M = molien_(gtheta, var)
# ans = primcand2Q(gtheta,vs,M,[])
# print(ans)