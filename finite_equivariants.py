import numpy as np
import sympy as sp

from internal import *
from built_ins import *
from head import *
from grob import *
from finite_molien import *
from finite_groups import *
from finite_invariants import * #(areprimaries,areinvpols,isinv,primaries2)
from variants import *
import settings


"""
equis:
isvecpol
arequivecs
isequivariant
proj2equi2
equimolien_
equivariants
vec2pol
vecnorming
equi_ofthisdegree
equis_
equis
"""

def vec2pol(v,slacks):
    """
    Input:
    v: vector of polynomials
    slacks: list of slack SymPy symbols variables

    Output:
    s: polynomial form of v
    """
    s = sp.sympify(0)
    for j in range(len(slacks)):
        s = s + v[j,0]*slacks[j] #v[j,:]
    return s

def isvecpol(v,vs):
    """
    Input:
    v: SymPy Matrix vector of polynomials or expressions
    vs: list of SymPy symbols variables

    Output:
    True if all entries of v are polynomials in vs, False otherwise
    """
    if type(v, 'vector') and not(all(is_type(v[:,i],'polynom',vs) for i in range(v.shape[1]))):
        return True
    else:
        return False
    
def areequivecs(lequi,gtheta,grho,vs):
    """
    Input:
    lequi: list of vectors of equivariant polynomials
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables

    Output:
    True if all elements of lequi are vectors of equivariant polynomials w.r.t g, False otherwise
    """
    if not(isinstance(lequi,(list,set)))  or not(all(is_type(elem,'vector') for elem in lequi)): 
        raise('areequivecs: List or set of vectors expected.')
    if not(all(isvecpol(elem,vs) for elem in lequi)):
        raise('areequivecs: Vectors of polynomials in',vs,'expected') 
    if not(all(isequivariant(elem,vs,gtheta,grho) for elem in lequi)): #src/equi/isequi
        raise('areequivecs: Vectors have to be equivariant') 
    return True

def isequivariant(q,gtheta,grho,vs):
    """
    Input:
    q: vector of equivariant polynomial
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables

    Output:
    True if q is a vector of equivariant polynomial w.r.t g, False otherwise
    """
    qequi = proj2equi2(q,vs,gtheta,grho)
    qequi = sp.Matrix(qequi*1/order(gtheta))
    if iszero(sp.Matrix(qequi-q)):
        return True
    else:
        return False

def proj2equi2(vm,vs,gtheta,grho):
    """
    Input:
    vm: vector of equivariant polynomial
    vs: list of SymPy symbols variables  
    gtheta: representation of g
    grho: representation of g
    
    Output:
    s: all group elements of g (rep gtheta) and their inverse (rep grho) applied to vm 
    """
    s = vector(dimension(grho))
    v = sp.Matrix(vs)
    elems = gtheta['allelements']
    for lhs, rhs in elems.items():
        v2 = convert(sp.Matrix(rhs*v),'list')
        ll = [(v[i,0],v2[i][0]) for i in range(len(v2))] #v[i,:] v2[i]
        vp = vm.subs(ll,simultaneous=True) #subs(ll,vm) 
        vp = sp.Matrix(getinverse(grho,lhs) * vp)
        vp = expand(vp)
        s = sp.Matrix(vp+s)
    return sp.Matrix(s)

def equimolien_(g1,g2,l):
    """
    Input:
    g1: representation of g
    g2: representation of g
    l: SymPy symbol
    
    Output:
    s: Hilbert series of the module of equivariants
    """
    print('equimolien_: Calculating equivariant Molien series.')
    s = 0
    n = dimension(g1)
    def a_fn(i,j):
        return 1 if i==j else 0 
    idx = matrix(n,n,a_fn)
    for lhs, rhs in g1['allelements'].items():
        t = trace(getinverse(g2,lhs))
        if t != 0:
            u = det(sp.Matrix(idx-l*rhs))
            s = s + t/u
            s = normal(s)
    s = s/order(g1)
    print('equimolien_: Starting poincaresimp.')
    s = poincaresimp(s,l)
    return s

def vecnorming(v,vs):
    """
    Input:
    v: vector of polynomials
    vs: list of SymPy symbols variables

    Output:
    normalised v
    """
    n = vectdim(v)
    i = 0
    norm = 0
    while i<n:
        if v[i,0] != 0: #v[i,:]
            norm = lcoeff(v[i,0],vs) #v[i,:]
            break
        else:
            i += 1
    if norm == 0:
        raise('vecnorming: Zero vector in vecnorming.')
    vv = v.copy()
    return sp.Matrix(vv*1/norm)

MONOMS = []

def equi_ofthisdegree(gtheta,grho,vs,d,m_in,m_out,cin,cout):
    """
    Inputs:
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables
    d: int degree
    m_in: SymPy monomial
    m_out: string
    cin: int index
    cout: string

    Output:
    equi: equivariant of degree d
    """
    m = m_in
    i = cin 
    N = dimension(grho)
    equi = vector(N)
    mv = equi.copy()
    # while iszero(equi) and mv is not False:
    #     if i == N-1: #N
    #         i = 0 #1
    #         m = nextmonom(m,vs,d)
    #         if m is False:  
    #             raise('equi_ofthisdegree: No more candidates.')
    #     else:
    #         i += 1
    #     mv[i,0] = m
    #     equi = proj2equi2(sp.Matrix(mv),vs,gtheta,grho)
    #     mv[i,0] = 0
    # m_out = m
    # cout = i
    # return sp.Matrix(equi),m_out,cout

    while iszero(equi) and mv is not False:
        m = nextmonom(m,vs,d)
        MONOMS.append((m,i))
        if m is False:
            mv[i,0] = 0
            if i < N-1:
                i += 1
                m = 0
            else:
                raise('equi_ofthisdegree: No more candidates of this degree.')
        else:
            # mcopy = m.copy()
            mv[i,0] = m #mcopy.as_expr()
            equi = proj2equi2(sp.Matrix(mv),vs,gtheta,grho)
    m_out = m
    cout = i
    return sp.Matrix(equi),m_out,cout


def equivariantsd(gtheta,grho,vs,prims,ll,k,Mol):
    """
    ll: list of candidates for equivariants
    """
    M = Mol
    varl = list(sp.Expr(M).free_symbols)
    if len(varl) > 1:
        raise('equivariantsd: Molien series has more than one variable.')
    else:
        var = varl[0]
    sec = []
    candis = ll
    # start at degree
    print(M.__class__)
    cd, q = tcoeff(M,var,'q')
    d = degree(q,var)
    j = 0
    # prepare memberofmoduletest
    grading = {str(elem):1 for elem in vs}
    grading['minint'] = 0
    grading['maxint'] = min(degree(M,var),k)
    tt = mktermorder(vs,'tdeg')
    gb = gradgroebner(prims,vs,tt,grading)
    normalfsec = []
    n = dimension(grho)
    slacks = [sp.Symbol('_z'+str(i)) for i in range(n)]
    tt = mktermorder(slacks + vs,'tdeg')
    # start computation 
    while M!=0 and d<=k:
        cd, q = tcoeff(M,var,'q')
        d = degree(q,var)
        if d>k:
            break
        m = 0
        c = 0 #1
        i = 0
        while i < len(candis) and cd>0:
            q = candis[i]
            if vecdegree(q)==d:
                norq = normalform(vec2pol(q,slacks), gb,tt[vs],tt)
                if norq != 0 and linalgindependent(norq,normalfsec,slacks + vs):
                    M = M-var**d
                    cd = cd-1
                    q = vecnorming(sp.Matrix(q),vs)
                    sec.append(sp.Matrix(q))
                    print('equivariantsd: Use user suggestion as equivariant ',q)
                    normalfsec.append(norq)
                    del candis[i]
                    if cd == 0:
                        i = len(candis)
            else:
                i += 1
        while cd>0:
            j += 1
            q, m, c = equi_ofthisdegree(gtheta,grho,vs,d,m,'m',c,'c')
            print('equivariantsd: ',j,'. equivariant candidate: ',q)
            if q is False:
                raise('equivariantsd: Something wrong.')
            norq = normalform(vec2pol(q,slacks),gb,tt['vs'],tt)
            print('equivariantsd: ',j,'. candidate normal form: ',norq)
            if norq != 0 and linalgindependent(norq,normalfsec,slacks + vs):
                M = M-var**d
                cd = cd-1
                q = vecnorming(sp.Matrix(q),vs)
                sec.append(sp.Matrix(q))
                f = open("D4RegularRep20.txt", "a")
                f.write("\n"+str(sp.Matrix(q))+"\n")
                f.close()
                userinfo('equivariantsd: Equivariant found: ',j,'. candidate ',q)
                normalfsec.append(norq)
    if M != 0:
        print('equivariantsd: Warning, some equivariants are still missing.')
    return sec

def equivariantsdMNISTD4(gtheta,grho,vs,prims,ll,k,Mol):
    """
    ll: list of candidates for equivariants
    """
    M = Mol
    varl = list(sp.Expr(M).free_symbols)
    if len(varl) > 1:
        raise('equivariantsd: Molien series has more than one variable.')
    else:
        var = varl[0]
    sec = []
    candis = ll
    # start at degree
    print(M.__class__)
    cd, q = tcoeff(M,var,'q')
    d = degree(q,var)
    j = 0
    # prepare memberofmoduletest
    grading = {str(elem):1 for elem in vs}
    grading['minint'] = 0
    grading['maxint'] = min(degree(M,var),k)
    tt = mktermorder(vs,'tdeg')
    gb = list(grob_inte_sympy(prims, *vs, order='grevlex').args[0])# gradgroebner(prims,vs,tt,grading)
    normalfsec = []
    n = dimension(grho)
    slacks = [sp.Symbol('_z'+str(i)) for i in range(n)]
    tt = mktermorder(slacks + vs,'tdeg')
    # start computation 
    while M!=0 and d<=k:
        cd, q = tcoeff(M,var,'q')
        d = degree(q,var)
        if d>k:
            break
        m = 0
        c = 0 #1
        i = 0
        while i < len(candis) and cd>0:
            q = candis[i]
            if vecdegree(q)==d:
                st = time.time()
                norq = normalform(vec2pol(q,slacks), gb,tt[vs],tt)
                print('equivariantsd: Time in normalform ',time.time()-st)
                st = time.time()
                if norq != 0 and linalgindependent(norq,normalfsec,slacks + vs):
                    print('equivariantsd: Time in linalgindependent ',time.time()-st)
                    M = M-var**d
                    cd = cd-1
                    q = vecnorming(sp.Matrix(q),vs)
                    sec.append(sp.Matrix(q))
                    print('equivariantsd: Use user suggestion as equivariant ',q)
                    normalfsec.append(norq)
                    del candis[i]
                    if cd == 0:
                        i = len(candis)
            else:
                i += 1
        while cd>0:
            j += 1
            q, m, c = equi_ofthisdegree(gtheta,grho,vs,d,m,'m',c,'c')
            print('equivariantsd: ',j,'. equivariant candidate: ',q)
            print('q',q)

            if d == 0:
                M = M-var**d
                cd = cd-1
                q = vecnorming(sp.Matrix(q),vs)
                if q not in sec:
                    sec.append(sp.Matrix(q))
                    f = open("D4MNISTequivariantsFINAL.txt", "a")
                    f.write("\n"+str(sp.Matrix(q))+"\n")
                    f.close()
            else:
                q = vecnorming(sp.Matrix(q),vs)
                if q not in sec:
                    sec.append(sp.Matrix(q))
                    f = open("D4MNISTequivariantsFINAL.txt", "a")
                    f.write("\n"+str(sp.Matrix(q))+"\n")
                    f.close()
            
            # if q is False:
            #     raise('equivariantsd: Something wrong.')
            # st = time.time()
            # norq = normalform(vec2pol(q,slacks),gb,tt['vs'],tt)
            # print('equivariantsd: Time in normalform ',time.time()-st)
            # print('equivariantsd: ',j,'. candidate normal form: ',norq)
            # st = time.time()
            # if norq != 0 and linalgindependent(norq,normalfsec,slacks + vs):
            #     print('equivariantsd: Time in linalgindependent ',time.time()-st)
            #     M = M-var**d
            #     cd = cd-1
            #     q = vecnorming(sp.Matrix(q),vs)
            #     sec.append(sp.Matrix(q))
            #     f = open("message.txt", "w")
            #     f.write("\n"+str(sp.Matrix(q))+"\n")
            #     f.close()
            #     userinfo('equivariantsd: Equivariant found: ',j,'. candidate ',q)
            #     normalfsec.append(norq)
    if M != 0:
        print('equivariantsd: Warning, some equivariants are still missing.')
    return sec

def equivariantsd64D4(gtheta,grho,vs,prims,ll,k,Mol):
    """
    ll: list of candidates for equivariants
    """
    M = Mol
    varl = list(sp.Expr(M).free_symbols)
    if len(varl) > 1:
        raise('equivariantsd: Molien series has more than one variable.')
    else:
        var = varl[0]
    sec = []
    candis = ll
    # start at degree
    cd, q = tcoeff(M,var,'q')
    d = degree(q,var)
    j = 0
    # prepare memberofmoduletest
    grading = {str(elem):1 for elem in vs}
    grading['minint'] = 0
    grading['maxint'] = min(degree(M,var),k)
    tt = mktermorder(vs,'tdeg')
    # gb = list(grob_inte_sympy(prims, *vs, order='grevlex').args[0])# gradgroebner(prims,vs,tt,grading)
    normalfsec = []
    n = dimension(grho)
    slacks = [sp.Symbol('_z'+str(i)) for i in range(n)]
    tt = mktermorder(slacks + vs,'tdeg')
    # start computation 
    while M!=0 and d<=k:
        cd, q = tcoeff(M,var,'q')
        d = degree(q,var)
        if d>k:
            break
        m = 0
        c = 0 #1
        i = 0
        # while i < len(candis) and cd>0:
        #     q = candis[i]
        #     if vecdegree(q)==d:
        #         st = time.time()
        #         norq = normalform(vec2pol(q,slacks), gb,tt[vs],tt)
        #         print('equivariantsd: Time in normalform ',time.time()-st)
        #         st = time.time()
        #         if norq != 0 and linalgindependent(norq,normalfsec,slacks + vs):
        #             print('equivariantsd: Time in linalgindependent ',time.time()-st)
        #             M = M-var**d
        #             cd = cd-1
        #             q = vecnorming(sp.Matrix(q),vs)
        #             sec.append(sp.Matrix(q))
        #             print('equivariantsd: Use user suggestion as equivariant ',q)
        #             normalfsec.append(norq)
        #             del candis[i]
        #             if cd == 0:
        #                 i = len(candis)
        #     else:
        #         i += 1
        while cd>0:
            j += 1
            q, m, c = equi_ofthisdegree(gtheta,grho,vs,d,m,'m',c,'c')
            if d == 0:
                M = M-var**d
                cd = cd-1
                # q = vecnorming(sp.Matrix(q),vs)
                if q not in sec:
                    print('equivariantsd: ',j,'. equivariant candidate: ',q)
                    sec.append(sp.Matrix(q))
                    f = open("D464equivariants.txt", "a")
                    f.write("\n"+str(sp.Matrix(q))+"\n")
                    f.close()
            else:
                # q = vecnorming(sp.Matrix(q),vs)
                if q not in sec:
                    print('equivariantsd: ',j,'. equivariant candidate: ',q)
                    sec.append(sp.Matrix(q))
                    f = open("D464equivariants.txt", "a")
                    f.write("\n"+str(sp.Matrix(q))+"\n")
                    f.close()
            
            # if q is False:
            #     raise('equivariantsd: Something wrong.')
            # st = time.time()
            # norq = normalform(vec2pol(q,slacks),gb,tt['vs'],tt)
            # print('equivariantsd: Time in normalform ',time.time()-st)
            # print('equivariantsd: ',j,'. candidate normal form: ',norq)
            # st = time.time()
            # if norq != 0 and linalgindependent(norq,normalfsec,slacks + vs):
            #     print('equivariantsd: Time in linalgindependent ',time.time()-st)
            #     M = M-var**d
            #     cd = cd-1
            #     q = vecnorming(sp.Matrix(q),vs)
            #     sec.append(sp.Matrix(q))
            #     f = open("message.txt", "w")
            #     f.write("\n"+str(sp.Matrix(q))+"\n")
            #     f.close()
            #     userinfo('equivariantsd: Equivariant found: ',j,'. candidate ',q)
            #     normalfsec.append(norq)
    if M != 0:
        print('equivariantsd: Warning, some equivariants are still missing.')
    return sec

def equivariants(gtheta,grho,vs,pri,ll,k,M):
    """
    ll: list of candidates for equivariants
    """
    return equivariantsd(gtheta,grho,vs,pri,ll,k,M)

def equis_(gtheta,grho,vs,pri,ll,k):
    """
    Inputs:
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables
    pri: list of primary invariants
    ll: list of
    k: max degree till which to search for equivariants

    Output:
    dict of primary gtheta invariants and gtheta grho equivariants
    """
    # determine primary invariants
    prims = pri
    if len(prims) == 0:
        print('equis_: Getting invariants.')
        prims = primaries2(gtheta,vs,[])
    # next
    var = sp.Symbol('_t')
    #WARNING HARD CODED FOR NOW IN degmultofsecondariesMNISTD4, M = (1 - var)**(-784) 
    M = equimolien_(gtheta,grho,var)
    print('equis_: Equivariant Molien series is ',M)
    sec = []
    #  determine degrees of equivariants
    #WARNING HARD CODED FOR NOW, M = degmultofsecondariesMNISTD4(M)
    M = degmultofsecondaries(M,prims,vs)
    # search for equivariants
    if M != 0: 
        #WARNING HARD CODED FOR NOW, sec = equivariantsdMNISTD4(gtheta,grho,vs,prims,ll,k,M)
        sec = equivariants(gtheta,grho,vs,prims,ll,k,M)
    return {'primary_invs':prims,'equivariants':sec}

def equis_D4MNIST(gtheta,grho,vs,pri,ll,k):
    """
    Inputs:
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables
    pri: list of primary invariants
    ll: list of
    k: max degree till which to search for equivariants

    Output:
    dict of primary gtheta invariants and gtheta grho equivariants
    """
    # determine primary invariants
    prims = pri
    # next
    var = sp.Symbol('_t')
    M = (1 - var)**(-784) 
    print('equis_: Equivariant Molien series is ',M)
    sec = []
    #  determine degrees of equivariants
    M = degmultofsecondariesMNISTD4(M)
    # search for equivariants
    if M != 0: 
        sec = equivariantsdMNISTD4(gtheta,grho,vs,prims,ll,k,M)
    return {'primary_invs':prims,'equivariants':sec}

def equis_D4smallMNIST(gtheta,grho,vs,pri,ll,k):
    """
    Inputs:
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables
    pri: list of primary invariants
    ll: list of
    k: max degree till which to search for equivariants

    Output:
    dict of primary gtheta invariants and gtheta grho equivariants
    """
    # determine primary invariants
    prims = pri
    # next
    var = sp.Symbol('_t')
    M = (1 - var)**(-64) 
    print('equis_: Equivariant Molien series is ',M)
    sec = []
    #  determine degrees of equivariants
    M = degmultofsecondariessmallMNISTD4(M)
    # search for equivariants
    if M != 0: 
        sec = equivariantsdMNISTD4(gtheta,grho,vs,prims,ll,k,M)
    return {'primary_invs':prims,'equivariants':sec}

def equis_D464(gtheta,grho,vs,pri,ll,k):
    """
    Inputs:
    gtheta: representation of g
    grho: representation of g
    vs: list of SymPy symbols variables
    pri: list of primary invariants
    ll: list of
    k: max degree till which to search for equivariants

    Output:
    dict of primary gtheta invariants and gtheta grho equivariants
    """
    # determine primary invariants
    prims = pri
    # next
    var = sp.Symbol('_t')
    M = (1 - var)**(-(64**2)) 
    print('equis_: Equivariant Molien series is ',M)
    sec = []
    #  determine degrees of equivariants
    M = degmultofsecondaries64D4(M)
    # search for equivariants
    if M != 0: 
        sec = equivariantsd64D4(gtheta,grho,vs,prims,ll,k,M)
    return {'primary_invs':prims,'equivariants':sec}

def equis(*args):
    """
    """
    nargs = len(args)
    if nargs>6:
        raise('equis: Too many arguments.')
    if nargs<3:
        raise('equis: Too few arguments.')
    gtheta, grho, vs = args[0], args[1], args[2]
    if not(isinstance(gtheta,dict)):
        raise('equis: First argument should contain the data structure of a finite group.')
    if not(isinstance(grho,dict)):
        raise('equis: Second argument should contain the data structure of a finite group.')
    if not(arerepofsamegroup(gtheta,grho)):
        raise('equis: First and second argument should be representations of the same group.')
    # 'symmetry/src/internal/arevars'(vs);
    n = dimension(gtheta)
    if (len(vs) != n):
        raise('equis: Number of vars should equal dimension of first representation.')
    k = sp.oo
    ll = []
    prims = []
    if nargs>3:
        if is_type(args[3],'integer') and args[3]>0: 
            k = args[3] 
            if nargs>4:
                raise('equis: Fourth and fifth argument in wrong ordering.')
        elif isinstance(args[3],list) or all(is_type(elem,'polynom',vs) for elem in args[3]):
            # areprimaries(args[3],gtheta,vs)
            prims = args[3]
            if nargs>4:
                if is_type(args[4],'integer') and args[4]>0:
                    k = args[4]
                    if nargs>5:
                        raise('equis: Fifth and sixth argument in wrong ordering.')
                else:
                    areequivecs(args[4],gtheta,grho,vs)
                    ll = args[4]
                    if nargs == 6:
                        if is_type(args[5],'integer') and args[5]>0:
                            k = args[5]
                        else:
                            raise('equis: Wrong sixth argument.')
        else:
            raise('equis: wrong third argument')
    return equis_(gtheta,grho,vs,prims,ll,k)



# gtheta = dihedral(4,[1,2,3,4,5,5])

# def roll(n):
#     res = sp.zeros(8)
#     l = [(j+n) % 8 for j in range(8)]
#     for i,j in enumerate(l):
#         res[i,j] = 1
#     return sp.Matrix(np.roll(np.eye(8),-n,0).tolist()) #res

# g = {}
# g['oname'] = 'permutation8'
# g['allelements']={}
# g['allelements']['_r0'] = sp.eye(8)
# g['allelements']['_r1'] = roll(1)
# g['allelements']['_r2'] = roll(2)
# g['allelements']['_r3'] = roll(3)
# g['allelements']['_s_r0'] = roll(4)
# g['allelements']['_s_r1'] = roll(5)
# g['allelements']['_s_r2'] = roll(6)
# g['allelements']['_s_r3'] = roll(7)
# g['invelements']={}
# g['invelements']['_r0'] = sp.Matrix(g['allelements']['_r0'].inv())
# g['invelements']['_r1'] = sp.Matrix(g['allelements']['_r1'].inv())
# g['invelements']['_r2'] = sp.Matrix(g['allelements']['_r2'].inv())
# g['invelements']['_r3'] = sp.Matrix(g['allelements']['_r3'].inv())
# g['invelements']['_s_r0'] = sp.Matrix(g['allelements']['_s_r0'].inv())
# g['invelements']['_s_r1'] = sp.Matrix(g['allelements']['_s_r1'].inv())
# g['invelements']['_s_r2'] = sp.Matrix(g['allelements']['_s_r2'].inv())
# g['invelements']['_s_r3'] = sp.Matrix(g['allelements']['_s_r3'].inv())
# g['generators']={}
# g['generators']['_s'] = g['allelements']['_s_r0']
# g['generators']['_r'] = g['allelements']['_r0']
