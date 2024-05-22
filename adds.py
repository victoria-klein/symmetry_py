
#-------------------------------------------------------------#
#-------------------------------------------------------------#
#--------------------------OFTHISDEGREE-----------------------#
#-------------------------------------------------------------#
#-------------------------------------------------------------#

def ofthisdegree(g,vs,d,m):
    inv = 0
    while inv == 0 and m != False:
        m = nextmonom(m,vs,d) #SymPy monomial
        if m != False:
            inv = proj2inv2(m,vs,g)
        else:
            inv = False
    return mout, inv

def nextmonom(m,vs,d):
    if m == 0:
        vs[0]**d
    if len(vs) == 1:
        return False
    else:
        n1 = degree(m, vs[0])
        m1 = nextmonom(expand(m/x1**n1),vs[1:],d-n1)
        if m1 != False:
            return expand(vs[0]**n1*m1)
        else:
            if n1 == 0:
                return False
            else:
                return expand(vs[0]**(n1-1)*vs[1]**(d-n1+1))

def proj2inv2(m,vs,g):
    s = 0
    v = convert(vs,'vector')
    mats = map(rhs,g['allelements'])
    for ma1 in mats:
        v2 = convert(ma1 * v, 'list')
        ll = [sp.Eq(v[i],v2[i]) for i in range(len(v2))]
        p = m.subs(ll)
        s = (p+s).expand
    return s

#-------------------------------------------------------------#
#-------------------------------------------------------------#
#--------------------------PRIMARYSECONDS---------------------#
#-------------------------------------------------------------#
#-------------------------------------------------------------#

def primaryseconds(g, vs, M, invs):
    Q = primcand2Q(g, vs, M, invs)
    sec, prims = Q2primesec(Q, vs)
    return sec,prims

def primcand2Q(g, vs, M, invs):
    n = len(vs)
    Q = []
    for i in range(len(invs)):
        if isradicalequal(vs, Q):
            return(Q)
        if not(ismemberofradical(invs[i], Q, vs)):
            print('Use suggestion by user as primary candidate ', invs[i],'.')
            Q = [op(Q),invs[i]]
    m = 0
    while not(isradicalequal(vs, Q)):
        update = False
        while not(update):
            m, p = nextcandidate(g, vs, Q, M, m)
            if not(ismemberofradical(p, Q, vs)):
                print('Candidate for primaries found ', p,'.')
                Q = [op(Q),p]
                update = True
    return(Q)

def isradicalequal(vs, Q):
    if nops(Q)<1:
        return False
    n = nops(vs)
    k = 1
    flag = True
    while flag and k < n:
        ch = [(vs[i],0) for i in range(k-1)]+[(vs[k-1],1)] #i:='i';
        polys = Q.subs(ch)
        vs2 = sp.indets(polys)
        if len(vs2) > 0:
            tt = mktermorder(vs2, 'tdeg')
            gb = gradgroebner(polys, vs2, op(tt))
            if not(gb=[1]):
                flag = False
        else:
            flag = False
        k += 1
    return(flag)


def nextcandidate(g,vs,Q,M,m):
    #g - dict, vs - list of sympy symbols, Q - list of polys, M - int, m - int
    var = op(indets(M))
    p = convert(series(M,sp.Eq(var,0),20),'polynom')-1
    if p == 0:
        raise('nextcandidate: Unexpected case.')
    if m == 0:
        d = tcoeff(p, var, 'd'):
        d = degree(d, var)
    else:
        d = degree(m,set(vs))
        c = coeff(p, var, d)
        for q in Q:
            if degree(q, set(vs)) == d:
                c = c-1
        if c <= 0:
            for i in range(1,d+1):
                p = p - coeff(p, var**i)*var**i
            d = tcoeff(p, var, 'd')
            d = degree(d, var)
            m = 0
    mout, q = ofthisdegree(g,vs,d,m)
    while not(q):
        for i in range(1,d+1):
            p = p - coeff(p, var**i)*var**i
        if p == 0:
            raise('nextcandidate: Unexpected case.')
        d = tcoeff(p, var, 'd')
        m = 0
        mout, q = ofthisdegree(g,vs,d,m)
    q = q / lcoeff(q,vs)
    return m, Q


def ismemberofradical(p,Q,vs):
    # p, Q, vs
    if Q == []:
        return False
    vars2 = union(indets(Q),indets(p))
    _Z = sp.Symbol('_Z')
    if not(member(_Z,vars2)):
        vars2 = [op(vars2),_Z]
    else:
        raise('ismemberofradical: Implement new slack variable.')
    tt = mktermorder(vars2, 'tdeg')
    gb = gradgroebner([op(Q),1-_Z*p],vars2,op(tt))
    if gb == [1]:
        return True
    else:
        return False
    return

def Q2primesec(Qq,vs):
    print('There are ',nops(Qq),' candidates for primaries.')
    Q = Qq
    sec = []
    n = nops(vs)
    while nops(Q) > n:
        for i in [nops(Q)-2-i for i in range(nops(Q)+2)]:
            if isradicalequal(vs,[op(Q[:i-1]),op(Q[i:])]):
                sec = [op(sec),Q[i]]
                print('Q2primsec: Drop candidate ', Q[i])
                Q = [op(Q[:i-1]),op(Q[i:])]
                i = -2
                break
        if i == 0 and nops(Q) > n:
            raise('Noether normalization/impl. of al.g for system of homogenous parameters necessary.')
    print('Q2primsec: Primaries found: ', Q)
    return sec, Q

def degmultofsecondaries(M, prims, vs):
    var = op(indets(M))
    pol = M
    for q in prims:
        pol = pol*(1-var**degree(q,set(vs)))
    pol = sp.expand(normal(sp.expand(pol)))
    if not(is_type(pol, 'polynom') and var in pol.free_symbols):
        raise('degmultofsecondaries: Pol not polynomial in var.')
    print('Pol with degs and multiplicities of generators of free module ', pol)
    return pol

def secondaries(g,vs,prims,sec,M,k):
    var = op(indets(M))
    sec = s
    cd = tcoeff(M,var,'q')
    d = degree(q,var)
    j = 0
    if d > k:
        return sec
    grading = table([sp.Eq(vs[i],1) for i in range(len(vs))])
    grading['minint'] = 0
    grading['maxint'] = min(degree(M,var),k)
    tt = mktermorder(vs,'tdeg')
    gb = gradgroebner(prims,vs,op(tt),op(grading))
    normalfsec = map(normalform,sec,gv,vs,op(tt))
    while M != 0 and d <= k:
        cd, q = tcoeff(M,var,'q')
        d = degree(q,var)
        if d > k:
            break
        m = 0
        while cd > 0:
            j += 1
            m, q = ofthisdegree(g,vs,d,m)
            print('Secondary candidate ',q,'.')
            if q == False:
                raise('secondaries: Something wrong.')
            normalfq = normalform(q,gb,vs,op(tt))
            print('Normal form of secondary candidate ',normalfq,'.')
            if normalfq != 0 and linalgindependent(normalfq,normalfsec,vs):
                M = M - var**d
                cd = cd - 1
                q = q/lcoeff(q,vs)
                print('Secondary found = candidate',j,q,'.')
                sec = [op(sec),q]
                normalfsec = [op(normalfsec),normalfq]
    if M != 0:
        print('Warning, some secondary invariants are still missing.')
    return sec


def linalgindependent(g,qs,vs):
    d = degree(g,set(vs))
    qds = []
    for i in range(len(qs)):
        if degree(qs[i],set(vs)) == d:
            qds = [op(qds),qs[i]]
    if nops(qds) == 0:
        return True
    pol = _b*g+convert([_a.i*qds[i] for i in range(len(qds))],'+')
    eqs = list(sp.expand(pol))
    for i in range(len(vs)):
        neweqs = []
        for pol in eqs:
            neweqs = [op(neweqs), coeffs(collect(pol,vs[i]),vs[i])]
        eqs = neweqs
    res = sp.solve(set(eqs),set([_b]+[_a.i for i in range(qds)]))
    if member(sp.Eq(_b,0),res):
        return True
    else:
        return False

def normalform(*args):
    nargs = len(args)
    if nargs < 4:
        raise('normalform: Too few arguments.')
    elif nargs > 4:
        raise('normalform: Too many arguments.')
    f,polys,X,to = args[0],args[1],args[2],args[3]
    if not(is_type(X,'list')):
        raise('normalform: Variables X must be a list.')
    if not(all(is_type(elem,'name') for elem in X)):
        raise('normalform: Third argument should be variable list.')
    # if not(istermorder(to)):
    #     raise('normalform: Fourth argument should be termorder.')
    if not(equal(X,to['vs'])):
        raise('normalform: Variables X should and those in termorder must be the same.')
    if not(is_type(polys,['list','set'])):
        raise('normalform: Second argument has to be a lisst or set of polynomials.')
    if not(is_type(f,'polynom') and f.free_symbols in X):
        raise('normalform: First argument must be a polynomial in X.')
    if not(all(is_type(elem,'polynom') for elem in polys) and all(elem.free_symbols in X for elem in polys)):
        raise('normalform: Polys input must be polynomials over X.')
    fb = expand(f)
    if fb == 0:
        return 0
    else
        fb, fcont = primpart(fb,X,'fcont')
        fb = fb.subs([sp.Eq(Catalan,sp.Catalan),sp.Eq(Pi,sp.Pi),sp.Eq(E,sp.EulerGamma),sp.Eq(gamma,sp.Gamma)])
    nzbasis = []
    Y = indets(fb)
    for p in polys.subs([sp.Eq(Catalan,sp.Catalan),sp.Eq(Pi,sp.Pi),sp.Eq(E,sp.EulerGamma),sp.Eq(gamma,sp.Gamma)]) do
        ep = expand(p)
        if ep != 0:
            nzbasis = [op(nzbasis),p]
            Y = union(Y,indets(ep))
    if nops(nzbasis) == 0:
        fb = fb.subs([sp.Eq(Catalan,sp.Catalan),sp.Eq(Pi,sp.Pi),sp.Eq(E,sp.EulerGamma),sp.Eq(gamma,sp.Gamma)])
        return fcont*collect(fb,X,distributed)
    if minus(Y,set(X)) != {} or minus(Y,set(X)) == sp.EmptySet:
        dom = 'poly'
    else:
        dom = 'inte'
    dom2 = arerationalpols(nzbasis,X)
    if dom2:
        dom2 = arerationalpols([fb],X)
        if dom == 'inte':
            answer = fcont*nf(fb,sp.expand(list(op(nzbasis))),X,to)
        else:
            G = map(ptotab,set(op(nzbasis)),[X for i in range(len(op(nzbasis)))])
            fb = ptotab(fb,X)
            fb = nf(fb,G,X,tp)
            fcont, fb = rmul(fcont,fb)
            answer = tabtop(fb,X,to)
    else:
        G = expand(list(op(nzbasis)))
        vv, pp, gg, G = convert([fb,op(G)],Y,'vv','pp','gg')
        fb = G[0]
    # test whether leading coefficients depend on roots
        HCs = map(hcoeff,G[1:],to['vs'],to)
        if nops(indets(HCs)) > 0: 
            raise('normalform: Case of leading coefficients depending on roots is not implemented yet.') 
    #   start division algorithm
        G = [op(G[1:]),op(pp)]
        tt = mktermorder([op(X),op(vv)],'blocked',to,mktermorder(vv,'plex'))
        Xx = [op(X),op(vv)]
        if dom == inte: 
            answer = fcont*nf(fb,G,Xx,op(tt))
        else
            G = map(ptotab,[op(G),op(pp)],[Xx for i in range(len([op(G),op(pp)]))])
            fb = ptotab(fb,X)
            fb = nf(fb,G,Xx,op(tt))
            fcont, fb = rmul(fcont,fb)
            answer = tabtop(fb,Xx,op(tt))
        answer = convertback1(answer,Xx,vv,pp,gg)
    answer = answer.subs([sp.Eq(Catalan,sp.Catalan),sp.Eq(Pi,sp.Pi),sp.Eq(E,sp.EulerGamma),sp.Eq(gamma,sp.Gamma)])
    return answer

import time

def head_poly(t,X,to):
    """
    Output:
    Leading monomial and coefficient
    """
    return thterm(map(op,{indices(t)}),X,to)

def thterm(s,X,to):
    """
    """
    newset = head2(s,X,to)
    if nops(newset) > 1:
       newset = head2(newset,X,{'vs':X,'ordername':plex})
    if nops(newset)>1:
        raise('thterm: Nonuniqueness.')
    hc, ht = lcoeff(op(1,newset),X,'ht')
    return ht

def head2(monomset,X,to):
    """
    """

def nf(f,G,X,to):
#    local i,k,rest,temp,temp2,j,HT,HC,Sugar,scale,oldcont,newcont,ht,lowest,st,stt,sres;
    if (indices(f) = NULL):
        return op(f)
    if nops(G) == 0:
        return op(f)
    stt = time.time()
    HT = [head(G[i],X,to) for i in range(len(G))]
    i = 'i'
    HC = [seq(G[i][ HT[i] ],range(len(G)))]
    Sugar = [sugardegree(G[i],X) for i in range(len(G))]
    lowest = HT[0]
    for j in range(1,len(G)+1):
        if isgreaterorder2(lowest,HT[j],to):
           lowest = HT[j]
    k = table(sparse)
    temp = table(sparse)
    rest = table(sparse)
    rest = f
    ht = head(rest,X,termorder) 
    oldcont = gcont(rest,ht)
    sres = degree(ht[2],set(X))
    while indices(rest) != NULL:
        if not(isgreaterorder2(ht,lowest,to)):
            oldcont = rmul(oldcont,rest)
            k = addto(k,rest,1)
            break
        scale, newcont, sres, temp2 = sred(rest,sres,G,HT,HC,Sugar,X,to,oldcont,'scale','newcont','sres')
        temp = op(1,temp2)
        if indices(temp) == NULL:
            break
        ht = head(temp,X,to)
        oldcont = normal(newcont/scale)
        k[ht] = normal(oldcont*temp[ht])
        temp[ht] = evaln(temp[ht])
        rest = op(temp)
        ht = head(rest,X,termorder)
    st = time.time()
    print('nf: Time: ',st,', elapsed ',st-stt)
    return op(k)


def rmul(r,p):
    """
    Input:
    r: coefficient that is rational SymPy function
    p: (list, set, sp.Matrix, np.ndarray or dict) of SymPy polynomials p

    Output:
    p: (list, set, sp.Matrix, np.ndarray or dict) normal fom of p rescaled by r
    """
    inds = map(op,indices(p))
    for x in inds:
        p[x] = normal(r*p[x])
    return p




def tabtop(t,X,to):
    """
    Input:
    t: (list, set, sp.Matrix, np.ndarray or dict) of SymPy polynomials p
    X: list of variables SymPy symbols
    to: dict of term order

    Output:
    p: sum of polynomials in t rescaled by indices in t
    """
    inds = map(op, indices(t))
    p = 0
    for z in inds:
        p = p + t[z]*z
    return p

def convertback1(p,X,algvars,algpols,alggls):
    """
    Input:
    p: SymPy polynomial p
    alggls: SymPy equations in variables X

    Output:
    p: SymPy polynomial p as a radical
    """
    return convert(p.subs(alggls),'radical')
