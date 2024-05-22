from internal import *
import numpy as np
import sympy
from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
from sympy.polys import polyoptions as options
from sympy.polys.polyerrors import (
    OperationNotSupported, DomainError,
    CoercionFailed, UnificationFailed,
    GeneratorsNeeded, PolynomialError,
    MultivariatePolynomialError,
    ExactQuotientFailed,
    PolificationFailed,
    ComputationFailed,
    GeneratorsError,
)
from sympy.polys.polytools import parallel_poly_from_expr, Poly, GroebnerBasis, _parallel_poly_from_expr
from grob_inte import grob_internal_inte
from sympy.polys.monomials import Monomial
from sympy.polys.rings import PolyRing
from sympy import Expr
from sympy.polys.groebnertools import s_poly, lbp, sig, Sign, Polyn, cp_key, critical_pair, f5_reduce, Num, is_rewritable_or_comparable
from sympy.external.pythonmpq import PythonMPQ

from math import gcd
import time as time
from tqdm import tqdm


import bisect
from types import FunctionType
from sympy.matrices.utilities import _iszero, _simplify
from sympy.core.singleton import S
from reductions import _row_reduce_list, _row_reduce_nump

def grob_inte_sympy(F, *gens, **args):
    """
    Computes the reduced Groebner basis for a set of polynomials.

    Use the ``order`` argument to set the monomial ordering that will be
    used to compute the basis. Allowed orders are ``lex``, ``grlex`` and
    ``grevlex``. If no order is specified, it defaults to ``lex``.

    For more information on Groebner bases, see the references and the docstring
    of :func:`~.solve_poly_system`.

    Examples
    ========

    Example taken from [1].

    >>> from sympy import groebner
    >>> from sympy.abc import x, y

    >>> F = [x*y - 2*y, 2*y**2 - x**2]

    >>> groebner(F, x, y, order='lex')
    GroebnerBasis([x**2 - 2*y**2, x*y - 2*y, y**3 - 2*y], x, y,
                  domain='ZZ', order='lex')
    >>> groebner(F, x, y, order='grlex')
    GroebnerBasis([y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y], x, y,
                  domain='ZZ', order='grlex')
    >>> groebner(F, x, y, order='grevlex')
    GroebnerBasis([y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y], x, y,
                  domain='ZZ', order='grevlex')

    By default, an improved implementation of the Buchberger algorithm is
    used. Optionally, an implementation of the F5B algorithm can be used. The
    algorithm can be set using the ``method`` flag or with the
    :func:`sympy.polys.polyconfig.setup` function.

    >>> F = [x**2 - x - 1, (2*x - 1) * y - (x**10 - (1 - x)**10)]

    >>> groebner(F, x, y, method='buchberger')
    GroebnerBasis([x**2 - x - 1, y - 55], x, y, domain='ZZ', order='lex')

    References
    ==========

    1. [Buchberger01]_

    """
    # GroebnerBasis(F, *gens, **args)
    # """Compute a reduced Groebner basis for a system of polynomials. """
    # print('grob_inte_sympy: Started parallel_poly_from_expr')
    try:
        polys, opt = parallel_poly_from_expr(F, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('groebner', len(F), exc)
    from sympy.polys.rings import PolyRing
    # print('grob_inte_sympy: PolyRing')
    ring = PolyRing(opt.gens, opt.domain, opt.order)
    
    polys = [ring.from_dict(poly.rep.to_dict()) for poly in polys if poly]

    G = groebner(polys, ring)

    G = [Poly._from_dict(g, opt) for g in G]
    # print('grob_inte_sympy: Finished')
    return GroebnerBasis._new(G, opt)

def groebner(seq, ring):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Wrapper around the (default) improved Buchberger and the other algorithms
    for computing Groebner bases. The choice of algorithm can be changed via
    ``method`` argument or :func:`sympy.polys.polyconfig.setup`, where
    ``method`` can be either ``buchberger`` or ``f5b``.

    """
    domain, orig = ring.domain, None
    # print('groebner: make domain')
    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            orig, ring = ring, ring.clone(domain=domain.get_field())
        except DomainError:
            raise DomainError("Cannot compute a Groebner basis over %s" % domain)
        else:
            seq = [ s.set_ring(ring) for s in seq ]

    G = _buchberger(seq, ring)

    if orig is not None:
        G = [ g.clear_denoms()[1].set_ring(orig) for g in G ]

    return G

def _buchberger(f, ring):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Given a set of multivariate polynomials `F`, finds another
    set `G`, such that Ideal `F = Ideal G` and `G` is a reduced
    Groebner basis.

    The resulting basis is unique and has monic generators if the
    ground domains is a field. Otherwise the result is non-unique
    but Groebner bases over e.g. integers can be computed (if the
    input polynomials are monic).

    Groebner bases can be used to choose specific generators for a
    polynomial ideal. Because these bases are unique you can check
    for ideal equality by comparing the Groebner bases.  To see if
    one polynomial lies in an ideal, divide by the elements in the
    base and see if the remainder vanishes.

    They can also be used to solve systems of polynomial equations
    as,  by choosing lexicographic ordering,  you can eliminate one
    variable at a time, provided that the ideal is zero-dimensional
    (finite number of solutions).

    Notes
    =====

    Algorithm used: an improved version of Buchberger's algorithm
    as presented in T. Becker, V. Weispfenning, Groebner Bases: A
    Computational Approach to Commutative Algebra, Springer, 1993,
    page 232.

    References
    ==========

    .. [1] [Bose03]_
    .. [2] [Giovini91]_
    .. [3] [Ajwa95]_
    .. [4] [Cox97]_

    """
    # print('_buchberger: Started.')
    order = ring.order

    monomial_mul = ring.monomial_mul
    monomial_div = ring.monomial_div
    monomial_lcm = ring.monomial_lcm

    def select(P):
        # # print('select: Starting select.')
        # normal selection strategy
        # select the pair with minimum LCM(LM(f), LM(g))
        pr = min(P, key=lambda pair: order(monomial_lcm(f[pair[0]].LM, f[pair[1]].LM)))
        # # print('select: Finished select.')
        return pr

    def normal(g, J):
        # # print('normal: Starting normal.')
        h = g.rem([ f[j] for j in J ])

        if not h:
            # # print('normal: Finished normal 1st.')
            return None
        else:
            # # print('normal: Starting normal 2nd.')
            h = h.monic()

            if h not in I:
                I[h] = len(f)
                f.append(h)
            # # print('normal: Finished normal 2nd.')
            return h.LM, I[h]

    def update(G, B, ih):
        # # print('update: Starting update.')
        # update G using the set of critical pairs B and h
        # [BW] page 230
        h = f[ih]
        mh = h.LM

        # filter new pairs (h, g), g in G
        C = G.copy()
        D = set()

        while C:
            # select a pair (h, g) by popping an element from C
            ig = C.pop()
            g = f[ig]
            mg = g.LM
            LCMhg = monomial_lcm(mh, mg)

            def lcm_divides(ip):
                # LCM(LM(h), LM(p)) divides LCM(LM(h), LM(g))
                m = monomial_lcm(mh, f[ip].LM)
                return monomial_div(LCMhg, m)

            # HT(h) and HT(g) disjoint: mh*mg == LCMhg
            if monomial_mul(mh, mg) == LCMhg or (
                not any(lcm_divides(ipx) for ipx in C) and
                    not any(lcm_divides(pr[1]) for pr in D)):
                D.add((ih, ig))

        E = set()

        while D:
            # select h, g from D (h the same as above)
            ih, ig = D.pop()
            mg = f[ig].LM
            LCMhg = monomial_lcm(mh, mg)

            if not monomial_mul(mh, mg) == LCMhg:
                E.add((ih, ig))

        # filter old pairs
        B_new = set()

        while B:
            # select g1, g2 from B (-> CP)
            ig1, ig2 = B.pop()
            mg1 = f[ig1].LM
            mg2 = f[ig2].LM
            LCM12 = monomial_lcm(mg1, mg2)

            # if HT(h) does not divide lcm(HT(g1), HT(g2))
            if not monomial_div(LCM12, mh) or \
                monomial_lcm(mg1, mh) == LCM12 or \
                    monomial_lcm(mg2, mh) == LCM12:
                B_new.add((ig1, ig2))

        B_new |= E

        # filter polynomials
        G_new = set()

        while G:
            ig = G.pop()
            mg = f[ig].LM

            if not monomial_div(mg, mh):
                G_new.add(ig)

        G_new.add(ih)
        # # print('update: Finished update.')
        return G_new, B_new
        # end of update ################################

    # print('_buchberger: 1st half.')
    if not f:
        return []

    # replace f with a reduced list of initial polynomials; see [BW] page 203
    f1 = f[:]
    # print('f1',f1)
    while True:
        f = f1[:]
        f1 = []

        for i in range(len(f)):
            p = f[i]
            r = p.rem(f[:i])
            # print('i',i)
            # print('p',p)
            # print('r',r)
            # print('f[:i]',f[:i])
            if r:
                f1.append(r.monic())
            # print('f1',f1)
        if f == f1:
            break
    # print('f1',f1)
    I = {}            # ip = I[p]; p = f[ip]
    F = set()         # set of indices of polynomials
    G = set()         # set of indices of intermediate would-be Groebner basis
    CP = set()        # set of pairs of indices of critical pairs

    for i, h in enumerate(f):
        I[h] = i
        F.add(i)

    #####################################
    # algorithm GROEBNERNEWS2 in [BW] page 232

    while F:
        # print('_buchberger: F loop')
        # select p with minimum monomial according to the monomial ordering
        h = min([f[x] for x in F], key=lambda f: order(f.LM))
        ih = I[h]
        F.remove(ih)
        G, CP = update(G, CP, ih)

    # count the number of critical pairs which reduce to zero
    reductions_to_zero = 0

    while CP:
        # print('_buchberger: CP loop')
        ig1, ig2 = select(CP)
        CP.remove((ig1, ig2))

        h = spoly(f[ig1], f[ig2], ring)
        # ordering divisors is on average more efficient [Cox] page 111
        G1 = sorted(G, key=lambda g: order(f[g].LM))
        ht = normal(h, G1)

        if ht:
            G, CP = update(G, CP, ht[1])
        else:
            reductions_to_zero += 1

    ######################################
    # now G is a Groebner basis; reduce it
    Gr = set()

    for ig in G:
        # print('_buchberger: ig loop')
        ht = normal(f[ig], G - {ig})

        if ht:
            Gr.add(ht[1])
    
    Gr = [f[ig] for ig in Gr]

    # # print('_buchberger: before sorted')
    # order according to the monomial ordering
    Gr = sorted(Gr, key=lambda f: order(f.LM), reverse=True)
    # print('_buchberger: Finished.')
    return Gr

def spoly(p1, p2, ring):
    """
    Compute LCM(LM(p1), LM(p2))/LM(p1)*p1 - LCM(LM(p1), LM(p2))/LM(p2)*p2
    This is the S-poly provided p1 and p2 are monic
    """
    # # print('spoly: Starting spoly.')
    LM1 = p1.LM
    LM2 = p2.LM
    LCM12 = ring.monomial_lcm(LM1, LM2)
    m1 = ring.monomial_div(LCM12, LM1)
    m2 = ring.monomial_div(LCM12, LM2)
    s1 = p1.mul_monom(m1)
    s2 = p2.mul_monom(m2)
    s = s1 - s2
    # # print('spoly: Finished spoly.')
    return s




def lexupdate(sfm, done, ones_m):
    if ones_m.ndim == 1: ones_m = ones_m[:,np.newaxis]

    ones_m = ones_m[~np.where((ones_m == done[:,None]).all(-1))[1]]

    sfm_sc, ones_m_sc = np.c_[np.sum(sfm,axis=1),-1*sfm[:,::-1]], np.c_[np.sum(ones_m,axis=1),-1*ones_m[:,::-1]]   
    
    sfm_sc = np.ndarray(sfm_sc.shape[0], dtype=[('', sfm_sc.dtype)] * sfm_sc.shape[1], buffer=sfm_sc)
    ones_m_sc = np.ndarray(ones_m_sc.shape[0], dtype=[('', ones_m_sc.dtype)] * ones_m_sc.shape[1], buffer=ones_m_sc)
    
    iil = sfm_sc.searchsorted(ones_m_sc, 'left')
    iir = sfm_sc.searchsorted(ones_m_sc, 'right')
    antimask = np.argwhere(iil == iir).squeeze()
    sfantimask = iil[antimask].squeeze()

    sfm = np.insert(sfm, sfantimask, ones_m[antimask], axis=0)
    
    return sfm, antimask.size


class GrobRadicalEqual:
    def __init__(self,vs,Qs):

        self.gens = vs

        try:
            self.polys, self.opt = parallel_poly_from_expr(Qs, *self.gens, **{'order':'grevlex'})
        except PolificationFailed as exc:
            raise ComputationFailed('groebner', len(Qs), exc)
        from sympy.polys.rings import PolyRing
        # print('grob_inte_sympy: PolyRing')
        self.ring = PolyRing(self.opt.gens, self.opt.domain, self.opt.order)

        self.polys = [self.ring.from_dict(poly.rep.to_dict()) for poly in self.polys if poly]

        self.flag = not self.ring.domain.is_Field or not self.ring.domain.has_assoc_Field

        if self.flag:
            self.orig, self.clone_ring = self.ring, self.ring.clone(domain=self.ring.domain.get_field())
            self.b_ring = self.clone_ring
        else:
            self.b_ring = self.ring
        
        self.Q_seq = self.polys.copy()
        if self.flag:
            self.Q_seq = [pg.set_ring(self.b_ring) for pg in self.Q_seq]
            

        self.order = self.b_ring.order

        self.monomial_mul = self.b_ring.monomial_mul
        self.monomial_div = self.b_ring.monomial_div
        self.monomial_lcm = self.b_ring.monomial_lcm
    
    def __call__(self):
        """
        Computes Groebner basis for a set of polynomials in `K[X]`.

        Wrapper around the (default) improved Buchberger and the other algorithms
        for computing Groebner bases. The choice of algorithm can be changed via
        ``method`` argument or :func:`sympy.polys.polyconfig.setup`, where
        ``method`` can be either ``buchberger`` or ``f5b``.

        """

        G = self._buchberger_radical_equal(self.Q_seq)

        if G is None:
            return False
        else:
            if self.flag and (self.orig is not None):
                g = G[0].clear_denoms()[1].set_ring(self.orig)
            g = Poly._from_dict(g, self.opt)
            if g != 1:
                return False
            else:
                return True

    def _buchberger_radical_equal(self,f):
        """
        Computes Groebner basis for a set of polynomials in `K[X]`.
        """
        print('_buchberger: Started.')

        def update_member_radical(G, B, ih):
            # # print('update: Starting update.')
            # update G using the set of critical pairs B and h
            # [BW] page 230
            h = f[ih]
            mh = h.LM

            # filter new pairs (h, g), g in G
            C = G.copy()
            D = set()

            while C:
                # select a pair (h, g) by popping an element from C
                ig = C.pop()
                g = f[ig]
                mg = g.LM
                LCMhg = self.monomial_lcm(mh, mg)

                def lcm_divides(ip):
                    # LCM(LM(h), LM(p)) divides LCM(LM(h), LM(g))
                    m = self.monomial_lcm(mh, f[ip].LM)
                    return self.monomial_div(LCMhg, m)

                # HT(h) and HT(g) disjoint: mh*mg == LCMhg
                if self.monomial_mul(mh, mg) == LCMhg or (
                    not any(lcm_divides(ipx) for ipx in C) and
                        not any(lcm_divides(pr[1]) for pr in D)):
                    D.add((ih, ig))

            E = set()

            while D:
                # select h, g from D (h the same as above)
                ih, ig = D.pop()
                mg = f[ig].LM
                LCMhg = self.monomial_lcm(mh, mg)

                if not self.monomial_mul(mh, mg) == LCMhg:
                    E.add((ih, ig))

            # filter old pairs
            B_new = set()

            while B:
                # select g1, g2 from B (-> CP)
                ig1, ig2 = B.pop()
                mg1 = f[ig1].LM
                mg2 = f[ig2].LM
                LCM12 = self.monomial_lcm(mg1, mg2)

                # if HT(h) does not divide lcm(HT(g1), HT(g2))
                if not self.monomial_div(LCM12, mh) or \
                    self.monomial_lcm(mg1, mh) == LCM12 or \
                        self.monomial_lcm(mg2, mh) == LCM12:
                    B_new.add((ig1, ig2))

            B_new |= E

            # filter polynomials
            G_new = set()

            while G:
                ig = G.pop()
                mg = f[ig].LM

                if not self.monomial_div(mg, mh):
                    G_new.add(ig)

            G_new.add(ih)
            # # print('update: Finished update.')
            return G_new, B_new
            # end of update ################################

        # SYMBOLIC PREPROCESSING
        def symbolic_preprocessing(CP,G):
            
            L = set().union(*(set([f[i].mul_monom(self.monomial_div(self.monomial_lcm(f[i].LM, f[j].LM),f[i].LT[0])),f[j].mul_monom(self.monomial_div(self.monomial_lcm(f[i].LM, f[j].LM),f[j].LT[0]))]) for i,j in tqdm(CP)))
            L_LMs = set(l.LM for l in L)
            
            G_LM = [g.LM for g in G]
            G1LM = np.array(G_LM, dtype=np.float64)
            temp = set().union(*(list(l.monoms('grevlex'))[1:] for l in L))

            done = np.array(list(L_LMs.copy()))
            comp = np.array(sorted(list(temp), key=lambda m: self.order(m)))
            
            while comp.size != 0:
                # print(comp.shape[0])
                m, comp = comp[-1], comp[:-1]
                done = np.r_[done,m[None,:]]
                
                ind = np.argmax((G1LM <= m).all(axis=1))
                if monomial_divides(G_LM[ind],tuple(m)):

                    new = G[ind].mul_monom(self.monomial_div(tuple(m),G_LM[ind]))
                    L.add(new)
                    L_LMs.add(new.LM)

                    comp, no_added = lexupdate(comp, done, np.array(list(new.monoms())))

            # while done != Mon_L:
            #     comp = Mon_L - done
            #     m = max(comp, key=lambda elem: self.order(elem))
            #     done.add(m)
            #     for g,g_lm in G_LMs:
            #         if monomial_divides(g_lm,m): 
            #             temp = g.mul_monom(self.monomial_div(m,g_lm))
            #             L.add(temp)
            #             L_LMs.add(temp.LM)
            #             Mon_L = Mon_L.union(set(temp.monoms()))
            #             break
                
            return L, L_LMs

        def _row_reduce_comb(M_np,M_symp,M_rows,M_cols):
            mat, pivot_cols = _row_reduce_nump(M_np,M_symp, M_rows, M_cols, normalize_last=True,
                                    normalize=True, zero_above=True, one=S.One,
                                    iszerofunc=_iszero, simpfunc=_simplify)

            return mat, pivot_cols
        
        def _div(a,b):
            ap, aq = a.numerator, a.denominator
            bp, bq = b.numerator, b.denominator
            # print(bp)
            x1 = gcd(ap, bp)
            x2 = gcd(bq, aq)
            p, q = ((ap//x1)*(bq//x2), (aq//x2)*(bp//x1))  
            return PythonMPQ(p,q)
        
        # REDUCTION
        def reduction(CP,G):
            
            L, L_LMs = symbolic_preprocessing(CP,G)
            print('reduction: Sorting ',len(L),' polynomials in L:')
            L = sorted(L, key=lambda l: self.order(l.LM), reverse=True)

            # Make matrix M
            print('reduction: Get all monomials.')
            M = [l.monoms() for l in tqdm(L)]
            print('reduction: Sort all monomials.')
            monoms = sorted(list(set().union(*(set(elem) for elem in M))), key=lambda elem: self.order(elem), reverse=True)
            monoms_d = {m:i for i,m in enumerate(monoms)}

            print('reduction: Constructing d.')
            d_l = list(itertools.chain(*[[[j, monoms_d[m], l.get(m).numerator, l.get(m).denominator] for m in M[j]] for j,l in enumerate(L)]))
            d = list(map(list, zip(*d_l)))
            print('reduction: Constructing M_np.')
            M_np  = np.empty((len(L),len(monoms)), dtype=[('n',np.int64),('d',np.int64)])
            M_np['n'], M_np['d'] = np.zeros((len(L),len(monoms))), np.ones((len(L),len(monoms)))
            M_np['n'][d[0],d[1]], M_np['d'][d[0],d[1]] = d[2], d[3]
            # Msymp = sp.matrices.sparse.SparseMatrix(1,len(L)*len(monoms),{(0,l[0]*len(monoms)+l[1]): l[2] for l in d_l}).tolist()[0]

            # Make M_ the RREF of M
            print('reduction: Calculating reduced row echelon form of M.')
            M_, pivot_cols = _row_reduce_comb(M_np,[[]],len(L),len(monoms)) #WARNING NOW NON-NORMALIZED
            
            # Check if M_[piv_row,piv_col] == 0, then all non-zero values come after piv_col
            assert(np.all(np.all(np.nonzero(M_[i])[0].tolist() >= pivot_cols[i]) for i in range(len(pivot_cols)) if PythonMPQ(M_[i,pivot_cols[i]]) == 0))
            # Get rid of 0 rows
            print('reduction: Get rid of 0 rows in M_ and pivot_cols.')
            nzr = ~np.all(M_ == 0, axis=1)
            M_, pivot_cols = M_[nzr], np.array(pivot_cols)[nzr[:len(pivot_cols)]]
            pivot_cols = [np.argmax(M_[i] != 0) if PythonMPQ(M_[i,pc]) == 0 else pc for i,pc in enumerate(pivot_cols)]
            assert(np.all(PythonMPQ(M_[i,pc]) != 0 for i,pc in enumerate(pivot_cols)))

            # Make polynomials L_ from M_
            print('reduction: Making polynomials from M_.')
            L_ = [(self.b_ring.from_dict({monoms[j]: _div(PythonMPQ(M_[i,j]),PythonMPQ(M_[i,pc])) for j in tqdm(np.nonzero(M_[i])[0].tolist(),leave=False)})) for i,pc in enumerate(tqdm(pivot_cols))]

            # Make G_ {f in L_ | LM(f) != LM(g) for any g in L}
            print('reduction: Making G_.')
            G_ = [l for l in tqdm(L_) if l.LM not in L_LMs]
            return G_
        
            sr = self.b_ring.zero
            sf = ssf.copy()
            if not sf:
                return sr
            custom_dtype = np.dtype([('m', int, (len(self.b_ring.gens))), ('n', int), ('d', int)])
            l = list(map(list, zip(*sorted(list(ssf.items()), key=lambda kv: self.order(kv[0])))))
            sf = np.empty(len(l[0]), dtype = custom_dtype)
            sf['m'], sf['n'], sf['d'] = np.array(l[0]), np.array([c.numerator for c in l[1]]), np.array([c.denominator for c in l[1]])
            G1s = np.empty(len(G1), dtype = custom_dtype)
            l = list(map(list, zip(*[g.LT for g in G1])))
            G1s['m'], G1s['n'], G1s['d'] =  np.array(l[0]), np.array([c.numerator for c in l[1]]), np.array([c.denominator for c in l[1]])
            used_G1s = {}

            # stime = time.time()
            # sf = np.array([(m,c.numerator,c.denominator) for m,c in sorted(list(ssf.items()), key=lambda kv: self.order(kv[0]))], dtype = custom_dtype)
            # G1s = np.array([(g.LT[0],g.LT[1].numerator,g.LT[1].denominator) for g in G1], dtype = custom_dtype)
            # print('Time in beg.: ',time.time()-stime)
            # assert(np.all(_sf == sf),np.all(_G1s == G1s))
            
            sltm, sltn, sltd = sf[-1]
            print('No. of monoms in sf',sf.size)

            while sf.size != 0:
                
                sg, sm, sc = None, None, None

                idx = np.argmax((sltm >= G1s['m']).all(axis=1))
                if np.all(sltm >= G1s['m'][idx]):
                    sg, sm, sc = G1[idx], tuple(sltm - G1s['m'][idx]), self.b_ring.domain.quo(PythonMPQ(int(sltn),int(sltd)), PythonMPQ(int(G1s['n'][idx]),int(G1s['d'][idx])))
                
                if (sg,sm,sc) != (None,None,None):
                    
                    if idx in used_G1s:
                        ones_ = used_G1s[idx].copy()
                        ones_['m'], ones_['n'], ones_['d'] = np.array(sm)+ones_['m'], -sc.numerator*ones_['n'], sc.denominator*ones_['d']
                    else:
                        l = list(map(list, zip(*sorted(list(sg.items()), key=lambda kv: self.order(kv[0])))))
                        ones_ = np.empty(len(l[0]), dtype=custom_dtype)
                        ones_['m'], ones_['n'], ones_['d'] = np.array(l[0]), np.array([cg.numerator for cg in l[1]]), np.array([cg.denominator for cg in l[1]])
                        used_G1s[idx] = ones_.copy()
                        ones_['m'], ones_['n'], ones_['d'] = np.array(sm)+ones_['m'], -sc.numerator*ones_['n'], sc.denominator*ones_['d']

                    # stime = time.time()
                    # ones_ = np.array([(self.b_ring.monomial_mul(mg, sm),-sc.numerator*cg.numerator,sc.denominator*cg.denominator) for mg,cg in sorted(list(sg.items()), key=lambda kv: self.order(kv[0]))], dtype = custom_dtype)
                    # times[1].append(time.time() - stime)
                    # assert(np.all(ones_ == _ones_))
                    
                    # New version of update
                    before = sf.size
                    mtemp, ntemp, dtemp = lexupdate(sf['m'],ones_['m'],sf['n'],ones_['n'],sf['d'],ones_['d'])
                    sf = np.empty(mtemp.shape[0] ,dtype=custom_dtype)
                    sf['m'],sf['n'],sf['d'] = mtemp, ntemp, dtemp
                    
                    if sf.size != 0:
                        sltm, sltn, sltd = sf[-1]
                    else:
                        sltm = None

                else:

                    if tuple(sltm) in sr:
                        sr[tuple(sltm)] += PythonMPQ(int(sltn),int(sltd))
                    else:
                        sr[tuple(sltm)] = PythonMPQ(int(sltn),int(sltd))

                    sf = sf[:-1]
                    
                    if sf.size != 0:
                        sltm, sltn, sltd = sf[-1]
                    else:
                        sltm = None

            return sr
        
            print('normal: Starting normal.')
            # st = time.time()
            # h_check = g.rem([ f[j] for j in J ])
            # print('Time in rem',time.time() - st)
            st = time.time()
            h = new_special_rem(g,[ f[j] for j in J ])
            print('Time in custom rem',time.time() - st)
            # assert(h == h_check)

            if not h:
                print('normal: Finished normal 1st.')
                return None
            else:
                # # print('normal: Starting normal 2nd.')
                h = h.monic()

                if h not in I:
                    I[h] = len(f)
                    f.append(h)
                print('normal: Finished normal 2nd.')
                return h.LM, I[h]
        
        print('_buchberger: 1st half.')
        if not f:
            raise('buchberger_member_radical: not f.')
            return []

        # replace f with a reduced list of initial polynomials; see [BW] page 203
        f1 = f[:]
        
        while True:
            f = f1[:]
            f1 = []

            for i in range(len(f)):
                p = f[i]
                r = p.rem(f[:i])
                if r:
                    f1.append(r.monic())

            if f == f1:
                break

        I = {}            # ip = I[p]; p = f[ip]
        F = set()         # set of indices of polynomials
        G = set()         # set of indices of intermediate would-be Groebner basis
        CP = set()        # set of pairs of indices of critical pairs

        for i, h in enumerate(f):
            I[h] = i
            F.add(i)

        #####################################
        # algorithm GROEBNERNEWS2 in [BW] page 232

        while F:
            # select p with minimum monomial according to the monomial ordering
            h = min([f[x] for x in F], key=lambda f: self.order(f.LM))
            ih = I[h]
            F.remove(ih)
            G, CP = update_member_radical(G, CP, ih)
        
        
        # F4
        while len(CP) != 0:
            print('No. of CP left:',len(CP))
            G1 = sorted([f[ig] for ig in G], key=lambda p: self.order(p.LM))
            G_ = reduction(CP,G1)
            CP = set()
            print('Checking ',len(G_),' new polynomials:')
            # count = 0
            if len(G_) != 0:
                for h in tqdm(G_):
                    h = h.monic()
                    if h != 0:
                        if h not in I:
                            I[h] = len(f)
                            f.append(h)
                    G, CP = update_member_radical(G, CP, I[h])

        ######################################
        # now G is a Groebner basis; reduce it
        Gr = set()
        Gr_count = 0
        G = [f[ig] for ig in G]
        for g in G:
            h = g.rem(set(G) - {g})
            if not h:
                continue
            else:
                Gr.add(h.monic())
                Gr_count += 1
            if Gr_count > 1:
                print(Gr)
                return None
        # print(Gr)
        return list(Gr)
            


class GrobMemberRadical:
    def __init__(self,vs,_Z,old_Qs,write=True):
        self.vs = vs
        self._Z = _Z
        self.gens = vs + [_Z]
        
        self.opt = sp.polys.polyoptions.build_options(self.gens,{'order':'grevlex', 'domain':'ZZ'})
        del self.opt['auto']
        self.opt.polys = False
        
        self.ring = PolyRing(self.opt.gens, self.opt.domain, self.opt.order)
        self.flag = not self.ring.domain.is_Field or not self.ring.domain.has_assoc_Field

        if self.flag:
            self.orig, self.clone_ring = self.ring, self.ring.clone(domain=self.ring.domain.get_field())
            self.b_ring = self.clone_ring
        else:
            self.b_ring = self.ring
        
        self.Q_seq = []
        self.Q = []
        if len(old_Qs) > 0:
            polys, _ = _parallel_poly_from_expr(old_Qs, self.opt)
            self.Q = polys.copy()
            self.Q_seq = [self.ring.from_dict(pg.rep.to_dict()) for pg in polys if pg]
        if self.flag:
            self.Q_seq = [pg.set_ring(self.b_ring) for pg in self.Q_seq]

        self.order = self.b_ring.order

        self.monomial_mul = self.b_ring.monomial_mul
        self.monomial_div = self.b_ring.monomial_div
        self.monomial_lcm = self.b_ring.monomial_lcm

    def __call__(self,p):
        """
        Computes the reduced Groebner basis for a set of polynomials.

        Use the ``order`` argument to set the monomial ordering that will be
        used to compute the basis. Allowed orders are ``lex``, ``grlex`` and
        ``grevlex``. If no order is specified, it defaults to ``lex``.

        For more information on Groebner bases, see the references and the docstring
        of :func:`~.solve_poly_system`.

        """
        # GroebnerBasis(F, *gens, **args)
        # """Compute a reduced Groebner basis for a system of polynomials. """

        pg, _ = sp.polys.polytools._poly_from_expr(1-self._Z*p, self.opt)
        Q = self.Q + [pg]

        pg = self.ring.from_dict(pg.rep.to_dict()) if pg else None
        seq = self.Q_seq + [pg]

        if self.flag:
            pg = pg.set_ring(self.b_ring) 
            seq = self.Q_seq + [pg]

        G =  self._buchberger_member_radical(seq) #self._f5b(seq)

        if G is None:
            # Add polynomial for next time
            p, _ = sp.polys.polytools._poly_from_expr(p, self.opt)
            self.Q_seq = self.Q_seq+[(self.ring.from_dict(p.rep.to_dict())).set_ring(self.b_ring)]
            return False
        else:
            if self.flag and (self.orig is not None):
                g = G[0].clear_denoms()[1].set_ring(self.orig)
            g = Poly._from_dict(g, self.opt)
            if g != 1:
                # Add polynomial for next time
                p, _ = sp.polys.polytools._poly_from_expr(p, self.opt)
                self.Q_seq = self.Q_seq+[(self.ring.from_dict(p.rep.to_dict())).set_ring(self.b_ring)]
                return False
            else:
                return True

    def _buchberger_member_radical(self,f):
        """
        Computes Groebner basis for a set of polynomials in `K[X]`.
        """
        print('_buchberger: Started.')

        def update_member_radical(G, B, ih):
            # # print('update: Starting update.')
            # update G using the set of critical pairs B and h
            # [BW] page 230
            h = f[ih]
            mh = h.LM

            # filter new pairs (h, g), g in G
            C = G.copy()
            D = set()

            while C:
                # select a pair (h, g) by popping an element from C
                ig = C.pop()
                g = f[ig]
                mg = g.LM
                LCMhg = self.monomial_lcm(mh, mg)

                def lcm_divides(ip):
                    # LCM(LM(h), LM(p)) divides LCM(LM(h), LM(g))
                    m = self.monomial_lcm(mh, f[ip].LM)
                    return self.monomial_div(LCMhg, m)

                # HT(h) and HT(g) disjoint: mh*mg == LCMhg
                if self.monomial_mul(mh, mg) == LCMhg or (
                    not any(lcm_divides(ipx) for ipx in C) and
                        not any(lcm_divides(pr[1]) for pr in D)):
                    D.add((ih, ig))

            E = set()

            while D:
                # select h, g from D (h the same as above)
                ih, ig = D.pop()
                mg = f[ig].LM
                LCMhg = self.monomial_lcm(mh, mg)

                if not self.monomial_mul(mh, mg) == LCMhg:
                    E.add((ih, ig))

            # filter old pairs
            B_new = set()

            while B:
                # select g1, g2 from B (-> CP)
                ig1, ig2 = B.pop()
                mg1 = f[ig1].LM
                mg2 = f[ig2].LM
                LCM12 = self.monomial_lcm(mg1, mg2)

                # if HT(h) does not divide lcm(HT(g1), HT(g2))
                if not self.monomial_div(LCM12, mh) or \
                    self.monomial_lcm(mg1, mh) == LCM12 or \
                        self.monomial_lcm(mg2, mh) == LCM12:
                    B_new.add((ig1, ig2))

            B_new |= E

            # filter polynomials
            G_new = set()

            while G:
                ig = G.pop()
                mg = f[ig].LM

                if not self.monomial_div(mg, mh):
                    G_new.add(ig)

            G_new.add(ih)
            # # print('update: Finished update.')
            return G_new, B_new
            # end of update ################################

        # SYMBOLIC PREPROCESSING
        def symbolic_preprocessing(CP,G):
            
            L = set().union(*(set([f[i].mul_monom(self.monomial_div(self.monomial_lcm(f[i].LM, f[j].LM),f[i].LT[0])),f[j].mul_monom(self.monomial_div(self.monomial_lcm(f[i].LM, f[j].LM),f[j].LT[0]))]) for i,j in tqdm(CP)))
            L_LMs = set(l.LM for l in L)
            
            G_LM = [g.LM for g in G]
            G1LM = np.array(G_LM, dtype=np.float64)
            temp = set().union(*(list(l.monoms('grevlex'))[1:] for l in L))

            done = np.array(list(L_LMs.copy()))
            comp = np.array(sorted(list(temp), key=lambda m: self.order(m)))
         
            while comp.size != 0:
                print(comp.shape[0])
                m, comp = comp[-1], comp[:-1]
                done = np.r_[done,m[None,:]]
                
                ind = np.argmax((G1LM <= m).all(axis=1))
                if monomial_divides(G_LM[ind],tuple(m)):

                    new = G[ind].mul_monom(self.monomial_div(tuple(m),G_LM[ind]))
                    L.add(new)
                    L_LMs.add(new.LM)

                    comp, no_added = lexupdate(comp, done, np.array(list(new.monoms())))

            # while done != Mon_L:
            #     comp = Mon_L - done
            #     m = max(comp, key=lambda elem: self.order(elem))
            #     done.add(m)
            #     for g,g_lm in G_LMs:
            #         if monomial_divides(g_lm,m): 
            #             temp = g.mul_monom(self.monomial_div(m,g_lm))
            #             L.add(temp)
            #             L_LMs.add(temp.LM)
            #             Mon_L = Mon_L.union(set(temp.monoms()))
            #             break
                
            return L, L_LMs

        def _row_reduce_comb(M_np,M_symp,M_rows,M_cols):
            mat, pivot_cols = _row_reduce_nump(M_np,M_symp, M_rows, M_cols, normalize_last=True,
                                    normalize=True, zero_above=True, one=S.One,
                                    iszerofunc=_iszero, simpfunc=_simplify)

            return mat, pivot_cols
        
        def _div(a,b):
            ap, aq = a.numerator, a.denominator
            bp, bq = b.numerator, b.denominator
            # print(bp)
            x1 = gcd(ap, bp)
            x2 = gcd(bq, aq)
            p, q = ((ap//x1)*(bq//x2), (aq//x2)*(bp//x1))  
            return PythonMPQ(p,q)
        
        # REDUCTION
        def reduction(CP,G):
            
            L, L_LMs = symbolic_preprocessing(CP,G)
            print('reduction: Sorting ',len(L),' polynomials in L:')
            L = sorted(L, key=lambda l: self.order(l.LM), reverse=True)

            # Make matrix M
            print('reduction: Get all monomials.')
            M = [l.monoms() for l in tqdm(L)]
            print('reduction: Sort all monomials.')
            monoms = sorted(list(set().union(*(set(elem) for elem in M))), key=lambda elem: self.order(elem), reverse=True)
            monoms_d = {m:i for i,m in enumerate(monoms)}

            print('reduction: Constructing d.')
            d_l = list(itertools.chain(*[[[j, monoms_d[m], l.get(m).numerator, l.get(m).denominator] for m in M[j]] for j,l in enumerate(L)]))
            d = list(map(list, zip(*d_l)))
            print('reduction: Constructing M_np.')
            M_np  = np.empty((len(L),len(monoms)), dtype=[('n',np.int64),('d',np.int64)])
            M_np['n'], M_np['d'] = np.zeros((len(L),len(monoms))), np.ones((len(L),len(monoms)))
            M_np['n'][d[0],d[1]], M_np['d'][d[0],d[1]] = d[2], d[3]
            # Msymp = sp.matrices.sparse.SparseMatrix(1,len(L)*len(monoms),{(0,l[0]*len(monoms)+l[1]): l[2] for l in d_l}).tolist()[0]

            # Make M_ the RREF of M
            print('reduction: Calculating reduced row echelon form of M.')
            M_, pivot_cols = _row_reduce_comb(M_np,[[]],len(L),len(monoms)) #WARNING NOW NON-NORMALIZED
            
            # Check if M_[piv_row,piv_col] == 0, then all non-zero values come after piv_col
            # assert(np.all(np.all(np.nonzero(M_[i])[0].tolist() >= pivot_cols[i]) for i in range(len(pivot_cols)) if PythonMPQ(M_[i,pivot_cols[i]]) == 0))
            # Get rid of 0 rows
            print('reduction: Get rid of 0 rows in M_ and pivot_cols.')
            # nzr = ~np.all(M_['n'] == 0, axis=1)
            M_ = M_[~np.all(M_['n'] == 0, axis=1)] # pivot_cols = np.array(pivot_cols)[nzr[:len(pivot_cols)]]
            # pivot_cols = [np.argmax(M_[i] != 0) if PythonMPQ(M_[i,pc]) == 0 else pc for i,pc in enumerate(pivot_cols)]
            # assert(np.all(PythonMPQ(M_[i,pc]) != 0 for i,pc in enumerate(pivot_cols)))

            # Make polynomials L_ from M_
            print('reduction: Making polynomials from M_.')
            L_ = [(self.b_ring.from_dict({monoms[j]: PythonMPQ(int(M_['n'][i,j]),int(M_['d'][i,j])) for j in tqdm(range(M_.shape[1]),leave=False)})) for i in tqdm(range(M_.shape[0]))]
            # np.nonzero(M_['n'][i])[0].tolist()

            # Make G_ {f in L_ | LM(f) != LM(g) for any g in L}
            print('reduction: Making G_.')
            G_ = [l for l in tqdm(L_) if l.LM not in L_LMs]
            return G_
        
            sr = self.b_ring.zero
            sf = ssf.copy()
            if not sf:
                return sr
            custom_dtype = np.dtype([('m', int, (len(self.b_ring.gens))), ('n', int), ('d', int)])
            l = list(map(list, zip(*sorted(list(ssf.items()), key=lambda kv: self.order(kv[0])))))
            sf = np.empty(len(l[0]), dtype = custom_dtype)
            sf['m'], sf['n'], sf['d'] = np.array(l[0]), np.array([c.numerator for c in l[1]]), np.array([c.denominator for c in l[1]])
            G1s = np.empty(len(G1), dtype = custom_dtype)
            l = list(map(list, zip(*[g.LT for g in G1])))
            G1s['m'], G1s['n'], G1s['d'] =  np.array(l[0]), np.array([c.numerator for c in l[1]]), np.array([c.denominator for c in l[1]])
            used_G1s = {}

            # stime = time.time()
            # sf = np.array([(m,c.numerator,c.denominator) for m,c in sorted(list(ssf.items()), key=lambda kv: self.order(kv[0]))], dtype = custom_dtype)
            # G1s = np.array([(g.LT[0],g.LT[1].numerator,g.LT[1].denominator) for g in G1], dtype = custom_dtype)
            # print('Time in beg.: ',time.time()-stime)
            # assert(np.all(_sf == sf),np.all(_G1s == G1s))
            
            sltm, sltn, sltd = sf[-1]
            print('No. of monoms in sf',sf.size)

            while sf.size != 0:
                
                sg, sm, sc = None, None, None

                idx = np.argmax((sltm >= G1s['m']).all(axis=1))
                if np.all(sltm >= G1s['m'][idx]):
                    sg, sm, sc = G1[idx], tuple(sltm - G1s['m'][idx]), self.b_ring.domain.quo(PythonMPQ(int(sltn),int(sltd)), PythonMPQ(int(G1s['n'][idx]),int(G1s['d'][idx])))
                
                if (sg,sm,sc) != (None,None,None):
                    
                    if idx in used_G1s:
                        ones_ = used_G1s[idx].copy()
                        ones_['m'], ones_['n'], ones_['d'] = np.array(sm)+ones_['m'], -sc.numerator*ones_['n'], sc.denominator*ones_['d']
                    else:
                        l = list(map(list, zip(*sorted(list(sg.items()), key=lambda kv: self.order(kv[0])))))
                        ones_ = np.empty(len(l[0]), dtype=custom_dtype)
                        ones_['m'], ones_['n'], ones_['d'] = np.array(l[0]), np.array([cg.numerator for cg in l[1]]), np.array([cg.denominator for cg in l[1]])
                        used_G1s[idx] = ones_.copy()
                        ones_['m'], ones_['n'], ones_['d'] = np.array(sm)+ones_['m'], -sc.numerator*ones_['n'], sc.denominator*ones_['d']

                    # stime = time.time()
                    # ones_ = np.array([(self.b_ring.monomial_mul(mg, sm),-sc.numerator*cg.numerator,sc.denominator*cg.denominator) for mg,cg in sorted(list(sg.items()), key=lambda kv: self.order(kv[0]))], dtype = custom_dtype)
                    # times[1].append(time.time() - stime)
                    # assert(np.all(ones_ == _ones_))
                    
                    # New version of update
                    before = sf.size
                    mtemp, ntemp, dtemp = lexupdate(sf['m'],ones_['m'],sf['n'],ones_['n'],sf['d'],ones_['d'])
                    sf = np.empty(mtemp.shape[0] ,dtype=custom_dtype)
                    sf['m'],sf['n'],sf['d'] = mtemp, ntemp, dtemp
                    
                    if sf.size != 0:
                        sltm, sltn, sltd = sf[-1]
                    else:
                        sltm = None

                else:

                    if tuple(sltm) in sr:
                        sr[tuple(sltm)] += PythonMPQ(int(sltn),int(sltd))
                    else:
                        sr[tuple(sltm)] = PythonMPQ(int(sltn),int(sltd))

                    sf = sf[:-1]
                    
                    if sf.size != 0:
                        sltm, sltn, sltd = sf[-1]
                    else:
                        sltm = None

            return sr
        
            print('normal: Starting normal.')
            # st = time.time()
            # h_check = g.rem([ f[j] for j in J ])
            # print('Time in rem',time.time() - st)
            st = time.time()
            h = new_special_rem(g,[ f[j] for j in J ])
            print('Time in custom rem',time.time() - st)
            # assert(h == h_check)

            if not h:
                print('normal: Finished normal 1st.')
                return None
            else:
                # # print('normal: Starting normal 2nd.')
                h = h.monic()

                if h not in I:
                    I[h] = len(f)
                    f.append(h)
                print('normal: Finished normal 2nd.')
                return h.LM, I[h]
        
        print('_buchberger: 1st half.')
        if not f:
            raise('buchberger_member_radical: not f.')
            return []

        # replace f with a reduced list of initial polynomials; see [BW] page 203
        f1 = f[:]
        
        while True:
            f = f1[:]
            f1 = []

            for i in range(len(f)):
                p = f[i]
                r = p.rem(f[:i])
                if r:
                    f1.append(r.monic())

            if f == f1:
                break

        I = {}            # ip = I[p]; p = f[ip]
        F = set()         # set of indices of polynomials
        G = set()         # set of indices of intermediate would-be Groebner basis
        CP = set()        # set of pairs of indices of critical pairs

        for i, h in enumerate(f):
            I[h] = i
            F.add(i)

        #####################################
        # algorithm GROEBNERNEWS2 in [BW] page 232

        while F:
            # select p with minimum monomial according to the monomial ordering
            h = min([f[x] for x in F], key=lambda f: self.order(f.LM))
            ih = I[h]
            F.remove(ih)
            G, CP = update_member_radical(G, CP, ih)
        
        
        # F4
         # f.copy()
        # G_LM = [g.LM for g in G]
        # G_LT =  [g.LT[0] for g in G]
        # k = len(G)
        while len(CP) != 0:
            print('No. of CP left:',len(CP))
            G1 = sorted([f[ig] for ig in G], key=lambda p: self.order(p.LM))
            # G_LM = [g.LM for g in G1]
            # G_LT =  [g.LT[0] for g in G1]
            G_ = reduction(CP,G1)
            CP = set()
            print('Checking ',len(G_),' new polynomials:')
            # count = 0
            if len(G_) != 0:
                for h in tqdm(G_):
                    h = h.monic()
                    if h != 0:
                        if h not in I:
                            I[h] = len(f)
                            f.append(h)
                        G, CP = update_member_radical(G, CP, I[h])
                # if h not in G:
                    # count += 1
                    # G.append(h)
                    # G_LM.append(h.LM)
                    # G_LT.append(h.LT[0])
                    # k += 1
                    # CP = CP.union((i,k-1) for i in range(k-1))
            # print(count,' new polynomials added to G.')

        ######################################
        # now G is a Groebner basis; reduce it
        Gr = set()
        Gr_count = 0
        G = [f[ig] for ig in G]
        for g in G:
            h = g.rem(set(G) - {g})
            if not h:
                continue
            else:
                Gr.add(h.monic())
                Gr_count += 1
            if Gr_count > 1:
                print(Gr)
                return None
        # print(Gr)
        return list(Gr)
            
        
   
# vs = list(sp.symbols('x1:785'))
# _Z = sp.Symbol('_Z')
# sp.var('x1:785')

# Q = [x1 + x28 + x757 + x784, x2 + x27 + x29 + x56 + x729 + x756 + x758 + x783, x26 + x3 + x57 + x701 + x728 + x759 + x782 + x84, x112 + x25 + x4 + x673 + x700 + x760 + x781 + x85, x113 + x140 + x24 + x5 + x645 + x672 + x761 + x780, x141 + x168 + x23 + x6 + x617 + x644 + x762 + x779, x169 + x196 + x22 + x589 + x616 + x7 + x763 + x778, x197 + x21 + x224 + x561 + x588 + x764 + x777 + x8, x20 + x225 + x252 + x533 + x560 + x765 + x776 + x9, x10 + x19 + x253 + x280 + x505 + x532 + x766 + x775, x11 + x18 + x281 + x308 + x477 + x504 + x767 + x774, x12 + x17 + x309 + x336 + x449 + x476 + x768 + x773, x13 + x16 + x337 + x364 + x421 + x448 + x769 + x772, x14 + x15 + x365 + x392 + x393 + x420 + x770 + x771, x30 + x55 + x730 + x755, x31 + x54 + x58 + x702 + x727 + x731 + x754 + x83, x111 + x32 + x53 + x674 + x699 + x732 + x753 + x86, x114 + x139 + x33 + x52 + x646 + x671 + x733 + x752, x142 + x167 + x34 + x51 + x618 + x643 + x734 + x751, x170 + x195 + x35 + x50 + x590 + x615 + x735 + x750, x198 + x223 + x36 + x49 + x562 + x587 + x736 + x749, x226 + x251 + x37 + x48 + x534 + x559 + x737 + x748, x254 + x279 + x38 + x47 + x506 + x531 + x738 + x747, x282 + x307 + x39 + x46 + x478 + x503 + x739 + x746, x310 + x335 + x40 + x45 + x450 + x475 + x740 + x745, x338 + x363 + x41 + x422 + x44 + x447 + x741 + x744, x366 + x391 + x394 + x419 + x42 + x43 + x742 + x743, x59 + x703 + x726 + x82, x110 + x60 + x675 + x698 + x704 + x725 + x81 + x87, x115 + x138 + x61 + x647 + x670 + x705 + x724 + x80, x143 + x166 + x619 + x62 + x642 + x706 + x723 + x79, x171 + x194 + x591 + x614 + x63 + x707 + x722 + x78, x199 + x222 + x563 + x586 + x64 + x708 + x721 + x77, x227 + x250 + x535 + x558 + x65 + x709 + x720 + x76, x255 + x278 + x507 + x530 + x66 + x710 + x719 + x75, x283 + x306 + x479 + x502 + x67 + x711 + x718 + x74, x311 + x334 + x451 + x474 + x68 + x712 + x717 + x73, x339 + x362 + x423 + x446 + x69 + x713 + x716 + x72, x367 + x390 + x395 + x418 + x70 + x71 + x714 + x715, x109 + x676 + x697 + x88, x108 + x116 + x137 + x648 + x669 + x677 + x696 + x89, x107 + x144 + x165 + x620 + x641 + x678 + x695 + x90, x106 + x172 + x193 + x592 + x613 + x679 + x694 + x91, x105 + x200 + x221 + x564 + x585 + x680 + x693 + x92, x104 + x228 + x249 + x536 + x557 + x681 + x692 + x93, x103 + x256 + x277 + x508 + x529 + x682 + x691 + x94, x102 + x284 + x305 + x480 + x501 + x683 + x690 + x95, x101 + x312 + x333 + x452 + x473 + x684 + x689 + x96, x100 + x340 + x361 + x424 + x445 + x685 + x688 + x97, x368 + x389 + x396 + x417 + x686 + x687 + x98 + x99, x117 + x136 + x649 + x668, x118 + x135 + x145 + x164 + x621 + x640 + x650 + x667, x119 + x134 + x173 + x192 + x593 + x612 + x651 + x666, x120 + x133 + x201 + x220 + x565 + x584 + x652 + x665, x121 + x132 + x229 + x248 + x537 + x556 + x653 + x664, x122 + x131 + x257 + x276 + x509 + x528 + x654 + x663, x123 + x130 + x285 + x304 + x481 + x500 + x655 + x662, x124 + x129 + x313 + x332 + x453 + x472 + x656 + x661, x125 + x128 + x341 + x360 + x425 + x444 + x657 + x660, x126 + x127 + x369 + x388 + x397 + x416 + x658 + x659, x146 + x163 + x622 + x639, x147 + x162 + x174 + x191 + x594 + x611 + x623 + x638, x148 + x161 + x202 + x219 + x566 + x583 + x624 + x637, x149 + x160 + x230 + x247 + x538 + x555 + x625 + x636, x150 + x159 + x258 + x275 + x510 + x527 + x626 + x635, x151 + x158 + x286 + x303 + x482 + x499 + x627 + x634, x152 + x157 + x314 + x331 + x454 + x471 + x628 + x633, x153 + x156 + x342 + x359 + x426 + x443 + x629 + x632, x154 + x155 + x370 + x387 + x398 + x415 + x630 + x631, x175 + x190 + x595 + x610, x176 + x189 + x203 + x218 + x567 + x582 + x596 + x609, x177 + x188 + x231 + x246 + x539 + x554 + x597 + x608, x178 + x187 + x259 + x274 + x511 + x526 + x598 + x607, x179 + x186 + x287 + x302 + x483 + x498 + x599 + x606, x180 + x185 + x315 + x330 + x455 + x470 + x600 + x605, x181 + x184 + x343 + x358 + x427 + x442 + x601 + x604, x182 + x183 + x371 + x386 + x399 + x414 + x602 + x603, x204 + x217 + x568 + x581, x205 + x216 + x232 + x245 + x540 + x553 + x569 + x580, x206 + x215 + x260 + x273 + x512 + x525 + x570 + x579, x207 + x214 + x288 + x301 + x484 + x497 + x571 + x578, x208 + x213 + x316 + x329 + x456 + x469 + x572 + x577, x209 + x212 + x344 + x357 + x428 + x441 + x573 + x576, x210 + x211 + x372 + x385 + x400 + x413 + x574 + x575, x233 + x244 + x541 + x552, x234 + x243 + x261 + x272 + x513 + x524 + x542 + x551, x235 + x242 + x289 + x300 + x485 + x496 + x543 + x550, x236 + x241 + x317 + x328 + x457 + x468 + x544 + x549, x237 + x240 + x345 + x356 + x429 + x440 + x545 + x548, x238 + x239 + x373 + x384 + x401 + x412 + x546 + x547, x262 + x271 + x514 + x523, x263 + x270 + x290 + x299 + x486 + x495 + x515 + x522, x264 + x269 + x318 + x327 + x458 + x467 + x516 + x521, x265 + x268 + x346 + x355 + x430 + x439 + x517 + x520, x266 + x267 + x374 + x383 + x402 + x411 + x518 + x519, x291 + x298 + x487 + x494, x292 + x297 + x319 + x326 + x459 + x466 + x488 + x493, x293 + x296 + x347 + x354 + x431 + x438 + x489 + x492, x294 + x295 + x375 + x382 + x403 + x410 + x490 + x491, x320 + x325 + x460 + x465, x321 + x324 + x348 + x353 + x432 + x437 + x461 + x464, x322 + x323 + x376 + x381 + x404 + x409 + x462 + x463, x349 + x352 + x433 + x436, x350 + x351 + x377 + x380 + x405 + x408 + x434 + x435, x378 + x379 + x406 + x407, x1**2 + x28**2 + x757**2 + x784**2, x1*x2 + x1*x29 + x27*x28 + x28*x56 + x729*x757 + x756*x784 + x757*x758 + x783*x784, x1*x3 + x1*x57 + x26*x28 + x28*x84 + x701*x757 + x728*x784 + x757*x759 + x782*x784]
# p = 1-_Z*(x1*x4 + x1*x85 + x112*x28 + x25*x28 + x673*x757 + x700*x784 + x757*x760 + x781*x784)
# vs2 = vs + [_Z]

# gmr = GrobMemberRadical(vs,_Z,Q,write=False)
# st = time.time()
# gbc = gmr(p)
# print(time.time() - st)
# print(gbc)


