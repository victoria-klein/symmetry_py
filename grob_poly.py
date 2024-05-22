import numpy as np
import sympy as sp
import time as time

from built_ins import *
from internal import *
from head import *
from head_poly import *
from sugar import *
from sugar_poly import *
from grob_helpers import *

#-------------------------------------------------------------#
# FUNCTIONS FOR GRADGROEBNER ON POLY DOMAIN
#-------------------------------------------------------------#


def minimalgb_poly(G,HT,HC,Sugar,X,to,lowest):
    """
    Input:
    G: Grobner basis
    e.t.c

    Output:
    Minimal Grobner basis
    """
    R = G
    HTr = HT
    HCr = HC
    Sugarr = Sugar
    n = len(G)
    P = []
    HTp = []
    HCp = []
    Sugarp = []
    while R != []:
        # print(R,HTr)
        h = R[0]
        s = Sugarr[0]
        h, su = reduce_poly(h,s,P,X,to,HTp,HCp,Sugarp,lowest,'su') 
        if len(indices(h)) != 0:
            J = set()
            for j in range(len(P)):
                if divide(HTp[j],HTr[0]):
                    J  = union(J,set([j]))
            P = deletelist(P,J)
            HTp = deletelist(HTp,J)
            HCp = deletelist(HCp,J)
            Sugarp = deletelist(Sugarp,J)
            P.append(h)
            HTp.append(HTr[0])
            HCp.append(h[HTr[0]])
            Sugarp.append(su)
        R = R[1:]
        HTr = HTr[1:]
        HCr = HCr[1:]
        Sugarr = Sugarr[1:]
        n = n-1
    return [P,HTp,HCp,Sugarp]

def reducedgb_poly(G,HT,HC,Sugar,X,to,lowest):
    """
    Input:
    G: Grobner basis (almost minimal)
    e.t.c

    Output:
    K: Reduced Grobner basis
    """
    K = []
    KHT = []
    n = len(G) 
    for j in range(n):
        G_no_j, HT_no_j, HC_no_j = G[:], HT[:], HC[:]
        G_no_j, HT_no_j, HC_no_j = subsop(G_no_j,(j,None)), subsop(HT_no_j,(j,None)), subsop(HC_no_j,(j,None))
        p, s = reduce_poly(G[j],Sugar[j],G_no_j,X,to,HT_no_j,HC_no_j,Sugar,lowest,'s')
        # p, s = reduce_poly(G[j],Sugar[j],subsop(G,sp.Eq(j,None)),
        #                                 X,to,subsop(HT,sp.Eq(j,None)),
        #                                 subsop(HC,sp.Eq(j,None)),Sugar,
        #                                 lowest,'s')
        if len(indices(p)) != 0:
            K.append(p)
            KHT.append(HT[j])
    J = [j for j in range(len(K))]
    J = sort_internal(J,isHTjgreater,KHT,to)
    K = [K[J[j]] for j in range(len(K))]
    return K


def postproc_poly(G,HT,HC,Sugar,X,to,lowest,reds):
    """
    Input:
    G: Grobner basis
    e.t.c

    Output:
    G1: reduced Grobner basis
    """
    # print(G,HT,reds)
    G1 = deletelist(G,reds)
    HT1 = deletelist(HT,reds)
    HC1 = deletelist(HC,reds)
    Sugar1 = deletelist(Sugar,reds)
    # print(G1,HT1)
    G1 = minimalgb_poly(G1,HT1,HC1,Sugar1,X,to,lowest)
    HT1 = G1[1]
    HC1 = G1[2]
    Sugar1 = G1[3]
    G1 = G1[0]

    G1 = reducedgb_poly(G1,HT1,HC1,Sugar1,X,to,lowest)
    G1 = [tabtop(G1[i],X,to) for i in range(len(G1))]
    return G1

def grob_internal_poly(*args):
    """
    Input:
    F: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    to: dict of term order
    args: args for addpairs

    Intermediates:
    G: list of polynomials that will form the basis (starts with F and is reduced)
    HT: heads of polynomials in G
    HC: leading coefficients of polynomials in G
    reds: set of indices of redundant elements of G
    P: ordered list of critical pairs with ghost head LCMHT
    memberP: critical pairs to be done, 
        memberP([i,j]) = True for index [i,j] on list
        memberP([i,j]) = 'finished' if S-pol. reduced or pair elim. by criteria
    SugarIJ: value of S-polynomial
    N: new list of critical pairs

    Output:
    G: list of polynomials that form reduced Grobner basis for F
    """
    F, X, to =args[0], args[1], args[2]
    stt = time.time()
    print('Computing Groebner basis for domain inte.')
    npolys = len(F)
    if member(1,F):
        return [1]
    # Prepocessing
    G = [ptotab(elem,X) for elem in F] #list of dicts
    npolys = len(F)
    HT = [head_poly(G[i],X,to) for i in range(npolys)]

    if not(len(list(set(HT)))==len(HT)): # some polynomials have same lt
        i = 0
        print('grob_internal_poly: Begin preprocessor loop ',npolys,'.')
        while i < npolys-1:
            if i == 0:
                J = [j for j in range(npolys)]
                J = sort_internal(J,isindexHTsmaller,HT,to)
                G = [G[J[j]] for j in range(npolys)]
                HT = [HT[J[j]]for j in range(npolys)]
            print('grob_internal_poly: Preprocessor loop ',i,'.')
            if HT[i+1]==HT[i]:
                fb = G[i+1].copy()
                fb = cmul(G[i][HT[i]],fb)
                fb = addto(fb, G[i], -G[i+1][HT[i+1]])
                if not(len(indices(fb)) == 0):
                    G = G[:i] + [fb] + G[i+1:]
                    HT = HT[:i] + [head_poly(G[i],X,to)] + HT[i+1:]
                    i += 1
                else:
                    G = G[:i] + G[i+1:]
                    HT = HT[:i] + HT[i+1:]
                    npolys -= 1
            else:
                i += 1

    HC = [G[i][HT[i]] for i in range(npolys)]
    Sugar = [sugardegree_poly(G[i],X) for i in range(npolys)]
    reds = set()
    memberP = SymDict() #symmetric - switch to matrix or array or factor in symmetry in conditions
    P = []
    LCMHT = {}
    bugfix = SymDict() #symmetric - switch to matrix or array or factor in symmetry in conditions
    for i in range(npolys):
        for j in range(i,npolys):
            bugfix[i,j] = [i,j]
    SugarIJ = {}
    lowest = HT[0]
    superfluous = set()
    statistics = {}
    statistics['crit2a'] = 0
    statistics['crit2b'] = 0
    statistics['crit2p'] = 0
    statistics['pairs'] = 0
    statistics['respairs'] = 0
    statistics['crit3'] = 0

    for j in range(1,npolys):
        P, LCMHT, memberP, statistics, SugarIJ = addpairs(j,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,*args[3:])
        if isgreaterorder2(lowest,HT[j],to):
            lowest = HT[j]

    while len(P) != 0:
        IJ = P[0]
        P = P[1:]
        i = IJ[0] 
        j = IJ[1]
        fb, sfb, LCMHT, SugarIJ, G = sp_poly(IJ,G,HT,HC,LCMHT,SugarIJ,bugfix,'sfb')
        memberP[i,j] = 'finished'
        if len(indices(fb)) != 0: 
            fb, sfb = reduce_poly(fb,sfb,G,X,to,HT,HC,Sugar,lowest,'sfb')
        if len(indices(fb)) != 0:
            npolys += 1
            G.append(fb)
            HT.append(head_poly(fb,X,to))
            print(G,HT)
            HC.append(fb[HT[npolys-1]])
            Sugar.append(sfb)
            if isgreaterorder2(lowest,HT[npolys-1],to):
                lowest = HT[npolys-1]
            print('grob_internal_poly: New polynomial ',npolys,' with head ',HT[npolys-1])
            if iscritredunt(i,j,HT,memberP,bugfix):
                superfluous = union(superfluous, set([i]))
                P, memberP, statistics = erase(P,i,memberP,bugfix,statistics)
            elif iscritredunt(j,i,HT,memberP,bugfix):
                superfluous = union(superfluous, set([j]))
                P, memberP, statistics = erase(P,j,memberP,bugfix,statistics)
            reds = union(reds,redundant(npolys-1,HT,reds))
            # print(P,memberP,LCMHT)
            P, LCMHT, memberP, statistics, SugarIJ  = addpairs(npolys-2,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,*args[3:])
            print('grob_internal_poly: Nr of remaining S-polynomials: ',len(P))
        else:
            print('grob_internal_poly: S-polynomial reduced to 0: ',IJ)
    # Final reduction
    print('grob_internal_poly: All S-polynomials have been reduced, start final reduction.')
    G = postproc_poly(G,HT,HC,Sugar,X,to,lowest,reds)
    print('grob_internal_poly: statistics:')
    print('Total number of treated pairs: ',statistics['pairs'])
    print('Pairs not treated wrt restriction: ',statistics['respairs'])
    print('Pairs eliminated because criteria: ', statistics['crit2a']+ statistics['crit2b']+ statistics['crit2p']+statistics['crit3'])
    print('= ',statistics['crit2a'],'+',statistics['crit2b'],'+',statistics['crit2p'],'+',statistics['crit3'])
    print('Time: ',time.time()-stt)
    return G


# x,y,z = sp.symbols('x:z')
# X = [x,y]
# F = [x**2+y**2,y**2+5,z**2]
# to = {'ordername':'plex', 'vs':X}
# print(grob_internal_poly(F,X,to))
# gb_check = sp.groebner(F, *X, order='lex')
# self.assertTrue(gb==list(gb_check.args[0]))

def polygrob(*args):
    raise('polygrob: Not yet implemented.')

def polyradgrob(*args):
    raise('polyradgrob: Not yet implemented.')