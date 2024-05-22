import numpy as np
import sympy as sp
import time as time

from internal import *
from built_ins import *
from head import *
from head_inte import *
from sugar import *
from sugar_inte import *
from grob_helpers import *
from MGH_helpers import *

import settings

#-------------------------------------------------------------#
# FUNCTIONS FOR GRADGROEBNER ON INTE DOMAIN
#-------------------------------------------------------------#

def minimalgb_inte(G,HT,HC,Sugar,X,to,lowest):
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
        h = R[0]
        s = Sugarr[0]
        print('h',h)
        h, su = reduce_inte(h,s,P,X,to,HTp,HCp,Sugarp,lowest,'su')
        print('h',h)
        if h!= 0:
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
            HCp.append(hcoeff_inte(h,X,to))
            Sugarp.append(su)
        R = R[1:]
        HTr = HTr[1:]
        HCr = HCr[1:]
        Sugarr = Sugarr[1:]
        n = n-1
    return [P,HTp,HCp,Sugarp]


def reducedgb_inte(G,HT,HC,Sugar,X,to,lowest):
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
        p, s = reduce_inte(G[j],Sugar[j],G_no_j,X,to,HT_no_j,HC_no_j,Sugar,lowest,'s')
        if p!=0:
            K.append(p)
            KHT.append(HT[j])
    J = [j for j in range(len(K))]
    J = sort_internal(J,isHTjgreater,KHT,to)
    K = [K[J[j]] for j in range(len(K))]
    return K

def postproc_inte(G,HT,HC,Sugar,X,to,lowest,reds):
    """
    Input:
    G: Grobner basis
    e.t.c

    Output:
    G1: reduced Grobner basis
    """
    G1 = deletelist(G,reds)
    HT1 = deletelist(HT,reds)
    HC1 = deletelist(HC,reds)
    Sugar1 = deletelist(Sugar,reds)
    G1 = minimalgb_inte(G1,HT1,HC1,Sugar1,X,to,lowest)
    HT1 = G1[1]
    HC1 = G1[2]
    Sugar1 = G1[3]
    G1 = G1[0]

    G1 = reducedgb_inte(G1,HT1,HC1,Sugar1,X,to,lowest)
    #G1 = [seq(inte/polsort(G1['j'],to),'j'=1..nops(G1))]
    return G1

class BugDict(dict):
    def __getitem__(self, key):
        i, j = key
        if min(i,j) == i:
            return [i, j]
        else:
            return [j, i]

def grob_internal_inte(*args):
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
    print('grob_internal_inte: Computing Groebner basis for domain inte.')
    G = F
    npolys = len(G)
    if member(1,G):
        return [1]
    HT = [hterm_inte(G[i],X,to) for i in range(npolys)]
    if not(len(list(set(HT)))==len(HT)): # some polynomials have same lt
        i = 0
        while i < npolys-1:
            if i == 0:
                J = [j for j in range(npolys)]
                J = sort_internal(J,isindexHTsmaller,HT,to)
                G = [G[J[j]] for j in range(npolys)]
                HT = [HT[J[j]]for j in range(npolys)]
            if HT[i+1]==HT[i]:
                fb = hcoeff_inte(G[i],X,to)*G[i+1]-hcoeff_inte(G[i+1],X,to)*G[i]
                if fb != 0:
                    G = G[:i] + [fb] + G[i+1:]
                    HT = HT[:i] + [hterm_inte(G[i],X,to)] + HT[i+1:]
                else:
                    G = G[:i] + G[i+1:]
                    HT = HT[:i] + HT[i+1:]
                    npolys -= 1
                i = 0
            else:
                i += 1

    HC = [hcoeff_inte(G[i],X,to) for i in range(npolys)]
    Sugar = [sugardegree(G[i],X) for i in range(npolys)]
    reds = set()
    memberP = SymDict() #symmetric dict
    P = []
    LCMHT = {}
    bugfix = BugDict() #symmetric dict returning indices
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
        fb, sfb, LCMHT, SugarIJ = sp_inte(IJ,G,HT,HC,LCMHT,SugarIJ,bugfix,'sfb')
        memberP[i,j] = 'finished'
        fb, sfb = reduce_inte(fb,sfb,G,X,to,HT,HC,Sugar,lowest,'sfb')
        if fb != 0:
            npolys += 1
            G.append(fb)
            HT.append(hterm_inte(fb,X,to))
            HC.append(hcoeff_inte(fb,X,to))
            Sugar.append(sfb)
            if isgreaterorder2(lowest,HT[npolys-1],to):
                lowest = HT[npolys-1]
            print('grob_internal_inte: New polynomial ',npolys,' with head ',HT[npolys-1],'.')
            if iscritredunt(i,j,HT,memberP,bugfix):
                superfluous = union(superfluous,set([i]))
                P, memberP, statistics = erase(P,i,memberP,bugfix,statistics)
            elif iscritredunt(j,i,HT,memberP,bugfix):
                superfluous = union(superfluous,set([j]))
                P, memberP, statistics = erase(P,j,memberP,bugfix,statistics)
            reds = union(reds,redundant(npolys-1,HT,reds))
            P, LCMHT, memberP, statistics, SugarIJ = addpairs(npolys-1,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,*args[3:])
            print('grob_internal_inte: nr of remaining S-polynomials: ', len(P),'.')
        else:
            print('S-polynomial reduced to 0: ',IJ,'.')
    print('G',G)
    # Final reduction
    print('grob_internal_inte: All S-polynomials have been reduced, start final reduction.')
    G = postproc_inte(G,HT,HC,Sugar,X,to,lowest,reds)
    print('G',G)
    print('grob_internal_inte: statistics:')
    print('Total number of treated pairs: ',statistics['pairs'])
    print('Pairs not treated wrt restriction: ',statistics['respairs'])
    print('Pairs eliminated because criteria: ', statistics['crit2a']+ statistics['crit2b']+ statistics['crit2p']+statistics['crit3'],
        '= ',statistics['crit2a'],'+',statistics['crit2b'],'+',statistics['crit2p'],'+',statistics['crit3'])
    print('Time: ',time.time()-stt)
    return G

def integrob(*args):
    """
    Input:
    F: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols
    to: dict of term order
    grads: list of gradings used for restriction
    Wei: list of gradings forming a weight system
    H: Hilbert series
    zvars: list of SymPy symbols variables

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
    HP: list of Hilbert series
    tHP1s, tH: tentative Hilbert series

    Output:
    G: list of polynomials that form reduced Grobner basis for F, where selection of S-polynomials
    occurs with weights and algorithm is Hilbert series driven (see ...,Robbiano ISSAC 96)
    """
    F, X, to, grads, Wei, H, zvars = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    stt = time.time()
    print('integrob: Computing Groebner basis for domain inte, Hilbert series driven version.')
    G = F
    npolys = len(G)
    HT = [hterm_inte(G[i],X,to) for i in range(npolys)]
    if not(len(list(set(HT)))==len(HT)): # some polynomials have same lt
        i = 0
        while i < npolys-1:
            if i == 0:
                J = [j for j in range(npolys)]
                J = sort_internal(J,isindexHTsmaller,HT,to)
                G = [G[J[j]] for j in range(npolys)]
                HT = [HT[J[j]]for j in range(npolys)]
            if HT[i+1]==HT[i]:
                fb = hcoeff_inte(G[i],X,to)*G[i+1]-hcoeff_inte(G[i+1],X,to)*G[i]
                if fb != 0:
                    G = G[:i] + [fb] + G[i+1:]
                    HT = HT[:i] + [hterm_inte(G[i],X,to)] + HT[i+1:]
                else:
                    G = G[:i] + G[i+1:]
                    HT = HT[:i] + HT[i+1:]
                    npolys -= 1
                i = 0
            else:
                i += 1

    HC = [hcoeff_inte(G[i],X,to) for i in range(npolys)]
    Sugar = [sugardegree(G[i],X) for i in range(npolys)]
    reds = set()
    memberP = SymDict() #symmetric - switch to matrix or array or factor in symmetry in conditions
    P = []
    LCMHT = {}
    bugfix = SymDict() #symmetric - switch to matrix or array or factor in symmetry in conditions
    for i in range(npolys+10):
        for j in range(i,npolys+10):
            bugfix[i,j] = [i,j]
    SugarIJ = {}
    lowest = HT[0]
    superfluous = set()
    statistics = MGH_initstatistics()

    for j in range(1,npolys):
        P, LCMHT, memberP, statistics, SugarIJ = MGH_addpairs(j,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,grads,Wei)
        if isgreaterorder2(lowest,HT[j],to):
            lowest = HT[j]
    # print(bugfix)
    # special Hilbert series depending part
    EE = {}
    s = MGH_nrgradsinweight(Wei,X)
    EE['_s'] = s
    print('integrob: Nr of gradings in minimal weight system ',s,'.')
    r = len(Wei)
    EE['_r'] = r
    print('integrob: Nr of gradings in full weight system ',r,'.')
    HP = MGH_fullHP(H,zvars,s)
    tHP1s = MGH_tentHP(HT[:len(HT)],X,Wei[:s])
    if len(P) != 0:
        d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
        if normal(HP[0]-tHP1s) != 0:
            _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
            d1s = EE['_d1s']
            if MGH_multideggreatprop(d1s,d1sa):
                P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                if len(P) != 0:
                    d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                    if MGH_multideggreatprop(d1sa,d1s):
                        EE = MGH_Eclean(EE,d1s)
                        _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                        d1s = EE['_d1s']
            print('integrob: Start at degree',d1s,' with dimension ',EE['_d1s'],'.')
            flag = True
            while len(P) != 0 and flag:
                dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                tHP = MGH_tentativeHP(HT[:len(HT)],X,Wei,s,d1s,tHP1s)
                flag = False
                flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                if flag:
                    EE = MGH_Eclean(EE,d1s)
                    _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1s[:s-1] + [d1s[s]+1])
                    d1s = EE['_d1s']
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                else:
                    d = EE['_da']
                    if r>s:
                        print('integrob: Starting at degree ',d,' with dimension ',EE[d],'.')
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
        else:
            len(P) == 0

    while len(P) != 0:
        IJ = P[0]
        P = P[1:]
        i = IJ[0]
        j = IJ[1]
        fb, sfb, LCMHT, SugarIJ = sp_inte(IJ,G,HT,HC,LCMHT,SugarIJ,bugfix,'sfb')
        memberP[i,j] = 'finished'
        fb, sfb = reduce_inte(fb,sfb,G,X,to,HT,HC,Sugar,lowest,'sfb')
        if fb != 0:
            npolys += 1
            G.append(fb)
            HT.append(hterm_inte(fb,X,to))
            HC.append(hcoeff_inte(fb,X,to))
            Sugar.append(sfb)
            if isgreaterorder2(lowest,HT[npolys-1],to):
                lowest = HT[npolys-1]
            print('integrob: New polynomial ',npolys,' with head ',HT[npolys-1],'.')
            if iscritredunt(i,j,HT,memberP,bugfix):
                superfluous = union(superfluous,set([i]))
                P, memberP, statistics = erase(P,i,memberP,bugfix,statistics)
            elif iscritredunt(j,i,HT,memberP,bugfix):
                superfluous = union(superfluous,set([j]))
                P, memberP, statistics = erase(P,j,memberP,bugfix,statistics)
            reds = union(reds,redundant(npolys-1,HT,reds))
            P, LCMHT, memberP, statistics, SugarIJ = MGH_addpairs(npolys-1,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,grads,Wei)
            # special Hilbert series depending part
            d = MGH_multideg(HT[len(HT)-1],X,Wei)
            if MGH_multideggreatprop(d[:s],d1s):
            # since we truncate wrt grads we need to make sure to work at right degree
                if len(P) == 0:
                    break
                EE = MGH_Eclean(EE,EE['_da'])
                tHP1s = MGH_tentHP(HT[:len(HT)],X,Wei[:s])
                if normal(tHP1s-HP[0]) == 0:
                    break
                d1sa = d[:s]
                _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                d1s = EE['_d1s']
                if MGH_multideggreatprop(d1s,d1sa):
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                    if len(P) == 0:
                        break 
                    d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                    if MGH_multideggreatprop(d1sa,d1s):
                        EE = MGH_Eclean(EE,d1s)
                        _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                        d1s = EE['_d1s']
                print('integrob: Now new degree is ',d1s,' with dim ',EE[d1s],'.')
                if len(P) == 0:
                    break 
                dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                tHP = MGH_tentativeHP(HT[:len(HT)],X,Wei,s,d1s,tHP1s)
                flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                while len(P) != 0 and flag:
                    _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1s[:s-1] + [d1s[s]+1])
                    d1s = EE['_d1s'] 
                    print('integrob: Now new degree is ',d1s,'.')
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d1s,statistics)
                    if len(P)==0:
                        break 
                    dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                    flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                d = EE['_da']
                P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT, Wei,X,d,statistics)
                if r>s:
                    print('integrob: Now current deg is ',d,' with dim ',EE[d],'.') 
            else:
                if MGH_multideggreatprop(d,EE['_da']):
                    dd = EE['_da']
                    for nu in range(s,r+1):
                        if [d[:nu]] == [dd[:nu]]:
                            k = nu
                        else:
                            del EE[dd[:nu]] #= evaln(EE[dd[1..nu]])
                            del EE[dd[:nu],'polynom'] #= evaln(EE[dd[1..nu],polynom])
                    flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,d,k+1,'flag')
                    if flag:
                        raise('integrob: Unusual case in integrob.')
                EE, flag, flagda = MGH_Eupdate(EE,HP,tHP,zvars,d,'flag','flagda')
                if flagda:
                    if flag:
                        EE = MGH_Eclean(EE,EE['_da'])
                        tHP1s = MGH_tentHP(HT[:len(HT)],X,Wei[:s])
                        if normal(tHP1s-HP[0]) == 0:
                            break 
                        P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d[:s-1] + [d[s]+1],statistics)
                        if len(P) == 0:
                            break 
                        d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                        _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                        d1s = EE['_d1s']
                        if MGH_multideggreatprop(d1s,d1sa):
                            P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                        if len(P) == 0:
                            break 
                        d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                        if MGH_multideggreatprop(d1sa,d1s):
                            EE = MGH_Eclean(EE,d1s)
                            _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                            d1s = EE['_d1s']
                        print('integrob: New degree is ',d1s,'.')
                        if len(P) == 0:
                            break 
                        dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                        tHP = MGH_tentativeHP(HT[:len(HT)],X,Wei,s,d1s,tHP1s) 
                        flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                        while len(P) != 0 and flag:
                            _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d[:s-1] + [d[s]+1])
                            d1s = EE['_d1s']
                            print('integrob: New degree is ',d1s,' of dim ',EE[d1s],'.')
                            P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                            if len(P) == 0:
                                break 
                            dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                            flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                        d = EE['_da']
                        if r>s:
                            P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
                            print('integrob: Current deg is ',d,' with dim ',EE[d],'.')
                    else:
                        d = EE['_da']
                        P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
                        if r>s:
                            print('integrob: Current degree is',d,'dim',EE[d]) 
            print('integrob: nr of remaining S-polynomials: ', len(P),'.')
        else:
            print('integrob: S-polynomial reduced to 0: ',IJ,'.')
# final reduction
    print('integrob: All S-polynomials have been reduced, start final reduction.')
    G = postproc_inte(G,HT,HC,Sugar,X,to,lowest,reds)
    MGH_printstatistic(statistics,stt)
    return G


def interadgrob(*args):
    """
    Input:
    F: list of SymPy polynomials or expressions
    X: list of vars SymPy symbols including slack variables for roots
    to: dict of term order
    grads: list of gradings used for restriction
    Wei: list of gradings forming a weight system
    H: Hilbert series
    zvars: list of SymPy symbols variables
    ws: list of SymPy symbols slack variables for roots

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
    HP: list of Hilbert series
    tHP1s, tH: tentative Hilbert series
    XX: variables not including slack variables for roots

    Output:
    G: list of polynomials that form reduced Grobner basis for F, where selection of S-polynomials
    occurs with weights and algorithm is Hilbert series driven (see ...,Robbiano ISSAC 96)
    """
    F, X, to, grads, Wei, H, zvars, ws = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
    XX = []
    for elem in X:
        if not(member(elem,set(ws))):
            XX.append(elem)
    stt = time.time()
    print('interadgrob: Computing Groebner basis for domain inte, Hilbert series driven version, treating radicals as additional variables.')
    G = F
    npolys = len(G)
    HT = [hterm_inte(G[i],X,to) for i in range(npolys)]
    if not(len(list(set(HT)))==len(HT)): # some polynomials have same lt
        i = 0
        while i < npolys-1:
            if i == 0:
                J = [j for j in range(npolys)]
                J = sort_internal(J,isindexHTsmaller,HT,to)
                G = [G[J[j]] for j in range(npolys)]
                HT = [HT[J[j]]for j in range(npolys)]
            if HT[i+1]==HT[i]:
                fb = hcoeff_inte(G[i],X,to)*G[i+1]-hcoeff_inte(G[i+1],X,to)*G[i]
                if fb != 0:
                    G = G[:i] + [fb] + G[i+1:]
                    HT = HT[:i] + [hterm_inte(G[i],X,to)] + HT[i+1:]
                else:
                    G = G[:i] + G[i+1:]
                    HT = HT[:i] + HT[i+1:]
                    npolys -= 1
                i = 0
            else:
                i += 1

    HC = [hcoeff_inte(G[i],X,to) for i in range(npolys)]
    Sugar = [sugardegree(G[i],X) for i in range(npolys)]
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
    statistics = MGH_initstatistics()

    for j in range(1,npolys):
        P, LCMHT, memberP, statistics, SugarIJ = MGH_addpairs(j,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,grads,Wei)
        if isgreaterorder2(lowest,HT[j],to):
            lowest = HT[j]

# special Hilbert series depending part
    EE = {}
    s = MGH_nrgradsinweight(Wei,XX)
    EE['_s'] = s
    print('interadgrob: Nr of gradings in minimal weight system',s)
    r = len(Wei)
    EE['_r'] = r
    print('interadgrob: Nr of gradings in full weight system',r)
    HP = MGH_fullHP(H,zvars,s)
    tHP1s = MGH_tentHP(MGH_select(HT[:len(HT)],ws),XX,Wei[:s])
    if len(P) != 0:
        d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
        if normal(HP[0]-tHP1s) != 0:
            _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
            d1s = EE[_d1s]
            if MGH_multideggreatprop(d1s,d1sa):
                P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                if len(P) != 0:
                    d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                    if MGH_multideggreatprop(d1sa,d1s):
                        EE = MGH_Eclean(EE,d1s)
                        _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                        d1s = EE[_d1s]
            print('interadgrob: Start at degree ',d1s,' with dimension ',EE[d1s],'.')
            flag = True
            while len(P) != 0 and flag:
                dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                tHP = MGH_tentativeHP(MGH_select(HT[:len(HT)],ws),XX,Wei,s,d1s,tHP1s)
                flag = False
                flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                if flag:
                    EE = MGH_Eclean(EE,d1s)
                    _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1s[:s-1] + [d1s[s]+1])
                    d1s = EE['_d1s']
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                else:
                    d = EE['_da']
                    if r>s:
                        print('interadgrob: Starting at degree',d,'with dimension',EE[d])
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
        else:
            len(P) == 0
    
    while len(P) != 0:
        IJ = P[0]
        P = P[1:]
        i = IJ[0]
        j = IJ[1]
        fb, sfb, LCMHT, SugarIJ = sp_inte(IJ,G,HT,HC,LCMHT,SugarIJ,bugfix,'sfb')
        memberP[i,j] = 'finished'
        fb, sfb = reduce_inte(fb,sfb,G,X,to,HT,HC,Sugar,lowest,'sfb')
        if fb != 0:
            npolys += 1
            G.append(fb)
            HT.append(hterm_inte(fb,X,to))
            HC.append(hcoeff_inte(fb,X,to))
            Sugar.append(sfb)
            if isgreaterorder2(lowest,HT[npolys-1],to):
                lowest = HT[npolys-1]
            print('interadgrob: New polynomial ',npolys,' with head ',HT[npolys-1],'.')
            if iscritredunt(i,j,HT,memberP,bugfix):
                superfluous = union(superfluous,set([i]))
                P, memberP, statistics = erase(P,i,memberP,bugfix,statistics)
            elif iscritredunt(j,i,HT,memberP,bugfix):
                superfluous = union(superfluous,set([j]))
                P, memberP, statistics = erase(P,j,memberP,bugfix,statistics)
            reds = union(reds,redundant(npolys-1,HT,reds))
            P, LCMHT, memberP, statistics, SugarIJ = MGH_addpairs(npolys-1,P,memberP,superfluous,LCMHT,SugarIJ,HT,Sugar,X,to,bugfix,statistics,grads,Wei)
    # special Hilbert series depending part
            d = MGH_multideg(HT[len(HT)],X,Wei)
    # ## special case for treatment of radicals
            if MGH_multideg(HT[i],X,Wei) != d and MGH_multideg(HT[j],X,Wei) != d:
                if MGH_multideggreatprop(d[:s],d1s):
                # since we truncate wrt grads we need to make sure to work at right degree
                    if len(P) == 0:
                        break
                    EE = MGH_Eclean(EE,EE['_da'])
                    tHP1s = MGH_tentHP(MGH_select(HT[:len(HT)],ws),XX,Wei[:s])
                    if normal(tHP1s-HP[0]) == 0:
                        break
                    d1sa = d[:s]
                    _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                    d1s = EE['_d1s']
                    if MGH_multideggreatprop(d1s,d1sa):
                        P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                        if len(P) == 0:
                            break
                        d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                        if MGH_multideggreatprop(d1sa,d1s):
                            EE = MGH_Eclean(EE,d1s)
                            _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                            d1s = EE['_d1s']
                    print('interadgrob: Now new degree is',d1s,'with dim',EE[d1s],'.')
                    if len(P) == 0:
                        break
                    dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                    tHP = MGH_tentativeHP(MGH_select(HT[:len(HT)],ws),XX,Wei,s,d1s,tHP1s)
                    flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                    while len(P) != 0 and flag:
                        _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1s[:s-1] + [d1s[s]+1])
                        d1s = EE['_d1s'] 
                        print('interadgrob: Now new degree is ',d1s,'.')
                        P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d1s,statistics)
                        if len(P) == 0:
                            break
                        dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                        flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                    d = EE['_da']
                    P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
                    if r>s:
                        print('interadgrob: Now current deg is',d,'with dim',EE[d])
                else:
                    if MGH_multideggreatprop(d,EE[_da]):
                        dd = EE['_da']
                        for nu in range(s,r+1):
                            if d[:nu] == dd[:nu]:
                                k = nu
                            else:
                                del EE[dd[:nu]] #= evaln(EE[dd[:nu]])
                                del EE[dd[:nu],'polynom'] #= evaln(EE[dd[:nu],polynom])
                        flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,d,k+1,'flag')
                        if flag:
                            raise('interadgrob: Unusual case in integrob.')
                    EE, flag, flagda = MGH_Eupdate(EE,HP,tHP,zvars,d,'flag','flagda')
                    if flagda:
                        if flag:
                            EE = MGH_Eclean(EE,EE['_da'])
                            tHP1s = MGH_tentHP(MGH_select(HT[:len(HT)],ws),XX,Wei[:s])
                            if normal(tHP1s-HP[0]) == 0:
                                break
                            P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,[op(d[:s-1]),d[s]+1],statistics)
                            if len(P) == 0:
                                break
                            d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                            _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                            d1s = EE['_d1s']
                            if MGH_multideggreatprop(d1s,d1sa):
                                P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                            if len(P) == 0:
                                break
                            d1sa = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei[:s])
                            if MGH_multideggreatprop(d1sa,d1s):
                                EE = MGH_Eclean(EE,d1s)
                                _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1sa)
                                d1s = EE['_d1s']
                            print('interadgrob: New degree is ',d1s,'.')
                            if len(P) == 0:
                                break
                            dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                            tHP = MGH_tentativeHP(MGH_select(HT[:len(HT)],ws),XX,Wei,s,d1s,tHP1s) 
                            flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                            while len(P) != 0 and flag:
                                _, EE = MGH_Einit(EE,HP[0],tHP1s,zvars[:s],d1s[:s-1] + [d1s[s]+1])
                                d1s = EE['_d1s']
                                print('interadgrob: New degree is ',d1s,' of dim ',EE[d1s],'.')
                                P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei[:s],X,d1s,statistics)
                                if len(P) == 0:
                                    break
                                dd = MGH_multideg(LCMHT[*bugfix[*P[0]]],X,Wei)
                                flag, EE = MGH_Enextdegdim(EE,HP,tHP,zvars,dd,s+1,'flag')
                            d = EE['_da']
                            if r>s:
                                P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
                                print('interadgrob: Current deg is',d,' with dim ',EE[d],'.') 
                        else:
                            d = EE['_da']
                            P, memberP, statistics = MGH_discard(P,memberP,bugfix,LCMHT,Wei,X,d,statistics)
                            if r>s:
                                print('interadgrob: Current degree is',d,'dim',EE[d])
            print('interadgrob: Nr of remaining S-polynomials:',nops(P))
        else:
            print('interadgrob: S-polynomial reduced to 0: ',IJ,'.')
# final reduction
    print('interadgrob: All S-polynomials have been reduced, start final reduction.')
    G = postproc_inte(G,HT,HC,Sugar,X,to,lowest,reds)
    MGH_printstatistic(statistics,stt)
    return G
             