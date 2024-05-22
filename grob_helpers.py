import numpy as np
import sympy as sp

from internal import *
from built_ins import *
from head import *
from sugar import *
from criteria import *

#-------------------------------------------------------------#
# FUNCTIONS FOR GRADGROEBNER
#-------------------------------------------------------------#

def iscritredunt(i,j,HT,memberP,bugfix):
    """
    Input:
    i: index int
    j: index int
    HT: leading terms of polynmoials in G
    memberP: dict
    bugfix: dict

    Output:
    True if leading terms of G[i] and G[j] are redundant and both have been finished, otherwise False
    """
    if divide(HT[i],HT[j]) and memberP[*bugfix[i,j]]=='finished':
        return True 
    else: 
        return False

def erase(P,i,memberP,bugfix,statistics):
    """
    Input:
    P: list of critical pairs
    i: index of a superfluous polynomial
    memberP: dict
    bugfix: dict
    statistics: dict

    Output:
    newP: P without pair corresponding to supefluous polynomial
    memberP, statistics: updated
    """
    newP = P
    r = 0
    n = len(P)
    for k in range(n):
        IJ = P[k]
        # print(IJ)
        if IJ[0]==i or IJ[1]==i:
            # print(k-r,k+1-r,n-r)
            newP = newP[:k-r] + newP[k+1-r:n-r]
            r += 1
            memberP[*bugfix[*IJ]] = 'finished'
            statistics['crit3'] = statistics['crit3']+1
    return newP, memberP, statistics

def redundant(k,HT,oldreds):
    """
    Input:
    k: indices of possible new redundants
    HT: list of leading terms of polynomials in G
    oldreds: list of indices of old redundants

    Output:
    reds: set of updated indices of redundant polynomials in reduced Groebner basis
    """
    reds = set()
    for i in range(0,k-1):
        if not(member(i,oldreds)) and divide(HT[i],HT[k]): 
            reds = union(reds,set([i]))
    return reds

def isindexHTsmaller(i,j,HT,to):
    """
    Input:
    i: int index
    j: int index
    HT: list of head/leading monomials of SymPy polynomials
    to: dict of term order

    Output:
    bool: True if leading monomial of jth polynomial is higher w.r.t term order than ith polynomial
    """
    if head3(HT[i]+HT[j],to['vs'],to) == HT[i]:
        return False
    else:
        return True

def isHTjgreater(i,j,HT,to):
    """
    Input:
    i: index int
    j: index int
    HT: list of leading terms of polynomials
    to: dict of term order

    Output:
    True if HT[i] > HT[j] w.r.t term order, otherwise False
    """
    if isgreaterorder2(HT[i],HT[j],to):
        return True
    else:
        return False

def newpairs(*args):
    """
    Input:
    k: index int
    HT:leading monomial term of each SymPy polynomials
    superfluous:
    LCMHT: dict
    X: list of vars SymPy symbols
    memberP: symmetric dict
    statistics: dict
    gradings: list of grading dicts

    Output:
    N: set of pairs of indices of polynomials that can be put in P
    LCMHT, memberP, statistics: updated 
    """
    k = args[0]
    HT = args[1]
    superfluous = args[2]
    LCMHT = args[3]
    X = args[4]
    memberP = args[5]
    statistics = args[6]
    N = set()
    for i in range(k):
        if not(i in superfluous):
            LCMHT[i,k] = termlcm(HT[i],HT[k],X)
            if isinbounds(LCMHT[i,k],X,*args[7:]):
                memberP[i,k] = True
                N = union(N ,{(i,k)})
                statistics['pairs'] = statistics['pairs']+1
            else:
                memberP[i,k] = 'restricted by grading' #added
                print('newpairs: S-polynomial ',i,',',k,' not considered because of restriction in grading.')
                statistics['respairs'] = statistics['respairs']+1
        else:
            memberP[i,k] = 'superfluous' #added
    return N, LCMHT, memberP, statistics

def addpairs(k,P,memberP,superfluous,LCMHT,sugarIJ,HT,Sugar,X,to,bugfix,statistics,*args):
    """
    Input:
    k: index int
    P: set of possible SymPy polynomials
    memberP: symmetric dict
    superfluous: set
    LCMHT: dict
    sugarIJ: dict
    HT: leading monomial term of each SymPy polynomials
    Sugar: list of sugar degree of each SymPy polynomial
    X: list of vars SymPy symbols
    to: dict of term order
    bugfix: symmetric dict
    statistics: dict

    Output:
    newP: updated set of possible SymPy polynomials 
    """
    # print(k,len(LCMHT),len(memberP),len(HT))
    neupairs, LCMHT, memberP, statistics = newpairs(k,HT,superfluous,LCMHT,X,memberP,statistics,*args)
    # print('newpairs done')
    newP, statistics, memberP = criteria2proper(P,k,LCMHT,memberP,bugfix,statistics)
    # print('criteria2proper done')
    neupairs, memberP, statistics = criteria2a(k,neupairs,HT,LCMHT,memberP,bugfix,statistics)
    # print('criteria2a done')
    for IK in neupairs:
        sugarIJ[*IK] = sugarSpol(IK,LCMHT,X,HT,Sugar)
    # print('sugarSpol done')
    neupairs = [list(pair) for pair in neupairs]
    ### fuzzy variant
    neupairs = sort_internal(neupairs,sugar,LCMHT,sugarIJ,bugfix,to)
    neupairs2b, statistics, memberP = criteria2b(k,neupairs,LCMHT,memberP,bugfix,statistics)
    # print('criteria2b done')
    ### sloppy variant
    # neupairs:=`moregroebner/src/internal/sort`(
    #                 neupairs,'`moregroebner/src/internal/strategy/normal`',
    #                             LCMHT,bugfix,termorder);
    # neupairs2b:=`moregroebner/src/internal/criteria2b`(k,neupairs,LCMHT,memberP,bugfix);
    # neupairs2b:=`moregroebner/src/internal/sort`(
    #                   neupairs2b,'moregroebner[critpairs,strategy,sugar]',
    #                             LCMHT,sugarIJ,bugfix,termorder);
    ### without criterion 2b
    # neupairs2b:=neupairs;
    newP = merge_internal(newP,neupairs2b,sugar,LCMHT,sugarIJ,bugfix,to)
    return newP, LCMHT, memberP, statistics, sugarIJ

def leadm(*args):
    """
    Input:
    p: SymPy polynomial or expression
    to: term order

    Output:
    leading monomial of p wrt to
    """
    nargs = len(args)
    if nargs < 2:
        raise('leadm: Too few arguments.')
    p, to = args[0], args[1]
    if not(isinstance(to,dict)): 
        raise('leadm: Second argument should be a dict including the termorder.')
    X = to['vs']
    if not(is_type(p,'polynom',X)):
        raise('leadm: ',p,'not a polynomial over',X)
    h = head_inte(p,X,to)
    return h[1]