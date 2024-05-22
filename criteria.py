import numpy as np
import sympy as sp

from internal import *
from built_ins import *

#-------------------------------------------------------------#
# CRITERIA FOR SELECTION
#-------------------------------------------------------------#


def criteria2proper(P,k,LCMHT,memberP,bugfix,statistics):
    """
    Input:
    P: list of pairs of indices that refer to polynomials
    k: index int
    LCMHT:
    memberP:
    bugfix:
    statistics:

    Output:
    res: list of pairs in P that satisfy criteria 2 proper
    memberP, statistics: updated 
    """
    for IJ in P:
        i = IJ[0]
        j = IJ[1]
        if member(memberP[i,k],{True,'finished'}) and member(memberP[j,k],{True,'finished'}):
            # print(i,j,bugfix,LCMHT)
            bl1, q1 = divide(LCMHT[*bugfix[i,j]],LCMHT[i,k],'q')
            bl2, q2 = divide(LCMHT[*bugfix[i,j]],LCMHT[j,k],'q')
            # print(q1,q2)
            if q1 != 1 and q2 != 1:
                memberP[*bugfix[i,j]] = 'finished'
                statistics['crit2p'] = statistics['crit2p']+1
    res = []
    for JK in P:
        if memberP[*bugfix[*JK]] == True:
            res.append(JK)
    return res, statistics, memberP

# x,y,z = sp.symbols('x:z')
# # X = [x,y,z]
# # memberP = SymDict()
# # bugfix = SymDict()
# # LCMHT = {}
# # HT = [x**2,x,y]
# # superfluous = {}
# # statistics = {}
# # statistics['pairs'] = 0
# # statistics['respairs'] = 0
# # k = 2
# # newpairs(k,HT,superfluous,LCMHT,X,memberP,statistics)

# neupairs = {(0,1)}
# LCMHT = {(0,1): x**2}
# statistics = {'pairs': 1, 'respairs': 0, 'crit2p': 0}
# memberP = SymDict()
# memberP[0,1] = True
# P = []
# bugfix = SymDict()
# bugfix[0,1] = [0,1]
# # print(neupairs, LCMHT, memberP, statistics)
# k = 1
# # print(memberP)
# print(criteria2proper(P,k,LCMHT,memberP,bugfix,statistics))
# # print(memberP)

def criteria2a(k,newpairs,HT,LCMHT,memberP,bugfix,statistics):
    """
    Input:
    k:index int
    newpairs: set of tuples/pairs of indices that refer to polynomials
    e.t.c

    Output:
    Nold: list of pairs in newpairs that satisfy criteria 2a
    memberP, statistics: updated 
    """
    Nold = newpairs
    Nnew = Nold 
    for i in range(k-1):
        if memberP[i,k]==True and intersect(indets(HT[i]),indets(HT[k])) in [set(),sp.EmptySet]:
            Nnew = Nold
            for JK in Nold:
                j = JK[0]
                if memberP[j,k]==True and LCMHT[j,k]==LCMHT[i,k] and (i == j or member(memberP[*bugfix[i,j]],{True,'finished'})):
                    Nnew = minus(Nnew,{(j,k)})
                    memberP[j,k] = 'finished'
                    statistics['crit2a'] = statistics['crit2a']+1
            Nold = Nnew
    return Nold, memberP, statistics

def criteria2b(k,newpairs,LCMHT,memberP,bugfix,statistics):
    """
    Input:
    k: index int
    newpairs: set of tuples/pairs of indices that refer to polynomials
    LCMHT:
    memberP:
    bugfix:
    statistics:

    Output:
    res: list of pairs in newpairs that satisfy criteria 2b
    memberP, statistics: updated 
    """
    # if k==13:
    #     op(newpairs)
    #     op(LCMHT)
    #     op(memberP)
    for j in range(1,len(newpairs)):
        i = 0
        jj = newpairs[j][0]
        while memberP[*bugfix[*newpairs[j]]] == True and i<j:
            if member(memberP[*bugfix[*newpairs[i]]],{True,'finished'}):
                ii = newpairs[i][0]
                if member(memberP[*bugfix[ii,jj]],{True,'finished'}) and divide(LCMHT[jj,k],LCMHT[ii,k]) and LCMHT[jj,k] != LCMHT[ii,k]:
                    memberP[jj,k] = 'finished'; 
                    statistics['crit2b'] = statistics['crit2b']+1
            i += 1
    res = []
    for JK in newpairs:
        # print(JK)
        # print(bugfix)
        # print(memberP)
        if memberP[*bugfix[*JK]] == True:
            res.append(JK)
    return res, statistics, memberP