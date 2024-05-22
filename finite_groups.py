import numpy as np
import sympy as sp

from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
from built_ins import *
from internal import *
import settings

"""
dimension
isonlist
isfinitegroup
getrepmatrix
arerepofsamegroup
getinverse
order_fg
fgchangecoords
fgadd
mkfinitegroup -bad loop

dihedral
dihedralreflection
dihedralrotation

cyclic
cyclicrotation

icosahedral -bad loop
icosahedral_c2
icosahedral_c2_i
icosahedral_c5
icosahedral_c5_i

tetrahedral
tetrahedralt
tetrahedralx
tetrahedraly
tetrahedralz

octrahedral
octrahedralabdc
octrahedralacbd
octrahedralbacd
octrahedraldbca
"""

#-------------------------------------------------------------#
# HELPER FUNCTIONS FOR FINITE GROUPS
#-------------------------------------------------------------#

def dimension(g):
    """
    Input:
    g: dict representation of group

    Output:
    dimension of g
    """
    return rowdim(g['generators']['_s'])

def isonlist(c,a):
    """
    Input:
    c: SymPy matrix
    a: list of (str,Matrix) pairs

    Output:
    True if c is value of some key in a, False if not
    """
    return True if c in a[:][1] else False

def isfinitegroup(g):
    """
    Input:
    g: dict representation of group

    Output:
    True if g is dictionary with finite group type and dictionary of generators and alelements
    """
    return isinstance(g, dict) and g['grouptype'] == 'finite' and isinstance(g['allelements'], dict) and isinstance(g['generators'], dict)

def getrepmatrix(g,ele):
    """
    Input:
    g: dict representation of group
    ele: name of element

    Output:
    matrix representation of ele of g
    """
    return g['allelements'][ele]

def arerepofsamegroup(g1,g2):
    """
    Input:
    g1: dict representation of group
    g2: dict representation of group

    Output:
    True if generators of the g1 and no. of elements of g1 are the same as g2, False otherwise
    """
    return g1['oname'] == g2['oname'] and list(g1['generators'].keys()) == list(g2['generators'].keys()) and len(g1['allelements'].keys()) == len(g2['allelements'].keys())

def getinverse(g,ele):
    """
    Input:
    g: dict representation of group
    ele: name of element

    Output:
    inverse of ele of g
    """
    inv = g['invelements'][ele]
    if isinstance(inv, str):
        inv = g['allelements'][inv]
    return inv

def order_fg(g):
    """
    Input:
    g: dict representation of group
    
    Output:
    order of group g
    """
    return len(g['allelements'].keys())

def fgchangecoords(*args):
    """
    Input:
    g: dict representation of group
    A: change of cooordinates matrix

    Output:
    T: group with generators and allelements in new coordinates
    """
    nargs = len(args)
    if nargs != 2:
        raise('fgchangecoords: Two arguments required.')
    g, A = args[0], args[1]
    if not(isfinitegroup(g)):
        raise('fgchangecoords: First argument should be valid dict representation of finite group.')
    if not(is_type(A, 'matrix')):
        raise('fgchangecoords: Second argument must be SymPy matrix.')
    if coldim(A) != coldim(list(g['generators'].values())[0]):
        raise('fgchangecoords: Matrix A must have dimension equal to generators of the group g.')
    T = g.copy()
    T['generators'] = {key : A * value * A.inv() for key,value in g['generators']}
    T['allelements'] = {key : A * value * A.inv() for key,value in g['allelements']}
    return T

def fgadd(*args):
    """
    Input:
    g: dict representation of finite group
    g2: dict representation of finite group

    Output:
    T: dict representation of direct sum of g and g2
    """
    nargs = len(args)
    if nargs != 2:
        raise('fgadd: Two arguments required.')
    g, g2 = args[0], args[1]
    if not(isfinitegroup(g)):
        raise('fgadd: First argument should be valid dict representation of finite group.')
    if not(isfinitegroup(g2)):
        raise('fgadd: Second argument should be valid dict representation of finite group.')
    if not(arerepofsamegroup(g,g2)):
        raise('fgadd: Representations of the same finite group are expected.')
    T = g.copy()
    if intersect(set(g['isotypics'].keys()),set(g2['isotypics'].keys())) not in [{}, set(),sp.EmptySet]:
        raise('fgadd: Problem merging isotypics of g and g2 with the same keys.')
    T['isotypics'] = g['isotypics'] | g2['isotypics']
    T['allelements'] = {key : diag(getrepmatrix(g,key),getrepmatrix(g2,key)) for key,value in g['allelements']}
    T['generators'] = {key : diag(value,g2['generators'][key]) for key,value in g['generators']}
    return T


def mkfinitegroup(gs,na=None):
    """
    Input:
    gs: dict of group generators
    na: string of name of new finite group

    Output:
    res: new group in dict representation
    """
    vs = list(gs.keys())
    ms = list(gs.values())
    if not(all(is_type(elem,'matrix') for elem in ms)):
        raise('mkfinitegroup: First argument should be a dict with generator matrices for values.')
    if len(union(set([coldim(elem) for elem in ms]),set([rowdim(elem) for elem in ms])))!= 1:
        raise('mkfinitegroup: Matrices should have the same dimension.')
    res = {}
    if na is not None:
        if not(isinstance(na,str)):
            raise('mkfinitegroup: Second argument should be a name.')
        res['oname'] = na
    res['grouptype'] = 'finite'
    res['generators'] = gs.copy()
    n = rowdim(list(gs.values())[0])
    a_fn = lambda i, j : 1 if i==j else 0
    a = [('id',matrix(n,n,a_fn))] + [(key, value) for key,value in gs.items()]
    i = 1
    while i < len(a):
        b = a[i][1]
        j = 0
        while j < len(a):
            c = sp.simplify(b*a[j][1])
            if not(isonlist(c,a)):
                a.append((a[i][0]+a[j][0],c))
            j += 1
            if j > 1000:
                raise('mkfinitegroup: Group is too large.')
        i += 1
        if i > 1000:
            raise('mkfinitegroup: Group is too large.')
    res['allelements'] = {key:value for key,value in a}
    ainv = {}
    while len(a.keys()) > 1:
        keys = list(a.keys()).remove('id')
        key0 = keys[0]
        b = a[key0]
        for keyj in keys:
            c = b*a[keyj]
            if equal(c,a['id']):
                if keyj == key0:
                    ainv[keyj] = key0
                    del a[key0]
                else:
                    ainv[keyj] = key0
                    ainv[key0] = keyj
                    del a[key0]
                    del a[keyj]
                break
        if key0 in a.keys():
            raise('mkfinitegroup: No inverse found for ',key0,'.')
    res['invelements'] = ainv.copy()
    res['invelements']['id'] = 'id'
    return res

#-------------------------------------------------------------#
# FINITE GROUPS
#-------------------------------------------------------------#

def dihedralreflection (h, n):
    """
    Input:
    h: no. corresponding to irrep of D_n
    n: for D_n

    Output:
    1D or 2D representation
    """
    if is_type(n, 'even'):
        if h == 1 or h == 3:
            ms = matrix(1, 1, [[1]])
        elif h == 2 or h == 4:
            ms = matrix(1, 1, [[-1]])
        else:
            ms = matrix(2, 2, [[1,0],[0,-1]])
    else:
        if h == 1:
            ms = matrix(1, 1, [[1]])
        elif h == 2:
            ms = matrix(1, 1, [[-1]])
        else:
            ms = matrix(2, 2, [[1,0],[0,-1]])
    return(ms)

def dihedralrotation(h, n):
    """
    Input:
    h: no. corresponding to irrep of D_n
    n: for D_n

    Output:
    1D or 2D representation
    """
    if is_type(n, 'even'):
        if h == 1 or h == 2:
            md = matrix(1, 1, [[1]])
        elif h == 3 or h == 4:
            md = matrix(1, 1, [[-1]])
        else:
            theta = 2 * sp.pi / n
            md = matrix(2, 2, [[sp.cos((h - 4) * theta),sp.sin((h - 4) * theta)],[-sp.sin((h - 4) * theta),sp.cos((h - 4) * theta)]])
    else:
        if h == 1:
            md = matrix(1, 1, [[1]])
        elif h == 2:
            md = matrix(1, 1, [[1]])
        else:
            theta = 2 * sp.pi / n
            md = matrix(2, 2, [[sp.cos((h - 2) * theta),sp.sin((h - 2) * theta)],[-sp.sin((h - 2) * theta),sp.cos((h - 2) * theta)]])
    return(md)

def dihedral(*args):
    """
    Input:
    n: for D_n
    h: no. corresponding to irrep of D_n

    Output:
    g: dict representation of group
    """
    nargs = len(args)
    if nargs < 1:
        raise("Argument missing")
    if 2 < nargs:
        raise("Too many arguments")
    n = args[0]
    if not(is_type(n, 'integer')) or n < 2:
        raise("First argument must be positive integer")
    if nargs == 2:
        h = args[1]
        if not(is_type(h, 'listinteger')):
            raise("Second argument must be list of integers.")
        if is_type(n, 'even'):
            if not(all(h1 <= n / 2 + 3 and 1 <= h1 for h1 in h)):
                raise("Second argument does not correspond to nrs of irreducible repr.")
        else:
            if not(all(h1 <= n / 2 + 0.3e1 / 0.2e1 and 1 <= h1 for h1 in h)):
                raise("Second argument does not correspond to nr. of irreducible repr.")
    else:
        if is_type(n, 'even'):
            h = [5]
            return(h)
        else:
            h = [3]
            return(h)
    res = {}
    res['oname'] = 'dihedral'+str(n)   
    res['grouptype'] = 'finite' 
    res['isotypics'] = h
    # print(list(map(dihedralreflection, h, [n for i in range(len(h))])))
    if len(h) >1:
        ms = sp.diag(*list(map(dihedralreflection, h, [n for i in range(len(h))])))
        md =  sp.diag(*list(map(dihedralrotation, h, [n for i in range(len(h))])))
    else:
        ms = diag(list(map(dihedralreflection, h, [n for i in range(len(h))])))
        md = diag(list(map(dihedralrotation, h, [n for i in range(len(h))])))
    # print(md)
    res['generators'] = {'_s': ms, '_r': md}
    a = [md**k for k in range(1,n)]
    a = list(map(lambda a1,s: s*a1,a,[ms for i in range(1,n)]))
    res['allelements'] = {'_r0': md ** n}
    for k in range(1,n):
        res['allelements']['_r'+str(k)] = md**k
    res['allelements']['_s_r0'] = ms
    for k in range(1,n):
        res['allelements']['_s_r'+str(k)] = a[k-1]
    res['invelements'] = {'_r0': res['allelements']['_r0']}
    for k in range(1,n):
        res['invelements']['_r'+str(k)] = '_r'+str(n-k)
    for k in range(n):
        res['invelements']['_s_r'+str(k)] = '_s_r'+str(k)
    if nargs == 2:
        if (is_type(n, 'even') and not(member(5,h))) or (is_type(n, 'odd') and not(member(3,h))):
            print('Warning, representation is not faithful.')
    return res


def cyclicrotation(h,n):
    """
    Input:
    h: no. corresponding to irrep of C_n
    n: for C_n

    Output:
    1D representation
    """
    return matrix(1,1,[[sp.exp(2*sp.pi*sp.I*(h-1)/n)]])

def cyclic(*args):
    """
    Input:
    n: for C_n
    h: no. corresponding to irrep of C_n

    Output:
    g: dict representation of group
    """
    nargs = len(args)
    if nargs < 1:
        raise("Argument missing")
    if 2 < nargs:
        raise("Too many arguments")
    n = args[0]
    if not(is_type(n, 'integer')) or n < 2:
        raise("First argument must be positive integer")
    if nargs == 2:
        h = args[1]
        if not(is_type(h, 'listinteger')):
            raise("Second argument must be list of integers.")
        if not(all(elem<=n and elem>=1 for elem in h)):
            raise("Second argument does not correspond to nrs of irreducible repr.")
    else:
        if is_type(n,'even'):
            h = [n/2+1]
        else:
            h = [2]
    res = {}
    res['oname'] = 'dihedral'+str(n)   
    res['grouptype'] = 'finite' 
    res['isotypics'] = h
    mr = diag([cyclicrotation(elem,n) for elem in h])
    res['generators'] = {'_r':mr}
    res['allelements'] = {'_r0':mr**n}|{'_r'+str(k):mr**k for k in range(1,n)}
    res['invelements'] = {'_r0':'_r0'}|{'_r'+str(k):'_r'+str(n-k) for k in range(1,n)}
    if nargs == 2:
        if not(member(True,[True if gcd(elem-1,n) == 1 else False for elem in h])):
            print('Warning, representation is not faithful.')
    return res

def icosahedral(*args):
    """
    Input:
    h: list of numbers corresponding to irreps of icosahedral group
    
    Output:
    res: dict representation of icosahedral group
    """
    nargs = len(args)
    if nargs > 1:
        raise('icosahedral: No more than one argument allowed.')
    if nargs == 0:
        h = [5]
    else:
        h = args[0]
        if not(isinstance(h, list)):
            raise('icosahedral: Argument must be a list.')
        if not(is_type(h, 'listinteger')):
            raise('icosahedral: Argument must be a list of integers.')
        if not(all(elem <= 5 and elem >= 1 for elem in h)):
            raise('icosahedral: Argument does not correspond to no. of irreducible representation.')
    mc2 = icosahedral_c2(h)
    mc5 = icosahedral_c5(h)
    res = mkfinitegroup({'c2':mc2, 'c5':mc5},'icosahedral')
    res['isotypics'] = h
    return res

def icosahedral_c2(h):
    """
    Input:
    h: list of numbers corresponding to irreps of icosahedral group
    
    Output:
    block diagonal representation of c2 for irreps h of icosahedral group
    """
    return diag(*[icosahedral_c2_i(elem) for elem in h])

def icosahedral_c5(h):
    """
    Input:
    h: list of numbers corresponding to irreps of icosahedral group
    
    Output:
    block diagonal representation of c5 for irreps h of icosahedral group
    """
    return diag(*[icosahedral_c5_i(elem) for elem in h])

def icosahedral_c2_i(i):
    """
    Input:
    i: single no. corresponding to single irrep of icosahedral group
    
    Output:
    matrix representation of c2 for single irrep i of icosahedral group
    """
    if i == 1:
        c2 = matrix(1,1,[[1]])
    elif i == 2 or i == 3:
        c2 = matrix(3,3,[[-1,0,0],[0,-1,0],[0,0,1]])
    elif i == 4:
        c2 = matrix(4,4,[ [1,0,0,0],
                              [0,-1/3,2*sp.sqrt(2)/3,0],
                              [0,2*sp.sqrt(2)/3,1/3,0],
                              [0,0,0,-1]])
    elif i ==5:
        c2 = matrix(5,5,[[-1/3,2*sp.sqrt(2)/3,0,0,0],
                              [2*sp.sqrt(2)/3,1/3,0,0,0],
                              [0,0,1,0,0],
                              [0,0,0,-1,0],
                              [0,0,0,0,-1]])
    else:
        raise('icosahedral_c2_i: Invalid no. for irrep, must be integer from 1 to 5.')
    return c2

def icosahedral_c5_i(i):
    """
    Input:
    i: single no. corresponding to single irrep of icosahedral group
    
    Output:
    matrix representation of c5 for single irrep i of icosahedral group
    """
    etam = 0.618034
    etap = 1.618034
    if i == 1:
        c5 = matrix(1,1,[[1]])
    elif i == 2:
        c5 = 1/2 * matrix(3,3,[[etam,-etap,1], 
                                [etap,1,etam],
                                [-1,etam,etap]])
    elif i == 3:
        c5 = 1/2 * matrix(3,3,[[-etap,-1,etam], 
                                 [1,-etam,etap],
                                 [-etam,etap,1]])
    elif i == 4:
        c5 = 1/12 * matrix(4,4,[[-3,3*sp.sqrt(15),0,0],
                               [-sp.sqrt(15),-1,8*sp.sqrt(2),0],
                               [-sp.sqrt(30),-sp.sqrt(2),-2,6*sp.sqrt(3)],
                               [-3*sp.sqrt(10),-sp.sqrt(6),-2*sp.sqrt(3),-6]])
    elif i ==5:
        c5 = 1/12 * matrix(5,5,[[-4,-3*sp.sqrt(3),4*sp.sqrt(6),0,0],
                               [-4*sp.sqrt(2),1,-sp.sqrt(3),-3*sp.sqrt(3),9],
                               [0,-3*sp.sqrt(3),-3,-9,-3*sp.sqrt(3)],
                               [-4*sp.sqrt(6),sp.sqrt(3),-3,3,-3*sp.sqrt(3)],
                               [0,-9,-3*sp.sqrt(3),3*sp.sqrt(3),3]])
    else:
        raise('icosahedral_c5_i: Invalid no. for irrep, must be integer from 1 to 5.')
    return c5

def tetrahedral(*args):
    """
    Input:
    h: list of numbers corresponding to irreps of tetrahedral group
    
    Output:
    res: dict representation of tetrahedral group
    """
    nargs = len(args)
    if nargs > 1:
        raise('tetrahedral: No more than one argument allowed.')
    if nargs == 1:
        h = args[0]
        if not(is_type(h, 'listinteger')) or not(all(elem>=1 and elem<=4 for elem in h)):
            raise('tetrahedral: Argument should be list of integers between 1 and 4.')
    else:
        h = [4]
    res = {}
    res['oname'] = 'tetrahedral'
    res['grouptype'] = 'finite'
    res['isotypics'] = h
    mta4 = diag(*[tetrahedralt(elem) for elem in h])
    mxa4 = diag(*[tetrahedralx(elem) for elem in h])
    mya4 = diag(*[tetrahedraly(elem) for elem in h])
    mza4 = diag(*[tetrahedralz(elem) for elem in h])
    res['generators'] = {'_ta4':mta4, '_xa4':mxa4, '_ya4':mya4, '_za4':mza4}
    res['allelements'] = {'_ida4': sp.Matrix(mta4**3),
                          '_ta4': mta4,
                          '_t2a4': sp.Matrix(mta4**2),
                          '_xa4': mxa4,
                          '_ya4': mya4,
                  '_za4': mza4,
                          '_txa4': mta4 * mxa4,
                          '_tya4': mta4 * mya4,
                          '_tza4': mta4 * mza4,
                          '_t2xa4': sp.Matrix(mta4 * mta4 * mxa4),
                          '_t2ya4': sp.Matrix(mta4 * mta4 * mya4),
                          '_t2za4': sp.Matrix(mta4 * mta4 * mza4)}
    res['invelements'] = {'_ida4' : '_ida4',
                          '_ta4'  : '_t2a4',
                          '_t2a4' : '_ta4',
                          '_xa4'  : '_xa4',
                          '_ya4'  : '_ya4',
                          '_za4'  : '_za4',
                          '_txa4' : '_t2za4',
                          '_tya4' : '_t2xa4',
                          '_tza4' : '_t2ya4',
                          '_t2xa4' : '_tya4',
                          '_t2ya4' : '_tza4',
                          '_t2za4' : '_txa4'}
    if not(member(4, set(h))):
        print('tetrahedral: Warning, representation is not faithful.')
    return res

def tetrahedralt(i):
    """
    Input:
    i: single no. corresponding to single irrep of tetrahedral group
    
    Output:
    ms: matrix representation of t element for single irrep i of tetrahedral group
    """
    if i ==1:
        ms = matrix(1,1,[[1]])
    elif i == 2:
        ms = matrix(1,1,[[-1/2+sp.I*sp.sqrt(3)/2]])
    elif i == 3:
        ms = matrix(1,1, [[-1/2-sp.I*sp.sqrt(3)/2]])
    elif i == 4:
        ms = matrix(3,3, [[0,0,-1],[-1,0,0],[0,1,0]])
    else:
        raise('tetrahedralt: Invalid no. for irrep, must be integer from 1 to 4.')
    return ms

def tetrahedralx(i):
    """
    Input:
    i: single no. corresponding to single irrep of tetrahedral group
    
    Output:
    ms: matrix representation of x element for single irrep i of tetrahedral group
    """
    if i ==1 or i == 2 or i == 3:
        ms = matrix(1,1,[[1]])
    elif i == 4:
        ms = matrix(3,3, [[-1,0,0],[0,1,0],[0,0,-1]])
    else:
        raise('tetrahedralx: Invalid no. for irrep, must be integer from 1 to 4.')
    return ms

def tetrahedraly(i):
    """
    Input:
    i: single no. corresponding to single irrep of tetrahedral group
    
    Output:
    ms: matrix representation of y element for single irrep i of tetrahedral group
    """
    if i ==1 or i == 2 or i == 3:
        ms = matrix(1,1,[[1]])
    elif i == 4:
        ms = matrix(3,3, [[1,0,0],[0,-1,0],[0,0,-1]])
    else:
        raise('tetrahedraly: Invalid no. for irrep, must be integer from 1 to 4.')
    return ms

def tetrahedralz(i):
    """
    Input:
    i: single no. corresponding to single irrep of tetrahedral group
    
    Output:
    ms: matrix representation of z element for single irrep i of tetrahedral group
    """
    if i ==1 or i == 2 or i == 3:
        ms = matrix(1,1,[[1]])
    elif i == 4:
        ms = matrix(3,3, [[-1,0,0],[0,-1,0],[0,0,1]])
    else:
        raise('tetrahedralz: Invalid no. for irrep, must be integer from 1 to 4.')
    return ms

def octrahedral(*args):
    """
    Input:
    h: list of numbers corresponding to irreps of octrahedral group
    
    Output:
    res: dict representation of octrahedral group
    """
    nargs = len(args)
    if nargs > 1:
        raise('octrahedral: No more than one argument allowed.')
    if nargs == 1:
        h = args[0]
        if not(is_type(h, 'listinteger')) or not(all(elem>=1 and elem<=5 for elem in h)):
            raise('octrahedral: Argument should be list of integers between 1 and 5.')
    else:
        h = [4]
    res = {}
    res['oname'] = 'octrahedral'
    res['grouptype'] = 'finite'
    res['isotypics'] = h
    mbacd = diag(*[octrahedralbacd(elem) for elem in h])
    macbd = diag(*[octrahedralacbd(elem) for elem in h])
    mabdc = diag(*[octrahedralabdc(elem) for elem in h])
    mdbca = diag(*[octrahedraldbca(elem) for elem in h])
    res['generators'] = {'_bacd':mbacd, '_acbd':macbd, '_abdc':mabdc, '_dbca':mdbca}
    res['allelements'] = {'_ids4' : sp.Matrix(mbacd**2), 
                            '_bacd' : mbacd,
                            '_abdc' : mabdc,
                            '_dbca' : mdbca,
                            '_cabd' : mbacd * macbd,
                            '_acbd' : macbd,
                            '_bcad' : macbd * mbacd,
                            '_dacb' : mdbca * mbacd,
                            '_bdca' : mbacd * mdbca,
                            '_dbac' : mabdc * mdbca,
                            '_cbda' : mdbca * mabdc,
                            '_adbc' : macbd * mabdc,
                            '_acdb' : mabdc * macbd,
                            '_badc' : mbacd * mabdc,
                            '_cdab' : mabdc * mbacd * macbd * mdbca,
                            '_dcba' : macbd * mdbca,
                            '_cbad' : mbacd * macbd * mbacd,
                            '_adcb' : mdbca * mbacd * mdbca,
                            '_bcda' : mabdc * macbd * mbacd,
                            '_bdac' : macbd * mbacd * mabdc,
                            '_cadb' : mabdc * mbacd * macbd,
                            '_dabc' : mbacd * macbd * mabdc,
                            '_cdba' : mbacd * macbd * mdbca,
                            '_dcab' : mabdc * macbd * mdbca}
    res['invelements'] = {'_ids4' : '_ids4',
                            '_dcab' : '_cdba',
                            '_dcba' : '_dcba',
                            '_dbac' : '_cbda',
                            '_dbca' : '_dbca',
                            '_dabc' : '_bcda',
                            '_dacb' : '_bdca',
                            '_cdab' : '_cdab',
                            '_cdba' : '_dcab',
                            '_cbad' : '_cbad',
                            '_cbda' : '_dbac',
                            '_cabd' : '_bcad',
                            '_cadb' : '_bdac',
                            '_bdac' : '_cadb',
                            '_bdca' : '_dacb',
                            '_bcad' : '_cabd',
                            '_bcda' : '_dabc',
                            '_bacd' : '_bacd',
                            '_badc' : '_badc',
                            '_adbc' : '_acdb',
                            '_adcb' : '_adcb',
                            '_acbd' : '_acbd',
                            '_acdb' : '_adbc',
                            '_abdc' : '_abdc'}
    if not(member(4,set(h))) and not(member(5,set(h))):
        print('octrahedral: Warning, representation is not faithful.')
    return res

def octrahedralbacd(i):
    """
    Input:
    i: single no. corresponding to single irrep of octrahedral group
    
    Output:
    ms: matrix representation of bacd element for single irrep i of octrahedral group
    """
    if i == 1:
        ms = matrix(1,1,[[1]])
    elif i == 2:
        ms = matrix(1,1,[[-1]])
    elif i == 3:
        ms = matrix(2,2,[[1/2,sp.sqrt(3)/2],[sp.sqrt(3)/2,-1/2]])
    elif i == 4:
        ms = matrix(3,3, [[0,0,-1],[0,1,0],[-1,0,0 ]])
    elif i == 5:
        ms = matrix(3,3, [[0,0, 1],[0,-1,0],[1,0,0 ]])
    else:
        raise('octrahedralbacd: Invalid no. for irrep, must be integer from 1 to 5.')
    return ms

def octrahedralacbd(i):
    """
    Input:
    i: single no. corresponding to single irrep of octrahedral group
    
    Output:
    ms: matrix representation of acbd element for single irrep i of octrahedral group
    """
    if i == 1:
        ms = matrix(1,1,[[1]])
    elif i == 2:
        ms = matrix(1,1,[[-1]])
    elif i == 3:
        ms = matrix(2,2,[[1/2,-sp.sqrt(3)/2],[-sp.sqrt(3)/2,-1/2]])
    elif i == 4:
        ms = matrix(3,3, [[0,-1,0],[-1,0,0],[0,0,1]])
    elif i == 5:
        ms = matrix(3,3, [[0,1,0],[1,0,0],[0,0,-1]])
    else:
        raise('octrahedralacbd: Invalid no. for irrep, must be integer from 1 to 5.')
    return ms

def octrahedralabdc(i):
    """
    Input:
    i: single no. corresponding to single irrep of octrahedral group
    
    Output:
    ms: matrix representation of abdc element for single irrep i of octrahedral group
    """
    if i == 1:
        ms = matrix(1,1,[[1]])
    elif i == 2:
        ms = matrix(1,1,[[-1]])
    elif i == 3:
        ms = matrix(2,2,[[1/2,sp.sqrt(3)/2],[sp.sqrt(3)/2,-1/2]])
    elif i == 4:
        ms = matrix(3,3, [[0,0,1],[0,1,0],[1,0,0]])
    elif i == 5:
        ms = matrix(3,3, [[0,0,-1],[0,-1,0],[-1,0,0]])
    else:
        raise('octrahedralabdc: Invalid no. for irrep, must be integer from 1 to 5.')
    return ms

def octrahedraldbca(i):
    """
    Input:
    i: single no. corresponding to single irrep of octrahedral group
    
    Output:
    ms: matrix representation of dbca element for single irrep i of octrahedral group
    """
    if i == 1:
        ms = matrix(1,1,[[1]])
    elif i == 2:
        ms = matrix(1,1,[[-1]])
    elif i == 3:
        ms = matrix(2,2,[[1/2,-sp.sqrt(3)/2],[-sp.sqrt(3)/2,-1/2]])
    elif i == 4:
        ms = matrix(3,3, [[0,1,0],[1,0,0],[0,0,1]])
    elif i == 5:
        ms = matrix(3,3, [[0,-1,0],[-1,0,0],[0,0,-1]])
    else:
        raise('octrahedraldbca: Invalid no. for irrep, must be integer from 1 to 5.')
    return ms

# print(dihedral(4,[5,1]))

