import itertools
import numpy as np
import scipy
import math
import time as time

from tqdm import tqdm
from built_ins import SparseDictNZ
from types import FunctionType

import sympy as sp
from sympy.matrices.utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from sympy.core.singleton import S
from sympy.external.pythonmpq import PythonMPQ
from sympy.core.numbers import Float, Integer

def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):

    newly_determined = []
    col = list(col)
    # a column that contains a mix of floats and integers
    # but at least one float is considered a numerical
    # column, and so we do partial pivoting
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            # just because iszerofunc returned True, doesn't
            # mean the value is numerically zero.  Make sure
            # to replace all entries with numerical zeros
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    # PASS 1 (iszerofunc directly)
    possible_zeros = []
    # print('Pass 1')
    for i, x in enumerate(col):
        is_zero = iszerofunc(x)
        # is someone wrote a custom iszerofunc, it may return
        # BooleanFalse or BooleanTrue instead of True or False,
        # so use == for comparison instead of `is`
        if is_zero == False:
            # we found something that is definitely not zero
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)

    # by this point, we've found no certain non-zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 2 (iszerofunc after simplify)
    # we haven't found any for-sure non-zeros, so
    # go through the elements iszerofunc couldn't
    # make a determination about and opportunistically
    # simplify to see if we find something
    print('Pass 2')
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero in (True, False):
            newly_determined.append((i, simped))
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero

    # after simplifying, some things that were recognized
    # as zeros might be zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 3 (.equals(0))
    # some expressions fail to simplify to zero, but
    # ``.equals(0)`` evaluates to True.  As a last-ditch
    # attempt, apply ``.equals`` to these expressions
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            # ``.iszero`` may return False with
            # an implicit assumption (e.g., ``x.equals(0)``
            # when ``x`` is a symbol), so only treat it
            # as proved when ``.equals(0)`` returns True
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        return (None, None, False, newly_determined)

    # at this point there is nothing that could definitely
    # be a pivot.  To maintain compatibility with existing
    # behavior, we'll assume that an illdetermined thing is
    # non-zero.  We should probably raise a warning in this case
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)

def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc,
                normalize_last=True, normalize=True, zero_above=True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = isimp(a*mat[p] - b*mat[p + q])

    isimp = _get_intermediate_simp(_dotprodsimp)
    piv_row, piv_col = 0, 0
    pivot_cols = []
    swaps = []

    # use a fraction free method to zero above and below each pivot
    # pbar = tqdm(desc = 'while loop', total = rows)
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val, \
        assumed_nonzero, newly_determined = _find_reasonable_pivot(
                get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        # _find_reasonable_pivot may have simplified some things
        # in the process.  Let's not let them go to waste
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset*cols + piv_col] = val

        if pivot_offset is None:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))

        # if we aren't normalizing last, we normalize
        # before we zero the other rows
        # if normalize_last is False:
        #     i, j = piv_row, piv_col
        #     mat[i*cols + j] = one
        #     for p in range(i*cols + j + 1, (i + 1)*cols):
        #         mat[p] = isimp(mat[p] / pivot_val)
        #     # after normalizing, the pivot value is 1
        #     pivot_val = one

        # zero above and below the pivot
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # don't zero above the pivot unless we're told.
            if zero_above is False and row < piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row*cols + piv_col]
            if iszerofunc(val):
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1
        # pbar.update(piv_row)

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i*cols + piv_j]
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)
    # print(swaps)
    return mat, tuple(pivot_cols), tuple(swaps)


def _find_reasonable_pivot_nump(col):
    
    arg_max = np.argmax(col['n'] != 0) #First non-zero value
    if col['n'][arg_max] == 0:
        return None, None
    return arg_max, col[arg_max]

isimp = _get_intermediate_simp(_dotprodsimp)

def _row_reduce_nump(MD, mat, rows, cols, normalize_last=True, normalize=True, zero_above=True, one=S.One, iszerofunc=_iszero, simpfunc=_simplify):
    # def get_col(i):
    #     return mat[i::cols]

    # def row_swap(i, j):
    #     mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
    #         mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    piv_row, piv_col = 0, 0
    pivot_cols = []

    # use a fraction free method to zero above and below each pivot
    pbar = tqdm(desc = 'while loop', total =rows)
    while piv_col < cols and piv_row < rows:
        #Numpy
        pivot_offset, pivot_val = _find_reasonable_pivot_nump(MD[piv_row:,piv_col])
        if pivot_offset != None:
            assert(pivot_val['n'] != 0 and pivot_val['d'] != 0)

        # #Sympy
        # s_pivot_offset, s_pivot_val, \
        # assumed_nonzero, newly_determined = _find_reasonable_pivot(
        #         get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        # #Check
        # assert(s_pivot_offset == pivot_offset)
        # if s_pivot_val is None:
        #     assert(pivot_val is None)
        # else:
        #     assert(sp.Rational(pivot_val) == s_pivot_val)

        if pivot_offset is None:
            piv_col += 1
            continue
        
        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            #Numpy
            MD[(piv_row, pivot_offset + piv_row),:] = MD[(pivot_offset + piv_row, piv_row),:]
            pivot_val = MD[piv_row,piv_col]
            # #SymPy
            # row_swap(piv_row, pivot_offset + piv_row)
            # #Check
            # assert(mat[piv_row*cols:(piv_row+1)*cols] == [sp.Rational(MD[piv_row,j]) for j in range(cols)])
            # assert(mat[(pivot_offset + piv_row)*cols:(pivot_offset + piv_row+1)*cols] == [sp.Rational(MD[pivot_offset + piv_row,j]) for j in range(cols)])      

        # # noramlize_last = False (normalize as we go)
        # nz = np.nonzero(MD['n'][piv_row])[0]
        # num, den = MD['n'][piv_row,nz]*pivot_val['d'], MD['d'][piv_row,nz]*pivot_val['n']
        # num[den < 0], den[den < 0] = num[den < 0]*-1,  den[den < 0]*-1
        # gcd = np.gcd(num.astype(np.int64),den.astype(np.int64))
        # MD['n'][piv_row,nz], MD['d'][piv_row,nz] = num/gcd, den/gcd
        # pivot_val = MD[piv_row, piv_col]
        
        # assert(np.all(MD['d'] != 0))
        # assert(MD['n'][piv_row, piv_col] == 1 and MD['d'][piv_row, piv_col] == 1)
        
                             
        #Numpy
        nz = np.nonzero(MD['n'][:,piv_col])[0]
        nz = nz[nz != piv_row]

        if len(nz) != 0:
            # #SymPy
            # for i in nz:
            #     s_a, i, s_b, j = s_pivot_val, i, mat[i*cols + piv_col], piv_row
            #     q = (j - i)*cols
            #     for p in range(i*cols, (i + 1)*cols):
            #         before = s_a*mat[p] - s_b*mat[p + q]
            #         mat[p] = isimp(before)
            #         assert(before == mat[p])

            #Numpy
            a, b = pivot_val, MD[nz,piv_col]
            # MD[nz] = a*MD[nz] - b[:, np.newaxis]*MD[piv_row]
            aMDnz, bMDpivrow = np.empty_like(MD[nz]), np.empty_like(MD[nz])
            num, den = a['n']*MD['n'][nz], a['d']*MD['d'][nz]
            gcd = np.gcd(num.astype(np.int64),den.astype(np.int64))
            aMDnz['n'], aMDnz['d']  = num/gcd, den/gcd
            
            num, den = b['n'][:, np.newaxis]*MD['n'][piv_row], b['d'][:,np.newaxis]*MD['d'][piv_row]
            gcd = np.gcd(num.astype(np.int64),den.astype(np.int64))
            bMDpivrow['n'], bMDpivrow['d']  = num/gcd, den/gcd

            lcm = np.lcm(np.abs(aMDnz['d'].astype(np.int64)),np.abs(bMDpivrow['d'].astype(np.int64)))
            num, den = aMDnz['n']*(aMDnz['d']/lcm) - bMDpivrow['n']*(bMDpivrow['d']/lcm), lcm
            gcd = np.gcd(num.astype(np.int64),den.astype(np.int64))
            MD['n'][nz], MD['d'][nz] = num/gcd, den/gcd

            assert(np.all(MD['d'] > 0))
            
                
            # #Check
            # assert([[mat[p] for p in range(i*cols, (i + 1)*cols)] for i in nz] == [[sp.Rational(MD[i,k]) for k in range(cols)] for i in nz])

        piv_row += 1
        pbar.update(1)

    # normalize each row
    for piv_row,piv_col in enumerate(pivot_cols):
        pivot_val = MD[piv_row, piv_col]
        nz = np.nonzero(MD['n'][piv_row])[0]
        num, den = MD['n'][piv_row,nz]*pivot_val['d'], MD['d'][piv_row,nz]*pivot_val['n']
        num[den < 0], den[den < 0] = num[den < 0]*-1,  den[den < 0]*-1
        gcd = np.gcd(num.astype(np.int64),den.astype(np.int64))
        MD['n'][piv_row,nz], MD['d'][piv_row,nz] = num/gcd, den/gcd
        assert(MD['n'][piv_row, piv_col] == 1 and MD['d'][piv_row, piv_col] == 1)

    assert(np.all(MD['d'] != 0))
    

    # print('Normalizing each row in M_:')
    # for piv_i,piv_j in enumerate(tqdm(pivot_cols)): 
    #     #SymPy
    #     pivot_val = out[piv_i*cols + piv_j]
    #     out[piv_i*cols + piv_j] = one
    #     # for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
    #     #     out[p] = isimp(out[p] / pivot_val)
    #     out[piv_i*cols + piv_j + 1:(piv_i + 1)*cols] = [out[p].__truediv__(pivot_val) if out[p] != 0 else 0 for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols)]
        
    return MD, pivot_cols
# print(PythonMPQ(8.4e-323))
# print(PythonMPQ(1).__truediv__(PythonMPQ(8.4e-323)))

"""
def numpy_gcd(a, b):
    a, b = np.broadcast_arrays(a, b)
    a = a.copy()
    b = b.copy()
    pos = np.nonzero(b)[0]
    while len(pos) > 0:
        b2 = b[pos]
        a[pos], b[pos] = b2, a[pos] % b2
        pos = pos[b[pos]!=0]
    return a

def _row_reduce_nump_nd(MDn, MDd, mat, rows, cols, normalize_last=True, normalize=True, zero_above=True, one=S.One, iszerofunc=_iszero, simpfunc=_simplify):
    # def get_col(i):
    #     return mat[i::cols]

    # def row_swap(i, j):
    #     mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
    #         mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    # def cross_cancel(a, i, b, j):
    #     q = (j - i)*cols
    #     for p in range(i*cols, (i + 1)*cols):
    #         mat[p] = isimp(a*mat[p] - b*mat[p + q])

    piv_row, piv_col = 0, 0
    pivot_cols = []

    # use a fraction free method to zero above and below each pivot
    pbar = tqdm(desc = 'while loop', total =rows)
    while piv_col < cols and piv_row < rows:
        #Numpy
        pivot_offset, pivot_val = _find_reasonable_pivot_nump_nd(MDn[piv_row:,piv_col],MDd[piv_row:,piv_col])
        #Sympy
        # s_pivot_offset, s_pivot_val, \
        # assumed_nonzero, newly_determined = _find_reasonable_pivot(
        #         get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        #Check
        # assert(s_pivot_offset == pivot_offset)
        # if s_pivot_val is None:
        #     assert(pivot_val[0] is None)
        # else:
        #     assert(sp.Rational(pivot_val[0],pivot_val[1]) == s_pivot_val)

        if pivot_offset is None:
            piv_col += 1
            continue
        
        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            #Numpy
            MDn[(piv_row, pivot_offset + piv_row),:] = MDn[(pivot_offset + piv_row, piv_row),:]
            MDd[(piv_row, pivot_offset + piv_row),:] = MDd[(pivot_offset + piv_row, piv_row),:]
            #SymPy
            # row_swap(piv_row, pivot_offset + piv_row)
            #Check
            # assert(mat[piv_row*cols:(piv_row+1)*cols] == [sp.Rational(MDn[piv_row,j],MDd[piv_row,j]) for j in range(cols)])
            # assert(mat[(pivot_offset + piv_row)*cols:(pivot_offset + piv_row+1)*cols] == [sp.Rational(MDn[pivot_offset + piv_row,j],MDd[pivot_offset + piv_row,j]) for j in range(cols)])
    

        #Numpy
        nz = np.nonzero(MDn[:,piv_col])[0]
        nz = nz[nz != piv_row]

        if len(nz) != 0:
            # #SymPy
            # for i in nz:
            #     a, i, b, j = s_pivot_val, i, mat[i*cols + piv_col], piv_row
            #     q = (j - i)*cols
            #     for p in range(i*cols, (i + 1)*cols):
            #         before = a*mat[p] - b*mat[p + q]
            #         mat[p] = isimp(before)
            #         assert(before == mat[p])

            #Numpy
            b_n, b_d = MDn[nz,piv_col], MDd[nz,piv_col]

            ln_inz, ld_inz = pivot_val[0]*MDn[nz], pivot_val[1]*MDd[nz]
            l_inz = np.nonzero(ln_inz)
            ln_inz, ld_inz, ln, ld = ln_inz[l_inz], ld_inz[l_inz], np.zeros_like(ln_inz), np.ones_like(ld_inz)
            gcd = numpy_gcd(ln_inz,ld_inz) #np.gcd(np.array(ln_inz,dtype=np.int64), np.array(ld_inz,dtype=np.int64)) 
            ln[l_inz], ld[l_inz] = ln_inz/gcd, ld_inz/gcd

            
            rn_inz, rd_inz = b_n[:, np.newaxis]*MDn[piv_row], b_d[:, np.newaxis]*MDd[piv_row]
            r_inz = np.nonzero(rn_inz)
            rn_inz, rd_inz, rn, rd = rn_inz[r_inz], rd_inz[r_inz], np.zeros_like(rn_inz), np.ones_like(rd_inz)
            gcd = numpy_gcd(rn_inz,rd_inz) #np.gcd(np.array(rn_inz,dtype=np.int64), np.array(rd_inz,dtype=np.int64)) 
            rn[r_inz], rd[r_inz] = rn_inz/gcd, rd_inz/gcd
            
            # mn = pivot_val[0]*b_d[:, np.newaxis]*MDn[nz]*MDd[piv_row] - pivot_val[1]*b_n[:, np.newaxis]*MDn[piv_row]*MDd[nz]
            # md = pivot_val[1]*b_d[:, np.newaxis]*MDd[nz]*MDd[piv_row]
            mn = ln*rd - rn*ld
            md = ld*rd
            
            MDn[nz], MDd[nz] = 0, 1
            inz = np.nonzero(mn)
            mn, md = mn[inz], md[inz]
            gcd = numpy_gcd(mn,md) #np.gcd(np.array(mn,dtype=np.int64), np.array(md,dtype=np.int64)) 
            MDn[nz[inz[0]],inz[1]], MDd[nz[inz[0]],inz[1]] = mn/gcd, md/gcd
                
            #Check
            # assert([[mat[p] for p in range(i*cols, (i + 1)*cols)] for i in nz] == [[sp.Rational(MDn[i,k], MDd[i,k]) for k in range(cols)] for i in nz])

        piv_row += 1
        pbar.update(1)

    # normalize each row
    for piv_i,piv_j in enumerate(pivot_cols): 
        #Numpy
        pivot_val_n, pivot_val_d = MDn[piv_i,piv_j], MDd[piv_i,piv_j]
        MDn[piv_i,piv_j], MDd[piv_i,piv_j] = 1, 1
        MDn[piv_i,piv_j+1:] = MDn[piv_i,piv_j+1:] * pivot_val_d
        MDd[piv_i,piv_j+1:] = MDd[piv_i,piv_j+1:] * pivot_val_n
        
    return MDn/MDd

def _row_reduce_scip(MDn, MDd, rows, cols):

    piv_row, piv_col = 0, 0
    pivot_cols = []
    times = {i:[] for i in range(7)}

    # use a fraction free method to zero above and below each pivot
    pbar = tqdm(desc = 'while loop', total =rows)
    while piv_col < cols and piv_row < rows:
        #Scipy
        st = time.time()
        pivot_offset, pivot_val = _find_reasonable_pivot_scip(MDn.tocsr()[piv_row:,:].tocsc()[:,piv_col],MDd.tocsr()[piv_row:,:].tocsc()[:,piv_col])
        ft = time.time()
        times[0].append(ft-st)

        if pivot_offset is None:
            piv_col += 1
            continue
        
        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            #Scipy
            st = time.time()
            I = scipy.sparse.eye(rows).tocoo()
            I.row = I.row[[pivot_offset + piv_row if k==piv_row else piv_row if k==pivot_offset + piv_row else k for k in range(rows)]]
            MDn, MDd = I.tocsr().dot(MDn.tocsr()).tolil(), I.tocsr().dot(MDd.tocsr()).tolil()
            # MDn[(piv_row, pivot_offset + piv_row),:] = MDn[(pivot_offset + piv_row, piv_row),:]
            # MDd[(piv_row, pivot_offset + piv_row),:] = MDd[(pivot_offset + piv_row, piv_row),:]
            ft = time.time()
            times[1].append(ft-st)

        #Scipy
        st = time.time()
        nz = MDn.tocsc()[:,piv_col].nonzero()[0]
        nz = nz[nz != piv_row]
        ft = time.time()
        times[2].append(ft-st)

        if len(nz) != 0:
            st = time.time()
            b_n, b_d = MDn.tocsr()[nz,:].tocsc()[:,piv_col], MDd.tocsr()[nz,:].tocsc()[:,piv_col]
            #Scipy vectorised
            mddnz, mdnnz, mddpr, mdnpr = MDd.tocsr()[nz], MDn.tocsr()[nz], MDd.tocsr()[piv_row], MDn.tocsr()[piv_row]
            mddnz1, mddpr1 = scipy.sparse.csr_matrix(np.ones(mddnz.shape)), scipy.sparse.csr_matrix(np.ones(mddpr.shape))
            mddnz1[mddnz.nonzero()], mddpr1[mddpr.nonzero()] = mddnz[mddnz.nonzero()], mddpr[mddpr.nonzero()]
            
            mn = pivot_val[0]*b_d.multiply(mdnnz.multiply(mddpr1)) - pivot_val[1]*b_n.multiply(mdnpr.multiply(mddnz1))
            md = pivot_val[1]*b_d.multiply(mddnz1.multiply(mddpr1))
            ft = time.time()
            times[3].append(ft-st)
            
            
            if mn.count_nonzero() == 0:
                MDn[nz], MDd[nz] = scipy.sparse.lil_matrix((len(nz),cols)), scipy.sparse.lil_matrix((len(nz),cols)) #0, 0
            else:
                st = time.time()
                inz = mn.nonzero()
                mn_nz, md_nz = np.array(mn[inz],dtype=np.int64), np.array(md[inz],dtype=np.int64)
                ft = time.time()
                times[4].append(ft-st)
                
                st = time.time()
                gcd = np.gcd(mn_nz, md_nz)
                ft = time.time()
                times[5].append(ft-st)
                
                st = time.time()
                mn, md = np.zeros(mn.shape), np.zeros(md.shape)
                mn[inz], md[inz] = mn_nz/gcd, md_nz/gcd
                MDn[nz], MDd[nz] = scipy.sparse.lil_matrix(mn), scipy.sparse.lil_matrix(md)
                ft = time.time()
                times[6].append(ft-st)


        piv_row += 1
        pbar.update(1)

        if piv_row % 100 == 0:
            print('_find_reasonable_pivot_nump:', sum(times[0])/len(times[0]))
            print('row_swap:', sum(times[1])/len(times[1]))
            print('nz:', sum(times[2])/len(times[2]))
            print('mn,md:', sum(times[3])/len(times[3]))
            print('before_gcd:', sum(times[4])/len(times[4]))
            print('gcd:', sum(times[5])/len(times[5]))
            print('after_gcd:', sum(times[6])/len(times[6]))
    # normalize each row
    for piv_i,piv_j in enumerate(pivot_cols): 
        #Scipy
        pivot_val_n, pivot_val_d = MDn[piv_i,piv_j], MDd[piv_i,piv_j]
        MDn[piv_i,piv_j], MDd[piv_i,piv_j] = 1, 1
        MDn[piv_i,piv_j+1:] = MDn[piv_i,piv_j+1:] * pivot_val_d
        MDd[piv_i,piv_j+1:] = MDd[piv_i,piv_j+1:] * pivot_val_n
        
    MDd[MDd == 0] = 1
    return MDn/MDd

def _gcd(a, b):
    if abs(a).max() == 0:
        return a, a
    # Set b to 0 where a is 0
    inz = a.nonzero()
    b[a == 0] = 0
    assert(set(b.nonzero()[0])==set(a.nonzero()[0]))
    assert(set(b.nonzero()[1])==set(a.nonzero()[1]))
    # Calculate nonzero gcd
    anz, bnz = np.array(a[inz],dtype=np.int64), np.array(b[inz],dtype=np.int64)
    gcd = np.gcd(anz,bnz)
    assert(gcd.shape == a[inz].shape)
    a[inz], b[inz] = a[inz]/gcd, b[inz]/gcd
    return a, b

def _row_reduce_comb(MDn,MDd,mat, rows, cols, normalize_last=True, normalize=True, zero_above=True, one=S.One, iszerofunc=_iszero, simpfunc=_simplify):

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = isimp(a*mat[p] - b*mat[p + q])

    piv_row, piv_col = 0, 0
    pivot_cols = []

    #Check
    mn,md = MDn.toarray(),MDd.toarray()
    md[mn == 0] = 1
    # print('Sympy=Scipy',[mat[i*cols:(i+1)*cols] for i in range(rows)]==[[sp.Rational(mn[i,j],md[i,j]) for j in range(cols)] for i in range(rows)])
    
    # use a fraction free method to zero above and below each pivot
    pbar = tqdm(desc = 'while loop', total =rows)
    while piv_col < cols and piv_row < rows:
        #Scipy
        pivot_offset, pivot_val = _find_reasonable_pivot_nump(MDn[piv_row:,piv_col],MDd[piv_row:,piv_col])
        #Sympy
        s_pivot_offset, s_pivot_val, \
        assumed_nonzero, newly_determined = _find_reasonable_pivot(
                get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        assert(s_pivot_offset == pivot_offset)
        if s_pivot_val is None:
            assert(pivot_val[0] is None)
        else:
            assert(sp.Rational(pivot_val[0],pivot_val[1]) == s_pivot_val)

        if pivot_offset is None:
            piv_col += 1
            continue
        
        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            #Scipy
            I = scipy.sparse.eye(rows).tocoo()
            I.row = I.row[[pivot_offset + piv_row if k==piv_row else piv_row if k==pivot_offset + piv_row else k for k in range(rows)]]
            MDn, MDd = I.dot(MDn), I.dot(MDd)
            #SymPy
            row_swap(piv_row, pivot_offset + piv_row)
            #Check
            # mn,md = MDn.toarray(),MDd.toarray()
            # md[mn == 0] = 1
            # assert(mat[piv_row*cols:(piv_row+1)*cols] == [sp.Rational(mn[piv_row,j],md[piv_row,j]) for j in range(cols)])
            # assert(mat[(pivot_offset + piv_row)*cols:(pivot_offset + piv_row+1)*cols] == [sp.Rational(mn[pivot_offset + piv_row,j],md[pivot_offset + piv_row,j]) for j in range(cols)])
    
        #Scipy
        nz = MDn[:,piv_col].nonzero()[0]
        nz = nz[nz != piv_row]
        #Sympy
        s_nz = []
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row*cols + piv_col]
            if iszerofunc(val):
                continue
            else:
                s_nz.append(row)
        #Check
        # print('NZ: ',nz.tolist() == s_nz)
        assert(nz.tolist() == s_nz)


        for i in nz:
            a, i, b, j = s_pivot_val, i, mat[i*cols + piv_col], piv_row
            q = (j - i)*cols
            # k = 0

            b_n, b_d = MDn[i,piv_col],MDd[i,piv_col]

            # print('a:',a,(pivot_val[0],pivot_val[1]))
            assert(a == sp.Rational(pivot_val[0],pivot_val[1])) #Non-zero denom
            # print('b:',b,(MDn[i,piv_col],MDd[i,piv_col]))
            assert(b == sp.Rational(b_n,b_d)) #Non-zero denom

            for p in range(i*cols, (i + 1)*cols):
                # SymPy
                # print('Before calc SymPy:',a,mat[p],b,mat[p + q])
                before = a*mat[p] - b*mat[p + q]
                mat[p] = isimp(before)
                #Check
                assert(before == mat[p])

                #Scipy
                # mddik,mddprk = MDd[i,k],MDd[piv_row,k]
                # if MDn[i,k] == 0: #Replace MDd[i,k] with 1
                #     mddik = 1
                # if MDn[piv_row,k] == 0: #Replace MDd[piv_row,k] with 1
                #     mddprk = 1
                # # print('Before calc Scipy n:',pivot_val[0],MDn[i,k],b_n,MDn[piv_row,k])
                # # print('Before calc Scipy d:',pivot_val[1],mddik,b_d,mddprk)
                # mn = (pivot_val[0]*b_d)*MDn[i,k]*mddprk - (b_n*pivot_val[1])*mddik*MDn[piv_row,k]
                # md = (pivot_val[1]*b_d)*mddik*mddprk
                # if mn == 0:
                #     # Check
                #     assert(mat[p] == 0)
                #     MDn[i,k], MDd[i,k] = 0, 0
                # else:
                #     assert(float(mn).is_integer(),float(md).is_integer())
                #     gcd = math.gcd(int(mn),int(md))
                #     # Check
                #     assert(mat[p] == sp.Rational(mn/gcd, md/gcd))
                #     MDn[i,k], MDd[i,k] = mn/gcd, md/gcd
                # k += 1

            #Scipy vectorised
            mddik,mddprk = MDd[i],MDd[piv_row]
            mddik[MDn[i] == 0] = 1 #Replace MDd[i,k] with 1
            mddprk[MDn[piv_row] == 0] = 1 #Replace MDd[piv_row,k] with 1
            mn = (pivot_val[0]*b_d)*MDn[i].multiply(mddprk) - (b_n*pivot_val[1])*mddik.multiply(MDn[piv_row])
            md = (pivot_val[1]*b_d)*mddik.multiply(mddprk)
            MDn[i], MDd[i] = _gcd(mn,md)
            MDn.eliminate_zeros()
            MDd.eliminate_zeros()

            #Check
            mdn,mdd = MDn[i].copy(),MDd[i].copy()
            mdd[mdn == 0] = 1
            assert(mat[i*cols:(i + 1)*cols] == [sp.Rational(mdn[0,i],mdd[0,i]) for i in range(cols)])

        piv_row += 1
        pbar.update(1)

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i,piv_j in enumerate(pivot_cols): 
            #Scipy
            pivot_val_n, pivot_val_d = MDn[piv_i,piv_j], MDd[piv_i,piv_j]
            MDn[piv_i,piv_j], MDd[piv_i,piv_j] = 1, 1
            # (MDn[i,j+1:]/MDd[i,j+1:]) / (pivot_val_n/pivot_val_d)
            # = MDn[i,j+1:]*pivot_val_d / MDd[i,j+1:]*pivot_val_n
            MDn[piv_i,piv_j+1:] = MDn[piv_i,piv_j+1:] * pivot_val_d
            MDd[piv_i,piv_j+1:] = MDd[piv_i,piv_j+1:] * pivot_val_n

            #SymPy
            s_pivot_val = mat[piv_i*cols + piv_j]
            #Check
            assert(sp.Rational(pivot_val_n,pivot_val_d) == s_pivot_val)
            
            #SymPy
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                before = mat[p] / s_pivot_val
                mat[p] = isimp(mat[p] / s_pivot_val)
                assert(mat[p] == before)
            #Check
            mdn,mdd = MDn[piv_i].copy(),MDd[piv_i].copy()
            mdd[mdn == 0] = 1
            assert(mat[piv_i*cols:(piv_i + 1)*cols] == [sp.Rational(mdn[0,i],mdd[0,i]) for i in range(cols)])
            
    MDd[MDd == 0] = 1
    MDn/MDd
    return MDn, tuple(pivot_cols), ()


def _find_reasonable_pivot_scip(coln,cold):
    
    # cold[coln == 0] = 1
    # cold_ = scipy.sparse.csr_matrix(np.ones(cold.shape))
    # cold_[cold.nonzero()] = cold[cold.nonzero()]
    # arg_max = (abs(coln)/abs(cold_) != 0).argmax()
    if coln.count_nonzero() == 0:
        return None, (None, None)
    arg_max =  np.sort(coln.nonzero()[0])[0] #First non-zero value
    return arg_max, (coln[arg_max,0],cold[arg_max,0])


"""