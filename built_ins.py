import numpy as np
import sympy as sp
import itertools

#---------------------------------------------------------------------------#


#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=op
#-------------------------------------------------------------#
def op(*args):
    if len(args)==1:
        e = args[0]
        # Equivalent to op(1..nops(e), e)
        if isinstance(e, dict) and any(not(is_type(elem,'integer')) for elem in e.keys()):
            return list(e.values())
        else:
            return op(range(1,nops(e)+1), e)
    else:
        i = args[0]
        e = args[1]
        if isinstance(i, int):
            if i > 0:
                if isinstance(e, list) or isinstance(e, tuple):
                    return e[i-1]
                elif isinstance(e, dict):
                    if i in e.keys():
                        return e[i]
                    else:
                        return None
                else:
                    return e.args[i - 1]
            elif i < 0:
                if isinstance(e, list) or isinstance(e, tuple):
                    return e[len(e)+i]
                elif isinstance(e, dict):
                    if i in e.keys():
                        return e[i]
                    else:
                        return None
                else:
                    return e.args[nops(e) + i]
            elif i == 0:
                if isinstance(e,sp.Mul):
                    return sp.Mul
                elif isinstance(e, sp.Add):
                    return sp.Add
                elif isinstance(e, sp.Function):
                    return e.__class__
                else:
                    return type(e)
        elif isinstance(i, range):
            if i.stop - i.start == 1:
                return op(i.start,e)
            if isinstance(e, list):
                return e[i.start-1:i.stop-1]
            elif isinstance(e, tuple):
                return tuple(e[i.start-1:i.stop-1])
            elif isinstance(e, dict):
                return [e[j] for j in range(i.start-1,i.stop-1)]
            else:
                return e.args[i.start-1:i.stop-1]
        elif isinstance(i, list):
            result = e
            for index in i:
                result = op(index, result)
            return result
        else:
            raise ValueError("Invalid argument for 'i'")

def nops(e):
    if isinstance(e, (list,tuple,set)):
        return(len(e))
    elif isinstance(e, sp.Basic):
        return(len(e.args))
    else:
        return 0


#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=indets
#-------------------------------------------------------------#

def indets(expr, typename=None):
    result = set()
    # Add in 
    def is_indeterminate(subexpr):
        a = isinstance(subexpr, sp.Symbol)
        c = isinstance(subexpr, sp.Pow) and not(isinstance(subexpr.args[1],sp.Integer))
        return ((a or c) and not(subexpr in result))

    def traverse(expression):
        if is_indeterminate(expression):
            result.add(expression)
        elif isinstance(expression, (tuple, list)):
            for subexpr in expression:
                traverse(subexpr)
        elif isinstance(expression, dict):
            for key, value in expression.items():
                traverse(value)
        elif isinstance(expression, (sp.Add, sp.Mul, sp.Pow)):
            for arg in expression.args:
                traverse(arg) 
        elif isinstance(expression, sp.Function) and not(isinstance(expression,(sp.Add, sp.Mul, sp.Pow))):
            flip = False
            for arg in expression.args:
                if not(arg.is_constant()):
                    flip = True
                    break
            if flip:
                result.add(expression)
                for arg in expression.args:
                    traverse(arg) 
        elif isinstance(expression,(sp.Number,float,int)):
            pass
        else:
            result.add(expression)

    traverse(expr)

    return result

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=MultiSet%2Fminus
#https://docs.sympy.org/latest/modules/sets.html#sympy.sets.sets.Complement
#-------------------------------------------------------------#

# 'a minus b' equivalent to 'minus(a,b)
def minus(set1, set2):
    set_list = (sp.FiniteSet, sp.Set, sp.Union, sp.Complement, sp.Intersection, set)
    if isinstance(set1, set) and isinstance(set2, set):
        return set1.difference(set2)
    # elif isinstance(set1, sp.FiniteSet) and isinstance(set1, sp.FiniteSet):
    #     print('Warning, SymPy sets being used.')
    #     return sp.Complement(set1, set2)
    elif all(isinstance(elem, (sp.Expr, sp.Symbol, sp.FiniteSet, sp.Set, sp.Union, sp.Complement, sp.Intersection, set)) for elem in [set1,set2]):
        # print('Warning, set minus between SymPy expressions.')
        return sp.Complement.reduce(*[sp.FiniteSet(elem) if not isinstance(elem, set_list) else elem for elem in [set1,set2]])
    else:
        raise('Set minus undefined on these two inputs.')

def union(*sets):
    set_list = (sp.FiniteSet, sp.Set, sp.Union, sp.Complement, sp.Intersection, set)
    if len(sets) < 2:
        raise('Set union taken of one input.')
    if all(isinstance(elem, set) for elem in sets):
        return sets[0].union(*sets)
    # elif all(isinstance(elem, sp.FiniteSet) for elem in sets):
    #     # print('Warning, SymPy sets being used.')
    #     return sp.Union(*sets)
    elif all(isinstance(elem, (sp.Expr, sp.Symbol, sp.FiniteSet, sp.Set, sp.Union, sp.Complement, sp.Intersection, set)) for elem in sets):
        # print('Warning, set union turns SymPy expressions into set of expression.')
        return sp.Union(*[sp.FiniteSet(elem) if not isinstance(elem, set_list) else elem for elem in sets])
    else:
        raise('Set union undefined on inputs.')

def intersect(*sets):
    set_list = (sp.FiniteSet, sp.Set, sp.Union, sp.Complement, sp.Intersection, set)
    if len(sets) < 2:
        raise('Set intersect taken of one input.')
    if all(isinstance(elem, set) for elem in sets):
        return sets[0].intersection(*sets)
    # elif all(isinstance(elem, sp.FiniteSet) for elem in sets):
    #     # print('Warning, SymPy sets being used.')
    #     return sp.Intersection(*sets)
    elif all(isinstance(elem, (sp.Expr, sp.Symbol, sp.FiniteSet, sp.Set, sp.Union, sp.Complement, sp.Intersection, set)) for elem in sets):
        # print('Warning, set intersect turns SymPy expressions into set of expression.')
        return sp.Intersection(*[sp.FiniteSet(elem) if not isinstance(elem, set_list) else elem for elem in sets])
    else:
        raise('Set intersect undefined on inputs.')

# ORDER OF PRECEDENCE: intersect, union and minus

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=icontent
#https://docs.sympy.org/latest/modules/polys/basics.html#divisibility-of-polynomials
#-------------------------------------------------------------#

def icontent(f, X=None, str=None):
    f = sp.sympify(f)
    if isinstance(f,(sp.Expr,sp.Poly)):
        if is_type(f, 'polynom',X):
            if isinstance(X,list):
                c, p = sp.polys.polytools.primitive(f,*X)
            else:    
                c, p = sp.polys.polytools.primitive(f)
        else:
            c, p = f.as_content_primitive()
        if str == None:
            return c
        else:
            return c, p
    if isinstance(f, (int, sp.Integer)):
        return f #f.as_content_primitive()
    else:
        raise('icontent: Argument must be SymPy expression or polynomial or integer.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=igcd
#https://docs.sympy.org/latest/modules/core.html#sympy.core.numbers.igcd
#-------------------------------------------------------------#
def igcd(*args):
    if len(args) == 0:
        return 0
    elif len(args)>=2:
        if all(isinstance(elem, (int,sp.Integer)) for elem in args):
            return sp.igcd(*args)
        # for elem in args:
        #     if not(isinstance(elem, (sp.Number, int, sp.Expr))):
        #         raise('Trying to call igcd on list of non-integers.')
        #     if isinstance(elem, (sp.Function,sp.Poly,sp.Symbol)):
        #         if isinstance(elem, (sp.Poly, sp.Symbol)):
        #             print('Trying to call igcd on polynomial, using gcd_list instead.')
        #             return sp.gcd_list(args)
        #         else:
        #             raise('Trying to call igcd on non-polynomial function.')
        else:
            raise('igcd: Arguments must be integers.')
    else:
        raise('Trying to call igcd on less than two numbers.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=coeff
#https://docs.sympy.org/latest/modules/core.html#sympy.core.numbers.igcd
#-------------------------------------------------------------#

def coeff(p, x, n=None):
    if n is not None:
        if not isinstance(n, (sp.Integer, int)):
            raise('Can only return coeff of x^n for integer n.')
        return sp.expand(p).coeff(x, n)
    else:
        return sp.expand(p).coeff(x)

#-------------------------------------------------------------#
#Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=lhs
#-------------------------------------------------------------#

def lhs(expr):
    if isinstance(expr, range):
        return expr.start
    elif isinstance(expr, (list, set, tuple)):
        if len(expr) == 2:
            return op(1,expr)
        else:
            raise('Too many values to call lhs.')
    elif isinstance(expr, (sp.Rel, sp.Eq, sp.Ge, sp.Le, sp.Ne, sp.Gt, sp.Lt)):
        return expr.lhs
    else:
        raise('lhs undefined on the input.')

def rhs(expr):
    if isinstance(expr, range):
        return expr.stop
    elif isinstance(expr, (list, set, tuple)):
        if len(expr) == 2:
            return op(2,expr)
        else:
            raise('Too many values to call rhs.')
    elif isinstance(expr, (sp.Rel, sp.Eq, sp.Ge, sp.Le, sp.Ne, sp.Gt, sp.Lt)):
        return expr.rhs
    else:
        raise('rhs undefined on the input.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=series
#https://docs.sympy.org/latest/modules/core.html#sympy.core.expr.Expr.series
#-------------------------------------------------------------#

def series(expr, eqn, n=None):

    if isinstance(eqn, tuple):
        x = eqn[0]
        x0 = eqn[1]
    elif isinstance(eqn, sp.Symbol):
        x = eqn
        x0 = 0

    if n == None:
        n = 6
    # print('Warning, max degree of series will be n, but may include more terms (from negative degree).')
    return sp.series(expr, x, x0, n) 

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=irem
#-------------------------------------------------------------#

def irem(a, b, str=None):
    if isinstance(a, (int, sp.Integer)) and isinstance(b, (int, sp.Integer)):
        # res = a % b
        # ret = res if not res else res-b if a<0 else res     
        ret = a - int(a/b) * b
        if str == None:
            return ret
        else:
            return int(a/b), ret
    else:
        raise('Calling irem for non-intgers.')

def iquo(a, b, str=None):
    if isinstance(a, (int, sp.Integer)) and isinstance(b, (int, sp.Integer)):
        # res = a % b
        # ret = res if not res else res-b if a<0 else res    
        ret = a - int(a/b) * b 
        if str == None:
            return int(a/b)
        else:
            return int(a/b), ret
    else:
        raise('Calling iquo for non-intgers.')


#-------------------------------------------------------------#
#-------------------------------------------------------------#
# NEED TO CONVERT args `something` INTO STRINGS 'something' OR WILL THROW ERROR

def userinfo(*args):
    if len(args) < 3:
        raise('Userinfo requires at least three arguments')
    else:
        print(*args[2:])

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=indices
#-------------------------------------------------------------#

def indices(t):
    if isinstance(t, sp.Matrix):
        return np.ndarray.tolist(np.stack(np.indices(t.shape), axis=-1))
    elif isinstance(t, (list, set)):
        if all(isinstance(elem, (sp.Eq, tuple)) for elem in t):
            return [[lhs(elem)] for elem in t]
        else:
            return [[i] for i in range(len(t))]
    elif isinstance(t, dict):
        return [key for key in t.keys()]
    elif isinstance(t, np.ndarray):
        return np.ndarray.tolist(np.stack(np.indices(t.shape), axis=-1))
    else:
        raise('Indices can only be called on list, set, sp.Matrix, np.ndarray or dict.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=map
#-------------------------------------------------------------#
function_list_map = [sp.Function, sp.Add, sp.Mul, sp.Pow]
def new_map(fcn, expr, args_alt=None):
    smb = list(fcn.free_symbols)
    if expr == []:
        return []
    elif isinstance(expr, sp.Expr):
        old_args = list(expr.args)
    elif isinstance(expr, set):
        old_args = list(expr)
    else:
        raise('Map only defined when expr is SymPy expression, a set or empty list.')

    if args_alt == None and len(smb) == 1:
        if isinstance(fcn, function_list_map):
            new_args = [fcn.subs([(smb[0],elem)]) for elem in old_args]
        else:
            new_args = [fcn(elem) for elem in old_args]
    elif len(args_alt) >= len(smb):
        if isinstance(fcn, function_list_map):
            new_args = [fcn.subs([(smb[0],elem)]+[(smb[i],args_alt[i]) for i in range(len(smb))]) for elem in old_args]
        else:
            new_args = [fcn(elem,*args_alt) for elem in old_args]
    else: 
        raise('Too few arguments provided for fcn in map.')   
    return expr.func(*new_args)

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=subs
#-------------------------------------------------------------#

def subs(args, expr):
    if isinstance(expr, sp.Expr):
        if isinstance(args, (tuple, sp.Eq)):
            if lhs(args) in expr.free_symbols or expr.args:
                return expr.subs(*args) 
            else:
                raise('Single substitution argument should be (SymPy symbol/expression, val) where the symbol/expression is in expr.')
        elif isinstance(args, (list, set)):
            if isinstance(args, set):
                args = list(args)
            return expr.subs(args,simultaneous=True)
        elif isinstance(args, dict):
            ll = []
            for elem in args:
                k,v = elem
                ll.append((k,v))
            return expr.subs(ll,simultaneous=True)
        else:
            raise('Subs first argument can only be tuple, SymPy equation, set, list or dict.')
    elif isinstance(expr, (int, float)):
        return expr
    else:
        raise('Subs only valid for SymPy expressions.')

# print(subs((a+b,y),(a+b+c)**2)) #CHECK THIS OUT

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=table
#-------------------------------------------------------------#

class SparseDict(dict):
    def __getitem__(self, key):
        if isinstance(key, int) and not(key in self.keys()):
            return 0
        else:
            return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

class SparseDictNZ(dict):
    def __getitem__(self, key):
        if isinstance(key, int) and not(key in self.keys()):
            return 0
        else:
            return dict.__getitem__(self, key)
    def __setitem__(self, key, value):
        if not(value == 0):
            dict.__setitem__(self, key, value)

class SymDict(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, frozenset(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, frozenset(key), value)

# def list_to_dict(ls, string=None):
#     if ls == []:
#         if string == 'sparse':
#             return SparseDict()
#         if string == 'symmetric':
#             return {}
#         else:
#             return {}
#     else:
#         if string == 'sparse':
#             d = SparseDict()
#         if string == 'symmetric':
#             d = {}
#         else:
#             d = {}
#         i = 1
#         for elem in ls:
#             if isinstance(elem, (tuple, sp.Eq)): 
#                 d[lhs(elem)] = rhs(elem)
#             else:
#                 d[i] = elem
#             i += 1
#         return d

# def table(*args):
#     if len(args) == 0:
#         return {}
#     elif len(args) == 1:
#         if isinstance(args[0], (list, set)):
#             return list_to_dict(args[0])
#         else:
#             raise('Can only make table/dict from list or set.')
#     elif len(args) == 2:
#         if args[0] == 'sparse':
#             print('Warning, sparse table/dict initialised.')
#             return list_to_dict(args[0], 'sparse')
#         elif args[0] == 'symmetric':
#             print('Warning, symmetric table/dict initialised.')
#             return list_to_dict(args[0], 'symmetric')
#         else:
#             raise('Only sparse and symmetric are valid string arguments for table')
    
#     else:
#         raise('Table can only be called on at most two arguments.')

# # check out op(t) op(op(t))

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=type
#-------------------------------------------------------------#

def is_from_K(p, K):      
    try:
        if K.from_sympy(p):
            return True
    except ValueError:
        return False

def is_rat(f):
    try:
        sp.Rational(f)
        return True
    except TypeError:
        return False

def is_type(e, t, vs=None):
    if not(isinstance(t, (list, set, tuple))):

        if t == 'listname' or t == 'setname':
            return isinstance(e, (list, set)) and all(isinstance(elem, str) for elem in e)
        elif t == 'name':
            return isinstance(e, str) or isinstance(e, sp.Symbol)

        elif t =='listinteger':
            return all(isinstance(elem, (sp.Integer, int)) for elem in e) and isinstance(e, list)
        elif t =='integer':
            if isinstance(e, (sp.Float, float)):
                return float(e).is_integer()
            else:
                return isinstance(e, (sp.Integer, int))

        elif t == 'listtable':
            return isinstance(e, list) and all(isinstance(elem, dict) for elem in e)
        elif t == 'table':
            return isinstance(e, dict)

        elif t == 'matrix':
            return isinstance(e, sp.Matrix)

        # POLYNOMIALS   
        elif t == 'listpolynom' or t == 'setpolynom':
            if not(isinstance(e, (set,list))):
                return False
            if vs == None:
                if all(elem.as_poly(list(elem.free_symbols)) == None for elem in e):
                    return False
                else:
                    return True
            else:
                if all(elem.as_poly(vs) == None for elem in e):
                    return False
                else:
                    return True
        elif t == 'polynom':
            e = sp.sympify(e)
            if vs == None:
                if e.as_poly(list(e.free_symbols)) == None:
                    return False
                else:
                    return True
            else:
                if e.as_poly(vs) == None:
                    return False
                else:
                    return True
        elif t == 'RootOf':
            return isinstance(e, (sp.RootOf,sp.CRootOf,MRootOf))
        elif t == 'rationalpolynom':
            if isinstance(e, (sp.RootOf,sp.CRootOf,MRootOf)):
                return False
            if vs == None:
                if e.as_poly(list(e.free_symbols)) != None:
                    K = sp.PolynomialRing(sp.QQ, list(e.free_symbols))
                    return is_from_K(e, K)
                else:
                    return False
            else:
                if e.as_poly(list(e.free_symbols)) != None:
                    # print('e',e,'vs',vs)
                    K = sp.PolynomialRing(sp.QQ, vs)
                    return is_from_K(e, K)
                elif is_type(e,'rational'):
                    return True
                else:
                    return False
        elif t == 'monomialinteger':
            if vs == None:
                p = e.as_poly(list(e.free_symbols))
                if p == None:
                    return False
                else:
                    return p.is_monomial and p.domain == sp.ZZ
            else:
                p = e.as_poly(vs)
                if p == None:
                    return False
                else:
                    return p.is_monomial and p.domain == sp.ZZ
        elif t == 'ratfunc':
            if vs == None:
                K = sp.RR[*list(e.free_symbols)].get_field()
                C = sp.CC[*list(e.free_symbols)].get_field()
                return is_from_K(e, K) or is_from_K(e, C)
            else:
                K = sp.RR[*vs].get_field()
                C = sp.CC[*vs].get_field()
                return is_from_K(e, K) or is_from_K(e, C)

        # NUMBERS
        elif t == 'even':
            return e % 2 == 0 and isinstance(e, (sp.Integer, int))
        elif t == 'odd':
            return e % 2 == 1 and isinstance(e, (sp.Integer, int))
        elif t == 'rational':
            if isinstance(e, (sp.Rational, sp.Integer, int)):
                return True
            elif isinstance(e, (sp.Float, float)) and is_rat(e):
                return True
            else:
                return False
        elif t == 'float':
            if isinstance(e, (sp.Float,float)):
                return True
            else:
                return False
        elif t == 'constant':
            if isinstance(e, bool):
                return True
            return e.is_constant() #e.args in (np.pi, 1j, sp.oo, sp.I, sp.pi, sp.gamma,)
        elif t == 'listpositive':
            return isinstance(e, list) and all(elem > 0 for elem in e)
        elif t == 'positive':
            return e > 0
        elif t == 'negative':
            return e < 0
        elif t == 'listnonneg':
            return isinstance(e, list) and all(elem >= 0 for elem in e)
        elif t == 'nonneg':
            return e >= 0        
        elif t == '+':
            return isinstance(e, sp.Add)
        elif t == '*':
            return isinstance(e, sp.Mul)
        
        # EQUATIONS AND VECTORS
        elif t == 'listequation':
            if isinstance(e, (list,set)) and any(isinstance(elem, tuple) for elem in e):
                raise('Please convert list of tuples to list of proper SymPy equation.')
            return all(isinstance(elem, sp.Eq) for elem in e) and isinstance(e, (list,set))
        elif t == 'equation':
            if isinstance(e, tuple):
                raise('Please convert tuple to proper SymPy equation.')
            return isinstance(e, sp.Eq)

        elif t == 'list':
            return isinstance(e, list)        
        elif t == 'set':
            return isinstance(e, set)

        elif t == 'listvector' or t == 'setvector':
            if isinstance(e, (list,set)) and any(isinstance(elem, list) for elem in e):
                raise('Please convert list of lists to list of proper SymPy nx1 Matrix i.e. vector.')
            return isinstance(elem, sp.Matrix) and (elem.shape[1]==1 for elem in e) and isinstance(e, (list,set))
        elif t == 'vector':
            if isinstance(e, list):
                raise('Please convert list to proper SymPy nx1 Matrix i.e. vector.')
            return isinstance(e, sp.Matrix) and e.shape[1]==1
        else:
            raise('Invalid string of type used for is_type.')
    else:
        return any(is_type(e, tp) for tp in t)

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=gcd
# https://docs.sympy.org/latest/modules/polys/basics.html
#-------------------------------------------------------------#
    
def gcd(a,b,*args):
    a, b = sp.sympify(a), sp.sympify(b)
    if is_type(a, 'integer') and is_type(b, 'integer'):
        if len(args) == 0:
            return igcd(a,b)
        else:
            return sp.polys.polytools.cofactors(a,b)[1], sp.polys.polytools.cofactors(a,b)[2]
    elif isinstance(a, sp.Expr) and isinstance(b, sp.Expr):
        if len(args) == 0:
            return sp.polys.polytools.gcd(a,b,extension=True)
        else:
            return sp.polys.polytools.cofactors(a,b)[1], sp.polys.polytools.cofactors(a,b)[2]
    else:
        raise('gcd: Arguments must either be two SymPy expressions/polynomials or two integers.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=member
#-------------------------------------------------------------#

def member(x, s, p=None):
    if isinstance(s, (list, set)):
        if x in s and p == None:
            return True
        elif x in s and p != None:
            return True, s.index(x)
        else:
            return False
    elif isinstance(s, dict):
        if x in s.values() and p == None:
            return True
        elif x in s.values() and p != None:
            return True, s.index(x)
        else:
            return False
    elif isinstance(s, sp.Function):
        if x in s.args and p == None:
            return True
        elif x in s.args and p != None:
            return True, s.args.index(x)
    else:
        raise('Member not defined for this structure.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=order
#-------------------------------------------------------------#

def is_rat(f):
    try:
        sp.Rational(f)
        return True
    except TypeError:
        return False

def numer(x):
    if isinstance(x, sp.Rational):
        return x.p
    elif isinstance(x, (sp.Integer, int)):
        return x
    elif isinstance(x, (sp.Float, float)):
        if is_rat(x):
            return sp.nsimplify(sp.Rational(x)).p
        else:
            return 1.0
    elif isinstance(x, complex):
        if is_rat(x.real) and is_rat(x.imag):
            r, i = sp.nsimplify(sp.Rational(x.real)), sp.nsimplify(sp.Rational(x.imag))
            return r.p*i.q + r.q*i.p*1j
        else:
            raise('Complex number is not rational, cannot call numer.')
    elif isinstance(x, sp.Expr):
        n, _ = sp.fraction(sp.factor(x))
        return n
    else:
        raise('Can only call numer on SymPy expression.')

def denom(x):
    if isinstance(x, sp.Rational):
        return x.q
    elif isinstance(x, (sp.Integer, int)):
        return 1
    elif isinstance(x, (sp.Float, float)):
        if is_rat(x):
            return sp.nsimplify(sp.Rational(x)).q
        else:
            return 1.0
    elif isinstance(x, complex):
        if is_rat(x.real) and is_rat(x.imag):
            r, i = sp.nsimplify(sp.Rational(x.real)), sp.nsimplify(sp.Rational(x.imag))
            return r.q*i.q
        else:
            raise('Complex number is not rational, cannot call denom.')
    if isinstance(x, sp.Expr):
        _, d = sp.fraction(sp.factor(x))
        return d
    else:
        raise('Can only call numer on SymPy expression.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=degree
#-------------------------------------------------------------#

def degree(a, x=None):
    if isinstance(a,int) or isinstance(a, sp.core.numbers.One):
        return 0
    if not isinstance(a, sp.polys.polytools.Poly):
        a = sp.sympify(a)
    # print('degree: Started.')
    if x == None:
        if is_type(a,'polynom'):
            # print('degree: Finished.')
            return (a.as_poly(list(a.free_symbols))).total_degree()
        elif is_type(a, 'ratfunc'):
            raise('Degree not yet implemented for rational functions, only polynomials.') 
        else:
            raise('Expression is not polynomial so cannot calculate degree.')
    elif isinstance(x, (list, set, sp.Symbol)):
        if isinstance(x, set):
            x = list(x)
        elif isinstance(x, sp.Symbol):
            x = [x]
        if is_type(a,'polynom',x) or (is_type(a, 'polynom') and member(x,a.free_symbols())):
            # print('degree: Finished.')
            return (a.as_poly(x)).total_degree()
        elif is_type(a, 'ratfunc'):
            raise('Degree not yet implemented for rational functions, only polynomials.')
        else:
            raise('Expression is not polynomial in x so cannot calculate degree.')
    else:
        raise('Second argument of degree must be a SymPy symbol, set or list.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=normal
#-------------------------------------------------------------#

def normal(f):
    if isinstance(f, sp.Add) and any(isinstance(elem, sp.O) for elem in f.args):
        return sp.Add(*[normal(elem) for elem in f.args])
    if isinstance(f, sp.Expr):
        return sp.factor(f)
    elif isinstance(f, sp.Eq):
        return sp.Eq(normal(f.lhs),normal(f.rhs))
    elif isinstance(f, sp.Lt):
        return sp.Lt(normal(f.lhs),normal(f.rhs))
    elif isinstance(f, sp.Le):
        return sp.Le(normal(f.lhs),normal(f.rhs))
    elif isinstance(f, sp.Gt):
        return sp.Gt(normal(f.lhs),normal(f.rhs))
    elif isinstance(f, sp.Ge):
        return sp.Ge(normal(f.lhs),normal(f.rhs))
    elif isinstance(f, sp.Unequality):
        return sp.Unequality(normal(f.lhs),normal(f.rhs))
    elif isinstance(f, set):
        return {normal(elem) for elem in f}
    elif isinstance(f, list):
        return [normal(elem) for elem in f]
    # elif isinstance(f, range):
    else:
        raise('Normal can only be called on SymPy expressions.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=coeffs
#         https://www.maplesoft.com/support/help/maple/view.aspx?path=lcoeff
#         https://www.maplesoft.com/support/help/maple/view.aspx?path=tcoeff
#-------------------------------------------------------------#

def coeffs(p, x=None, t=None):
    if isinstance(p, sp.Expr):
        if x == None:
            smb = list(p.free_symbols)
        elif isinstance(x, (list, tuple)):
            smb = x
        elif isinstance(x, set):
            smb = list(x)
        elif isinstance(x, sp.Expr):
            smb = [x]
        else:
            raise('Can only call coeffs with SymPy expression, list or set as seccond argument.')
    
        p = p.as_poly(smb)
        args = [sp.Monomial(m) for m in p.monoms()]
        l = [p.coeff_monomial(m) for m in args]
        # print(args,l)
        if t == None:
            return l
        else:
            return l, [args[i].as_expr(*smb) for i in range(len(args)) if l[i] != 0] 
    else:
        raise('Can only call coeffs on SymPy expressions.')

def lcoeff(p, x=None, t=None):
    if isinstance(p, sp.Expr):
        if x == None:
            smb = list(p.free_symbols)
        elif isinstance(x, list):
            smb = x
        elif isinstance(x, set):
            smb = list(x)
        elif isinstance(x, sp.Expr):
            smb = [x]
        else:
            raise('Can only call lcoeff with SymPy expression, list or set as seccond argument.')
        if t == None:
            return (p.as_poly(smb)).LC(order=sp.polys.orderings.GradedLexOrder())
        else:
            return (p.as_poly(smb)).LC(order=sp.polys.orderings.GradedLexOrder()), (p.as_poly(smb)).LM(order=sp.polys.orderings.GradedLexOrder()).as_expr(*smb)
    else:
        raise('Can only call lcoeff on SymPy expressions.')

def tcoeff(p, x=None, t=None):
    if isinstance(p, sp.Expr):
        if x == None:
            smb = list(p.free_symbols)
        elif isinstance(x, list):
            smb = x
        elif isinstance(x, set):
            smb = list(x)
        elif isinstance(x, sp.Expr):
            smb = [x]
        else:
            raise('Can only call tcoeff with SymPy expression, list or set as seccond argument.')
        p = p.as_poly(smb)
        tm = [sp.Monomial(m) for m in p.monoms(order=sp.polys.orderings.GradedLexOrder())][-1]
        tc = p.coeff_monomial(tm)
        if t == None:
            return tc
        else:
            return tc, tm.as_expr(*smb)
    elif isinstance(p, sp.polys.polytools.Poly):
        if t == None:
            return p.EC()
        else:
            return p.EC(), p.EM().as_expr()
    else:
        raise('Can only call tcoeff on SymPy expressions.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=sign
#-------------------------------------------------------------#

def sign(expr):
    if isinstance(expr, (int, float, sp.Float, sp.Number)):
        if expr == 0:
            return 1
        else:
            return expr/abs(expr)
    elif isinstance(expr, sp.Expr):
        return sign(lcoeff(expr))
    else:
        raise('Can only call sign on SymPy expressions or numbers.') 

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=convert
#-------------------------------------------------------------#

class MRootOf(sp.Expr):

    @classmethod
    def __new__(cls, f, x, index):

        poly = sp.Poly(f,x)
        root = sp.rootof(poly,x,index)

        obj = sp.Expr.__new__(cls)
        obj.poly = sp.PurePoly(poly)
        obj.index = index
        obj.expr = root

        return obj

def convert(expr, form):

    if form == '+':
        if isinstance(expr, (sp.Expr, list, set, sp.Function)):
            if isinstance(expr, (list,set)):
                return sp.Add(*expr)
            else:
                return sp.Add(*op(expr))
        else:
            raise('Expression not a valid argument of convert.')

    elif form == '*':
        if isinstance(expr, (sp.Expr, list, set, sp.Function)):
            if isinstance(expr, (list,set)):
                return sp.Mul(*expr)
            else:
                return sp.Mul(*op(expr))
        else:
            raise('Expression not a valid argument of convert.')

    elif form == 'set':
        if isinstance(expr, (sp.Add, sp.Mul)):
            return set(expr.args)
        elif isinstance(expr, set):
            return list(expr)
        elif isinstance(expr, sp.Pow):
            return set([expr.args[0] for i in range(expr.args[1])])
        else:
            raise('Convert to set only valid on SymPy Add and Mul expressions.') #check if list or set

    elif form == 'list':
        if isinstance(expr, sp.Matrix):
            # if expr.shape[1] == 1:
            #     print('Warning, argument to convert could be a vector but choosing to treat as matrix.')
            return np.ndarray.tolist(sp.matrix2numpy(expr))

    elif form == 'radical':
        if isinstance(expr, (sp.RootOf, sp.CRootOf)):
            f, i = expr.args
            smb = list(f.free_symbols)
            return sp.rootof(f, *smb, i, radicals=True)
        elif isinstance(expr, MRootOf):
            print('convert: Warning, converting multivariate root to radical returns original equation.')
            return expr.expr
        else:
            raise('Convert to radical only valid on RootOfs.')

    elif form == 'RootOf':
        if isinstance(expr, sp.Poly):
            expr = sp.expand(expr.as_expr)
        if isinstance(expr, sp.Pow):
            if is_type(expr.args[1], 'integer'):
                return expr
            elif is_type(expr.args[1], 'rational') and not(is_type(expr.args[1], 'integer')):
                frac = sp.nsimplify(sp.Rational(expr.args[1]))
                p, q = frac.p, frac.q
                _Z = sp.Symbol('_Z')
                if len((expr.args[0]).free_symbols) == 0:
                    poly = (_Z**q-expr.args[0]).as_poly([_Z])
                    return sp.CRootOf(poly,0)
                else:
                    poly = sp.Poly(_Z**q-expr.args[0],_Z)
                    return MRootOf.__new__(poly,_Z,0) #sp.rootof(poly,_Z,0) would return radical
            else:
                raise('Convert to RootOf only valid for powers that are rational.')
        elif isinstance(expr, sp.Expr):
            return expr
        else:
            raise('Convert to RootOf only valid on radicals and SymPy expressions.')
    # elif form == 'polynom':
    # elif form == 'vector':
    else:
        raise('Second argument of convert is not a valid string.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/Maple/view.aspx?path=assigned
#-------------------------------------------------------------#

def assigned(n,x,str=None):  
    if isinstance(n,(set,list,dict,sp.Matrix,np.ndarray)):
        if isinstance(n,dict):
            if x in n.keys():
                bl = True
            else:
                bl = False
        else:
            if x < len(n):
                bl = True
            else:
                bl = False
        if str == None:
            return bl
        else:
            if bl:
                return bl, n[x]
            else:
                return bl, None
    else:
        raise('Assigned can only be called to check dict, list, set, Matrix or np array.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=evaln
#-------------------------------------------------------------#
import inspect

def evaln(a, x):
    callers_local_vars = inspect.currentframe().f_locals.items() #.f_back
    ls = [var_name for var_name, var_val in callers_local_vars if var_val is a]
    if len(ls) == 1:
        print(ls[0])
        if isinstance(x, list):
            s = '_'.join([str(elem) for elem in x])
        else:
            s = str(x)
        return sp.Symbol(ls[0]+'_'+ s)
    else:
        print(ls)
        raise('evaln: More than one variable name for a.')
    # if isinstance(old_str, str):
    # else:
    #     raise('evaln can only be called on string.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=expand
#-------------------------------------------------------------#

def expand(expr,*args):
    expr = sp.sympify(expr)
    if isinstance(expr, (sp.Expr,sp.Poly)):
        if len(args) == 0:
            return sp.expand(expr)
        else:
            raise('Expand not yet implemented to factor out additional arguments.')
    else:
        raise('Expand can only be called on SymPy expressions')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=prem
#-------------------------------------------------------------#

def prem(a,b,x,str1=None,str2=None):
    if isinstance(a, (sp.Poly,sp.Expr)) and isinstance(b, (sp.Poly,sp.Expr)):
        if isinstance(x, (sp.Symbol, sp.Expr, sp.Poly)):
            if str1 == None:
                if str2 == None:
                    return sp.prem(a,b)
                else:
                    return sp.prem(a,b), sp.pquo(a,b)
            else:
                m = lcoeff(b,x)**(degree(a,x)-degree(b,x)+1)
                if str2 == None:
                    return sp.prem(a,b), m
                else:
                    return sp.prem(a,b), m, sp.pquo(a,b)
        else:
            raise('prem: Third argument must be SymPy single polynomial term')
    else:
        raise('prem: First and second arguments to prem must be SymPy polynomials or expressions.')


#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=subsop
#-------------------------------------------------------------#
def subsop(expr, *args): #with first arg NULL
    if isinstance(expr, (sp.Expr, sp.Poly)):
        if all(isinstance(elem, tuple) for elem in args):
            for elem in args:
                spec, elem = lhs(elem), rhs(elem)
                expr.subs([(op(spec,expr),elem)])
            return expr
        else:
            raise('subsop: All arguments other than the first must be tuples.')
    elif isinstance(expr, list):
        if all(isinstance(elem, tuple) for elem in args):
            for elem in args:
                spec, elem = lhs(elem), rhs(elem)
                if elem == None:
                    del expr[spec]
                else:
                    expr[spec] = elem
            return expr
        else:
            raise('subsop: All arguments other than the first must be tuples.')
    else:
        raise('subsop: First argument must be SymPy expression or polynomial.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=Divide
#-------------------------------------------------------------#

def does_it_divide(a,b):
    try:
        sp.pexquo(a,b,list(union(a.free_symbols,b.free_symbols)))
        return True
    except Exception:
        return False
    

def divide(a, b, q=None, X=None):
    if is_type(a, ['float','rational']) and is_type(b, ['float','rational']):
        if is_type(a/b,'integer'):
            if q == None:
                return True
            else:
                return True, int(a/b)
        else:
            if q == None:
                return False
            else:
                return False, 1
    elif (isinstance(a, (sp.Expr, sp.Poly)) and is_type(b, ['float','rational'])) or (isinstance(b, (sp.Expr, sp.Poly)) and is_type(a, ['float','rational'])):
        if sp.div(a,b)[1]==0:
            if q == None:
                return True
            else:
                return True, sp.div(a,b)[0]
        else:
            if q == None:
                return False
            else:
                return False, 1
    elif isinstance(a, (sp.Expr, sp.Poly)) and isinstance(b, (sp.Expr, sp.Poly)):
        if not(isinstance(a, sp.Poly)):
            a = a.as_poly()
        if not(isinstance(b, sp.Poly)):
            b = b.as_poly()
        a, b = a.set_domain(sp.ZZ), b.set_domain(sp.ZZ)
        if sp.div(a,b)[1]==0:
            if q == None:
                return True
            else:
                return True, sp.div(a,b)[0].as_expr()
        else:
            if q == None:
                return False
            else:
                return False, sp.sympify(1)
    else:
        raise('divide: First two arguments must be SymPy expressions / polynomials or integers.')


def lcm(m1,m2):
    m1 = sp.sympify(m1)
    m2 = sp.sympify(m2)
    if isinstance(m1, sp.Expr) and isinstance(m2,sp.Expr):
        return sp.lcm(m1,m2)
    else:
        raise('lcm: Both arguments must be SymPy polynomials.')
    
    
#-------------------------------------------------------------#
#-------------------------------------------------------------#
#--------------------------LINALG-----------------------------#
#-------------------------------------------------------------#
#-------------------------------------------------------------#

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2fmatrix
#https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html
#-------------------------------------------------------------#
import types
function_list = (sp.Function, sp.Add, sp.Mul, sp.Pow)

def matrix(*args):
    if len(args) == 1:
        if isinstance(args, list):
            # if all(isinstance(elem, sp.vector.vector.Vector) for elem in args):
            #     raise('Matrix from vectors not implemented yet.')
            if all(isinstance(elem, list) for elem in args):
                return sp.Matrix(args)
            else:
                raise('Trying to construct matrix from unsupported type.')
        else:
            raise('Matrix can only be constructed from list with one argument.')
    elif len(args) == 2:
        if all(isinstance(elem, (int, sp.Integer)) and elem>0 for elem in args):
            return sp.zeros(args[0], args[1])
        else:
            raise('Need two pos. integers for constructing mxn Matrix.')
    elif len(args) == 3:
        if not(isinstance(args[0], (int, sp.Integer)) and isinstance(args[1], (int, sp.Integer))):
            raise('Need two pos. integers for constructing mxn Matrix.')
        if args[2] == 0:
            return sp.zeros(args[0], args[1])
        elif isinstance(args[2], list):
            # if all(isinstance(elem, sp.vector.vector.Vector) for elem in args):
            #     raise('Matrix from vectors not implemented yet.')
            if all(isinstance(elem, list) for elem in args[2]):
                return sp.Matrix([l[:args[1]] for l in args[2][:args[0]]])
            else:
                m, n = args[0], args[1]
                if len(args[2]) < args[0]*args[1]:
                    l = args[2] + [0 for i in range(m*n-len(args[2]))]
                    return sp.Matrix([l[i*n:(i+1)*n] for i in range(m)])
                else:
                    return sp.Matrix([args[2][i*n:(i+1)*n] for i in range(m)])
        #   elif isinstance(args, sp.vector.vector.Vector):
        #       raise('Matrix from vectors not implemented yet.')
        elif isinstance(args[2], function_list):
            m, n = args[0], args[1]
            smb = list(args[2].free_symbols)
            print('Warning, careful of argument order when calling function on Matrix.')
            return sp.Matrix([[args[2].subs([(smb[0],i),(smb[1],j)]) for j in range(1,n+1)] for i in range(1,m+1)])
        elif isinstance(args[2], types.FunctionType):
            m, n = args[0], args[1]
            return sp.Matrix([[args[2](i,j) for j in range(1,n+1)] for i in range(1,m+1)])
        else:
            raise('Third argument for matrix must be list or 0 or function.')
    else:
        raise('Too many arguments to construct Matrix.')

def rowdim(input):
    if isinstance(input, sp.Matrix):
        return input.shape[0]
    else:
        raise('Not valid type of matrix.')

def coldim(input):
    if isinstance(input, sp.Matrix):
        return input.shape[1]
    else:
        raise('Not valid type of matrix.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=LinearAlgebra%2FRank
#https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixReductions.elementary_row_op
#-------------------------------------------------------------#

def rank(input):
    if not(isinstance(input, sp.Matrix)):
        raise('Can only call rank on SymPy Matrix.')
    else:
        return input.rank()

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fdiag
# https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html#matrix-constructors
#-------------------------------------------------------------#

def diag(*args):
    if all(isinstance(elem, sp.Matrix) for elem in args):
        # print('here')
        return sp.Matrix(sp.BlockDiagMatrix(*args))
    else:
        return sp.diag(*args)

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=linalg(deprecated)/det
#https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixDeterminant.det
#-------------------------------------------------------------#

def det(M, sparse=None):
    if not(isinstance(M, sp.Matrix)):
        raise('Can only call det on SymPy Matrix.')
    else:
        return M.det()

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fdotprod
#https://docs.sympy.org/latest/modules/matrices/matrices.html#basic-manipulation
#-------------------------------------------------------------#

def dotprod(u, v, str=None):
    if not(isinstance(u, (list, sp.Matrix)) and isinstance(v, (list, sp.Matrix))):
        raise('Can only call dotprod on two lists.')
    if str == 'orthogonal':
        if isinstance(u, (sp.Matrix)) and isinstance(v, (sp.Matrix)):
            return u.dot(v)
        if isinstance(u, list) and isinstance(v, list):
            if len(u) == len(v):
                return sum([u[i]*v[i] for i in range(len(u))])
            else:
                raise('Lists must have same length to call dotprod.')
    elif str == None:
        if isinstance(u, (sp.Matrix)) and isinstance(v, (sp.Matrix)):
            return u.dot(v, conjugate_convention='right')
        if isinstance(u,list) and isinstance(v, list):
            if len(u) == len(v):
                return sum([u[i]*(sp.conjugate(v[i])) for i in range(len(u))])
            else:
                raise('Lists must have same length to call dotprod.')
    else:
        raise('dotprod not defined on str.')    

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fexponential
#https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixBase.exp
#-------------------------------------------------------------#

def exponential(M, t=None):
    if t == None:
        if isinstance(M, sp.Matrix):
            return M.exp()
        else:
            raise('Matrix exponential can only be called on SymPy matrix.')
    else:
        if isinstance(t, sp.Symbol):
            return exponential(M*t)
        else:
            raise('Matrix exponential called for invalid t.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Finverse
#https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixBase.inv
#-------------------------------------------------------------#

def inverse(M):
    if isinstance(M, sp.Matrix):
        return M.inv()
    else:
        raise('Matrix inverse can only be called on SymPy matrix.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fvector#:~:text=Important%3A%20The%20linalg%20package%20has,packages%2C%20see%20examples%2FLinearAlgebraMigration.&text=The%20vector%20function%20is%20part%20of%20the%20linalg%20package.
#-------------------------------------------------------------#
function_list = (sp.Function, sp.Add, sp.Mul, sp.Pow)

def vector(*args):

    if len(args) == 1:
        if isinstance(args[0], list):
            return sp.Matrix(args[0]) #nx1
        elif isinstance(args[0], (int, sp.Integer)):
            return sp.zeros(args[0],1)
        else:
            raise('Can only call vector with single argument on integer or list.')
    elif len(args) == 2:
        if isinstance(args[0], (int, sp.Integer)):
            if isinstance(args[1], list):
                return sp.Matrix(args[1]) #nx1
            elif isinstance(args[1], function_list):
                n = args[0]
                smb = list(args[1].free_symbols)
                if len(smb) != 1:
                    raise('Vector called with function with more than one argument.')
                else:
                    print('Warning, careful of argument order when calling function on Vector.')
                    return sp.Matrix([args[1].subs([(smb[0],i)]) for i in range(1,n+1)])
            else:
                raise('Calling vector the second argument of two must be list or function.')
        else:
            raise('Calling vector the first argument of two must be integer.')
    else:
        raise('Vector called on too many arguments.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fmultiply
#-------------------------------------------------------------#

def multiply(*args):
    if len(args) < 2:
        raise('Need at least two arguments for matrix multiply.')
    else:
        if all(isinstance(args[i], sp.Matrix) for i in range(len(args))):
            k = 1
            M = args[0]
            while k < len(args):
                M = M*args[k]
                k += 1
            return M 
        else:
            raise('Can only call multiply on SymPy matrices.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fiszero
#-------------------------------------------------------------#

def iszero(M):
    if isinstance(M, sp.Matrix):
        return M == sp.zeros(*M.shape)
    else:
        raise('Can only call iszero on SymPy matrix.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fequal
#-------------------------------------------------------------#

def equal(A, B):
    if isinstance(A, sp.Matrix) and isinstance(B, sp.Matrix):
        if A.shape == B.shape:
            return A == B
        else:
            raise('Can only call equal on matrices of same shape.')
    else:
        raise('Can only call equal on SymPy matrices.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Ftrace
#https://docs.sympy.org/latest/modules/matrices/common.html#sympy.matrices.common.MatrixCommon.trace
#-------------------------------------------------------------#

def trace(M):
    if isinstance(M, sp.Matrix):
        return M.trace()
    else:
        raise('Can only call trace on SymPy matrix.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Fvectdim
#-------------------------------------------------------------#

def vectdim(arg):
    if isinstance(arg, sp.Matrix):
        if arg.shape[1] == 1:
            return arg.shape[0]
        else:
            raise('Can only call trace on nx1 SymPy matrix.')
    elif isinstance(arg, list):
        return len(arg)
    else:
        raise('Can only call trace on SymPy matrix or list.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/view.aspx?path=linalg(deprecated)%2Finnerprod
#-------------------------------------------------------------#

def innerprod(*args):
    if len(args) < 2:
        raise('Can only call innerprod on at least two arguments.')
    else:
        if all(isinstance(elem, sp.Matrix) for elem in args):
            if len(args) == 2:
                return dotprod(args[0],args[1])
            if len(args) == 3:
                return dotprod(args[0],multiply(args[1],args[2]))
            else:
                return dotprod(args[0],multiply(multiply(args[1:-1]),args[-1]),'orthogonal')
        else:
            raise('Can only call innerprod on SymPy matrices.')

#-------------------------------------------------------------#
# Source: https://www.maplesoft.com/support/help/maple/view.aspx?path=linalg(deprecated)%2Fgrad#:~:text=Important%3A%20The%20linalg%20package%20has,packages%2C%20see%20examples%2FLinearAlgebraMigration.&text=The%20function%20grad%20computes%20the%20gradient%20of%20expr%20with%20respect%20to%20v.
#https://docs.sympy.org/latest/modules/core.html#sympy.core.function.diff
#-------------------------------------------------------------#

def grad(expr, v):
    if isinstance(expr, sp.Expr):
        if isinstance(v, list):
            return [sp.diff(expr,v[i]) for i in range(len(v))]
        if isinstance(v, sp.Matrix):
            return vector([sp.diff(expr,v[i,0]) for i in range(v.shape[0])])
        else:
            raise('Second argument of grad must be SymPy matrix or list.')
    else:
        raise('First argument of grad must be SymPy expression.')


