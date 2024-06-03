 # SymmetryPy
This is a repository for the Python package symmetrypy built on SymPy based off of the Maple package symmetry by Karin Gatermann.

## One-line overview :zap: ...
SymmetryPy is a Python package that allows you to calculate generators of group-invariant/equivariant functions (known as fundamental invariants/equivariants) and evaluate them on large datasets for use in ML models!

## Quick installation
The package has only two dependencies:
- SymPy
- Numpy
<!--- ### Pip
Quick installation with `pip` is possible:
```
pip install symmetrypy
``` --->

Manual installation is currently the installation of choice! PyPip installation is coming soon!

# How do I use it?
Below are some [quick-fire examples](#quickfire-examples) on how to:
- construct a finite group with a representation
- calculate the fundamental invariants and/or equivariants
- evaluate the invariants on a dataset
- use in any ML context you like!

More involved references on [constructing groups](#constructing-groups) as well as [calculating fundamental invariants](#calculating-fundamental-invariants) and/or [equivariants](#calculating-fundamental-equivariants) can be found below.

## Quick-fire examples :fire:
Let's pick a simple example: suppose I want to consider functions $f:\mathbb{R}^2\to\mathbb{R}^2$ that are $D_4$-invariant. I want to calculate the set of fundamental invariants $\{\pi_1,...,\pi_r\}$ with respect to the 2-dimensional irreducible representation $\rho:D_4\times\mathbb{R}^2\to\mathbb{R}^2$ of $D_4$. By definition, this set generates the set $D_4$-invariant polynomials
$$\{f|f(x)=f(\rho(g)x)\forall x\in\mathbb{R}^2\forall g\in D_4\}.$$

First, let's import the necessary packages and modules.
```
import symmetrypy
import sympy as sp
```
To calculate the fundamental invariants, we first create the group $D_4$ using `dihedral`, where elements are given in the 2-dimensional irrep (see [below](#dihedral) for details about the representation).
```
D4 = dihedral(n)
```
Then we construct two SymPy symmbols $x_1$ and $x_2$ to represent variables in our domain such that $x=(x_1,x_2)\in\mathbb{R}^2$.
```
x_1, x_2 = sp.symbols('x_1,x_2')
```
Now let's make some toy data for this example in our domain $\mathbb{R}^2$. 

Primary and secondary invariants can be calculated up to degree $k\in\mathbb{N}$ using `CMBasis` (see [below](#calculating-fundamental-invariants) for more details). In this case suppose we choose $k=4$. Our code returns a dictionary with keys 'primary_invariants' and 'secondary_invariants'.
```
k = 4
invariants = CMBasis(D4,[x_1,x_2],k)
# {'primary_invariants': [x_1**2 + x_2**2, x_1**4 + x_2**4], 'secondary_invariants': [1]}
```
The set of primary invariants are the set of fundamental invariants. To evaluate each primary invariant on our toy dataset `x_train`, we use SymPy's `evalf`.
```
pi = [inv.evalf(x_train) for inv in invariants['primary_invariants']]
```

## Constructing Groups 
There are 5 types of finite groups that are already implemented:
1) [Dihedral](#dihedral)
2) [Cyclic](#cyclic)
3) [Tetrahedral](#tetrahedral)
4) [Octrahedral](#octrahedral)
5) [Icosahedral](#icosahedral)

See below for information on how to construct the groups below. [Custom groups](#custom-groups) can also be implemented.

### Dihedral
The $n$-Dihedral group can be instantiated for any $n\geq2$, even or odd. Recall that
- for even $n$, there are 4 1-dimensional irreducible representations and $\frac{1}{2}(n-2)$ 2-dimensional representations,
- for odd $n$, there are 2 1-dimensional irreducible representations and $\frac{1}{2}(n-1)$ 2-dimensional representations.

There are 2 ways to create the $n$-Dihedral group:

a. <ins>Standard irreducible representation:</ins> Calling `dihedral` with one argument $n$ creates $D_n$ where elements are given in the first 2-dimensional irrep of the group (i.e. the 5th irrep if $n$ is even or 3rd irrep if $n$ is odd). For example, let $n=4$, then
```
n = 4
g = dihedral(4)
```
creates $D_4$ with elements represented in the 5th irrep that is 2-dimensional.

b. <ins>Specific representation:</ins> Calling `dihedral` with two arguments, $n$ and a list of the integers $[i_1,...,i_r]$ for some $r\geq1$ and $1\leq i_k\leq\frac{1}{2}(n-2)$ if $n$ is even or $1\leq i_k\leq\frac{1}{2}(n-1)$ if $n$ is odd, creates $D_n$ where elements are given in the representation $\rho:D_n\times\mathbb{R}^m\to\mathbb{R}^m$ such that
$$\bigoplus_{k=1}^r\rho_{i_k}=\rho,$$
where $\rho_{i_k}$ is the $i_k$-th irrep of $D_n$ for $k=1,...,r$ and $m=\sum_{k=1}^r\text{dim}(\rho_{i_k})$.

For example, let $n=4$, then 
   ```
   n = 4
   irreps = [1,5,5]
   g = dihedral(n,irreps)
   ```
creates the $m=5$ dimensional representation $\rho$ of $D_4$ given by the $r=3$ irreps $[i_1,i_2,i_3] = [1,5,5]$ i.e. $\rho = \rho_1 \oplus\rho_5\oplus\rho_5$.

### Cyclic
The $n$-Cyclic group can be instantiated for any $n\geq2$, even or odd. Recall that
- for even $n$, there are 2 1-dimensional irreducible representations and $\frac{1}{2}(n-2)$ 2-dimensional representations,
- for odd $n$, there is 1 1-dimensional irreducible representations and $\frac{1}{2}(n-1)$ 2-dimensional representations.

There are 2 ways to create the $n$-Cyclic group:

a. <ins>Standard irreducible representation:</ins> Calling `cyclic` with one argument $n$ creates $C_n$ where elements are given in the the $n/2+1$-th irrep if $n$ is even or 2nd irrep if $n$ is odd. For example, let $n=3$, then
```
n = 3
g = cyclic(3)
```
creates $C_3$ with elements represented in the 2nd irrep that is 2-dimensional.

b. <ins>Specific representation:</ins> Calling `cyclic` with two arguments can be done as explained for the [Dihedral group](#dihedral) above.

For example, let $n=6$, then 
   ```
   n = 6
   irreps = [1,2,3]
   g = cyclic(n,irreps)
   ```
creates the $m=4$ dimensional representation $\rho$ of $C_6$ given by the $r=3$ irreps $[i_1,i_2,i_3] = [1,2,3]$ i.e. $\rho = \rho_1 \oplus\rho_2\oplus\rho_3$.

### Tetrahedral
The tetrahedral group $T_1$ is the group of symmetries of a tetrahedron that is isomorphic to the Symmetric group $S_4$. Recall that $T_1$ has 4 irreducible representations, 3 1-dimensional irreducible representations and 1 3-dimensional irreducible representation.

There are 2 ways to create the Tetrahedral group:

a. <ins>3-dimensional irreducible representation:</ins> Calling `tetrahedral` with no arguments creates $T_1$ where elements are given in the 3-dimensional irrep.

b. <ins>Specific representation:</ins> Calling `tetrahedral` with one argument that is a list of integers $[i_1,...,i_r]$ for some $r$ where $1\leq i_k\leq 4$ for $k=1,...,r$ creates $T_1$ where elements are given in the $m$-dimensional representation $\rho=\bigoplus_{k=1}^r\rho_{i_k}$ and $\rho_{i_k}$ is the $i_k$-th representation of $T_1$, while $m=\sum_{k=1}^{r}\text{dim}(\rho_{i_k})$, similar to the [Dihedral group](#dihedral).

For example, 
   ```
   irreps = [4,4]
   g = tetrahedral(irreps)
   ```
creates the $m=3+3=6$ dimensional representation $\rho$ of $T_1$ given by the $r=2$ irreps $[i_1,i_2] = [4,4]$ i.e. $\rho = \rho_4\oplus\rho_4$.

### Octrahedral
The octrahedral group $O$ is the group of symmetries of an octrahedron. Recall that $O$ has 5 irreducible representations in total: 2 1-dimensional, 1 2-dimensional and 2 3-dimensional irreducible representations.

There are 2 ways to create the Octrahedral group:

a. <ins>4th irreducible representation:</ins> Calling `octrahedral` with no arguments creates $O$ where elements are given in the 4th irreducible representation, that is a 3-dimensional representation.

b. <ins>Specific representation:</ins> Calling `octrahedral` with one argument that is a list of integers $[i_1,...,i_r]$ for some $r$ where $1\leq i_k\leq 5$ for $k=1,...,r$ creates $O$ where elements are given in the $m$-dimensional representation $\rho=\bigoplus_{k=1}^r\rho_{i_k}$ and $\rho_{i_k}$ is the $i_k$-th representation of $O$, while $m=\sum_{k=1}^{r}\text{dim}(\rho_{i_k})$, as with the [Tetrahedral group](#tetrahedral).

For example, 
   ```
   irreps = [2,3,5]
   g = octrahedral(irreps)
   ```
creates the $m=1+2+3=6$ dimensional representation $\rho$ of $O$ given by the $r=3$ irreps $[i_1,i_2,i_3] = [2,3,5]$ i.e. $\rho = \rho_2\oplus\rho_3\oplus\rho_5$.

### Icosahedral
The icosahedral group $I$ is the group of symmetries of an icosahedron. Recall that $I$ has 5 irreducible representations in total: 1 1-dimensional, 2 3-dimensional, 1 4-dimensional and 1 5-dimensional irreducible representations.

There are 2 ways to create the Icosahedral group:

a. <ins>5-dimensional irreducible representation:</ins> Calling `icosahedral` with no arguments creates $I$ where elements are given in the 5th irreducible representation, that is a 5-dimensional representation.

b. <ins>Specific representation:</ins> Calling `icosahedral` with one argument that is a list of integers $[i_1,...,i_r]$ for some $r$ where $1\leq i_k\leq 5$ for $k=1,...,r$ creates $I$ where elements are given in the $m$-dimensional representation $\rho=\bigoplus_{k=1}^r\rho_{i_k}$ and $\rho_{i_k}$ is the $i_k$-th representation of $I$, while $m=\sum_{k=1}^{r}\text{dim}(\rho_{i_k})$, as with the [Tetrahedral group](#tetrahedral).

For example, 
   ```
   irreps = [5,3]
   g = icosahedral(irreps)
   ```
creates the $m=5+4=9$ dimensional representation $\rho$ of $I$ given by the $r=2$ irreps $[i_1,i_2] = [5,3]$ i.e. $\rho = \rho_5\oplus\rho_3$.

### Custom groups
Finite groups that have not already been implemented can be constructed by hand. For the purpose of calculating fundamental invariants and equivariants, a finite group $G$ is a nested Python dictionary with the following key/value structure:
```
{'generators': {'_s1': _s1
                '_s2': _s2
                ...
                '_sm': _sm},
 'allements': {'_r1': _r1,
               '_r2': _r2,
               ...
               '_rn': _rn},
 'invelements': {'_r1': _r1,
                 '_r2': _r2,
               ...
               '_rn': _rn},
'order': n}
```
where `n` is an integer that is the order (number of elements) of $G$ and `_sj` and `_ri` are SymPy matrices for $i=0,...,$`n`$-1$ and $j=1,...,m$ where $m$ is the number of generators of $G$, that are all elements of $G$ given in the same representation.

## Calculating fundamental invariants
Primary and secondary invariants of a finite group $G$ are calculated using the function `CMBasis`.

It returns a Python dictionary with keys `'primary_invariants'` and `'secondary_invariants'`, with values of a list of primary invariants and a list of secondary invariants respectively. 

```
D4 = dihedral(n,[5]) # This representation of D4 has dimension 2

x1, x2 =  sp.smbols('x1:4)
k = 4 # We want to generate a basis of secondary invariants up to degree 4

invariants = CMBasis(D4,[x_1,x_2,x_3],k)
# {'primary_invariants': [x_1**2 + x_2**2, x_1**4 + x_2**4], 'secondary_invariants': [1]}
```

## Calculating fundamental equivariants

A set of equivariants of a finite group $G$ are calculated using the function `equis`. For two representations of the same group '`gtheta'` and `'grho'`, it calculates a basis of equivariants F such that $g_\theta(g) F(x)=F(g_\rho(g) x)$ for every $g\in G$. In order to run, it therefore either requires a list of primary invariants for `'grho'` in the form of SymPy expresssions in the variables, or it will calculate them itself.

It returns a Python dictionary with keys `'primary_invariants'` and `'equivariants'`, with values of a list of primary invariants and equivariants respectively. These are all given as SymPy matrices of SymPy expressions. The matrices are dimension dim$(g_\theta)$ and the variables are in $x_1,\hdots,x_n$ where $n=\text{dim}g_\rho$.

```
gtheta = dihedral(n,[5,5]) # This representation of D4 has dimension 4
grho = dihedral(n,[5]) # This representation of D4 has dimension 2

x1, x2 =  sp.smbols('x1:4) # These are the variables of grho
k = 4 # We want to generate a basis of equivariants up to degree 4
primaries = [x_1**2 + x_2**2, x_1**4 + x_2**4] # If we alreay know some primary ivnariants we can input them here

invs_and_equis = equis(D4,[x_1,x_2,x_3] ,prims)
# {'primary_invariants': [x_1**2 + x_2**2, x_1**4 + x_2**4], 'equivariants':[]}
```
<!-- ## High-dimensional -->

