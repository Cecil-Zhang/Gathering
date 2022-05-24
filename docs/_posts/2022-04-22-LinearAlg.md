---
layout: single
title:  "Linear Algebra Essence"
date:   2022-04-22 12:00:04 -0700
categories: Math
toc: true
usemathjax: true
---

# Vector

## Vector

### Convention

- All vectors are column vectors by default $$\forall x \in R$$

### Addition and Subtraction

![Vector Addition and Subtraction]({{ site.baseurl }}/assets/images/math/vecOps.png)

### Length (norm)

- $$\|\vec{a}\|_2=\sqrt{a_1^2+\cdots+a_n^2}=\sqrt{\vec{a}\cdot\vec{a}}$$ is called the length of a vector $$\vec{a}$$

### Angle

- The angle between two vectors is $$\theta=\arccos\frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|_2\|\vec{b}\|_2}$$

### Dot Product

- For $$\vec{a}=(a_1,\dots,a_n), \vec{b}=(b_1,\dots,b_n)\in\mathbb{R}^n$$, $$\vec{a}\cdot\vec{b}=\sum_{i=1}^na_ib_i$$
- Projection: the length of $$\vec a$$ projection on $$\vec b$$ = $$|\vec a|\cos(\theta)=\frac{\vec a\cdot \vec b}{|\vec b|}$$
    
    ![projection]({{ site.baseurl }}/assets/images/math/projection.png)

### Vector Projection

- The projection of $$\vec a$$ on $$\vec b$$ is $$\frac{\vec a\cdot \vec b}{\vec b \cdot \vec b}\times\vec b$$
    - The length of $$\vec a$$ projection on $$\vec b$$ = $$|\vec a|\cos(\theta)=\frac{\vec a\cdot \vec b}{|\vec b|}$$
    - The direction of projection is parallel to $$\vec b$$, which is $$\frac{\vec b}{|\vec b|}$$
    - Thus, the projection vector is $$\frac{\vec a\cdot \vec b}{|\vec b|} \times \frac{\vec b}{|\vec b|}=\frac{\vec a\cdot \vec b}{|\vec b||\vec b|}\times\vec b==\frac{\vec a\cdot \vec b}{\vec b\cdot \vec b}\times\vec b$$

### Null Space

- Defition: $$null(A)=\{x\vert Ax=0\}$$

## Vector Space

### Vector Space

- A **vector space** ***V = {a collection of vectors $$\vec{v}$$ }*** satisfies
    - All $$\vec{v},\vec{w}\in V$$ can be added and multiplied by $$a\in \mathbb{R}$$:
        - $$\vec{v}+\vec{w}\in V, \qquad a\cdot \vec{v} \in V$$
    - The operations `+` and `.` must satisfy the following axioms
        
        ![Axioms]({{ site.baseurl }}/assets/images/math/LinearAlg0.png)
        
    - Example:
        - $$\mathbb{R}^n$$ is a vector space.
        - Polynomials is a vector space.
- **Subspace**
    - A subspace is a nonempty subset $$S\subseteq V$$  of a vector space ***V*** over field ***F*** that satisfies the conditions
        - $$\alpha \vec{v}\in S,\forall \vec{v}\in S,\forall \alpha\in \mathbb{F}$$
        - $$\vec{v}+\vec{u}\in S, \forall \vec{v},\vec{u}\in S$$
    - In other words S is closed scalar multiplication and under addition.

### Span

<aside>
💡 Intuition: All linear combinations of the vectors.

</aside>

- $$\vec{v}$$ is called a **linear combination** of $$\vec{v_1},\dots,\vec{v_n}$$ if $$\vec{v}=a_1\vec{v_1}+\dots+a_n\vec{v_n}$$
- For a set of vectors $$S=\{\vec{v_i}:i\in I\}$$, all its **linear combinations** define

$$
span(S)=\{\sum_ia_i\vec{v_i}:\vec{v_i}\in S \text{ and }a_i\in \mathbb{R}\}
$$

- The set $$\vec{v_1},\dots,\vec{v_n}\subseteq V$$ is called a **spinning se**t if $$span(\vec{v_1},\dots,\vec{v_n})=V$$. (i.e. every vector in ***V*** can be written as a linear combination of  $$\vec{v_1},\dots,\vec{v_n}$$)

### Linear Independence

- Vector $$\vec{v_1},\dots,\vec{v_n}$$ are called **linearly independent** if $$c_1\vec{v_1}+c_2\vec{v_2}+\dots+c_n\vec{v_n}=\vec{0}$$ implies that $$c_1=\dots=c_n=0$$.
- On the contrary, if there exists scalars $$c_1,\dots,c_n$$ not all zero such that $$c_1\vec{v_1}+c_2\vec{v_2}+\dots+c_n\vec{v_n}=\vec{0}$$, we call the vectors **linearly dependent**.
    - In this case, at least one vector is redundant and we can form a smaller spanning set.
    - At least one vector can be a linear transformation of other vectors

### **Basis**

- A set of vectors $$\{\vec{v_1},\dots,\vec{v_n}\}$$ that are linearly independent and span the full vector space ***V***, (i.e. $$span(\vec{v_1},\dots,\vec{v_n})=V$$), are called a **basis** of ***V***.
- Basis are a central concept in linear algebra
    - For a given vector space V there are usually an infinite number of bases to choose from
    - For many problems, the key solution step is transforming into the right basis
- Given a basis $$\{\vec{v_1},\dots,\vec{v_n}\}$$ for ***V***, there is a unique way to write any $$\tilde{\vec{v}}\in V$$ as: $$\tilde{\vec{v}}=\alpha_1\vec{v_1}+\alpha_2\vec{v_2}+\dots+\alpha_n\vec{v_n}$$

![Implicit Assumption]({{ site.baseurl }}/assets/images/math/LinearAlg1.png)

### Dimension

- If a vector space ***V*** has a basis consisting of *n* vectors, then V is said to have **dimension** *n*.

## Linear Transformation


💡 Intuition: A matrix can be interpreted as a certain transformation of a vector space.

$$
\begin{bmatrix}a&b\\c&c\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=x\begin{bmatrix}a\\b\end{bmatrix}=y\begin{bmatrix}b\\d\end{bmatrix}=\begin{bmatrix}ax+by\\cx+dy\end{bmatrix}
$$ 

Check out this [3Blue1Brown video](https://www.youtube.com/watch?v=kYB8IZa5AuE) for more info.


### Linear Transformation

- Given two vector spaces $$V, V’$$, **Linear transformation** is a mapping/function $$L:V\to V’$$ such that
    
    $$
    L(\alpha \vec{v}+\beta \vec{u})=\alpha L(\vec{v})+\beta L(\vec{u})\quad \forall \vec{v},\vec{u} \in V
    $$
    
- Given a basis $$\{\vec{v_1},\dots,\vec{v_n}\}$$ for ***V***, we have
    
    $$
    L(c_1\vec{v_1}+\dots+c_n\vec{v_n})=c_1L(\vec{v_1})+\dots+c_nL(\vec{v_n})
    $$
    

### Representation of Linear Transformation

- All linear maps $$L: \mathbb{R}^n\to \mathbb{R}^m$$ can be expressed as $$L(\vec{v})=A\vec{v}$$
- Evaluating the coordinates of $$L(\vec{v})$$ in basis ***C*** for an arbitrary $$\vec{v}$$

$$
\begin{equation}
\begin{split}   [L(\vec{v})]_C &=[L(x_1\vec{v_1}+\dots+x_n\vec{v_n})]_C\\
      &=\sum_{j=1}^n x_j[L(\vec{v_j})]_C=\sum_{j=1}^n\begin{bmatrix}
   a_{1j} \\
   \vdots \\
   a_{mj}
\end{bmatrix}x_j\\
 &=\begin{bmatrix}
   a_{1j} \cdots a_{1n} \\ 
   \vdots \ddots \vdots\\
   a_{m1} \cdots a_{mn}
\end{bmatrix}
\begin{bmatrix}
   x_{1} \\
   \vdots \\
   x_{n}
\end{bmatrix}\\
&:=A[\vec{v}]_B
\end{split}
\end{equation}
$$

### Interpretation of Matrix-Vector Multiplication

- $$\vec{a}=A\vec{b}$$
    - Assume $$\hat{e_1},\dots,\hat{e_n}$$ is the basis of vector space of $$\vec{b}$$, then each column of ***A*** is where the transformed vectors of $$\hat{e_1},\dots,\hat{e_n}$$ landed in the vector space of $$\vec{a}$$ (i.e. forming a basis)
    
    ![Interpretation]({{ site.baseurl }}/assets/images/math/LinearAlg2.png)
    
    - Example
        
        ![linear combination]({{ site.baseurl }}/assets/images/math/LinearAlg3.png)
        
        ![illustration]({{ site.baseurl }}/assets/images/math/LinearAlg4.png)
        

# Matrix

## Matrix Operation

### Matrix Multiplication

- Let $$A\in\mathbb{R}^{p\times m}$$ and $$B\in\mathbb{R}^{m\times n}$$.
- The matrix dimension should match!
- The matrix multiplication operation $$C=AB$$ is defined as: $$c_{ij}=\sum_{k=1}^m a_{ik}b_{kj}$$
    - Each column of C is a linear combination of the columns of A
        
        ![column]({{ site.baseurl }}/assets/images/math/LinearAlg5.png)
        
    - Each row of *C* is a linear combination of the rows of *B*
        
        ![row]({{ site.baseurl }}/assets/images/math/LinearAlg6.png)
        
- Properties
    - Associative: $$A(BC)=(AB)C$$
    - Distributive: $$A(B+C)=AB+AC$$
    - Not commutative: $$AB\neq BA$$
- Interpretation: $$\vec{u}=AB\vec{v}=C\vec{v}$$ ($$C=AB$$), matrix multiplication is the composition of linear transformation
    - The effect of first applying linear transformation ***A***, then applying linear transformation ***B***
    - Overall it is the same as applying the linear transformation ***C***.

### Transpose

- Def: $$(A^T)_{ij}=A_{ji}$$
- Properties
    - $$(A^T)^T=A$$
    - $$(A+B)^T=A^T+B^T$$
    - $$(AB)^T=B^TA^T$$

### Inverse

- A square matrix ***X*** is an **inverse** of $$A\in\mathbb{R}^{m\times m}$$ if and only if ***AX=I*** and ***XA=I***.
    - We write $$X=A^{-1}$$, ***A*** is said to be **invertible** or **nonsingular** when it has an inverse.
- If A and B are invertible, so are $$A^{-1},A^T,AB$$
- $$(A^{-1})^{-1}=A, (A^{-1})^{T}=(A^{T})^{-1}, (AB)^{-1}=B^{-1}A^{-1}$$
- Theorem: the following conditions for square matrix $$A\in\mathbb{F}^{m\times m}$$ are equivalent
    - A has an inverse
    - A is a nonsingular matrix
    - det(A)≠0 (if det(A)=0, then the transformation squishes into lower dimension)
    - A is of full rank m
    - $$range(A)=\mathbb{F}^{m}$$ 
    - $$\mathbb{N}(A)=\{\vec{0}\}$$ 
    - 0 is not an eigenvalue of A
    - 0 is not a singular value of A

## Denotation

### Trace

- Sum of its diagonal element: $$tr(A)=\sum_{j=1}^m a_{jj}$$
- Properties: $$tr(A+B)=tr(A)+tr(B), tr(cA)=ctr(A), tr(AB)=tr(BA)$$
- Cyclic Property: $$tr(ABCD)=tr(BCDA)=tr(CDAB)=tr(DABC)$$

### Range / Column Space

- The **range / column space** of $$A\in\mathbb{R}^{m\times n}$$ is the image of $$\mathbb{R}^n$$ under ***A***
    - $$\text{range(A)}:=\lbrace{A\vec{x}|\vec{x}\in}\mathbb{R}^{n}\rbrace$$
    - i.e. the set of all possible outputs of ***Ax***

### Matrix Rank

<aside>
💡 Intuition: rank = the number of dimensions in the output / column space

</aside>

- The **column rank** of ***A*** is the dimension of the column space range(A)
- The **row rank** of ***A*** is the dimension of the row space (the span of ***A***’s rows)
    - Column and row ranks always equal, so we can simply refer to the **rank** of a matrix
- **Full rank**: Matrix ***A*** is **full rank** if it has the greatest possible rank, *min(m, n)*
    - A full-rank matrix A with m≥n must have n linearly independent columns
    - Theorem: a matrix A with m≥n has full rank n iff $$Ax\neq Ay ,\;\forall x\neq y\in \mathbb{R}^n$$ (no distinct vectors get mapped to the same vector)
- Rank-One
    - The transformation becomes a line
    - The rank of a matrix $$A\in\mathbb{R}^{m\times n}$$ equals 1 if and only if it can be written in the form $$xy^T$$
- Rank-Two: the transformation becomes a plane
- rank(AB) ≤ min(rank(A), rank(B))

### Determinants

<aside>
💡 Intuition: change in volume of the unit hypercube when it is transformed by A

</aside>

- The determinant of a square matrix, , measuring the “volume change” produced by the corresponding linear map.
- Axioms
    - Axioms1: $$\det(A)$$ is a multilinear function of the columns of A
        - Intuition: in 3D, scale the length by *a* then add it by *b*, the volume changes to $$\alpha V + V_b$$

            ![Axiom1]({{ site.baseurl }}/assets/images/math/LinearAlg-det1.png)
            
    - Axiom2: $$\det(A)$$ vanishes if any columns are repeated
        - Intuition: in 3D, when a dimension collapse, the cube falls into a plane.

            ![Axiom2]({{ site.baseurl }}/assets/images/math/LinearAlg-det2.png)
            
    - Axiom3: $$\det(I)=1$$
- Properties
    - Property1: Adding a multiple of another column doesn’t change the determinant
        - $$\begin{equation}
\begin{split}   &\det(e_1,\dots,e_k+\alpha e_j,\dots, e_m)\\&=\det(e_1,\dots,e_k,\dots, e_m)
      +\underbrace{\alpha\det(e_1,\dots, e_j,\dots, e_m)}_{=0}
\end{split}
\end{equation}$$
    - Property2: Swapping two adjacent columns negates
        - $$\begin{equation}
\begin{split}
&\det(e_1,\dots,e_j+e_{j+1},e_j+e_{j+1},\dots, e_m)\\ &=\underbrace{\det(e_1,\dots,e_j,e_j,\dots, e_m)}_{=0}+\det(e_1,\dots,e_j,e_{j+1},\dots,e_m)\\ &+\det(e_1,\dots,e_{j+1},e_{j},\dots,e_m)+\underbrace{\det(e_1,\dots,e_{j+1},e_{j+1},\dots, e_m)}_{=0}\\ &=0
\end{split}
\end{equation}$$
    - Property3: det(A) = 0 if A’s columns are linearly dependent
        - $$\det(e_1,\dots,e_{m-1},\sum_{i=1}^{m-1}c_ie_i)=\sum_{i=1}^{m-1}c_i\det(e_1,\dots,e_{m-1},e_i)=0$$
- $$\det(\text{triangular matrix})=\prod_{i=1}^m a_{ii}$$
- $$det(\alpha A)=\alpha^m det(A)$$
- $$det(A^T)=det(A)$$
- $$det(AB)=det(A)det(B)$$
- $$det(A^{-1})=\frac{1}{det(A)}$$

### Eigenvalue and Eigenvector

- Given a square matrix $$A\in \mathbb{C}^{m\times m}$$, a nonzero vector $$\vec{x}\in \mathbb{C}^m$$ is an **eigenvector**, and $$\lambda\in \mathbb{C}$$ is its corresponding **eigenvalue**, if: $$A\vec{x}=\lambda\vec{x}$$
- The set of all eigenvalues of A is called the matrix **spectrum** $$\Lambda(A)$$
- More refer to [eigenvalue decomposition](#eigenvalue-decomposition-1)

### Singular Value

- Refer to [singular value decompostion](#singular-value-decomposition)

## Special Matrix

### Identity Matrix

$$
I_n=[\vec{e_1},\dots,\vec{e_n}]=\begin{bmatrix}
   1 & 0 & \dots & 0 \\
   0 & 1 & \ddots & \vdots \\
   \vdots & \ddots & \ddots & 0 \\
   0 & \dots & 0 & 1
\end{bmatrix} \text{ such that } I_n\vec{v}=\vec{v}
$$

### Triangular Matrix

- Lower triangular: a square matrix is called lower triangular if all the entries above the main diagonal are zero.
- Upper triangular: a square matrix is called upper triangular if all the entries below the main diagonal are zero.
- Properties
    - The product of two upper/lower triangular matrix is still upper/lower triangular.
    - The inverse of upper/lower triangular matrix, if it exists, is still upper/lower triangular.

### Positive Definite Matrix

- A is positive definite if  $$A^T=A\text{ and }x^TAx>0,\forall x\neq 0$$

### Symmetric Matrix

- A is symmetric if $$A^T=A$$

### Hermitian Matrix

- $$A^H=(\bar{A})^T$$
- Compare to Symmetric
    - Symmetric for matrices $$\in\mathbb{R}^{m\times n}$$
    - Hermitian for matrices $$\in\mathbb{C}^{m\times n}$$

### Orthogonal Matrix

- Vectors $$\vec{a}, \vec{b}$$ are orthogonal if $$\vec{a}\cdot\vec{b}=0=\cos90$$
- A Square Matrix $$A\in\mathbb{R}^{n\times n}$$ is orthogonal if $$A^TA=AA^T=I$$
- $$det(A)=\pm1$$ (Proof: $$det(I)=det(A^TA)=det(A^T)det(A)=1$$)

### Unitary Matrix

- Matrix $$A\in\mathbb{C}^{n\times n}$$ is unitary if $$A^HA=AA^H=I$$
- Compare to Orthogonal
    - Orthogonal matrices $$\in\mathbb{R}^{m\times n}$$
    - Unitary matrices $$\in\mathbb{C}^{m\times n}$$

# Norm

## Vector Norms

- **Norms** are an indispensable tool to provide vectors and matrices (and functions) with measures of size, length and distance.

### Definition

- A **vector** **norm** on $$\mathbb{C}^n$$ is a mapping that maps each $$\vec{x}\in\mathbb{C}^n$$ to a real number $$\|\vec{x}\|\in\mathbb{R}$$satisfying the following properties:
    - Positive Definite Property: $$\|\vec{x}\|>0\text{ for } \vec{x}\neq\vec{0} \text{ and } \|\vec{0}\|=0$$
    - Absolute Homogeneity: $$\|\alpha \vec{x}\|=\vert\alpha\vert\|\vec{x}\| \forall \alpha \in \mathbb{C}$$
    - Triangle inequality: $$\|\vec{x}+\vec{y}\|\leq\|\vec{x}\|+\|\vec{y}\|$$

### p-norm

- **Vector p-norm**
    - Let $$p\in\mathbb{R}$$ and $$1\leq p \leq \infty$$, define $$\|\vec{x}\|_p=(\sum_{i=1}^m\vert x_i\vert^p)^{\frac{1}{p}}$$
    - Then $$\|\vec{x}\|_p$$ is a norm and is called the vector p-norm.
- Frequently used norms
    - “Taxicab” or “Manhattan distance”: $$p=1\implies \|\vec{x}\|_1=\sum_{i=1}^m\vert x_i\vert$$
    - “Euclidean length”: $$p=2\implies \|\vec{x}\|_2=\sqrt{\sum_{i=1}^m\vert x_i\vert^2}=\sqrt{\vec{x}^H\vec{x}}$$
    - $$p=\infty\implies \|x\|_\infty=\max_{1\leq i\leq m}\vert x_i\vert$$
    - The geometry of vector norms
        
        ![Geometry]({{ site.baseurl }}/assets/images/math/LinearAlg7.png)
        
- Nice Property of 2-norm
    - invariance under unitary transformation: $$\|Qx\|_2=\|x\|_2$$ if $$Q^HQ=I$$
    - differentiable: $$\Delta\|x\|_2=\frac{x}{\|x\|_2}$$
- Norm Equivalence
    - Let $$\|\cdot\|_p,\|\cdot\|_q$$ be any two vector norms, then there are constants $$c_1,c_2>0$$ such that
    
    $$
    c_1\|\cdot\|_p\leq \|\cdot\|_q\leq c_2\|\cdot\|_p
    $$
    
    - By the norm equivalence, the convergence (error, distance) in one norm implies convergence (error, distance) in any other norm.

## Matrix Norms

### Definition

- A matrix **norm** on $$\mathbb{C}^{m\times n}$$ is a mapping that maps each $$A\in\mathbb{C}^{m\times n}$$ to a real number $$\|A\|\in\mathbb{R}$$satisfying the following properties:
    - Positive Definite Property: $$\|A\|>0\text{ for } A\neq\vec{0} \text{ and } \|{0}\|=0$$
    - Absolute Homogeneity: $$\|\alpha A\|=\vert\alpha\vert\|A\|$$ for $$\alpha\in\mathbb{C}$$
    - Triangle inequality: $$\|A+B\|\leq\|A\|+\|B\|$$
    - *Consistency: $$\|AB\|\leq \|A\|\|B\|$$

### Induced Norms

**Induced Matrix Norm**: A vector norm $$\|\cdot\|$$ induces a matrix norm, denoted by

$$
\|A\|_{(m,n)}:=\max_{\substack{\vec x\in \mathbb{C}^n\\\vec x\neq \vec 0}}\frac{\|A\vec x\|_{(m)}}{\|\vec x\|_{(n)}}=\max_{\substack{\vec x\in \mathbb{C}^n\\\|\vec x\|_{(n)}=1}} \|A\vec x\|_{(m)}
$$

- Property: $$\|Ax\|\leq \|A\|\|x\|$$. Therefore, $$\|A\|$$ is the maximal factor by which ***A*** can “stretch” a vector.

### Frequently Used Matrix Norms

- **Frobenius norm** $$\|A\|_F$$ of A is defined as

$$
\|A\|_F=(\sum_{i=1}^m\sum_{j=1}^n|a_{ij}|^2)^{\frac{1}{2}}=\sqrt{tr(A^HA)}\quad \text{where } A=(a_{ij})\in\mathbb{C}^{m\times n}
$$

- **Matrix p-norms**
    - $$\|A\|_1=\max_{\vec{x}\neq 0}\frac{\|A\vec x\|_1}{\|\vec x\|_1}=\max_{1\leq j\leq n}\{\sum_{i=1}^m\vert a_{ij}\vert\}$$ = max absolute column sum
    - $$\|A\|_2=\max_{\vec{x}\neq 0}\frac{\|A\vec x\|_2}{\|\vec x\|_2}=\sqrt{\text{the largest eigenvalue of }A^HA}$$ = the largest singular value of A
    - $$\|A\|_\infty=\max_{\vec{x}\neq 0}\frac{\|A\vec x\|_\infty}{\|\vec x\|_\infty}=\max_{1\leq i\leq m}\{\sum_{j=1}^n\vert a_{ij}\vert\}$$ = max absolute row sum

### Properties

- $$\|\cdot\|_2, \|\cdot\|_F$$ are **unitarily invariant**:
    - $$\|UAV\|_2=\|A\|_2,\quad\|UAV\|_F=\|A\|_F$$ for all unitary matrices ***U*** and ***V***.
- Norm Inequalities
    - $$\|A\|_2^2\leq \|A\|_1\|A\|_\infty$$
- Norm Equivalence
    - Let $$\|\cdot\|_p,\|\cdot\|_q$$ be any two matrix norms, then there are constants $$c_1,c_2>0$$ such that

$$
c_1\|\cdot\|_p\leq \|\cdot\|_q\leq c_2\|\cdot\|_p
$$

# Matrix Decomposition

## LU Decomposition

### Definition

- Factor a matrix $$A\in \mathbb{C}^{m\times m}$$ into the form: *A=LU*, where *L* is lower triangular and *U* is upper triangular.
    - The algorithm is Gaussian elimination in matrix form
    
    ![LU]({{ site.baseurl }}/assets/images/math/LinearAlg8.png)
    
- Gaussian Elimination
    
    ![Gaussian elimination]({{ site.baseurl }}/assets/images/math/LinearAlg9.png)
    

### Pivoting

- Without pivoting (Plain LU): The version of Gaussian elimination presented is neither backward stable nor stable — very dangerous
    - The backward stability issue is closely related to zeros on the diagonal leading to $$l_{ik}=\frac{a_{ik}}{0}$$
    - Solution: swap the rows/columns (**pivoting**)
    - Pivoting brings backward stability
- Complete Pivoting: $$PAQ=LU$$ (P, Q are permutation matrices)
    - At step k, we ideally pick the largest $$x_{ij}$$ in submatrix $$i,j\ge k$$ as the pivot, reordering rows and columns
    - Backward Stable
    - Computation complexity: $$O(m^3)$$
- Partial Pivoting: $$PA=LU$$(P is a permutation matrix)
    - Only reorder the **rows**. Choose the largest $$x_{ik}\;\text{ in column k} (i\ge k)$$ as the pivot
    - Technically backward stable, though theoretically with backward error growing exponentially as we enlarge the matrix $$\rho=2^{m-1}$$
    - Computation complexity: $$O(m^2)$$
- Partial Pivoting is truly helpful and appears to be sufficient for acceptable (non-exponential) growth factors except in vanishingly rare pathological cases.

### Special Cases

- Cholesky decomposition
    - A symmetric matrix A is positive definite if and only if $$A=LL^T$$
    - *L* is a unique nonsingular lower triangular matrix with positive diagonal entries
- $$LDL^T$$ factorization
    - If a symmetric matrix A is nonsingular, then there exists a permutation P, a unit lower triangular matrix L, and a block diagonal matrix D with 1-by-1 and 2-by-2 blocks such that $$PAP^T=LDL^T$$

### Application

- Solve $$Ax=b$$
- Compute $$det(A)$$
- Compute $$A^{-1}$$, if applicable

## QR Decomposition

### Definition

- Let $$A\in \mathbb{R}^{m\times n}$$ with m ≥ n and rank(A) = n. Then there exists an orthogonal matrix $$Q\in \mathbb{R}^{m\times n} (Q^TQ=I)$$, and an upper triangular matrix $$R\in \mathbb{R}^{n\times n}$$ such that $$A=QR$$
- QR decomposition is Gram-Schmidt orthogonalization process in matrix form

![QR decomposition]({{ site.baseurl }}/assets/images/math/LinearAlg10.png)

### Gram-Schmidt
![gram-schmidt-1]({{ site.baseurl }}/assets/images/math/gramschmidt1.png)
![gram-schmidt-2]({{ site.baseurl }}/assets/images/math/gramschmidt2.png)

### Application

- Find an orthonormal basis of the subspace spanned by the columns of A, namely the Gram-Schmidt orthogonalization process.
- Solve the linear least squares problem $$\min_x\|Ax-b\|_2$$

## Eigenvalue Decomposition

### Schur decomposition

- For a square matrix $$A\in \mathbb{R}^{n\times n}$$, there is an n x n unitary matrix U ($$U^HU=I$$) such that $$A=UTU^H$$ where T is upper triangular
- Always exists

### Spectral decomposition

- For a Hermitian matrix $$A\in \mathbb{R}^{n\times n}(A^H=A)$$, we have $$A=U\Lambda U^H$$ where $$\Lambda$$ is real and diagonal, and U is unitary($$U^HU=I$$).

### Eigenvalue decomposition

- A square matrix $$A\in \mathbb{C}^{n\times n}$$ is **diagonalizable** if and only if $$A=X\Lambda X^{-1}$$, where $$\Lambda$$ is a diagonal matrix and X is nonsingular.

![eigenvalue decomposition]({{ site.baseurl }}/assets/images/math/LinearAlg11.png)

- Not always exists

### Application

- spectral embedding

## Singular Value Decomposition

> Swiss Army Knife of linear algebra.
> 

### Definition

The (full) **singular value decomposition** (SVD) of $$A\in \mathbb{R}^{m\times n}$$

$$A=U\Sigma V^T$$

where

- $$U$$ is an $$m\times m$$ orthogonal matrix $$U^TU=I$$
- $$V$$ is an $$n\times n$$ orthogonal matrix $$V^TV=I$$
- $$\Sigma$$ is an $$m\times n$$ diagonal matrix with the ordered nonnegative diagonal elements $$\sigma_1\ge \sigma_2\dots \sigma_r>\sigma_{r+1}=\dots=\sigma_n=0$$

Furthermore,

- $$\sigma_1, \sigma_2,\dots,\sigma_n$$ are called **singular values**
- The columns $$u_1, u_2,\dots,u_m$$ of $$U$$ are called **left singular vectors**
- The columns $$v_1, v_2,\dots,v_m$$ of $$V$$ are called **right singular vectors**
- Always exists

### Compact/Thin/Reduced SVD

Let *r=rank(A)*, then $$A=U_r\Sigma_r V_r^T$$, where

- $$U_r$$ is an $$m\times r$$ orthogonal matrix $$U_r^TU_r=I$$
- $$V_r$$ is an $$n\times r$$ orthogonal matrix $$V_r^TV_r=I$$
- $$\Sigma_r=Diag(\sigma_1,\dots,\sigma_r)$$

### Application

- *rank(A) = r =* the number of nonzero singular values
- The range/column space of *A*: $$col(A)=span\{u_1,u_2,\dots,u_r\}$$
- The null space of *A*: $$null(A)=span\{v_{r+1},\dots,v_n\}$$
- SVD as the sum of rank-1 matrices (”components”)
    - Let $$r=rank(A)$$, then  $$A=E_1+E_2+\dots+E_r$$
    - where $$E_k$$ for $$k=1,2,\dots,r$$ is a rank-one matrix of the form $$E_k=\sigma_ku_kv_k^T$$
    - $$E_k$$ is referred to as the k-th **component** of A
- Echkart-Young Theorem (optimal low-rank approximation)
    - Let $$A=U\Sigma V^T$$ be the SVD of A. Then the matrix  $$A_k=E_1+E_2+\dots+E_k$$ is the **best** rank-k approximatin of A. That is,
    
    $$
    \min_{rank(B)=k}\|A-B\|=\|A-A_k\|
    $$
    
    - where $$k≤ r = rank(A)$

## Solving Least Squares

- Problem: $$\min_x\|Ax-b\|_2$$ for $$A\in\mathbb{C}^{m\times n},\;m\geq n$$
- Solution
    - Cholesky (normal equations)
        - $$A^HAx=A^Hb,\quad A^HA=LL^H$$
    - Reduced QR
        - $$A=\hat Q\hat R,Ax=b\implies \underbrace{\hat R^H\hat Q^H}_{A^H} \underbrace{\hat Q\hat R}_A x=\underbrace{\hat R^H\hat Q^H}_{A^H}b\implies \hat Rx=\hat Q^Hb$$
    - Reduced SVD
        - $$A=\hat U\hat \Sigma V^H,Ax=b\implies \underbrace{V\hat \Sigma\hat U^H}_{A^H} \underbrace{\hat U\hat \Sigma V^H}_A x=\underbrace{V\hat \Sigma\hat U^H}_{A^H}b\implies x=V\hat\Sigma^{-1}\hat U^Hb$$