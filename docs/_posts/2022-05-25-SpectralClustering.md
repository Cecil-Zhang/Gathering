---
layout: single
title:  "Spectral Clustering"
date:   2022-05-25 18:00:00 -0700
categories: 
- Artificial Intelligence
tags:
- Unsupervised Learning
toc: true
usemathjax: true
---

# Prerequisite

## Graph

- An (undirected) **graph** is $$G=(V,E)$$, where
    - $$V=\{v_i\}$$ is a set of vertices;
    - $$E=\{(v_i,v_j)|v_i,v_j\in V\}$$ is a subset of $$V\times V$$
    
    ![graph1]({{ site.baseurl }}/assets/images/ML/spectral_clustering.png)
    
    graph1
    
- Simple graph: there is at most one edge from $$v_i$$ to $$v_j$$

### Degree

- degree: for any $$v_i\in V$$, the **degree** $$d(v_i)$$ of $$v_i$$ is the number of edges adjacent to  $$v_i$$: $$d(v_i)=|\{v_j\in V|\{v_j,v_i\}\in E|$$
- degree matrix: $$D=diag(d(v_1),\dots,d(v_n))$$
    - The degree matrix of *graph1* is $$D=\begin{bmatrix}2 & 0 & 0 & 0\\0 & 3 & 0 & 0\\0 & 0 & 3 & 0\\0 & 0 & 0 & 2\end{bmatrix}$$

### Adjacency Matrix

- Given $$G=(V,E)$$, with $$|V|=n$$ and $$|E|=m$$, the **adjacency matrix** $$A=(a_{ij})$$ of $$G$$ is **symmetric** $$n\times n$$ matrix with

$$$$
a_{ij} = \begin{cases}
   1, &\text{if } \{v_i,v_j\}\in E \\
   0, &\text{otherwise }
\end{cases}
$$$$

- The adjacency matrix of *graph1* is $$A(G)=\begin{bmatrix}0 & 1 & 1 & 0\\1 & 0 & 1 & 1\\1 & 1 & 0 & 1\\0 & 1 & 1 & 0\end{bmatrix}$$

## Weighted Graph

- A weighted graph is $$G=(V,W)$$, where
    - $$V=\{v_i\}$$ is a set of vertices and $$|V|=n$$;
    - $$W\in \mathbb{R}^{n\times n}$$ is called weight matrix with
        
        $$$$
        w_{ij} = \begin{cases}
        w_{ij}\ge 0, &\text{if } i\neq j \\
        0, &\text{if } i=j
        \end{cases}
        $$$$
        
- The underlying graph of $$G$$ is $$\hat G=(V,E)$$ with
    - $$E=\{(v_i,v_j)|w_{ij}>0\}$$
- If $$w_{ij}\in\{0,1\}, W=A$$, the adjacency matrix of $$\hat G$$

### Degree

- degree: for any $$v_i\in V$$, the **degree** $$d(v_i)$$ of $$v_i$$ is the sum of the weights of the edges adjacent to  $$v_i$$: $$d(v_i)=\sum_{j=1}^n w_{ij}$$
- degree matrix: $$D=diag(d(v_1),\dots,d(v_n))$$

### Volumn

- $$|A|\coloneqq$$ the number of vertices in ***A***
- Given a subset of vertices $$V_1\subseteq V$$, we define the **volumn** by
    
    $$$$
    vol(V_1)=\sum_{v_i\in V_1}d(v_i)=\sum_{v_i\in V_1}(\sum_{j=1}^n w_{ij})
    $$$$
    
    - It sums over the weights of all edges attached to vertices in ***A***.
    
    ![Untitled]({{ site.baseurl }}/assets/images/ML/spectral_clustering_1.png)
    
- Remarks
    - If $$vol(V_1)=0$$, all the vertices in $$V_1$$ are isolated

### Links

- Given two subsets of vertices $$V_1,V_2\subseteq V$$, we define the **links** by
    
    $$$$
    W(V_1,V_2)=links(V_1,V_2)=\sum_{v_i\in V_1,v_j\in V_2}w_{ij}
    $$$$
    
- Remarks
    - $$V_1, V_2$$ are not necessarily distinct
    - $$links(V_1,V_2)=links(V_2,V_1)$$, since ***W*** is symmetric
    - $$vol(V_1)=links(V_1,V)$$

### Cut

- The quantity $$cut(V_1)$$ is defined by $$cut(V_1)=links(V_1,V-V_1)$$
- The quantity $$assoc(V_1)$$ is defined by $$assoc(V_1)=links(V_1,V_1)$$
- Remarks
    - $$cut(V_1)$$ measures how many links **escape** from $$V_1$$
    - $$assoc(V_1)$$ measures how many links **stay** within $$V_1$$
    - $$cut(V_1)+assoc(V_1)=vol(V_1)$$

## Graph Laplacian

### Unnormalized Graph Laplacian

- Given a weighted graph $$G=(V,W)$$, the **(graph) Laplacian** ***L*** of ***G*** is defined by

$$$$
L=D-W
$$$$

- where ***D*** is the degree matrix of ***G***, i.e., $$D=diag(W\cdot \bold 1)$$

### Properties of unnormalized Laplacian *L*

- $$x^TLx=\frac{1}{2}\sum_{i,j=1}^n w_{ij}(x_i-x_j)^2,\quad \forall x\in\mathbb{R}^n$$
- $$L\ge 0$$ if $$w_{ij}\ge 0$$ for all $$i,j$$.
- $$L$$ is symmetric and positive semi-definite (if $$w_{ij}\ge 0$$).
- $$L\cdot \bold 1=\bold 0$$
    - Proof: $$L\cdot \bold 1=(D-W)\cdot\bold 1=D\cdot\bold 1-W\cdot\bold 1=\bold d-\bold d=\bold 0$$
- If the underlying graph of ***G*** is connected
    - $$0=\lambda_1<\lambda_2\leq\lambda_3\leq\dots\leq\lambda_n$$, where $$\lambda_i$$ are the eigenvalues of ***L***. $$\lambda_2$$ is called a Fiedler value/eigenvalue.
    - The smallest eigenvalue of ***L*** is 0, the corresponding eigenvector is the constant one vector $$\bold 1$$.
    - the dimension of the nullspace of ***L*** is **1.**
- The multiplicity ***k*** of the eigenvalue ***0*** of L equals **the number of connected components** $$A_1,\dots,A_k$$ in the graph. The eigenspace of eigenvalue 0 is spanned by the indicator vectors $$\bold 1_{A_1},\dots,\bold 1_{A_k}$$ of those components.

### Normalized Graph Laplacian

- $$L_{sym}\coloneqq D^{-1/2}LD^{1/2}=I-D^{-1/2}WD^{1/2}$$
    - Denote as $$L_{sym}$$ as it’s a symmetric matrix
- $$L_{rm}\coloneqq D^{-1/2}L=I-D^{-1}W$$
    - Denote as $$L_{rw}$$ as it’s closely related to a random walk.

### Properties of normalized Laplacian $$L_{sym}, L_{rw}$$

- $$x^TL_{sym}x=\frac{1}{2}\sum_{i,j=1}^n w_{ij}(\frac{x_i}{\sqrt d_i}-\frac{x_j}{\sqrt d_j})^2,\quad \forall x\in\mathbb{R}^n$$
- $$\lambda$$ is an eigenvalue of $$L_{rw}$$ with eigenvector $$u$$ if and only if $$\lambda$$ is an eigenvalue of $$L_{sym}$$ with eigenvector $$w=D^{1/2}u$$
- $$\lambda$$ is an eigenvalue of $$L_{rw}$$ with eigenvector $$u$$ if and only if $$\lambda\text{ and }u$$ solves the generalized eigen-problem $$Lu=\lambda Du$$
- 0 is an eigenvalue of $$L_{rw}$$ with the constant one vector $$\bold 1$$ as eigenvector. 0 is an eigenvalue of $$L_{sym}$$ with eigenvector $$D^{1/2}\bold 1$$.
- $$L_{sym}$$ and $$L_{rw}$$ are positive semi-definite and have n non-negative real-valued eigenvalues $$0=\lambda_1<\lambda_2\leq\lambda_3\leq\dots\leq\lambda_n$$.
- The multiplicity ***k*** of the eigenvalue ***0*** of both $$L_{sym}$$ and $$L_{rw}$$ equals **the number of connected components** $$A_1,\dots,A_k$$ in the graph. For $$L_{rw}$$, the eigenspace of eigenvalue 0 is spanned by the indicator vectors $$\bold 1_{A_i}$$ of those components. For $$L_{sym}$$, the eigenspace of eigenvalue 0 is spanned by the vectors $$D^{1/2}\bold 1_{A_i}$$.

## Rayleigh quotient

- Let $$A,B\in \mathbb R^{n\times n}, A^T=A, B^T=B>0$$ and $$\lambda_1\leq \lambda_2\leq \dots \leq \lambda_n$$ be the eigenvalues of $$(A,B)$$ with corresponding eigenvectors $$u_1,u_2,\dots,u_n$$, then
    - then $$\min_x\frac{x^TAx}{x^TBx}=\lambda_1,\arg\min_x\frac{x^TAx}{x^TBx}=u_1$$,
    - and $$\min_{x^TBu_1=0}\frac{x^TAx}{x^TBx}=\lambda_2,\arg\min_{x^TBu_1=0}\frac{x^TAx}{x^TBx}=u_2$$,

# Graph Clustering

## Graph Partition

### Connected Components

- A subset $$A \subset V$$ of a graph is **connected** if any two vertices in $$A$$ can be joined by a path such that all intermediate points also lie in $$A$$. (All vertices in $$A$$ are connected)
- A subset $$A$$ is called a **connected component** if it is **connected** and if there are no connections between vertices in $$A$$ and $$\bar A (V-A)$$.

### k-way partitioning

- Given a weighted graph $$G=(V,W)$$, find a **partition** $$V_1,V_2,\dots,V_k$$ of $$V$$, such that
    - $$V_i\cap V_j=\empty \text{ for } i\neq j$$
    - $$V_1\cup V_2\cup \dots \cup V_k=V$$
    - **for any $$i,j$$, the edges between $$(V_i,V_j)$$ have low weight (similarity) and the edges within $$V_i$$ and $$V_j$$ have high weight (similarity).**
- If k=2, it is a two-way partioning (bi-partitioning)
    - (two-way) $$cut(V_1)=W(V_1,V-V_1)=\sum_{v_i\in V_1,v_j\in V-V_1}w_{ij}$$

### mincut

- The **mincut** of k subsets is defined by

$$$$
\min_{A_1,\dots,A_k} cut(A_1,\dots,A_k):=\min_{A_1,\dots,A_k}\frac{1}{2}\sum_{i=1}^kW(A_i,\bar A_i)
$$$$

- Issue: In practice, the mincut typically yields **unbalanced** partitions. Usually it yields a trivial solution which simply separates one individual vertex from the rest of the graph.
    - In this example, $$\min cut(V_1)=1+2=3$$
    
    ![Untitled]({{ site.baseurl }}/assets/images/ML/spectral_clustering_2.png)
    
    - Solution: normalized cut
        - RatioCut
        - Ncut
        - normalization makes sure that the clusters are “balanced”, but makes them NP-hard
    - Spectral clustering is a way to solve **relaxed** versions of those problems.
    
    <aside>
    💡 We will see that **relaxing** **Ncut** leads to **normalized spectral clustering**, while **relaxing RatioCut** leads to **unnormalized spectral clustering**
    
    </aside>
    

## Normalized Cut

### Ratio Cut

- Normalized by the number of vertices

$$$$
\min_{A_1,\dots,A_k}RatioCut(A_1,\dots,A_k):=\min_{A_1,\dots,A_k}\frac{1}{2}\sum_{i=1}^k\frac{W(A_i,\bar A_i)}{|A_i|}
$$$$

- The minimum of $$\sum_{i=1}^k\frac{1}{|A_i|}$$ is achieved if all $$|A_i|$$ coincide

### Ncut

- Normalized by the weights of its edges $$vol(A_i)$$

$$$$
\min_{A_1,\dots,A_k}Ncut(A_1,\dots,A_k):=\min_{A_1,\dots,A_k}\frac{1}{2}\sum_{i=1}^k\frac{W(A_i,\bar A_i)}{vol(A_i)}
$$$$

- The minimum of $$\sum_{i=1}^k\frac{1}{vol(A_i)}$$ is achieved if all $$vol(A_i)$$ coincide
- For example, $$\min Ncut(V_1)=\frac{4}{3+6+6+3}+\frac{4}{3+6+6+3}=\frac{4}{9}$$
    
    ![Untitled]({{ site.baseurl }}/assets/images/ML/spectral_clustering_3.png)
    

# Spectral Clustering

<aside>
💡 The eigenvectors of the Laplacian Matrices can be proven to be the solution of the relaxed graph cut optimization problems.

</aside>

## Unnormalized Spectral Clustering

### Approximating RatioCut for k=2

- Easiest to start with
- Objective function: $$\min_{A\subset V} RatioCut(A,\bar A)$$
- Rewrite the objective function
    - Define the vector $$f=(f_1,\dots,f_n)'\in\mathbb{R}^n$$ with entries $$f_i=\begin{cases}\sqrt{|\bar A|/|A|},\text{ if } v_i\in A\\ -\sqrt{|A|/|\bar A|},\text{ if } v_i\in \bar A\end{cases}$$
    - It can be proven that $$f'Lf=|V|\cdot RatioCut(A,\bar A)$$
    - Additionally, we have $$\sum_{i=1}^n f_i=\sum_{i\in A}\sqrt{\frac{|\bar A|}{|A|}}-\sum_{i\in \bar A}\sqrt{\frac{|A|}{|\bar A|}}=|A|\sqrt{\frac{|\bar A|}{|A|}}-|\bar A|\sqrt{\frac{|A|}{|\bar A|}}=0$$
        - In other words, $$f$$ is orthognal to the constant one vector $$\bold 1$$
    - Finally, note that $$f$$ satisfies $$\|f\|^2=\sum_{i=1}^nf_i^2=|\bar A|+|A|=n$$
    - Put them together, we have
        
        $$$$
        \min_{A\subset V}f'Lf \text{ subject to } f\perp\bold 1,f_i=\begin{cases}\sqrt{|\bar A|/|A|}\text{ if } v_i\in A\\ -\sqrt{|A|/|\bar A|}\text{ if } v_i\in \bar A\end{cases},\|f\|=\sqrt n
        $$$$
        
- Issue: NP hard, solution: relax the discreteness condition and instead allow that $$f_i$$ takes arbitrary values in $$\mathbb{R}$$
- Leads to the relaxed optimization problem
    
    $$$$
    \min_{f\in\mathbb{R}^n}f'Lf \text{ subject to } f\perp\bold 1,\|f\|=\sqrt n
    $$$$
    
- By the Rayleigh-Ritz theorem, the solution is given by the vector $$f$$ which is the eigenvector corresponding to the second smallest eigenvalue of $$L$$
    - recall that the smallest eigenvalue of $$L$$ is 0 with eigenvector $$\bold 1$$
    - So we can approximate a minimizer of RatioCut by the second eigenvector of L
- To obtain a partition of the graph, we need to convert solution vector $$f$$ of the relaxed problem into a discrete indicator vector.
    - The simplest way is to choose $$\begin{cases}v_i\in A\text{ if } f_i\ge 0\\ v_i\in \bar A\text{ if } f_i<0\end{cases}$$
    - Standard way: consider the coordinates $$f_i$$ as points in $$\mathbb R$$ and cluster them into two groups by the k-means clustering algorithm.

### Approximating RatioCut for arbitrary k

- Follows a similar principle as k=2
- Given a partition of ***V*** into k sets $$A_1,\dots,A_k$$, we define k indicator vectors $$h_j=(h_{1,j},\dots,h_{n,j})'$$ by $$h_{i,j}=\begin{cases}1/\sqrt{|A_j|},\text{ if } v_i\in A_j\\ 0,\text{ otherwise}\end{cases}\quad (i=1,\dots,n;\;j=1,\dots,k)$$
- Then set the matrix $$H\in\mathbb R^{n\times k}$$ as the matrix containing those k indicator vectors as columns
    - The columns are orthogonal to each other, i.e. $$H’H=I$$
- Rewrite the objective function
    - Similarly, we have $$h_i'Lh_i=\frac{cut(A_i,\bar A_i)}{|A_i|}$$.
    - Moreover, $$h_i'Lh_i=(H’LH)_{ii}$$
    - Combining above facts, we get $$RatioCut(A_1,\dots,A_k)=\sum_{i=1}^kh_i’Lh_i=\sum_{i=1}^k(H’LH)_{ii}=Tr(H’LH)$$, where Tr denotes the trace of a matrix.
    - Put them together, we have
    
    $$$$
    \min_{A_1,\dots,A_k}Tr(H'LH)\text{ subject to }H'H=I,h_{i,j}=\begin{cases}1/\sqrt{|A_j|},\text{ if } v_i\in A_j\\ 0,\text{ otherwise}\end{cases}
    $$$$
    
- Similarly, we can relax the the problem by allowing the entries of H to take arbitrary real values, which leads to the relaxed optimization problem
    
    $$$$
    \min_{H\in \mathbb R^{n\times k}}Tr(H'LH)\text{ subject to }H'H=I
    $$$$
    
- By the Rayleigh-Ritz theorem, the solution is given by choosing H as the matrix which contains the first $$k$$ eigenvectors of $$L$$ as columns.
- To obtain a partition of the graph, we need to convert solution matrix $$H$$ of the relaxed problem into discrete indicator vectors.
    - Apply k-means algorithms on the rows of $$H$$

### Complete Algorithm

![Untitled]({{ site.baseurl }}/assets/images/ML/spectral_clustering_4.png)

## Normalized Spectral Clustering

### Approximating Ncut for k=2

- Objective function: $$\min_{A\subset V} Ncut(A,\bar A)$$
- Rewrite the objective function
    - Define the indicator vector $$f=(f_1,\dots,f_n)'\in\mathbb{R}^n$$ with entries $$f_i=\begin{cases}\sqrt{vol(\bar A)/vol(A)},\text{ if } v_i\in A\\ -\sqrt{vol(A)/vol(\bar A)},\text{ if } v_i\in \bar A\end{cases}$$
    - It can be proven that $$f'Lf=vol(V)\cdot Ncut(A,\bar A)$$
    - Additionally, we have $$(Df)'\bold 1=0$$
        - In other words, $$Df$$ is orthognal to the constant one vector $$\bold 1$$
    - Finally, note that $$f'Df=vol(V)$$
    - Put them together, we have
        
        $$$$
        \min_{A\subset V}f'Lf \text{ subject to } Df\perp\bold 1,f'Df=vol(V),f_i=\begin{cases}\sqrt{vol(\bar A)/vol(A)},\text{ if } v_i\in A\\ -\sqrt{vol(A)/vol(\bar A)},\text{ if } v_i\in \bar A\end{cases}
        $$$$
        
- Issue: NP hard, solution: relax the discreteness condition and instead allow that $$f_i$$ takes arbitrary values in $$\mathbb{R}$$
- Leads to the relaxed optimization problem
    
    $$$$
    \min_{f\in\mathbb{R}^n}f'Lf \text{ subject to } Df\perp\bold 1,f'Df=vol(V)
    $$$$
    
- Now we substitute $$g:=D^{1/2}f$$, we have

$$$$
\min_{g\in\mathbb{R}^n}g'D^{-1/2}LD^{-1/2}g \text{ subject to } g\perp\bold D^{1/2}1,\|g\|^2=vol(V)
$$$$

- Observe that $$D^{-1/2}LD^{-1/2}=L_{sym},D^{1/2}\bold 1$$ is the first eigenvector of $$L_{sym}$$, and $$vol(V)$$ is a constant.
- By the Rayleigh-Ritz theorem, the solution $$g$$ is given by the second eigenvector of $$L_{sym}$$
    - Re-substituting $$f=D^{-1/2}g$$, we see that $$f$$ is the second eigenvector of $$L_{rw}$$, or equivalently the generalized eigenvector of $$Lu=\lambda Du$$
    - So we can approximate a minimizer of Ncut by the second eigenvector of $$L_{rw}$$
- To obtain a partition of the graph, we convert solution vector $$f$$ of the relaxed problem into a discrete indicator vector similarly.

### Approximating Ncut for arbitrary k

- Objective function: $$\min_{A\subset V} Ncut(A,\bar A)$$
- Given a partition of ***V*** into k sets $$A_1,\dots,A_k$$, we define k indicator vectors $$h_j=(h_{1,j},\dots,h_{n,j})'$$ by $$h_{i,j}=\begin{cases}1/\sqrt{vol(A_j)},\text{ if } v_i\in A_j\\ 0,\text{ otherwise}\end{cases}\quad (i=1,\dots,n;\;j=1,\dots,k)$$
- Then set the matrix $$H\in\mathbb R^{n\times k}$$ as the matrix containing those k indicator vectors as columns
    - The columns are orthogonal to each other, i.e. $$H’H=I$$
    - $$h_i’Dh_i=1$$
- Rewrite the objective function
    - Similarly, we have $$h_i'Lh_i=\frac{cut(A_i,\bar A_i)}{vol(A_i)}$$.
    - Moreover, $$h_i'Lh_i=(H’LH)_{ii}$$
    - Combining above facts, we get $$Ncut(A_1,\dots,A_k)=\sum_{i=1}^kh_i’Lh_i=\sum_{i=1}^k(H’LH)_{ii}=Tr(H’LH)$$, where Tr denotes the trace of a matrix.
    - Put them together, we have
    
    $$$$
    \min_{A_1,\dots,A_k}Tr(H'LH)\text{ subject to }H'DH=I,h_{i,j}=\begin{cases}1/\sqrt{vol(A_j)},\text{ if } v_i\in A_j\\ 0,\text{ otherwise}\end{cases}
    $$$$
    
- Similarly, we can relax the the problem by allowing the entries of H to take arbitrary real values and substituting $$T=D^{1/2}H$$, which leads to the relaxed optimization problem
    
    $$$$
    \min_{H\in \mathbb R^{n\times k}}Tr(T'D^{-1/2}LD^{-1/2}T)\text{ subject to }T'T=I
    $$$$
    
- By the Rayleigh-Ritz theorem, the solution $$g$$ is given by the second eigenvector of $$L_{sym}$$
    - Re-substituting $$f=D^{-1/2}g$$, we see that $$f$$ is the second eigenvector of $$L_{rw}$$, or equivalently the generalized eigenvector of $$Lu=\lambda Du$$
    - So we can approximate a minimizer of Ncut by the second eigenvector of $$L_{rw}$$
- To obtain a partition of the graph, we convert solution vector $$f$$ of the relaxed problem into a discrete indicator vector similarly.

- By the Rayleigh-Ritz theorem, the solution is given by choosing H as the matrix which contains the first $$k$$ eigenvectors of $$L_{sym}$$ as columns.
    - Re-substituting $$H=D^{-1/2}T$$, we see that solution $$H$$ contains the first k eigenvectors of $$L_{rw}$$, or equivalently the first k generalized eigenvectors of $$Lu=\lambda Du$$.
    - So we can approximate a minimizer of RatioCut by the second eigenvector of L

### Complete Algorithm

![Untitled]({{ site.baseurl }}/assets/images/ML/spectral_clustering_5.png)

- A variant
    
    ![Untitled]({{ site.baseurl }}/assets/images/ML/spectral_clustering_6.png)
    

## **Comments**

- Most importantly, there is no guarantee whatsoever on the quality of the solution of the relaxed problem compared to the exact solution. In general it is known that efficient algorithms to approximate balanced graph cuts up to a constant factor do not exist. To the contrary, this approximation problem can be NP hard itself (Bui and Jones, 1992).
- **The reason why the spectral relaxation is so appealing is not that it leads to particularly good solutions. Its popularity is mainly due to the fact that it results in a standard linear algebra problem which is simple to solve.**

# Reference

- A Tutorial on Spectral Clustering [arXiv:0711.0189]