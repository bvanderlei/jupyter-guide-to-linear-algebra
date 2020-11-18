# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:07:23 2020

@author: Ben Vanderlei

The purpose of this module is to contain all code that is reused
throughout the LA Guide.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def RowSwap(A,k,l):
    ''' 
    RowSwap performs a single row operation on the matrix A.
    The positions of rows k and l are swapped.
    No error checking on the range of k and l.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    k : int
    l : int
    scale : float
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''

    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            B[i][j] = A[i][j]
        
    for j in range(n):
        temp = B[k][j]
        B[k][j] = B[l][j]
        B[l][j] = temp
        
    return B

def RowScale(A,k,scale):
    ''' 
    RowScale performs a single row operation on the matrix A.
    Row k is mulitiplied by scale, resulting in a new entries in row k.
    No error checking on the range of k.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    k : int
    scale : float
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            B[i][j] = A[i][j]
    for j in range(n):
        B[k][j] *= scale
        
    return B

def RowAdd(A,k,l,scale):
    ''' 
    RowAdd performs a single row operation on the matrix A.
    Row k is mulitiplied by scale and added to row l, replacing row l.
    No error checking on the range of k and l.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    k : int
    l : int
    scale : float
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''

    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            B[i][j] = A[i][j]
        
    for j in range(n):
        B[l][j] += B[k][j]*scale
        
    return B
    
def BackSubstitution(U,Y):
    '''
    BackSubstitution performs back substitution to find the solution to a
    square upper triangular system UX = Y.  There is no error checking to
    ensure that U is of full rank.

    Parameters
    ----------
    U : NumPy array object of dimension mxm
    Y : NumPy array object of dimension mx1

    Returns
    -------
    X : NumPy array object of dimension mx1
    '''

    m = U.shape[0]  # m is number of rows and columns in U
    X = np.zeros((m,1))
    
    for i in range(m-1,-1,-1):  # Calculate entries backward from m-1 to 0
        X[i] = Y[i]
        for j in range(i+1,m):
            X[i] -= U[i][j]*X[j]
        X[i] /= U[i][i]
    return X

def RowReduction(A):
    ''' 
    RowReduction performs steps of elimination with no pivot strategy and
    returns a row echelon form of the matrix A.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A

    B = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            B[i][j] = A[i][j]

    if m < n:
        elimination_steps = m
    else:
        elimination_steps = n

    # For each step of elimination, we find a suitable pivot, move it into
    # position and create zeros for all entries below.
    
    for k in range(elimination_steps):
        # Set pivot as (k,k) entry
        pivot = B[k][k]
        pivot_row = k
        
        # Find a suitable pivot if the (k,k) entry is zero
        while(pivot == 0 and pivot_row < m):
            pivot_row += 1
            pivot = B[pivot_row][k]
            
        # Swap row if needed
        if (pivot_row != k):
            B = RowSwap(B,k,pivot_row)
            
        # If pivot is nonzero, carry on with elimination in column k
        if (pivot != 0):
            B = RowScale(B,k,1./B[k][k])
            for i in range(k+1,m):    
                B = RowAdd(B,k,i,-B[i][k])
    return B

def SolveSystem(A,B):
    ''' 
    SystemSolve computes the solution to AX=B by elimination in the case that
    A is a square nxn matrix
    
    Parameters
    ----------
    A : NumPy array object of dimension nxn
    B : NumPy array object of dimension nx1
    
    Returns
    -------
    X: NumPy array object of dimension nx1
    '''
    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("SolveSystem accepts only square arrays.")
        return None
    n = A.shape[0]  # n is number of rows and columns in A

    # Join A and B to make the augmented matrix
    A_augmented = np.hstack((A,B))
    
    # Carry out elimination    
    R = RowReduction(A_augmented)

    # Split R back into nxn piece and nx1 piece
    B_reduced = R[:,n:n+1]
    A_reduced = R[:,0:n]

    # Do back substitution
    X = BackSubstitution(A_reduced,B_reduced)
    return X


def DeterminantIteration(A):
    ''' 
    DeterminantIteration computes the determinant of an nxn A matrix
    by using the recursive formula.

    Parameters
    ----------
    A: NumPy array object of dimension nxn
    
    Returns
    -------
    D: int
    '''
    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Determinant only defined for square arrays.")
        return None
    n = A.shape[0]  # n is number of rows and columns in A
    
    size = A.shape[0]
    if size == 2:
        return A[0,0]*A[1,1]-A[0,1]*A[1,0]
    
    else:
        m=0  # Determinant expansion along row 0
        D=0. # Set determinant to zero and add contributions

        for n in range(size):
            minor = []
            k=-1
            # Construct (m,n) minor array (row m, column n deleted)
            for i in range(size):
                if(i != m): 
                    minor.append([])    
                    k += 1
                    for j in range(size):
                        if(j != n):
                            minor[k].append(A[i,j])
            Minor_array = np.array(minor)
            cofactor = (-1)**(m+n)*DeterminantIteration(Minor_array)
            D += cofactor*A[m,n]
        return D

def DrawGraph(A, pos = None):
    '''
    Draws a directed graph based on adjacency matrix A.

    Parameters
    ----------
    A : NumPy array object.
    pos: Optional dictionary to specify node coordinates
    
    Returns
    -------
    pos: Dictionary of node coordinates used to draw graph

    '''
    plt.figure(figsize=(6,6))
    G = nx.DiGraph()
    
    N = A.shape[0]
    edge_list = []
    
    for i in range(N):
        for j in range(N):
            if(A[i,j] == 1):
                edge_list.append((i,j))
    
    G.add_edges_from(edge_list)
    if (pos == None):
        pos = nx.spring_layout(G)
    
    options = {"node_size" : 500, "with_labels": True,"font_size":20}
    nx.draw(G, pos,connectionstyle='arc3, rad = 0.1',arrowsize=30,**options)
    return pos

def HighlightSubgraph(A,pos,subgraph):
    '''
    Draws directed graph based on adjacency matrix A, with node positions pos,
    then colors a subgraph containing nodes in nodelist and edges connecting
    connecting nodes in nodelist    

    Parameters
    ----------
    A : NumPy array object
    
    pos : dictionary of node positions
    
    nodelist : list of ints representing the nodes in the subgraph

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(6,6))
    G = nx.DiGraph()
    
    N = A.shape[0]
    edge_list = []
    subgraph_edges = []
    
    for i in range(N):
        for j in range(N):
            if(A[i,j] == 1):
                edge_list.append((i,j))
 
    for edge in edge_list:           
        if (edge[0] in subgraph and edge[1] in subgraph):
                    subgraph_edges.append(edge)
    

    G.add_edges_from(edge_list)
    
    options = {"with_labels": True,"font_size":20}
    nx.draw(G, pos,connectionstyle='arc3, rad = 0.1',arrowsize=30,**options)
    
    node_options = {"node_color":'r',"node_size" : 400}
    nx.draw_networkx_nodes(G, pos, nodelist=subgraph, **node_options)

    
    edge_options = {"width" : 8,"alpha" : 0.5, "edge_color" : 'r',
                    "connectionstyle":"arc3, rad=0.1"}    
    nx.draw_networkx_edges(G,pos,edgelist=subgraph_edges, **edge_options)
