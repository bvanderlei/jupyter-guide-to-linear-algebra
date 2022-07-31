# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14, 2020

@author: Ben Vanderlei

This module contains all code that is reused
throughout the Jupyter Guide to Linear Algebra.

Testing in progress to check that all functions
work with arrays of both shape (m,1) and (m,) as
inputs intended to represent vectors.  

Functions that have been tested have BSV in docstring.
"""

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def BackSubstitution(U,Y):
    '''
    BackSubstitution(U,Y)
    (BSV)
    
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
        if (U[i][i] != 0):
            X[i] /= U[i][i]
        else:
            print("Zero entry found in U pivot position",i,".")
    return X

def DeterminantIteration(A):
    ''' 
    DeterminantIteration(A)
    
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

def DotProduct(U,V):
    ''' 
    DotProduct(U,V)
    
    DotProduct computes the Euclidean product of U and V
    
    Parameters
    ----------
    U : NumPy array object of dimension nx1
    V : NumPy array object of dimension nx1
    
    Returns
    -------
    product: float
    '''

    # Check shapes of U and V
    if (U.shape[1] != 1 or V.shape[1] != 1):
        print("Dot product only accepts column vectors.")
        return
    # Check shape of V
    if (U.shape[0] != V.shape[0]):
        print("Dot product only accepts column vectors of equal length.")
        return

    n = U.shape[0]
    product = 0.0
    
    for i in range(n):
        product += U[i,0]*V[i,0]

    return product

def DrawGraph(A, pos = None):
    '''
    DrawGraph(A, pos = None)
    
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

def FullRowReduction(A, tol = 1e-14):
    ''' 
    FullRowReduction(A, tol = 1e-14)
    
    Produces RREF for matrix of any shape.  No pivot strategy implemented.
    Entries with abs value < tol are set to zero to account for roundoff
    errors.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    tol: optional float

    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A

    B = np.copy(A).astype('float64')

    # Set initial pivot search position
    pivot_row = 0
    pivot_col = 0
    
    # Continue steps of elimination while possible pivot positions are 
    # within bounds of the array.
    
    while(pivot_row < m and pivot_col < n):

        # Set pivot value to current pivot position
        pivot = B[pivot_row,pivot_col]
        
        # If pivot is zero, search down current column, and then subsequent
        # columns (at or beyond pivot_row) for the next nonzero entry in the 
        # array is found, or the last entry is reached.

        row_search = pivot_row
        col_search = pivot_col
        search_end = False

        while(pivot == 0 and not search_end):
            if(row_search < m-1):
                row_search += 1
                pivot = B[row_search,col_search]
            else:
                if(col_search < n-1):
                    row_search = pivot_row
                    col_search += 1
                    pivot = B[row_search,col_search]
                else:  
                    # col_search = n-1 and row_search = m-1
                    search_end = True
                        
        # Swap row if needed to bring pivot to position for rref
        if (pivot != 0 and pivot_row != row_search):
            B = RowSwap(B,pivot_row,row_search)
            pivot_row, row_search = row_search, pivot_row
            
        # Set pivot position to search position
        pivot_row = row_search
        pivot_col = col_search
            
        # If pivot is nonzero, carry on with elimination in pivot column 
        if (pivot != 0):
            
            # Set pivot entry to one
            B = RowScale(B,pivot_row,1./B[pivot_row,pivot_col])

            # Create zeros above pivot
            for i in range(pivot_row):    
                B = RowAdd(B,pivot_row,i,-B[i][pivot_col])
                # Force known zeros
                B[i,pivot_col] = 0

            # Create zeros below pivot
            for i in range(pivot_row+1,m):    
                B = RowAdd(B,pivot_row,i,-B[i][pivot_col])
                # Force known zeros
                B[i,pivot_col] = 0

            # Force small numbers to zero to account for roundoff error
            for i in range(m):
                for j in range(n):
                    if abs(B[i,j])< tol :
                        B[i,j] = 0

        # Advance to next possible pivot position
        pivot_row += 1
        pivot_col += 1
        
    return B

def HighlightSubgraph(A,pos,subgraph):
    '''
    HighlightSubgraph(A,pos,subgraph)
    
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

def Inverse(A):
    '''
    Inverse(A)
    
    A is a NumPy array that represents a matrix of dimension n x n.
    Inverse computes the inverse matrix by solving AX=I where I is the identity.
    If A is not invertible, Inverse will not return correct results.

    Parameters
    ----------
    A: NumPy array object of dimension nxn
    
    Returns
    -------
    Inverse: NumPy array object of dimension nxn
    '''

    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Inverse accepts only square arrays.")
        return
    n = A.shape[0]  # n is number of rows and columns in A

    I = np.eye(n)
    
    # The augmented matrix is A together with all the columns of I.  RowReduction is
    # carried out simultaneously for all n systems.
    A_augmented = np.hstack((A,I))
    R = RowReduction(A_augmented)
    
    Inverse = np.zeros((n,n))
    
    # Now BackSubstitution is carried out for each column and the result is stored 
    # in the corresponding column of Inverse.
    A_reduced = R[:,0:n]
    for i in range(0,n):
        B_reduced = R[:,n+i:n+i+1]
        Inverse[:,i:i+1] = BackSubstitution(A_reduced,B_reduced)
    
    return(Inverse)



def Magnitude(U):
    ''' 
    Magnitude(U)
    
    Magnitude computes the magnitude of U based on the Euclidean inner product
    
    Parameters
    ----------
    U : NumPy array object of dimension nx1
    
    Returns
    -------
    product: float
    '''
    # Check shape of U
    if (U.shape[1] != 1):
        print("Magnitude only accepts column vectors.")
        return
    
    magnitude = math.sqrt(DotProduct(U,U))
    return magnitude    

def QRFactorization(A):
    ''' 
    QRFactorization(A)
    
    A is a Numpy array that represents a matrix of dimension m x n.
    QRFactorization returns matrices Q and R such that A=QR, Q is orthogonal
    and R is upper triangular.  The factorization is carried out using classical
    Gram-Schmidt and the results may suffer due to numerical instability.
    QRFactorization may not return correct results if the columns of A are 
    linearly dependent.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    
    Returns
    -------
    Q : NumPy array object of dimension mxn
    R : NumPy array object of dimension nxn
    '''

    # Check shape of A
    if (A.shape[0] < A.shape[1]):
        print("A must have more rows than columns for QR factorization.")
        return

    m = A.shape[0]
    n = A.shape[1]
    
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    
    for i in range(n):
        W = A[:,i:i+1]
        for j in range(i):
                W = W - DotProduct(A[:,i:i+1],Q[:,j:j+1])*Q[:,j:j+1]
        Q[:,i:i+1] = W/Magnitude(W)
        
    R = Q.transpose()@A
    
    return (Q,R)

def RowSwap(A,k,l):
    ''' 
    RowSwap(A,k,l)
    
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
    
    B = np.copy(A).astype('float64')

    for j in range(n):
        temp = B[k][j]
        B[k][j] = B[l][j]
        B[l][j] = temp
        
    return B

def RowScale(A,k,scale):
    ''' 
    RowScale(A,k,scale)
    
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
    
    B = np.copy(A).astype('float64')

    for j in range(n):
        B[k][j] *= scale
        
    return B

def RowAdd(A,k,l,scale):
    ''' 
    RowAdd(A,k,l,scale)
    
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
    
    B = np.copy(A).astype('float64')
        
    for j in range(n):
        B[l][j] += B[k][j]*scale
        
    return B

def RowReduction(A):
    ''' 
    RowReduction(A)
    
    RowReduction performs steps of elimination with no pivot strategy to
    produce a row echelon from of the matrix A.  It is assumed that A
    is the augemented matrix associated with a linear system that has
    a unique solution.  RowReduction may not return correct results if A
    does not have dimensions n x (n+1) or does not have a pivot in each '
    column.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''
    
    m = A.shape[0]  # A has m rows 
    n = A.shape[1]  # It is assumed that A has m+1 columns
    
    B = np.copy(A).astype('float64')

    # For each step of elimination, we find a suitable pivot, move it into
    # position and create zeros for all entries below.
    
    for k in range(m):
        # Set pivot as (k,k) entry
        pivot = B[k][k]
        pivot_row = k
        
        # Find a suitable pivot if the (k,k) entry is zero
        while(pivot == 0 and pivot_row < m-1):
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
        else:
            print("Pivot could not be found in column",k,".")
            
    return B


def ScaleMatrixRows(A):
    ''' 
    ScaleMatrixRows(A)
    
    ScaleMatrix rows accepts an mxn array where each row has been scaled
    to unit length.
    
    Parameters
    ----------
    A : NumPy array object of dimension mxn
    
    Returns
    -------
    B: NumPy array object of dimension mxn
    '''    
    
    m = A.shape[0]
    n = A.shape[1]
    
    B = np.copy(A).astype('float64')

    for i in range(m):
        row = A[i:i+1,:]
        row_magnitude = Magnitude(row.transpose())
        if (row_magnitude != 0):
            for j in range(n):
                B[i,j] = B[i,j]/row_magnitude
    
    return B

def SolveSystem(A,B):
    ''' 
    SolveSystem(A,B)
    BSV:  Accepts both (n,1) and (n,) for B.  Returns shape (n,1)
    
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
    B.shape = (n,1)
    
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



