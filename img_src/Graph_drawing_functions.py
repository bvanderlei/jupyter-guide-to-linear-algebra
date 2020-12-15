# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:22:19 2020

@author: Ben Vanderlei
"""

import matplotlib.pyplot as plt
import networkx as nx

def DrawGraph(A, pos = None):
    '''
    Draws a directed graph based on adjacency matrix A.

    Parameters
    ----------
    A : NumPy array object.

    Returns
    -------
    pos: dictionary of node positions

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
    
    options = {"with_labels": True,"font_size":20}
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
    plt.figure(figsize=(8,8))
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
