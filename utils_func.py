import numpy as np

def adjMatrix(edges, numNodes, selfLoops=True):
    """
    Function to calculate Adjacency MMatrix given an edge Array
    :param edges:  Dimensions 2 x m with the fst of the two being source and the snd being target
    :param numNodes: Number of different nodes in the graph, used to create correctly sized matrix
    :param selfLoops: bool to decide weather to add selfloops or not
    :return: adjacency Matrix with size numNodes x numNodes
    """
    adj = np.zeros((numNodes, numNodes))

    if selfLoops: adj += np.identity(numNodes)

    for edge in edges.T:
        adj[edge[0], edge[1]] += 1

    return adj

def degMatrix(adj_t):
    deg = np.zeros(adj_t.shape)
    for column in range(adj_t.shape[1]):
        deg[column][column] += np.sum(adj_t[:,column])

    return deg