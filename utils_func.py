import numpy as np
import torch_geometric.utils


def adjMatrix(edges, numNodes, selfLoops=True):
    adj = np.zeros((numNodes, numNodes))

    if selfLoops: adj +=np.identity(numNodes)

    for edge in edges.T:
        adj[edge[0], edge[1]] += 1

    return adj

