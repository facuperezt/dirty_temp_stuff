import numpy as np

import LRP_modded
import dataLoader
import utils
import utils_func

data = dataLoader.main(full_dataset=False)
x, edges = data
data = data[0]
size = data["x"].shape[0]
edges = data["edge_index"]
#edges = np.array([[0,1],[1,0],[1,2],[2,1],[1,3],[3,1]])
#print(edges)
adj = utils_func.adjMatrix(edges,size)
#print(adj)
#deg = utils_func.degMatrix(adj)
#deg_inv_sqrt = deg ** (-0.5)
#deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#print(deg_inv_sqrt)
#print(np.matmul(np.matmul(deg_inv_sqrt,adj),deg_inv_sqrt))
#print(deg_inv_sqrt*adj*deg_inv_sqrt)


graph = {}
layout = utils.layout(adj,None)
graph["layout"] = layout
graph['adjacency'] = adj
graph["target"] = 1
LRP_modded.plot([1],graph)
