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
adj = utils_func.adjMatrix(edges,size)
print(edges.size())
print(adj)

test = np.array([[0,1],[1,0],[1,2],[2,1],[3,1],[1,3]])
print(utils_func.adjMatrix(test.T,4,False))
print(utils_func.adjMatrix(test.T,4))
print(test.shape)

graph = {}
adj = np.array([[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,1,0,0]])
layout = utils.layout(adj,None)
graph["layout"] = layout
graph['adjacency'] = adj
graph["target"] = 1
LRP_modded.plot([1],graph)