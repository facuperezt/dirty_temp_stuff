import numpy as np

import LRP_modded
import dataLoader
import utils

data = dataLoader.main(full_dataset=False)
data = data[0]
size = data["x"].shape[0]
edges = data["edge_index"]
array = np.arrange(0,size,1)
adj = np.zeros([size,size])
for
print(adj)


graph = {}
adj = np.array([[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,1,0,0]])
layout = utils.layout(adj,None)
graph["layout"] = layout
graph['adjacency'] = adj
graph["target"] = 1
LRP_modded.plot([1],graph)