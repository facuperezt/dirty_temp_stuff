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


graph = {}
layout = utils.layout(adj,None)
graph["layout"] = layout
graph['adjacency'] = adj
graph["target"] = 1
LRP_modded.plot([1],graph)