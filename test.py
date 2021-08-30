import igraph
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
dataset = PygLinkPropPredDataset(name="ogbl-citation2", root="resources/dataset/")
graph = dataset[0]
edges = np.array(graph["edge_index"])
split = dataset.get_edge_split()
print(graph["edge_index"])
train = split["valid"]
print(train)
train = split["test"]
print(train)
train_s = np.array(train["source_node"])
train_t = np.array(train["target_node"])
train = np.vstack((train_s,train_t))
v1 = 2302382
v2 = 245742
print(train_t)
print(train.T)
"""
print(edges)
print(train)
print(train.shape,edges.shape)
print(np.all(edges==train))
"""

"""
graph = igraph.Graph()
graph['name'] ="test"
print(graph)

graph.add_vertex()
print(graph)

graph.add_vertex()
graph.add_edge(1,0)

print(graph)
print(graph.DictList())
"""