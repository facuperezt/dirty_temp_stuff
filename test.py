import igraph
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
from collections import Counter
import pandas as pd

df = pd.read_csv("data/index_2")
print(df.head())

dataset = PygLinkPropPredDataset(name="ogbl-citation2", root="resources/dataset/")
graph = dataset[0]
edges = np.array(graph["edge_index"])
split = dataset.get_edge_split()
print(split["valid"])
valid = np.array(split["valid"]["target_node_neg"])
test = np.array(split["test"]["target_node_neg"])
print(valid,test)
print(np.allclose(valid,split))


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