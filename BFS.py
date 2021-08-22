from ogb.linkproppred import PygLinkPropPredDataset
import pandas as pd
import numpy as np
from collections import Counter


dataset = PygLinkPropPredDataset(name="ogbl-citation2", root="resources/dataset/")
graph = dataset[0]
edges = np.array(graph["edge_index"])

# times cited :3886, node id: 716145,  mag id: 2258584306
fifo = [716145]
visited, out = [],[]
l = 0

# Simple Breadth First search
while l <= 100000:
    idx = fifo.pop()
    print(idx)
    tmp = [x for x in edges.T if idx == x[0] or idx ==x[1] ] # we add all nodes that cite cur index and nodes that are cited by it
    #print(tmp)
    out += tmp
    #print(out)

    tmp = np.array(tmp).flatten()
    unique = set(tmp)

    unique = [x for x in unique if x not in visited]
    fifo += unique
    #print(fifo)
    visited.append(idx)

    l = len(out)
    print(l)

#print(out)