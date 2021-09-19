import igraph
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
from collections import Counter
import pandas as pd
rand = torch.randint(12,(12,1))

one = torch.hstack((rand[0:4],torch.ones((4,1))))
two = torch.hstack((rand[4:8],torch.zeros((4,1))))
three = torch.hstack((rand[8:12],torch.zeros((4,1))))
main = torch.zeros((3,4,2))
#print(one,two,three)

main[0,:,:] = one
main[1,:,:] = two
main[2,:,:] = three
print(main)
print(main[0])
#tmp = main[:,main[torch.argsort(main[:,0], descending=True,dim=0)]]
values = []
for row in range(0, 4):
    test = main[:, row, :]
    #print("unsorted",test)
    test = test[torch.argsort(test[:,0], descending=True,dim=0)]
    #test,idx = torch.sort(test, descending=True, dim=0)
    print(test[:,1] ==1)
    maybe = test[:,1] ==1
    values.append(maybe.nonzero())
    #print("sorted",test)
    main[:,row,:] = test
print(values)
print(main)
uzff= main[:,:,0] ==1
print(uzff.nonzero())
"""
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