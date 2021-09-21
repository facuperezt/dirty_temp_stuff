import igraph
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

y = np.arange(0,50)
v = torch.randint(50,(50,1))
t = torch.randint(50,(50,1))
l = torch.randint(50,(50,1))


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Error')
plt.ylim(50)
plt.xlim(50)
ax1.plot(y, v,label="Valid MRR")
ax1.plot(y,t,label="Test MRR")
ax2.plot(y, l,label="Trainings Error")
ax1.legend(),ax2.legend()
ax1.grid(True),ax2.grid(True)

plt.show()
"""
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