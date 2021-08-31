from ogb.linkproppred import PygLinkPropPredDataset
import pandas as pd
import numpy as np
from collections import Counter
from numpy.random import default_rng
import torch
dataset = PygLinkPropPredDataset(name="ogbl-citation2", root="resources/dataset/")
graph = dataset[0]
edges = np.array(graph["edge_index"])

"""
# times cited :3886, node id: 716145,  mag id: 2258584306
#2032357
fifo = [2032357]
visited, out = [],[]
l = 0

# Simple Breadth First search
while l <= 30000:
    if fifo :
        idx = fifo.pop()
    else :
        idx = np.random.randint(0,graph.num_nodes+1) # TODO rewrite this one dead end somehow found a cluster and cant leave
    print(idx)
    tmp = [x for x in edges.T if idx == x[0] or idx ==x[1] ] # we add all nodes that cite cur index and nodes that are cited by it
    #print(tmp)
    out += tmp
    #print(out)

    unique = np.array(tmp).flatten()
    unique = np.unique(unique)


    unique = [x for x in unique if x not in visited]
    fifo += unique
    #print(fifo)
    visited.append(idx)

    l = len(out)
    print(l)

df = pd.DataFrame(np.array(out))
df.to_csv("data/Data_small_2")

df = pd.DataFrame(np.unique(out))
df.to_csv("data/index_2")

"""
# source,target
edges = np.array(pd.read_csv("data/Data_small_2",index_col='index'))
index = np.array(pd.read_csv("data/index_2",index_col='index' ))

features = np.squeeze(graph["x"][index].numpy())
year = graph["node_year"][index].numpy().flatten()

# train/valid/test splits : 98/1/1 --> valid & test newest papers
# that we have 22064 nodes : 21600/220/220 roughly

# create helper array with nodeid, year
year_helper = np.vstack((index.flatten(),year))
# use only nodes that are from  2019
# nodeId,year
candidates = np.array([x[0] for x in year_helper.T if x[1] >= 2018])


# -------------------------- test for enough edges
edges_tmp = np.array([x for x in edges if x[0] in candidates])
source = np.array([x[0] for x in Counter(edges_tmp[:,0]).items() if x[-1] >= 3])

# -------------------------- reshape and save dataset
# for everx index in source collect all edges in edge tmp, random sample 2 in train/valid and remove these from edge pool
rng = default_rng()
valid, test,=[],[]
neg_valid, neg_test = [],[]
print(edges.shape)
for src in source:
    tmp = edges[:, 0].tolist()

    # find all instances
    idx = [i for i,x in enumerate(tmp) if x == src]
    # choose two randomly
    v,t = rng.choice(idx,2)
    valid.append(edges[v])
    test.append(edges[t])

    # TODO redo because we also ignore already sampled for other sources
    neg_tmp = [i for i in range(len(tmp)) if i not in idx]
    neg_sample = rng.choice(neg_tmp, 20) #20 is arbiteray for testing right now
    neg_valid.append(edges[neg_sample])
    neg_sample = rng.choice(neg_tmp, 20) #20 is arbiteray for testing right now
    neg_test.append(edges[neg_sample])

    edges = np.delete(edges,(v),axis=0)
    edges = np.delete(edges, (t),axis=0)

#TODO check if correct 
neg_valid = np.array(neg_valid)[:,:,1]
neg_test = np.array(neg_test)[:,:,1]

#TODO add test and train
t = np.array(valid)[:,0]
v = np.array(valid)[:,1]
print(t,v)

valid_dict = {"source":t, "target":v, "target_neg":neg_valid}
torch.save(valid_dict,"data/test_save.pt")

pd.DataFrame(np.array(valid)).to_csv("data/Data_small_2_valid",header=['source','target'])
pd.DataFrame(np.array(test)).to_csv("data/Data_small_2_test",header=['source','target'])
pd.DataFrame(np.array(neg_valid)).to_csv("data/Data_small_2_neg_valid")
pd.DataFrame(np.array(neg_test)).to_csv("data/Data_small_2_neg_test")
pd.DataFrame(features).to_csv("data/Data_small_2_features")
pd.DataFrame(year).to_csv("data/Data_small_2_node_year")


