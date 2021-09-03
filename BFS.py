import pandas as pd
import numpy as np
from collections import Counter
from numpy.random import default_rng
import torch
import dataLoader

rng = default_rng()


data, split, years = dataLoader.main(full_dataset=True, use_year=True)
edges = np.array(data["edge_index"])
x = data["x"]
"""
# times cited :3886, node id: 716145,  mag id: 2258584306
# 2032357
fifo = [2032357]
visited, out = [], []
count = 0

# Simple Breadth First search
while count <= 30000:
    if fifo:
        idx = fifo.pop()
    else:
        # TODO rewrite this one dead end somehow found a cluster and cant leave
        idx = rng.integers(0, x.shape[0], size=1)

    # we add all nodes that cite cur index and nodes that are cited by it
    tmp = [x for x in edges.T if idx == x[0] or idx == x[1]]
    out += tmp

    unique = np.unique(np.array(tmp).flatten())

    unique = [x for x in unique if x not in visited]
    fifo += unique

    visited.append(idx)

    count = len(out)
    print(count, " von min. 30000")
#reindexing so nodefeature references will make sense
edge_small = np.array(out)

df = pd.DataFrame(np.unique(out))
df.to_csv("data/index_2")

index = df.to_numpy()
for x in index:
    oldNodeId = x[1]
    swap = np.where(edge_small==x[1])
    edge_small[swap] = x[0]

pd.DataFrame(edge_small).to_csv("data/Data_small_2")

"""
# source,target
edges = pd.read_csv("data/Data_small_2", index_col='index').to_numpy()
index = pd.read_csv("data/index_2",index_col='index').to_numpy()

features = np.squeeze(x[index].numpy())
year = years[index].numpy().flatten()
pd.DataFrame(features).to_csv("data/Data_small_2_features", index=False)
pd.DataFrame(year).to_csv("data/Data_small_2_node_year", index=False)

# train/valid/test splits : 98/1/1 --> valid & test newest papers
# that we have 22064 nodes : 21600/220/220 roughly

# create helper array with nodeId, year
year_helper = np.vstack((index.flatten(), year))
# use only nodes that are from  2019
# nodeId,year
candidates = np.array([x[0] for x in year_helper.T if x[1] >= 2018])

# -------------------------- test for enough edges
edges_tmp = np.array([x for x in edges if x[0] in candidates])
source = np.array([x[0] for x in Counter(edges_tmp[:, 0]).items() if x[-1] >= 3])

# -------------------------- reshape and save dataset
# for every idx in source collect all edges in edge tmp, random sample 2 in train/valid and remove these from edge pool
valid, test, = [], []
neg_valid, neg_test = [], []
train = edges.tolist()
edges = edges.tolist()
for src in source:
    # find all instances
    idx = [i for i, x in enumerate(train) if x == src]
    # choose two randomly
    v, t = rng.choice(idx, 2)
    valid.append(train[v])
    test.append(train[t])

    neg_tmp = [i for i in edges if i not in train[idx]]
    neg_sample = rng.choice(neg_tmp, 20)  # 20 is arbitrary for testing right now
    neg_valid.append(edges[neg_sample])
    neg_sample = rng.choice(neg_tmp, 20)  # 20 is arbitrary for testing right now
    neg_test.append(edges[neg_sample])

    train = np.delete(train, v, axis=0)
    edges = np.delete(train, t, axis=0)

# TODO check if correct
neg_valid = np.array(neg_valid)[:, :, 1]
neg_test = np.array(neg_test)[:, :, 1]

valid_dict = {"source": np.array(valid)[:, 0], "target": np.array(valid)[:, 1], "target_neg": neg_valid}
test_dict = {"source": np.array(test)[:, 0], "target": np.array(test)[:, 1], "target_neg": neg_test}
train_dict = {"source": edges[:, 0], "target": edges[:, 1]}
torch.save(valid_dict, "data/valid_small.pt")
torch.save(test_dict, "data/test_small.pt")
torch.save(train_dict, "data/train_small.pt")
