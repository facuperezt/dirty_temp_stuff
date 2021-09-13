import pandas as pd
import numpy as np
import torch_geometric.data as data
import torch


def main(full_dataset=False, use_year=False):
    if not full_dataset:
        # Load all dataset parts
        edges = pd.read_csv("data/Data_small_2reindexd")
        features = pd.read_csv("data/Data_small_2_features")
        valid = torch.load("data/valid_small.pt")
        test = torch.load("data/test_small.pt")
        train = torch.load("data/train_small.pt")

        year_path = "data/Data_small_2_node_year"

    else:
        edges = pd.read_csv("data/edge.csv", header=None, names=['source', 'target'])
        features = pd.read_csv("data/node-feat.csv", header=None)
        test = torch.load("data/test.pt")
        train = torch.load("data/train.pt")
        valid = torch.load("data/valid.pt")

        year_path = "data/node_year.csv"

    # edge_index = [2, num_nodes] as tensor with [0,:] as source and [1,:] as target
    edge_index = torch.from_numpy(np.vstack((edges["source"], edges["target"])))
    # features = [num_nodes, node_feature] as tensor
    x = torch.from_numpy(features.to_numpy())
    x = x.to(torch.float32)  # more for consistency than necessity
    dataset = data.Data(x, edge_index)

    for key in valid.keys():
        valid[key] = torch.from_numpy(valid[key])
    for key in test.keys():
        test[key] = torch.from_numpy(test[key])
    for key in train.keys():
        train[key] = torch.from_numpy(train[key])
    split = {"train": train, "valid": valid, "test": test}

    if use_year:
        year = pd.read_csv(year_path, header=None)
        return dataset, split, torch.from_numpy(year.to_numpy())

    return dataset, split


if __name__ == "__main__":
    main(full_dataset=False, use_year=False)
