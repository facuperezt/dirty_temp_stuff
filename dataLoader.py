import pandas as pd
import numpy as np
import torch_geometric.data as Data
import torch

def main(full_dataset=False,use_year=False):

    if full_dataset == False:
        # Load all dataset parts
        edges = pd.read_csv("data/Data_small_2")
        features = pd.read_csv("data/Data_small_2_features", header=None)
        test = pd.read_csv("data/Data_small_2_test", header=None)
        #TODO add test and train with torch.load
        valid = torch.load("data/test_save.pt")

        year_path = "data/Data_small_2_node_year"


    else:
        edges = pd.read_csv("data/edge.csv", header=None, names=['source', 'target'])
        features = pd.read_csv("data/node-feat.csv", header=None)
        test = torch.load("data/test.pt")
        train = torch.load("data/train.pt")
        valid = torch.load("data/valid.pt")
        print(train)
        year_path = "data/node_year.csv"

    # edge_index = [2, num nodes] as tensor with [0,:] as source and [1,:] as target TODO verify if i named rows or columns
    edge_index = torch.from_numpy(np.vstack((edges["source"],edges["target"])))
    # features = [num_noides, node_feature] as tensor
    x = features.to_numpy()

    dataset= Data.Data(x,edge_index)
    #TODO turn arrays into tensors
    split = {"train" :train, "valid": valid, "test": test}

    print(dataset)
    if use_year:
        year = pd.read_csv(year_path, header=None)
        return (dataset, split,year)
        #TODO maybe just add to node features...

    return (dataset,split)

if __name__ == "__main__":
    main(full_dataset=False, use_year=False)
