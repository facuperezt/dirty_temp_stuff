import pandas as pd
import numpy as np
import torch_geometric.data as Data

def main(full_dataset=False):

    if full_dataset == False:
        # Load all dataset parts
        edges = pd.read_csv("data/Data_small_2", header=None, names=['target', 'source'])
        print(edges.head())
        year = pd.read_csv("data/Data_small_2_node_year", header=None)
        features = pd.read_csv("data/Data_small_2_features", header=None)
        test = pd.read_csv("data/Data_small_2_test", header=None)
        valid = pd.read_csv("data/Data_small_2_valid", header=None)


    else:
        edges = pd.read_csv("data/edge.csv", header=None, names=['target', 'source'])
        print(edges.head())
        year = pd.read_csv("data/node_year.csv", header=None)
        features = pd.read_csv("data/node-feat.csv", header=None)
        # train, valid and test to be implemented

# edge_index = [2, num nodes] as tensor with COO format -->
    edge_index = np.vstack((edges["target"],edges["source"]))
    print(edge_index)
    # x = [num_noides, node_feature] as tensor
    data = Data.data(x,edge_index)
    return data
if __name__ == "__main__":
    main(full_dataset=False)
