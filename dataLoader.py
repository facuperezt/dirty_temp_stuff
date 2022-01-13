import pandas as pd
import numpy as np
from  torch_geometric import data, transforms,utils
import torch

class LinkPredData():
    #TODO repair use year and add self. attributes
    def __init__(self,root_dir,transform=transforms.ToSparseTensor(), use_year=False, use_small=True,):

        self.root_dir = root_dir
        self.transform = transform
        self.use_small = use_small
        self.use_year = use_year

        #TODO  dataset features



    def __len__(self):
        return self.dataset["x"].shape[0]

    def load(self, transform=True,explain=False,other=None):
        #TODO other ?
        if other:
            pass
        else:
            if self.use_small:
                # Load all dataset parts
                if explain:  # WHY did i do this again ????
                    edges = pd.read_csv(self.root_dir+"Data_small_2reindexd", names=['source', 'target'])
                else:
                    edges = pd.read_csv(self.root_dir+"Data_small_edgeIndex", names=['source', 'target'])
                features = pd.read_csv(self.root_dir+"Data_small_2_features")

                year_path = self.root_dir+"Data_small_2_node_year"

            else:
                edges = pd.read_csv(self.root_dir+"edge.csv", header=None, names=['source', 'target'])
                features = pd.read_csv(self.root_dir+"node-feat.csv", header=None)
                year_path = self.root_dir+"node_year.csv"

            edge_index = torch.from_numpy(np.vstack((edges["source"], edges["target"])))
            # features = [num_nodes, node_feature] as tensor
            x = torch.from_numpy(features.to_numpy())
            x = x.to(torch.float32)  # more for consistency than necessity
            dataset = data.Data(x, edge_index)

        self.num_edges = edge_index.shape[1]

        if transform :
            return self.transform(dataset)
        else: return dataset

    def get_edge_split(self,):

        if self.use_small:
            valid = torch.load(self.root_dir+"valid_small.pt")
            test = torch.load(self.root_dir+"test_small.pt")
            train = torch.load(self.root_dir+"train_small.pt")
        else :
            test = torch.load(self.root_dir+"test.pt")
            train = torch.load(self.root_dir+"train.pt")
            valid = torch.load(self.root_dir+"valid.pt")

        for key in valid.keys():
            valid[key] = torch.from_numpy(valid[key])
        for key in test.keys():
            test[key] = torch.from_numpy(test[key])
        for key in train.keys():
            train[key] = torch.from_numpy(train[key])

        split = {"train": train, "valid": valid, "test": test}

        return split

def main():
    dataset = LinkPredData("data/")
    data = dataset.load(transform=False,explain=True)
    print(data.x.shape, data.edge_index.shape)

if __name__ == "__main__":
    main()