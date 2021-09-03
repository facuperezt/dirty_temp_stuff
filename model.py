import numpy as np
import pandas as pd
import torch.nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, Module
import dataLoader
import LRP_modded


class GNN(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self, data):
        # build GNN here
        super(GNN, self).__init__()  # from torch documentation TODO look up what it does
        self.input = GCNConv(128, 256, normalize=False)
        self.hidden = GCNConv(256, 256, normalize=False)
        self.output = GCNConv(256, 256, normalize=False)
        # self.data =
        # self.edges =

    def prep(self, data):
        # do data transformation adjcent and stuff here
        # self.
        pass

    def forward(self):
        # do computations of layers here
        # TODO x = x_i + x_j why from paper
        # data , edges
        h = self.input()
        X = ReLU(h)

        h = self.hidden(X, )
        X = ReLU(h)

        h = self.output()
        return ReLU(h)

    def lrp(self):
        pass


class MLP(Module, num_predictions=1):  # from torch documentation TODO look up what it does
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self, num_predictions):
        # build MLP here
        super(MLP, self).__init__()  # from torch documentation TODO look up what it does
        self.input = Linear(256, 256)
        self.hidden = Linear(256, 256)
        self.output = (256, num_predictions)

    def forward(self, x):
        # do computations of layers here
        h = self.input(x)
        X = ReLU(h)

        h = self.hidden(X)
        X = ReLU(h)

        h = self.output(X)

        return ReLU(h)

    def lrp(self):
        pass


def train(batchsize, train, valid, gnn, mlp):
    permutation = torch.randperm(train["source"].size[0])

    # how many validation samples we can use to have some for all
    perms = train["source"].size[0] // valid["source"].size[0]

    for i in range(0, train["source"].size[0], batchsize):
        # Set up the batch
        idx = permutation[i:i + batchsize]
        train_src, train_tar = train["source"][idx], train["target"][idx]

        mid = gnn.forward()  # features, edgeindex
        out = mlp.forward()  # ref says src,dst

        # compute error

        # backward pass

        # Set up validation

    # training procedure of GNN&MLP here
    pass


def test():
    # testing procedure of model here
    pass


def mrr(pos, neg):
    arr = np.concatenate(pos, neg)
    lst = arr.tolist().sort(reverse=True)

    return 1 / lst.index(pos)


def main(batchsize=None, epochs=3, full_dataset=False):
    data, split, year = dataLoader.main(full_dataset=full_dataset, use_year=False)
    # TODO Prep data
    gnn = GNN()
    mlp = MLP()
    # call train
    # call test
    pass


if __name__ == "__main__":
    main()
