import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, Module


class GNN(Module):  # from torch documentation TODO look up what it does
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()  # from torch documentation TODO look up what it does
        self.input = GCNConv(128, 256, normalize=False)
        self.hidden = GCNConv(256, 256, normalize=False)
        self.output = GCNConv(256, 256, normalize=False)
        self.data = data
        self.edges =

    def prep(self, data):
        # do data transformation adjcent and stuff here
        # self.
        pass

    def forward(self):
        # do computations of layers here
        # TODO x = x_i + x_j why from paper
                    #data , edges
        h = self.input()
        X = ReLU(h)

        h = self.hidden(X, )
        X = ReLU(h)

        h = self.output()
        return ReLU(h)


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

    def forward(self):
        # do computations of layers here
        h = self.input()
        X = ReLU(h)

        h = self.hidden(X, )
        X = ReLU(h)

        h = self.output

        return ReLU(h)


def train():
    # training procedure of GNN&MLP here
    pass


def test():
    # testing procedure of model here
    pass


def main(batchsize=None, epochs=3):
    # call data loader
    # initialize GNN & MLP
    # call train
    # call test
    pass


if __name__ == "__main__":
    main()
