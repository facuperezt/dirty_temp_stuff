import numpy as np
import torch.nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU
import dataLoader
import utils_func as util
import LRP_modded



class GNN(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()  # from torch documentation TODO look up what it does
        self.input = GCNConv(128, 256, normalize=False)
        self.hidden = GCNConv(256, 256, normalize=False)
        self.output = GCNConv(256, 256, normalize=False)

    def forward(self,x,x_adj):
        # do computations of layers here
        # TODO x = x_i + x_j why from paper
        h = self.input(x,x_adj)
        X = ReLU(h)

        h = self.hidden(X,x_adj)
        X = ReLU(h)

        h = self.output(X,x_adj)
        return ReLU(h)

    def lrp(self):
        pass


class MLP(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self):
        # build MLP here
        super(MLP, self).__init__()  # from torch documentation TODO look up what it does
        self.input = Linear(256, 256)
        self.hidden = Linear(256, 256)
        self.output = (256, 1)


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


def train(batchsize, train_set, valid_set, gnn, mlp):
    permutation = torch.randperm(train_set["source"].size[0])

    # how many validation samples we can use to have some for all
    perms = train_set["source"].size[0] // valid_set["source"].size[0]

    for i in range(0, train_set["source"].size[0], batchsize):
        # Set up the batch
        idx = permutation[i:i + batchsize]
        train_src, train_tar = train_set["source"][idx], train_set["target"][idx]

        mid = gnn.forward()  # features, edgeindex
        out = mlp.forward()  # ref says src,dst

        # compute error

        # backward pass

        # Set up validation

    # training procedure of GNN&MLP here
    pass

@torch.no_grad
def test(batchsize, test_set, gnn, mlp):
    # testing procedure of model here

    #TODO use prefixed evaluator ?
    pass



#TODO rerad paper before finish this one
def mrr(pos, neg):
    arr = np.concatenate(pos, neg)
    lst = arr.tolist().sort(reverse=True)

    return 1 / lst.index(pos)


def main(batchsize=None, epochs=3, full_dataset=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, split, year = dataLoader.main(full_dataset=full_dataset, use_year=False)
    x , edges = data
    gnn = GNN()
    mlp = MLP()
    gnn.to(device),mlp.to(device),data.to(device),split.to(device)

    adj = util.adjMatrix(edges,x.size[0])

    if batchsize is None :
        batchsize = x.size()[0]


    for epoch in epochs:

        train(batchsize,split["train"],split["valid"],gnn,mlp)
        #TODO logging for less blackbox
        loss =  test(batchsize,split["test"],gnn, mlp)
        # TODO logging for less blackbox

        #TODO stopping criteria
        if loss <= 0.01 :
            break

if __name__ == "__main__":
    main()
