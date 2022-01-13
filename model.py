import torch.nn
import torch_geometric as tg
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch.nn.functional import relu
from ogb.linkproppred import Evaluator

import numpy as np
import matplotlib.pyplot as plt

import copy

import utils

import utils_func
import dataLoader
import baselines


# TODO reduce the whole numpy pytorch swapping

class GNN(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()  # from torch documentation TODO look up what it does
        self.input = GCNConv(128, 256, bias=False)
        self.hidden = GCNConv(256, 256, bias=False)
        self.output = GCNConv(256, 256, bias=False)

    def forward(self, x, edge_index):
        h = self.input(x, edge_index)
        X = relu(h)

        h = self.hidden(X, edge_index)
        X = relu(h)

        h = self.output(X, edge_index)
        return h

    def lrp(self, x, edge_index, walk, r_src, r_tar, src, tar, epsilon=0.0, gamma=0.0):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:, :] = cp.lin.weight + gamma * torch.clamp(cp.lin.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        A[0] = x
        A[0] = relu(A[0]).data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0], edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1], edge_index)).data.clone().requires_grad_(True)

        if walk[-1] == tar:
            R[-1] = r_tar
        else:
            R[-1] = r_src

        z = epsilon + roh(self.output).forward(A[2], edge_index)
        z = z[walk[2]]
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[1] = A[2] * c
        R[1] = R[1][walk[1]]

        z = epsilon + roh(self.hidden).forward(A[1], edge_index)
        z = z[walk[1]]
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c
        R[0] = R[0][walk[0]]

        z = epsilon + roh(self.input).forward(A[0], edge_index)
        z = z[walk[1]]
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        test = A[0] * c

        return R[0]


class MLP(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self):
        # build MLP here
        super(MLP, self).__init__()  # from torch documentation TODO look up what it does
        self.input = torch.nn.Linear(256, 256, bias=False)
        self.hidden = torch.nn.Linear(256, 256, bias=False)
        self.output = torch.nn.Linear(256, 1, bias=False)

    def forward(self, src, tar):
        x = src + tar  # TransE model embedding

        h = self.input(x)
        X = relu(h)  # if save : self.a[0] = X

        h = self.hidden(X)
        X = relu(h)
        # if save: self.a[1] = X

        h = self.output(X)
        # if save: self.a[2] = X

        # if save : self.R[-1] = h #TODO is it correct that a and R are the same ????
        return h

    def lrp(self, src, tar, r, epsilon=0.0, gamma=0.0):
        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.weight[:, :] = cp.weight + gamma * torch.clamp(cp.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        R[-1] = r

        src = src.data.clone().requires_grad_(True)
        tar = tar.data.clone().requires_grad_(True)

        A[0] = relu(src + tar)
        A[0] = A[0].data.clone().requires_grad_(True)
        A[1] = relu(self.input(src + tar)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1])).data.clone().requires_grad_(True)

        z = epsilon + roh(self.output).forward(A[2])
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[1] = A[2] * c

        z = epsilon + roh(self.hidden).forward(A[1])
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c

        """
        z = epsilon + roh(self.input).forward(A[0])
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        test = A[0] * c
        """

        z = epsilon + roh(self.input).forward(relu(src + tar))
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        src_grad = src.grad
        tar_grad = tar.grad

        return src * src_grad, tar * tar_grad


def train(batchsize, train_set, valid_set, gnn, mlp, adj, x, rng, optimizer):
    # generating random permutation
    permutation = torch.randperm(train_set["source_node"].shape[0])
    total_loss = []
    num_sample = 0

    for i in range(0, train_set["source_node"].shape[0], batchsize):
        # TODO which optimizer
        optimizer.zero_grad()
        # Set up the batch
        idx = permutation[i:i + batchsize]
        train_src, train_tar = train_set["source_node"][idx], train_set["target_node"][idx]
        # forward passes

        """
        # removing positive link for training
        tmp = adj.to_dense()
        tmp[train_src,train_tar] = 0
        tmp = SparseTensor.from_dense(tmp)
        """

        mid = gnn(x, adj)  # features, edgeindex
        # positive sampling
        out = torch.sigmoid(mlp(mid[train_src], mid[train_tar]))
        pos_loss = - torch.mean(torch.log(out + 1e-15))

        neg_tar = torch.randint(low=0, high=22064, size=train_src.size(), dtype=torch.long)  # 30657
        out = torch.sigmoid(mlp(mid[train_src], mid[neg_tar]))
        neg_loss = torch.log(1 - out + 1e-15)
        neg_loss = - torch.mean(neg_loss)

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize
        # print("Num of samples done: ", num_sample, "/", train_set["source_node"].shape[0], " batch loss :", loss)

    return sum(total_loss) / num_sample


def explains(train_set, test_set, gnn, mlp, adj, x, epoch):
    src, tar = test_set["source_node"], test_set["target_node"]
    walks_all = utils.walks(adj)

    # forward passes
    mid = gnn(x, adj)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    samples = [0, 1, 2, 3, 8, 9]
    for i in samples:
        print(i, src[i])
        walks = utils_func.find_walks(src[i], tar[i], walks_all)
        r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i])
        p = []
        for walk in walks:
            p.append(gnn.lrp(x, adj, walk, r_src, r_tar, src[i], tar[i]))
        utils_func.plot_explain(p, src[i], tar[i], walks, "pos", epoch, i)


@torch.no_grad()
def test(batchsize, data_set, gnn, mlp, edge_index, x, evaluator):
    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]

    mid = gnn(x, edge_index)  # features, edgeindex
    """
    
    #positive sampling
    pos_preds = []
    for i in range(0, data_set["source_node"].shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        pos_preds += [mlp(mid[src[idx]], mid[tar[idx]]).squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0)

    neg_preds = []
    src = src.view(-1, 1).repeat(1, 20).view(-1)
    tar_neg = tar_neg.view(-1)
    for i in range(0, data_set["source_node"].shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        #TODO batching
        neg_preds += [mlp(mid[src], mid[tar_neg]).squeeze().cpu()]
    
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 20)
    """
    pos_preds, neg_preds = [], []
    for i in range(0, src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        # positive sampling
        pos_preds += [mlp(mid[src[idx]], mid[tar[idx]]).squeeze().cpu()]

        # negative sampling
        src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        tar_neg_tmp = tar_neg_tmp.view(-1)
        neg_preds += [mlp(mid[src], mid[tar_neg]).squeeze().cpu()]

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 20)

    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()


def main(batchsize=None, epochs=1, full_dataset=False, explain=False, use_year=False, save=False, load=True):
    # ----------------------- Set up
    # globals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng()

    # loading the data
    if use_year:
        dataset = dataLoader.LinkPredData("data/", use_year)
    else:
        dataset = dataLoader.LinkPredData("data/")

    data = dataset.load()
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    # TODO how to degree matrix lets do out for now
    tmp = data.adj_t.set_diag()
    deg = tmp.sum(dim=0).pow(-0.5)
    deg[deg == float('inf')] = 0
    tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
    data.adj_t = tmp

    # manipulating train for mrr computation
    permutation = torch.randperm(int(np.array(train_set["source_node"].shape[0])))[0:126]
    mrr_train = {"source_node": train_set["source_node"][permutation],
                 "target_node": train_set["target_node"][permutation],
                 "target_node_neg": valid_set["target_node_neg"]
                 }

    # initilaization models
    gnn = GNN()
    mlp = MLP()

    if load == True:
        gnn.load_state_dict(torch.load("model/gnn"))
        mlp.load_state_dict(torch.load("model/mlp"))

    gnn.to(device), mlp.to(device), data.to(device)

    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(mlp.parameters()))
    evaluator = Evaluator(name='ogbl-citation2')

    # adjusting batchsize for full Dtataset
    if batchsize is None:
        batchsize = dataset.num_edges
    if explain:
        # to plot proper plot the the LRP-method we need all walks:
        explain_data = dataset.load(transform=True, explain=True)

    # ----------------------- training & testing
    valid_mrr = torch.zeros((epochs, 1))
    test_mmr = torch.zeros((epochs, 1))
    loss = torch.zeros((epochs, 1))
    old = 0

    for i in range(0, epochs):
        # TODO testing routine
        print(i)
        # loss[i] = train(batchsize, train_set, valid_set, gnn, mlp, data.adj_t, data.x, rng, optimizer).detach()

        test_mmr[i] = test(batchsize, test_set, gnn, mlp, data.adj_t, data.x, evaluator)
        valid_mrr[i] = test(batchsize, valid_set, gnn, mlp, data.adj_t, data.x, evaluator)
        print(test_mmr[i], valid_mrr[i])

        if valid_mrr[i] > old:
            old = valid_mrr[i]
            tmp_gnn = copy.deepcopy(gnn.state_dict())
            tmp_nn = copy.deepcopy(mlp.state_dict())

        if i == epochs - 1 and explain:
            explains(train_set, valid_set, gnn, mlp, explain_data.adj, explain_data.x, i)

    if save:
        torch.save(tmp_gnn, "model/gnn")
        torch.save(tmp_nn, "model/mlp")

    if 2 == 3:
        utils_func.plot_curves(epochs, [valid_mrr, test_mrr, loss],
                               ["Valid MRR", "Test MRR", "Trainings Error"], 'Model Error', )


if __name__ == "__main__":
    main()
