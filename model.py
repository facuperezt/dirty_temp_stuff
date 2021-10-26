import numpy as np
import torch.nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn.functional import relu
import dataLoader
import utils_func
import utils_func as util
import LRP_modded
import utils
import matplotlib.pyplot as plt
import copy

# TODO reduce the whole nmumpy pytorch swapping

class GNN(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()  # from torch documentation TODO look up what it does
        self.input = GCNConv(128, 256)
        self.hidden = GCNConv(256, 256)
        self.output = GCNConv(256, 256)

    def forward(self, x, edge_index):
        h = self.input(x, edge_index)
        X = relu(h)

        h = self.hidden(X, edge_index)
        X = relu(h)

        h = self.output(X, edge_index)
        return h

    def lrp(self,x,edge_index,walks,r,epsilon=0.1,gamma=0.1,):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:,:] = cp.lin.weight + gamma * torch.clamp(cp.lin.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        R[-1] = r

        A[0] = x
        A[0] = relu(A[0]).data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0],edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1],edge_index)).data.clone().requires_grad_(True)

        for walk in walks:

            R[-1] = r

            z = epsilon + roh(self.output).forward(A[2],edge_index)
            z = z[walk[2]]
            R[2]= R[2][walk[2]]
            s = R[2]/(z+1e-15)
            (z*s.data).sum().backward()
            c = A[2].grad
            R[1] = A[2] * c
            R[1] = R[1][walk[1]]

            z = epsilon + roh(self.hidden).forward(A[1],edge_index)
            z = z[walk[1]]
            #R[1] = R[1][walk[1]]
            s = R[1]/(z+1e-15)
            (z*s.data).sum().backward()
            c = A[1].grad
            R[0] = A[1] * c
            R[0] = R[0][walk[0]]

            z = epsilon + roh(self.input).forward(A[0],edge_index)
            z = z[walk[1]]
            #R[0] = R[0][walk[0]]
            s = R[0] / (z + 1e-15)
            (z * s.data).sum().backward()
            c = A[0].grad
            test = A[0] * c

            print("done walk")
        return R

class MLP(torch.nn.Module):  # from torch documentation TODO look up what it does
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self, batchsize):
        # build MLP here
        super(MLP, self).__init__()  # from torch documentation TODO look up what it does
        self.input = Linear(256, 256)
        self.hidden = Linear(256, 256)
        self.output = Linear(256, 1)

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

    def lrp(self, src, tar, r, epsilon=0.1, gamma=0.1):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.weight[:,:] = cp.weight + gamma * torch.clamp(cp.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        R[-1] = r

        A[0] = src + tar
        A[0] = relu(A[0]).data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0])).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1])).data.clone().requires_grad_(True)

        z = epsilon + roh(self.output).forward(A[2])
        s = R[2]/(z+1e-15)
        (z*s.data).sum().backward()
        c = A[2].grad
        R[1] = A[2] * c

        z = epsilon + roh(self.hidden).forward(A[1])
        s = R[1]/(z+1e-15)
        (z*s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c

        z = epsilon + roh(self.input).forward(A[0])
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        test = A[0] * c

        #print(test)
        #print(R)
        return R


def train(batchsize, train_set, valid_set, gnn, mlp, adj, x, rng, optimizer):
    # generating random permutaion
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
        mid = gnn(x, adj)  # features, edgeindex

        # positive sampling
        out = torch.sigmoid(mlp(mid[train_src], mid[train_tar]))  # ref says src,dst
        pos_loss = - torch.mean(torch.log(out + 1e-15))  # TODO  why inf if not added 1e-15

        # TODO shortend and make mor efficient
        # negativ sampling TODO --> do we care if we hit double ?
        neg_tar = rng.choice(x.shape[0], (train_src.shape[0], 20))
        neg_loss = torch.zeros((train_src.shape[0], 1))
        for i in range(neg_tar.shape[1]):
            out = torch.sigmoid(mlp(mid[train_src], mid[neg_tar[:, i]]))  # ref says src,dst
            neg_loss += torch.log(1 - out + 1e-15) / 20

        neg_loss = - torch.mean(neg_loss)

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize
        print("Num of samples done: ", num_sample, "/", train_set["source_node"].shape[0], " batch loss :", loss)

    return sum(total_loss) / num_sample

def explains(train_set, test_set, gnn, mlp, edge_index, x,adj):

    src, tar = test_set["source_node"], test_set["target_node"]
    train_src, train_tar = train_set["source_node"], train_set["target_node"]

    walks_all = utils.walks(adj)

    helper = np.vstack((train_src,train_tar)).T.tolist()
    tmp = np.vstack((src,tar)).T.tolist()
    idx = [helper.index(tmp[n]) for n in range(len(tmp))]
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[train_src], mid[train_tar])

    p = []

    for node in range(len(idx)):
        walks = utils_func.find_walks(train_src[node],train_tar[node],walks_all)
        p += gnn.lrp(x, edge_index, walks,mid)
        R = mlp.lrp(mid[train_src[node]], mid[train_tar[node]], pos_pred)
    return R

@torch.no_grad()
def test(batchsize, data_set, gnn, mlp, edge_index, x,adj):
    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    total_loss = []
    num_sample = 0

    for i in range(0, data_set["source_node"].shape[0], batchsize):
        # Set up the batch
        idx = permutation[i:i + batchsize]
        src, tar, tar_neg = data_set["source_node"][idx], data_set["target_node"][idx], data_set["target_node_neg"][idx]
        # forward passes
        mid = gnn(x, edge_index)  # features, edgeindex
        print(src.size())
        # positive sampling
        pos_pred = torch.sigmoid(mlp(mid[src], mid[tar])).detach()
        tmp = torch.ones(pos_pred.size())
        pos_helper = torch.hstack((pos_pred, tmp))

        tmp = torch.zeros((21, 126, 2))
        tmp[0, :, :] = pos_helper
        # neg sampling
        for i in range(tar_neg.shape[1]):
            prediction = torch.sigmoid(mlp(mid[src], mid[tar_neg[:, i]])).detach()
            neg_helper = torch.hstack((prediction, torch.zeros(prediction.size())))
            tmp[i + 1, :, :] = neg_helper

        values = []
        # TODO shortend and make mor efficient
        for row in range(0, 22):
            tmp2 = tmp[:, row, :]
            tmp2 = tmp2[torch.argsort(tmp2[:, 0], descending=True, dim=0)]
            maybe = tmp2[:, 1] == 1
            values.append(maybe.nonzero())
            tmp[:, row, :] = tmp2

        for z in range(len(values)):
            values[z] = 1 / (values[z] + 1)
        val = sum(values) / 126
        print(val)

    return (tmp, val)


def main(batchsize=None, epochs=1, full_dataset=False, explain=False, use_year=False):
    # ----------------------- Set up
    # global stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng()

    # loading the data
    if use_year:
        data, split, year = dataLoader.main(full_dataset=full_dataset, use_year=use_year)
    else:
        data, split = dataLoader.main(full_dataset=full_dataset, use_year=use_year)

    x, edges = data["x"], data["edge_index"]

    x_adj = util.adjMatrix(edges, x.shape[0])
    x_adj = torch.from_numpy(x_adj)


    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    # manipulating train for mrr computation
    permutation = torch.randperm(int(np.array(train_set["source_node"].shape[0])))[0:126]
    mrr_train = {"source_node": train_set["source_node"][permutation],
                 "target_node": train_set["target_node"][permutation],
                 "target_node_neg": valid_set["target_node_neg"]
                 }

    # initilaization models
    gnn = GNN()
    mlp = MLP(batchsize)
    gnn.to(device), mlp.to(device), data.to(device)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(mlp.parameters()))
    # adjusting batchsize for full Dtataset
    if batchsize is None:
        batchsize = edges.shape[1]


    # initilization of LRP
    if explain:
        graph = {"layout": utils.layout(x_adj, None), 'adjacency': x_adj, "target": 1}
        LRP_modded.plot([1], graph)

    # ----------------------- training & testing
    old = 0
    v = torch.zeros((50, 1))
    t = torch.zeros((50, 1))
    l = torch.zeros((50, 1))
    for i in range(0, epochs):

        loss = train(batchsize, train_set, valid_set, gnn, mlp, edges, x, rng, optimizer)

        # TODO logging for less blackbox
        R = explains(train_set, mrr_train, gnn, mlp, edges, x,x_adj)


        train_mmr = test(batchsize, mrr_train, gnn, mlp, edges, x,x_adj)
        test_mmr = test(batchsize, test_set, gnn, mlp, edges, x,x_adj)
        valid_mrr = test(batchsize, valid_set, gnn, mlp, edges, x,x_adj)
        # TODO logging for less blackbox

        old = test_mmr[1]
        # TODO stopping criteria
        if valid_mrr[1] >= 50:
            print("Reached treshold ")
            break

        l[i] = loss.detach()
        v[i] = (valid_mrr[1].detach())
        t[i] = (test_mmr[1].detach())

    if 2 == 3 :
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all")
        fig.suptitle('Model Error')
        plt.xlim(50)
        y = np.arange(0, 50)
        ax1.plot(y, v, label="Valid MRR")
        ax1.plot(y, t, label="Test MRR")
        ax2.plot(y, l, label="Trainings Error")
        ax1.legend(), ax2.legend()
        ax1.grid(True), ax2.grid(True)
        plt.xlim(50)
        plt.savefig("plots/errors.pdf")
        plt.show()


if __name__ == "__main__":
    main()
