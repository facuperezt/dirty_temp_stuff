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


# TODO reduce the whole nmumpy python swapping

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

    def lrp(self,x,edge_index,adj,epsilon=0.1,gamma=0.1,):

        def roh(layer):
            return layer + gamma * torch.clamp(layer, min=0)
    
        h = [None] * 3
        R = [None] * 3

        #R[-1] = out

        w1 = list(self.input.parameters())[-1].to(torch.float64)
        w2 = list(self.hidden.parameters())[-1].to(torch.float64)

        h[0] = relu(self.input(x, edge_index))
        h[1] = relu(self.hidden(h[0],edge_index))
        h[2] = relu(self.hidden(h[1],edge_index))

        # z = sum(J sum(jeJ labda h w)
        #s = Rk / z
        # cj sum(k  labda W) *s
        #rj = hj *cJ


        adj.requires_grad_(True)
        walks = utils.walks(adj)

        # R = nn.lrp(g['laplacian'], gamma, t, (j, k))[i].sum() TODO [i].sum()
        """
        R[-1] = r #Todo MEan ???

        A[0] = src + tar
        A[1] = roh(relu(self.input(A[0])))
        A[2] = roh(relu(self.hidden(A[1])))


        z = epsilon + sum(A[2])
        s = R[2] / (z + 1e-15)
        c = sum(torch.matmul(s, z))
        R[1] = A[2] * c

        z = epsilon + sum(A[1])
        s = R[1] / (z + 1e-15)
        c = sum(torch.matmul(s, z))
        R[0] = A[1] * c


        """
        for (i, j, k) in walks:
            print(i,j,k)
            mj = torch.FloatTensor(np.eye(len(adj))[j][:, np.newaxis])
            mk = torch.FloatTensor(np.eye(len(adj))[k][:, np.newaxis])

            w = list(self.input.parameters())[-1].to(torch.float64)
            print(torch.matmul(relu(self.input(x, edge_index)).to(torch.float64),w))
            z = relu(self.input(x, edge_index)).to(torch.float64)
            zp = torch.matmul(adj,torch.matmul(relu(self.input(x, edge_index)).to(torch.float64),roh(w)))
            h = torch.clamp(zp * (z/(zp +1e-15)).data,min=0)
            h = (h*mj) +(h.data *(1-mk))


            """
            z = torch.matmul(adj,h[0].to(torch.float64))
            tmp = roh(list(self.input.parameters())[-1]).to(torch.float64) # First entry should be bias
            p = torch.matmul(z,tmp)
            print(p.size(),z.size())
            q = (p * (z/(p+1e-15)).data).clamp(min=0)
            r = (q * mj) +(q.data *(1-mj))


            z = torch.matmul(adj,h[1])
            p = torch.matmul(z, roh(list(self.hidden.parameters())[-1]))  # First entry should be bias
            q = (p * (z/(p+1e-15)).data).clamp(min=0)
            r = (q * mk) +(q.data *(1-mk))
            
            """

            print(h)
            y = h.mean(dim=0)[1]
            print("Y", y)
            y.backward()

        return adj.data *adj.grad


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

        #TODO RELU Somewhere ??????

        def roh(layer):
            return layer.data + gamma * torch.clamp(layer.data, min=0)

        A = [None] * 3
        R = [None] * 3

        R[-1] = r

        A[0] = src + tar
        A[1] = relu(self.input(A[0]))
        A[2] = relu(self.hidden(A[1]))

        z = epsilon + roh(self.output).forward(A[2])
        s = R[2]/(z+1e-15)
        (z*s.data).sum().backward()
        c = A[2].grad
        R[1] = A[2] * c

        z = epsilon + roh(self.output).forward(A[1])
        s = R[1]/(z+1e-15)
        (z*s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c

        z = epsilon + roh(self.output).forward(A[0])
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        test = A[0] * c

        print(R,test)
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

    p = gnn.lrp(x,edge_index,adj)
    R = mlp.lrp(mid[src], mid[tar], pos_pred)
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
    print(x_adj)


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
