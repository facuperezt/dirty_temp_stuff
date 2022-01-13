import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.nn.functional import relu
from ogb.linkproppred import Evaluator
import dataLoader
import utils_func
import matplotlib.pyplot as plt
import copy


class MLP_base(torch.nn.Module):

    def __init__(self):
        super(MLP_base, self).__init__()
        self.input = torch.nn.Linear(768, 768)
        self.hidden = torch.nn.Linear(768, 768)
        self.output = torch.nn.Linear(768, 1)

    def forward(self, x):
        h = self.input(x)

        X = relu(h)
        h = self.hidden(X)

        X = relu(h)
        h = self.output(X)

        return torch.sigmoid(h)


def cn(src, tar, adj):

    src = torch.reshape(torch.cat((adj[src],adj[:,src].T)),(1,adj.shape[0]*2))
    tar = torch.reshape(torch.cat((adj[tar], adj[:, tar].T)),(1,adj.shape[0]*2))

    src = torch.nonzero(src)
    tar = torch.nonzero(tar)
    print(src,tar)
    if src.shape[0]==0 or tar.shape[0]==0:
        return torch.Tensor(0)
    src = src.numpy()[:,1]
    tar = tar.numpy()[:,1]
    res = torch.Tensor(np.intersect1d(src,tar).shape[0])
    if res.shape[0]==0 :
        return 0
    """
    #TODO directed neighbouring counting
    print()
    def nonzero(arr):
        res = [np.nonzero(arr[x].flatten().numpy()) if torch.count_nonzero(arr[x]) > 0 else np.array([[-1]]) for x in
               arr]
        print("nonzer",res)
        return res
    
    def intersect_len(arr):
        res = [len(np.intersect1d(arr[row, 0], arr[row, 1])) if (arr[row, 0][0] != -1) or (arr[row, 1][0] != -1) else 0 for
               row in range(arr.shape[0])]
        print("inter",res)
        return torch.from_numpy(np.asarray(res))

    res = np.hstack((nonzero(adj[src]),nonzero(adj[tar])))
    print(res)
    return intersect_len(res)

    l = src.shape[0]
    print(l)
    src = torch.reshape(torch.cat((adj[src],adj[:,src].T)),(2,adj.shape[0]*2))
    tar = torch.reshape(torch.cat((adj[tar], adj[:, tar].T)),(2,adj.shape[0]*2))
    print(src)
    if torch.count_nonzero(src) > 0 :
        src = torch.nonzero(src)
    else: src = -1 #maybe just break i guess
    if torch.count_nonzero(tar) > 0:
        tar = torch.nonzero(tar)
    else:
        src = -1  # maybe just break i guess

    src_idx = [max(torch.nonzero(src[:,0]==i))for i in range(l)]
    tar_idx = [max(torch.nonzero(tar[:,0]==i))for i in range(l)]
    res = []
    old = (0,0)
    for i,j in src_idx,tar_idx:
        if i ==[] or j ==[]:
            res +=0
        else:
            res += np.intersect1d(src[old[0]:i,1],tar[old[1]:j,1]).shape[1]
    print(res)

    print(src,src_idx)
    """
    return res
def test_cn(evaluator, data_set, batchsize, adj):
    # TODO efficient batching maybe
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]
    permutation = torch.randperm(src.shape[0])
    print(src.shape)
    pos_preds, neg_preds = [], []

    for i in range(src.shape[0]):
        # positive sampling
        pos_preds += [cn(src[i], tar[i], adj)]
        #print(i)
        # negative sampling
        #src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        #tar_neg_tmp = tar_neg_tmp.view(-1)
        for j in range(0,20):
            #print(j)
            neg_preds += [cn(src[i], tar_neg[i,j], adj)]
    print(len(neg_preds))
    """    
    for i in range(0, src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        # positive sampling
        pos_preds += [cn(src_tmp, tar_tmp, adj).squeeze().cpu()]

        # negative sampling
        src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        tar_neg_tmp = tar_neg_tmp.view(-1)
        neg_preds += [cn(src_tmp, tar_neg_tmp, adj).squeeze().cpu()]
    """
    pos_pred = torch.cat(pos_preds, dim=0)
    print(torch.cat(neg_preds),neg_preds)
    neg_pred = torch.cat(neg_preds).view(-1, 20)
    print(evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'])
    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()


def create_Dataset(adj, data):
    adj = adj.astype(dtype=float)
    adj -= np.identity(data.shape[0], dtype=float)
    save = np.zeros([data.shape[0], 256])
    for i in range(adj.shape[0]):
        one = 0
        two = 0
        for h1 in np.where(adj[i]):
            for value in h1:
                one += data[value]
                for h2 in np.where(adj[value]):
                    for tmp in h2:
                        two += data[tmp]
                save[i, 128:256] = two
        save[i, 0:128] = one

    pd.DataFrame(save).to_csv("data/baseline_NN", index=False)
    return save


def train(optimizer, train_set, rep, data, mlp, batchsize):
    optimizer.zero_grad()
    train_src, train_tar = train_set["source_node"], train_set["target_node"]
    total_loss = []
    num_sample = 0
    permutation = torch.randperm(train_src.shape[0])

    def helper(rep, data, src, tar):
        x = torch.cat((rep[src, 128:256], rep[src, 0:128], data[src], data[tar], rep[tar, 0:128], rep[tar, 128:256]))
        return torch.reshape(x, (src.shape[0], 128 * 6)).to(torch.float32)

    for i in range(0, train_src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src = train_src[idx]
        tar = train_tar[idx]
        tar_neg = torch.randint(low=0, high=22064, size=tar.size(), dtype=torch.int64)

        # forward passes
        # positive sampling
        out = mlp(helper(rep, data, src, tar))
        pos_loss = - torch.mean(torch.log(out + 1e-15))

        # negative sampling
        out = mlp(helper(rep, data, src, tar_neg))
        neg_loss = torch.log(1 - out + 1e-15)
        neg_loss = - torch.mean(neg_loss)

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize

    print(sum(total_loss) / num_sample)

    return sum(total_loss) / num_sample


@torch.no_grad()
def test(evaluator, data_set, mlp, rep, data, batchsize):
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]
    permutation = torch.randperm(src.shape[0])

    pos_preds, neg_preds = [], []

    def helper(rep, data, src, tar):
        x = torch.cat((rep[src, 128:256], rep[src, 0:128], data[src], data[tar], rep[tar, 0:128], rep[tar, 128:256]))
        return torch.reshape(x, (src.shape[0], 128 * 6)).to(torch.float32)

    for i in range(0, src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        # positive sampling
        pos_preds += [mlp(helper(rep, data, src_tmp, tar_tmp)).squeeze().cpu()]

        # negative sampling
        src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        tar_neg_tmp = tar_neg_tmp.view(-1)
        neg_preds += [mlp(helper(rep, data, src_tmp, tar_neg_tmp)).squeeze().cpu()]
    print(neg_preds[0].shape,neg_preds)
    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 20)

    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()


def runNN(create_dataset, epochs, load, save, batchsize=None):
    dataset = dataLoader.LinkPredData("data/")
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if create_dataset:
        data = dataset.load(transform=False)
        x_adj = utils_func.adjMatrix(data.edge_index, data.x.shape[0], selfLoops=False)
        rep = torch.from_numpy(create_Dataset(x_adj, data.x))
    else:
        data = dataset.load(transform=True)
        rep = torch.from_numpy(np.asarray(pd.read_csv("data/baseline_NN")))

    # initilaization models
    NN = MLP_base()
    if batchsize is None:
        batchsize = dataset.num_edges

    if load:
        NN.load_state_dict(torch.load("model/NN_baseline"))

    NN.to(device), data.to(device)
    optimizer = torch.optim.Adam(list(NN.parameters()))

    evaluator = Evaluator(name='ogbl-citation2')

    # ----------------------- training & testing
    valid_mrr = torch.zeros((epochs, 1))
    test_mmr = torch.zeros((epochs, 1))
    loss = torch.zeros((epochs, 1))
    old = 0
    for i in range(0, epochs):
        loss[i] = train(optimizer, train_set, rep, data.x, NN, batchsize).detach()

        test_mmr[i] = test(evaluator, test_set, NN, rep, data.x, batchsize)
        valid_mrr[i] = test(evaluator, valid_set, NN, rep, data.x, batchsize)

        if valid_mrr[i] > old:
            tmp_nn = copy.deepcopy(NN.state_dict())

    if save:
        torch.save(tmp_nn, "model/NN_baseline")

    if 2 == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all")
        fig.suptitle('Model Error')
        plt.xlim(epochs + 5)
        y = np.arange(0, epochs)
        ax1.plot(y, valid_mrr, label="Valid MRR")
        ax1.plot(y, test_mmr, label="Test MRR")
        ax2.plot(y, loss, label="Trainings Error")
        ax1.legend(), ax2.legend()
        ax1.grid(True), ax2.grid(True)
        plt.xlim(epochs + 5)
        plt.savefig("plots/errors_NN_baseline.pdf")
        plt.show()

    if create_dataset :
        test_cn(evaluator,test_set,batchsize,x_adj)
    else :
        adj = data.adj_t.to_symmetric().to_dense()
        test_cn(evaluator, test_set, batchsize, adj)


def main():
    # run all baseline and return results
    runNN(create_dataset=False, epochs=1, load=True, save=False, batchsize=None)


if __name__ == "__main__":
    main()
