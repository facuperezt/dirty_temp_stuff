import numpy as np
import pandas as pd
import torch.nn
from torch.nn.functional import relu
from ogb.linkproppred import Evaluator
import dataLoader
import utils_func

def cn(src,tar,adj):
    row_src = adj[src]
    row_tar = adj[tar]

    n = 0
    for i in range(row_tar.shape[0]):
        if row_src[i] == row_tar[i] :
            n+= 1

def create_Dataset( adj, data):
    adj = adj.astype(dtype=float)
    adj -= np.identity(data.shape[0],dtype=float)
    save = np.zeros([data.shape[0],2])
    for i in range(adj.shape[0]):
        one = 0
        two = 0
        for h1 in np.where(adj[i]):
            for value in h1 :
                one += data[value].sum()
                for h2 in  np.where(adj[value]):
                    for tmp in h2:
                        two += data[tmp].sum()
                save[i,1] = two
        save[i, 0] = one

    pd.DataFrame(save).to_csv("data/baseline_NN", index=False)
    return save

def train(optimizer,train_set,rep,data,mlp):
    optimizer.zero_grad()
    train_src, train_tar = train_set["source_node"], train_set["target_node"]
    total_loss = 0
    for src,tar in train_src,train_tar:
    # forward passes

        # positive sampling
        x = np.array([[rep[src,1],rep[src,0],data[src].sum(),data[tar].sum(),rep[tar,0],rep[tar,1]]])

        out = mlp(x)
        pos_loss = - torch.mean(torch.log(out + 1e-15))

        #negative sampling
        tar = torch.randint(low=0, high=22064, size=train_src.size(), dtype=torch.long)  # 30657
        x = np.array([[rep[src, 1], rep[src, 0], data[src].sum(), data[tar].sum(), rep[tar, 0], rep[tar, 1]]])

        out = mlp(x)
        neg_loss = torch.log(1 - out + 1e-15)
        neg_loss = - torch.mean(neg_loss)

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss+= loss

    return total_loss/data.shape[0]
def test(evaluator,data_set,mlp,rep,data):
    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]

    #positive sampling
    pos_preds = []
    for i in range(0, data_set["source_node"].shape[0]):
        x = np.array([[rep[src,1],rep[src,0],data[src].sum(),data[tar].sum(),rep[tar,0],rep[tar,1]]])
        pos_preds += [mlp(x).squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0)

    #TODO negative sampling not adjusted yet
    neg_preds = []
    src = src.view(-1, 1).repeat(1, 20).view(-1)
    tar_neg = tar_neg.view(-1)
    for i in range(0, data_set["source_node"].shape[0]):
        idx = permutation[i:i + batchsize]
        #TODO batching
        neg_preds += [mlp(mid[src], mid[tar_neg]).squeeze().cpu()]
    #print(neg_preds[0].size(),len(neg_preds))
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 20)
    #print(neg_pred.size())

    return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

class MLP_base(torch.nn.Module):

    def __init__(self):
        super(MLP_base, self).__init__()  # from torch documentation TODO look up what it does
        self.input = torch.nn.Linear(6, 6)
        self.hidden = torch.nn.Linear(6, 6)
        self.output = torch.nn.Linear(6, 1)

    def forward(self,x):
        h = self.input(x)

        X = relu(h)
        h = self.hidden(X)

        X = relu(h)
        h = self.output(X)

        return torch.sigmoid(h)

def runNN(dataset,epochs,load,save):
    data, split = dataLoader.main(full_dataset=False, use_year=False, explain=True)
    x, edges = data["x"], data["edge_index"]
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset:
        x_adj = utils_func.adjMatrix(edges, x.shape[0], selfLoops=False)

        rep = create_Dataset(x_adj, x)
    else:
        rep = pd.read_csv("data/baseline_NN")
        print(rep)

        # initilaization models
        mlp = MLP_base()

    if load == True:
        mlp.load_state_dict(torch.load("model/mlp_baseline"))

    MLP_base.to(device), data.to(device)
    optimizer = torch.optim.Adam(list(mlp.parameters()))

    evaluator = Evaluator(name='ogbl-citation2')

    # ----------------------- training & testing
    valid_mrr = torch.zeros((epochs, 1))
    test_mmr = torch.zeros((epochs, 1))
    loss = torch.zeros((epochs, 1))

    for i in range(0, epochs):
        loss[i] = train(optimizer, train_set, rep, x).detach()

        test_mmr[i] = test(batchsize, test_set, gnn, mlp, edges, x, evaluator)
        valid_mrr[i] = test(batchsize, valid_set, mlp, edges, x, evaluator)

    if save == True:
        torch.save(mlp.state_dict(), "model/mlp_baseline")


def main():
    runNN(dataset=False,epochs=50,load=False,save=True)

if __name__ == "__main__":
    main()
