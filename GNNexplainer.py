import numpy as np
import torch.nn.functional as F
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv
from utils import utils_func, utils
from tqdm import tqdm
import torch_sparse
import torch_geometric


def sigm(z):
    return torch.tanh(0.5 * z) * 0.5 + 0.5


def gnnexplainer(adj, gnn, nn, edge, H0=None,mask=None, steps=10, lr=0.5, lambd=0.01, verbose=False):
    # Assumption we only use base the modified adj map
    num_layer = 3
    bar = tqdm(range(steps)) if verbose else range(steps)
    adj_t = utils_func.adj_t(adj).to_dense()
    #TODO set adj info in mask for walk between src tar
    if mask is None :
        z = (torch.ones(adj.shape) * adj * 2)
    else :
        z = mask * 2
    for i in bar:
        z.requires_grad_(True)
        emb = gnn.forward(H0, adj_t, [sigm(z)] * num_layer)
        score = nn(emb[edge[0]]+emb[edge[1]],  classes=False)  # src,tar
        emp = -score
        reg = lambd * (z ** 2).sum()  # constant
        if i in [j ** 5 for j in range(10)] and verbose: print('%5d %8.3f %8.3f' % (i, emp.item(), reg.item()))
        (emp + reg).backward()
        with torch.no_grad():
            # Check if we can set constants
            _, tmp = torch_geometric.utils.dense_to_sparse(z.grad)
            tmp = list(np.asarray(tmp.flatten()))
            tmp = set([x for x in tmp if tmp.count(x) > 1])
            #assert len(tmp) ==1, "multiple duplicate values"

            if i == steps-1:
                #masking = z.grad.eq(tmp.pop())
                z = (z - lr * z.grad)
                #z[masking] = 0
            else:
                z = (z - lr * z.grad)

        z.grad = None

    return z.data


from captum.attr import IntegratedGradients


def get_top_edges_edge_ig(gnn, nn, H0, adj, target, drop_selfloop=False):
    # target index for whicxh computed ..> best ouse tar and src

    def model_edge_forward(edge_mask, gnn, nn, target, adj):
        edges = adj.nonzero()
        a = torch.zeros_like(adj)
        for mask, edge in zip(edge_mask, edges):
            # if edge[0] > edge[1]: continue # What does this do ?
            a[edge[0]][edge[1]] = mask
            a[edge[1]][edge[0]] = mask
        pred = torch.sigmoid(
            nn(target[0], target[1], emb=gnn(H0, torch_sparse.SparseTensor.from_dense(a)), classes=False))
        return pred

    ig = IntegratedGradients(model_edge_forward)
    if drop_selfloop:
        edges = (adj - torch.eye(adj.size[0])).nonzero()
        edge_num = edges.to_sparse().nnz
        input_mask = torch.ones(len((adj - torch.eye(adj.size[0])).nonzero())).requires_grad_(True)
    else:
        edges = adj.nonzero()
        edge_num = edges.shape[0]
        input_mask = torch.ones(len(adj.nonzero())).requires_grad_(True)
    ig_mask = ig.attribute(input_mask, additional_forward_args=(gnn, nn, target, adj),
                           internal_batch_size=len(input_mask))

    edge_mask = ig_mask.cpu().detach().numpy()

    edges_sort = []
    for i in (-edge_mask).argsort()[:edge_num]:
        edges_sort.append(tuple(edges[i].tolist()))
    return edges_sort


def CAM(adj, gnn, H0=None, masks=None):
    if masks is None:
        masks = adj

    adj = torch_sparse.SparseTensor.from_dense(adj)

    #A = torch_sparse.SparseTensor.from_dense(torch.eye(H0.shape[0]).squeeze(0))
    H = gnn.forward(H0, adj,None)
    print("H pre sum", H.shape)
    H = H.sum(dim=1) / 20 ** .5
    print("H post sum", H.shape)
    return H
