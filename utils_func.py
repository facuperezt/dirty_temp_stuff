import igraph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import copy
import utils
import matplotlib.lines as mlines


def adjMatrix(edges, numNodes, selfLoops=True):
    """
    Function to calculate Adjacency MMatrix given an edge Array
    :param edges:  Dimensions 2 x m with the fst of the two being source and the snd being target
    :param numNodes: Number of different nodes in the graph, used to create correctly sized matrix
    :param selfLoops: bool to decide weather to add selfloops or not
    :return: adjacency Matrix with size numNodes x numNodes
    """
    adj = np.zeros((numNodes, numNodes))

    if selfLoops: adj += np.identity(numNodes)

    for edge in edges.T:
        adj[edge[0], edge[1]] += 1

    return adj


def degMatrix(adj_t):
    deg = np.zeros(adj_t.shape)
    for column in range(adj_t.shape[1]):
        deg[column][column] += np.sum(adj_t[:, column])

    return deg


def find_walks(src, tar, walks):
    # walks might be a bit weird retund walk 22063,22063,22063 --> artafect of model
    # TODO combine walks and plt
    arr = np.asarray(walks)
    tmp = []
    """
    x,y = [],[]

    graph = igraph.Graph()
    test =[]
    s = {}
    for n in range(arr.shape[0]):
        if arr[n][2] == tar or arr[n][2] == src :
            # list of walk returns
            test.append(arr[n])

            # add edges and nodes to graph
            graph.add_vertices(str(arr[n][0]))
            graph.add_vertices(str(arr[n][1]))
            graph.add_vertices(str(arr[n][2]))

            graph.add_edges([(str(arr[0]), str(arr[1])), (str(arr[1]), str(arr[2]))])

            # create list for coordinates
            s.update(arr[n][0],str(arr[n][1],arr[n][2]))
            x.append(s.index(arr[0])), y.append(s.index(arr[1]))
            x.append(s.index(arr[1])), y.append(s.index(arr[2]))
    
    """
    for n in range(arr.shape[0]):
        if arr[n][2] == tar:
            tmp.append(arr[n])

        elif arr[n][2] == src:
            tmp.append(arr[n])

    # t1 = np.asarray(tmp)
    # t2 = np.asarray(test)
    # print(np.all(t1==t2))

    # return (walks,graph,x,y)
    return tmp


def get_nodes(adj, start):
    adj = adj.to_symmetric().to_dense()
    oneHop, twoHop = set(), set()

    oneHop.update(np.flatnonzero(adj[start]))
    for h1 in oneHop:
        twoHop.update(np.flatnonzero(adj[h1]))
    return list(oneHop), list(twoHop)



def grid_e_g(test_set, gnn, mlp, adj, x,edge_index):
    src, tar = test_set["source_node"], test_set["target_node"]
    walks_all = utils.walks(adj)
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    samples = pd.read_csv("data/samples_smoll").to_numpy(dtype=int)[:,1]
    e = np.array((0,0.33,0.66,1,1.33,1.66,2))
    g = np.array((0,0.33,0.66,1,1.33,1.66,2))
    X,Y = np.meshgrid(e,g)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    res = []
    n= 43
    l = [8,14,107,453,450,410,210,328,331,54]#331
    print(positions,len(positions))
    for pair in positions[43:-1]:
        abs_R = 0
        s=0
        for i in l:
            print(n,s)
            walks = find_walks(src[i], tar[i], walks_all)
            r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i],epsilon=pair[0], gamma=pair[1])
            p = []

            for walk in walks:
                p.append(gnn.lrp(x, edge_index, walk, r_src, r_tar, tar[i],epsilon=pair[0], gamma=pair[1]))
            abs_R += plot_explain(p, src[i], tar[i], walks, "pos", i)
            s+=1
        res.append(np.asarray((pair[0],pair[1],abs_R.numpy())))
        if n%7 == 0:
            pd.DataFrame(res).to_csv("data/tmp"+str(n)+".csv")
        n+= 1
    pd.DataFrame(res).to_csv("data/tmp.csv")


def masking(gnn,nn,x,src,tar,edge_index,adj,walk,gamma=0):

    def roh(layer,gamma):
        with torch.no_grad():
            cp = copy.deepcopy(layer)
            cp.lin.weight[:, :] = cp.lin.weight + gamma * torch.clamp(cp.lin.weight, min=0)
            return cp

    def roh_lin(layer,gamma):
        with torch.no_grad():
            cp = copy.deepcopy(layer)
            cp.weight[:, :] = cp.weight + gamma * torch.clamp(cp.weight, min=0)
            return cp

    torch.autograd.set_detect_anomaly(True)
    x.requires_grad_(True)
    H = [None]*6

    Mj = torch.FloatTensor(np.eye(len(adj))[walk[0]][:, np.newaxis])
    Mk = torch.FloatTensor(np.eye(len(adj))[walk[1]][:, np.newaxis])
    Ml = torch.FloatTensor(np.eye(len(adj))[walk[2]][:, np.newaxis])

    Z = roh(gnn.input,gamma=0).forward(x,edge_index)  # Adjacency *  (H * W1)
    Zp= roh(gnn.input,gamma=gamma).forward(x,edge_index)#
    H[0] = (Zp * (Z / (Zp + 1e-15)).data).clamp(min=0)
    H[0] = H[0] * Mj + (1 - Mj) * (H[0].data)

    Z = roh(gnn.hidden, gamma=0).forward(H[0], edge_index)  # Adjacency *  (H * W1)
    Zp = roh(gnn.hidden, gamma=gamma).forward(H[0], edge_index)  #
    H[1] = (Zp * (Z / (Zp + 1e-15)).data).clamp(min=0)
    H[1] = H[1] * Mk + (1 - Mk) * (H[1].data)

    Z = roh(gnn.output, gamma=0).forward(H[1], edge_index)  # Adjacency *  (H * W1)
    Zp = roh(gnn.output, gamma=gamma).forward(H[1], edge_index)
    H[2] = (Zp * (Z / (Zp + 1e-15)).data)
    H[2] = H[2] * Ml + (1 - Ml) * (H[2].data)

    x_nn = H[2][src]+H[2][tar]
    H[3] = roh_lin(nn.input,gamma=0).forward(x_nn).clamp(min=0)
    H[4] = roh_lin(nn.hidden, gamma=0).forward(H[3]).clamp(min=0)

    #Z = roh_lin(nn.output, gamma=0).forward(H[4])  # Adjacency *  (H * W1)
    #Zp = roh_lin(nn.output, gamma=gamma).forward(H[4])

    #H[5] = (Zp * (Z / (Zp + 1e-6)).data)
    #H[5] = H[5] * Ml + (1 - Ml) * (H[5].data)
    H[5] = roh_lin(nn.output, gamma=0).forward(H[4])
    H[5].backward()

    print("masking",(x.grad*x.data).sum())

    return x.grad*x.data