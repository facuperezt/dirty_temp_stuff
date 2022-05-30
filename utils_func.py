import igraph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import copy
import utils
import matplotlib.lines as mlines
import plots


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


def grid_e_g(test_set, gnn, mlp, adj, x, edge_index):
    src, tar = test_set["source_node"], test_set["target_node"]
    walks_all = utils.walks(adj)
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    samples = pd.read_csv("data/samples_smoll").to_numpy(dtype=int)[:, 1]
    e = np.array((0, 0.33, 0.66, 1, 1.33, 1.66, 2))
    g = np.array((0, 0.33, 0.66, 1, 1.33, 1.66, 2))
    X, Y = np.meshgrid(e, g)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    res = []
    n = 43
    l = [8, 14, 107, 453, 450, 410, 210, 328, 331, 54]  # 331
    print(positions, len(positions))
    for pair in positions[43:-1]:
        abs_R = 0
        s = 0
        for i in l:
            print(n, s)
            walks = find_walks(src[i], tar[i], walks_all)
            r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i], epsilon=pair[0], gamma=pair[1])
            p = []

            for walk in walks:
                p.append(gnn.lrp(x, edge_index, walk, r_src, r_tar, tar[i], epsilon=pair[0], gamma=pair[1]))
            abs_R += plot_explain(p, src[i], tar[i], walks, "pos", i)
            s += 1
        res.append(np.asarray((pair[0], pair[1], abs_R.numpy())))
        if n % 7 == 0:
            pd.DataFrame(res).to_csv("data/tmp" + str(n) + ".csv")
        n += 1
    pd.DataFrame(res).to_csv("data/tmp.csv")


def masking(gnn, nn, input, src, tar, edge_index, adj, walk, gamma=0):

    x = input.data
    x.requires_grad_(True)

    M = [None] * 4
    M[0] = torch.FloatTensor(np.eye(x.shape[0])[walk[0]][:, np.newaxis])
    M[1] = torch.FloatTensor(np.eye(x.shape[0])[walk[1]][:, np.newaxis])
    M[2] = torch.FloatTensor(np.eye(x.shape[0])[walk[2]][:, np.newaxis])
    M[3] = torch.FloatTensor(np.eye(x.shape[0])[walk[3]][:, np.newaxis])
    H = [None] * 6

    y = x * M[0] + (1 - M[0]) * x.data
    #x.retain_grad()

    H[0] = copy.deepcopy(gnn.input).forward(y, edge_index).clamp(min=0)
    H[0] = H[0] * M[1] + (1 - M[1]) * H[0].data

    H[1] = copy.deepcopy(gnn.hidden).forward(H[0], edge_index).clamp(min=0)
    H[1] = H[1] * M[2] + (1 - M[2]) * H[1].data

    H[2] = copy.deepcopy(gnn.output).forward(H[1], edge_index)
    H[2] = H[2] * M[3] + (1 - M[3]) * H[2].data

    x_n = H[2][src] + H[2][tar]
    H[3] = copy.deepcopy(nn.input).forward(x_n).clamp(min=0)
    H[4] = copy.deepcopy(nn.hidden).forward(H[3]).clamp(min=0)
    H[5] = copy.deepcopy(nn.output).forward(H[4])
    H[5].backward()
    print("Masking", (x.grad * x.data).sum(), x.grad.sum())
    return x.grad * x.data


def walks(A, src, tar):
    w = []
    for v1 in [src, tar]:
        for v2 in np.where(A[:, v1])[0]:
            for v3 in np.where(A[:, v2])[0]:
                for v4 in np.where(A[:, v3])[0]:
                    w += [[v4, v3, v2, v1.numpy().flatten()[0]]]

    return w


def validation(walks:list,relevance:list,node_src,pruning :bool, activation:bool,plot:bool):
    nodes = list(set(np.asarray(walks).flatten()))
    print(len(nodes))
    res_pruning,res_activation = [],[]
    if activation:
        R_g = 0
        res_activation = []
        for i in range(25):
            R_max,old = (0,0),0
            for node in nodes:
                s = np.asarray([relevance[x].sum() for x in range(len(walks)) if node in walks[x]])
                if (s.sum() + R_g) > old :
                    R_max = (node,s.sum())
                    old = s.sum() + R_g
            nodes.remove(R_max[0])
            res_activation.append((R_max[0],R_max[1]+R_g))
            R_g += R_max[1]
        if plot : plots.validation(res_activation,node_src)

    elif pruning:
        print(nodes)
        R_g = np.asarray(relevance).sum().sum()
        print(R_g)
        res_pruning = []
        for i in range(len(nodes)):
            print(i,nodes)
            R_min, old = (0, 0), np.infty
            for node in nodes:
                s = np.asarray([relevance[x].sum() for x in range(len(walks)) if node not in walks[x]])
                print("current node",node,np.abs(s.sum() - R_g),s.sum())
                print("current old", old)
                if np.abs(s.sum() - R_g) < old:
                    R_min = (node, s.sum())
                    print("     Rmin",R_min)
                    old = np.abs(s.sum() - R_g)
                    print("     new old",old)
            print("toremove",R_min[0])
            nodes.remove(R_min[0])
            res_pruning.append((R_min[0],np.abs(R_min[1] - R_g)))
            print("pre minus", R_g)
            #R_g += np.abs(R_min[1] - R_g)
            print("post minus",R_g)

            print("------------------------------")
        print(res_pruning)

        if plot: plots.validation(res_pruning,node_src)

    return res_activation,res_pruning