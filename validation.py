import torch_sparse
import numpy as np
import torch
import matplotlib.pyplot as plt
def validation_avg_plot(relevances: list, l):
    avg = np.zeros((1,l))
    for i in relevances :
        tmp = np.zeros((1,l))
        tmp[0,0:i[1]] = i[0]
        tmp[0,i[1]:] = i[0][-1]
        avg += tmp

    avg = avg/300
    fig, axs = plt.subplots()
    axs.fill_between(np.arange(0, l, 1), avg.flatten())
    tick_max = 5 * round(float(l) / 5)

    ticks = np.arange(0,tick_max+1, 5)
    axs.set_xticks(ticks, labels=ticks)

    axs.set_title('f(s^c)-f(s_i)')

    plt.savefig("plots/validation_pru_new_avg")
    plt.show()


def validation_plot(relevances: list, node, l):
    relevances = np.asarray(relevances).flatten()
    print(relevances)
    fig, axs = plt.subplots()
    axs.fill_between(np.arange(0, l, 1), relevances)
    tick_max = 5 * round(float(l) / 5)
    ticks = np.arange(0,tick_max+1, 5)
    axs.set_xticks(ticks, labels=ticks)

    plt.savefig("plots/validation_act_abs_new" + str(node.numpy()))
    plt.show()


def restore_edges(adj, nodes, new_node, adj_new):
    adj = adj.to_dense()
    adj_new = adj_new.to_dense()

    for node in nodes:
        adj_new[node, new_node] = adj[node, new_node]
        adj_new[new_node, node] = adj[new_node, node]


    adj_new = torch_sparse.SparseTensor.from_dense(adj_new)
    return adj_new


def remove_edges(adj_new, nodes, new_node):
    adj_new = adj_new.to_dense()

    for node in nodes:
        adj_new[node, new_node] = 0
        adj_new[new_node, node] = 0


    adj_new = torch_sparse.SparseTensor.from_dense(adj_new)
    return adj_new

def clear_edges(adj,walks):
    adj_tmp = adj.to_dense()
    nodes = list(set(np.asarray(walks).flatten()))
    for node in nodes:
        for other in nodes:
            adj_tmp[node, other] = 0
            adj_tmp[other, node] = 0

    adj_tmp = torch_sparse.SparseTensor.from_dense(adj_tmp)
    return adj_tmp


def validation_list(walks: list, relevance: list, pruning: bool, activation_bool: bool):
    nodes = list(set(np.asarray(walks).flatten()))
    print(len(nodes))
    if activation_bool:
        R_g = 0
        activation = []
        for i in range(len(nodes)):
            R_max, old = (0, 0), -np.infty
            for node in nodes:
                #print(node,R_max,len(nodes))
                s = np.asarray([relevance[x].sum() for x in range(len(walks)) if node in walks[x]])
                #print(s.sum() + R_g)
                if (s.sum() + R_g) > old:
                    R_max = (node, s.sum())
                    old = s.sum() + R_g
                    #print("new old",old)
            #print("to remove", R_max[0])
            nodes.remove(R_max[0])
            activation.append(R_max[0])
            R_g += R_max[1]
            #print("------------------------------")
        return activation

    elif pruning:
        R_g = np.asarray(relevance).sum().sum()
        res_pruning = []
        for i in range(len(nodes)):
            R_min, old = (0, 0), np.infty
            for node in nodes:
                s = np.asarray([relevance[x].sum() for x in range(len(walks)) if node not in walks[x]])
                if np.abs(s.sum() - R_g) < old:
                    R_min = (node, s.sum())
                    old = np.abs(s.sum() - R_g)

            nodes.remove(R_min[0])
            res_pruning.append(R_min[0])

    return res_pruning


def validation_results(gnn, mlp, x,adj,walks,relevances,src,tar, pruning=False,activaton= False):
    node_list = validation_list(walks,relevances,pruning=pruning,activation_bool=activaton)
    if pruning: adj_tmp = adj
    else : adj_tmp= clear_edges(adj,walks)
    print(adj)
    mid = gnn(x, adj_tmp)
    ref = mlp(mid[src], mid[tar]).detach().numpy().sum()
    graph = []
    predictions = []
    for node in node_list:
        graph.append(node)
        if pruning:
            adj_tmp= remove_edges(adj_tmp,graph,node)
        else:
            adj_tmp = restore_edges(adj, graph, node, adj_tmp)
        mid = gnn(x, adj_tmp)
        out = mlp(mid[src], mid[tar]).detach().numpy()
        #print(ref,out.sum())
        #print(np.abs(ref- out.sum()), np.abs(out.sum()-ref))
        #print(ref - out.sum(),out.sum() - ref)

        if pruning: predictions.append(ref-out.sum())
        else: predictions.append(out.sum()-ref)

    #validation_plot(predictions,src,len(node_list))
    return (predictions, len(node_list))