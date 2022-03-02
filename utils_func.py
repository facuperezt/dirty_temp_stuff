import igraph
import numpy as np
import matplotlib.pyplot as plt
import torch

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


def plot_explain(r, src, tar, walks, pos, node):
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))

    for node in nodes:
        graph.add_vertices(str(node))

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))

    place = np.array(list(graph.layout_kamada_kawai()))

    # edges plotting
    n = 0
    fig, axs = plt.subplots()
    for walk in walks:
        a = [place[nodes.index(walk[0]), 0], place[nodes.index(walk[1]), 0], place[nodes.index(walk[2]), 0]]
        b = [place[nodes.index(walk[0]), 1], place[nodes.index(walk[1]), 1], place[nodes.index(walk[2]), 1]]
        tx, ty = utils.shrink(a, b)

        R = r[n].detach()

        R = R.sum()
        #print(walk, "with relevance of ", R)
        axs.plot([place[x, 0], place[y, 0]], [place[x, 1], place[y, 1]], color='gray', lw=0.2, ls='dotted', alpha=0.3)

        if R > 0.0:
            alpha = np.clip(5 * R.data.numpy(), 0, 1)
            axs.plot(tx, ty, alpha=alpha, color='red', lw=1.2)
            # print("     and alpha of", alpha)
        if R < -0.0:
            alpha = np.clip(-5 * R.data.numpy(), 0, 1)
            axs.plot(tx, ty, alpha=alpha, color='blue', lw=1.2)
            # print("     and alpha of", alpha)
        n += 1

    # nodes plotting
    axs.plot(place[:, 0], place[:, 1], 'o', color='black', ms=3)
    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o', color='green', ms=6, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o', color='yellow', ms=6, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], color='blue', label="negative relevance")
    axs.plot([], [], color='red', label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")

    node = str(node)
    name = "plots/LRP_plot_" + pos + "_example_" + node + ".svg"
    fig.savefig(name)
    fig.show()


def plot_explain_nodes(walks,src,tar,oneHopSrc, twoHopSrc, oneHopTar, twoHopTar,R):
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))

    for node in nodes:
        graph.add_vertices(str(node))

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))

    place = np.array(list(graph.layout_kamada_kawai()))

    # edges plotting
    n = 0
    fig, axs = plt.subplots()

    axs.plot([place[x, 0], place[y, 0]], [place[x, 1], place[y, 1]], color='gray', lw=0.2, ls='dotted',
                 alpha=0.8)

    # nodes plotting
    lists = [oneHopSrc, twoHopSrc, oneHopTar, twoHopTar]
    relevances = [R[0:128].sum(),R[128:256].sum(),R[500:628].sum(),R[628:756].sum()]
    for node in nodes:
        r = 0
        for i in range(len(lists)):
            if node in lists[i]:
                r += relevances[i]
        if r < 0.0:
            alpha = np.clip(-r.detach().numpy(), 0.1, 1)
#            alpha = np.clip(-r, 0.1, 1)
            axs.plot(place[nodes.index(node), 0], place[nodes.index(node), 1], 'o', color='blue', ms=3, alpha=alpha)
        if r > 0.0:
            alpha = np.clip(r.detach().numpy(), 0.1, 1)
#            alpha = np.clip(r, 0.1, 1)
            axs.plot(place[nodes.index(node), 0], place[nodes.index(node), 1], 'o', color='red', ms=3, alpha=alpha)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o', color='green', ms=6, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o', color='yellow', ms=6, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], 'o',color='blue', label="negative relevance")
    axs.plot([], [], 'o',color='red',label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")

    node = str(src)
    name = "plots/LRP_plot_" + "_example_" + node +"baseline"+ ".svg"
    fig.savefig(name)
    fig.show()


def plot_curves(epochs, curves, labels, title, file_name="errors.pdf", combined=True):
    # we assume all curves have the same length
    # if we use combined we also assume that loss is always the last
    if combined:
        fig, (axs, ax2) = plt.subplots(1, 2, sharex="all")
        ax2.grid(True)
    else:
        fig, axs = plt.subplots()
    fig.suptitle(title)
    axs.grid(True)

    x = np.arange(0, epochs)
    plt.xlim([0,epochs + 5])

    for i in range(len(curves)):
        if i == len(curves) - 1 and combined:  # last elem
            ax2.plot(x, curves[i], label=labels[i])

        else:
            axs.plot(x, curves[i], label=labels[i])
            axs.legend()

    plt.legend()
    plt.savefig("plots/" + file_name)
    plt.show()


def accuracy(pos_preds,neg_preds):

    tresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    labels = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99']
    pos = np.zeros((2,len(tresholds))) # [true positiveves,false negatives]
    neg = np.zeros((2,len(tresholds))) # [true negatives,false positives]
    n = 0
    for treshold in tresholds:
        for res in pos_preds:
            if res > treshold:
                pos[0,n] += 1
            else:
                pos[1,n] += 1
        for res in neg_preds:
            if res > treshold:
                neg[1,n] += 1
            else:
                neg[0,n] += 1
        n += 1
    sens = pos[0] /(pos[1]+pos[0])
    spec = neg[0] / (neg[0]+neg[1])
    acc = (sens + spec) / 2
    fig, ax = plt.subplots()
    plt.plot(tresholds,acc,'o-')
    #plt.axhline(y=426, linewidth=1, color='r')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Treshold for positive classification')
    ax.set_title('Accuracy of test set, GNN Model')
    ax.grid(True)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig("plots/Accuracy.jpg")
    plt.show()


def get_nodes(adj, start):
    adj = adj.to_symmetric().to_dense()
    oneHop, twoHop = set(), set()

    oneHop.update(np.flatnonzero(adj[start]))
    for h1 in oneHop:
        twoHop.update(np.flatnonzero(adj[h1]))
    return list(oneHop), list(twoHop)

def NN_res(R,sample):
    R= R.detach().numpy()
    keys = ['s2','s1','src','tar','t1','t2']
    relevances = [R[0:128].sum(), R[128:256].sum(),R[256:384].sum(),R[382:512].sum(), R[512:640].sum(), R[640:768].sum()]
    width = 0.35
    ind = np.arange(len(relevances))

    fig, ax = plt.subplots()
    for i in range(len(relevances)):
        if relevances[i] < 0:
            c = 'b'
        else : c = 'r'
        p1 = ax.bar(ind[i], relevances[i], width,color=c)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Relevance')
    ax.set_title('Relevance per vector')
    ax.set_xticks(ind,labels=keys)

    plt.savefig("plots/barplot_"+str(sample)+".png")
    plt.show()

    # scaled version
def accuracy_overtrain(pos_preds,neg_preds,epochs):

    tresholds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    labels = ['0.2', '0.4', '0.6', '0.8', '0.9', '0.99']
    pos = np.zeros((epochs//10,2,len(tresholds))) # [true positiveves,false negatives]
    neg = np.zeros((epochs//10,2,len(tresholds))) # [true negatives,false positives]
    n = 0
    for axis in range(0,epochs//10):
        n = 0
        epoch = axis*10
        for treshold in tresholds:
            for res in pos_preds[epoch,:]:
                if res > treshold:
                    pos[axis,0,n] += 1
                else:
                    pos[axis,1,n] += 1
            for res in neg_preds[epoch,:]:
                if res > treshold:
                    neg[axis,1,n] += 1
                else:
                    neg[axis,0,n] += 1
            n += 1
    print(pos.shape, pos)
    sens = pos[0] /(pos[1]+pos[0])
    spec = neg[0] / (neg[0]+neg[1])
    acc = (sens + spec) / 2
    print(acc.shape,acc)
    fig, ax = plt.subplots()
    plt.plot(np.arange(0,epochs,10),acc,'o-')
    ax.legend(labels)
    #plt.axhline(y=426, linewidth=1, color='r')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch of training')
    ax.set_title('Accuracy of test set, GCN model')
    ax.grid(True)

    plt.savefig("plots/Accuracy_epochs.jpg")
    plt.show()