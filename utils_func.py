import igraph
import numpy as np
import matplotlib.pyplot as plt
import utils
from datetime import datetime

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
    # TODO walks might be a bit weird retund walk 22063,22063,22063 --> artafect of model
    arr = np.asarray(walks)
    src = src.numpy()
    tar = tar.numpy()

    tmp = [arr[n] for n in range(arr.shape[0]) if src in arr[n] or tar in arr[n]]

    return tmp

def plot_explain(r,src,tar, walks,pos, epoch):
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))

    for node in nodes:
        graph.add_vertices(str(node))

    x,y = [],[]
    for walk in walks :
        graph.add_edges([(str(walk[0]),str(walk[1])),(str(walk[1]),str(walk[2]))])
        x.append(nodes.index(walk[0])),y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])),y.append(nodes.index(walk[2]))

    place = np.array(list(graph.layout_kamada_kawai()))

    # nodes plotting
    plt.plot(place[:, 0], place[:, 1], 'o', color='black', ms=3)
    plt.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o', color='green', ms=3)
    plt.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o', color='yellow', ms=3)

    # edges plotting
    n = 0
    for walk in walks:
        a = [place[nodes.index(walk[0]),0],place[nodes.index(walk[1]),0],place[nodes.index(walk[2]),0]]
        b = [place[nodes.index(walk[0]),1],place[nodes.index(walk[1]),1],place[nodes.index(walk[2]),1]]
        tx, ty = utils.shrink(a,b)

        R = r[n].detach()
        R = R.sum()

        plt.plot([place[x, 0], place[y, 0]], [place[x, 1], place[y, 1]], color='gray',lw=0.5, ls='dotted')

        if R > 0.0:
            alpha = np.clip(20 * R.data.numpy(), 0, 1)
            plt.plot(tx, ty, alpha=alpha, color='red', lw=1.2)

        if R < -0.0:
            alpha = np.clip(-20 * R.data.numpy(), 0, 1)
            plt.plot(tx, ty, alpha=alpha, color='blue', lw=1.2)
        n+= 1

    now = datetime.now()
    plt.show()
    epoch = str(epoch)
    plt.savefig("plots/posExample_explain"+pos+epoch)