import igraph
import numpy as np
import matplotlib.pyplot as plt
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
    # TODO walks might be a bit weird retund walk 22063,22063,22063 --> artafect of model
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

    #t1 = np.asarray(tmp)
    #t2 = np.asarray(test)
    #print(np.all(t1==t2))

    #return (walks,graph,x,y)
    return tmp

def plot_explain(r,src,tar, walks,pos, epoch,node):
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

    # edges plotting
    n = 0
    fig,axs = plt.subplots()
    for walk in walks:
        a = [place[nodes.index(walk[0]),0],place[nodes.index(walk[1]),0],place[nodes.index(walk[2]),0]]
        b = [place[nodes.index(walk[0]),1],place[nodes.index(walk[1]),1],place[nodes.index(walk[2]),1]]
        tx, ty = utils.shrink(a,b)

        R = r[n].detach()

        R = R.sum()
        print(walk,"with relevance of ",R)
        axs.plot([place[x, 0], place[y, 0]], [place[x, 1], place[y, 1]], color='gray',lw=0.2, ls='dotted',alpha=0.3)

        if R > 0.0:
            alpha = np.clip(2*R.data.numpy(), 0, 1)
            axs.plot(tx, ty, alpha=alpha, color='red', lw=1.2)
            print("     and alpha of", alpha)
        if R < -0.0:
            alpha = np.clip(-2*R.data.numpy(), 0, 1)
            axs.plot(tx, ty, alpha=alpha, color='blue', lw=1.2)
            print("     and alpha of", alpha)
        n+= 1

    # nodes plotting
    axs.plot(place[:, 0], place[:, 1], 'o', color='black', ms=3)
    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o', color='green', ms=6,label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o', color='yellow', ms=6, label="target node")

    #legend shenenigans & # plot specifics
    axs.plot([], [], color='blue',label = "negative relevance")
    axs.plot([], [], color='red',label="positive relevance")

    axs.legend(loc= 2,bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")


    epoch = str(epoch)
    node = str(node)
    name = "plots/LRP_plot_"+pos+"_example_"+node+".svg"
    fig.savefig(name)
    fig.show()

