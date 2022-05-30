import numpy as np
import matplotlib.pyplot as plt
import igraph
import utils


def dreiD():
    df1 = pd.read_csv("data/tmp.csv")
    df2 = pd.read_csv("data/tmp42.csv")

    df = pd.concat((df2, df1))
    df = df.drop_duplicates(subset=["0", "1"])

    x = np.reshape(df["0"].to_numpy(), (6, 6))
    y = np.reshape(df["1"].to_numpy(), (6, 6))
    z = np.reshape(df["2"].to_numpy(), (6, 6)) / 10

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_zlim(0, 10)
    ax.set_xticks([0, 0.33, 0.66, 1, 1.33, 1.66])
    ax.set_yticks([1.66, 1.33, 1, 0.33, 0.66, 0])
    ax.set_xlabel(r'$\epsilon \ value$')
    ax.set_ylabel(r'$\gamma \ value$')
    ax.set_zlabel("absolut values of relevances")

    fig.savefig("plots/3dplot_absR_lim10")
    df.to_csv("data/hyperparam_e_g")
    plt.show()


def plt_sum(walks, gnn, r_src, r_tar, tar, x, edge_index):
    arr = np.zeros((4, 1))
    arr[0] = (r_src.detach() + r_tar.detach()).sum()

    walks = np.asarray(walks)
    l = set(walks[:, 2])
    for node in l:
        res = gnn.lrp(x, edge_index, [node, node, node], r_src, r_tar, tar)
        arr[1] += res[0].detach().numpy()
    l = set([tuple((walks[x, 1], walks[x, 2])) for x in range(walks.shape[0])])
    for node in l:
        res = gnn.lrp(x, edge_index, [node[0], node[0], node[1]], r_src, r_tar, tar)
        arr[2] += res[1].detach().numpy()
    for walk in walks:
        res = gnn.lrp(x, edge_index, walk, r_src, r_tar, tar)
        arr[3] += res[2].detach().numpy()

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2, 3], arr.flatten().T, 0.35)
    ax.set_ylabel('Absolut Relevance')
    ax.set_title(r'Absolut Relevance values, $\epsilon =0.0$,$\gamma =0.0$')
    ax.set_xticks([0, 1, 2, 3], labels=["R_0", "R_J", "R_JK", "R_JKL"])

    plt.savefig("plots/abs_r_sum.jpg")
    plt.show()


def accuracy_overtrain(pos_preds, neg_preds, epochs):
    tresholds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    labels = ['0.2', '0.4', '0.6', '0.8', '0.9', '0.99']
    pos = np.zeros((epochs // 10, 2, len(tresholds)))  # [true positiveves,false negatives]
    neg = np.zeros((epochs // 10, 2, len(tresholds)))  # [true negatives,false positives]
    n = 0
    for axis in range(0, epochs // 10):
        n = 0
        epoch = axis * 10
        for treshold in tresholds:
            for res in pos_preds[epoch, :]:
                if res > treshold:
                    pos[axis, 0, n] += 1
                else:
                    pos[axis, 1, n] += 1
            for res in neg_preds[epoch, :]:
                if res > treshold:
                    neg[axis, 1, n] += 1
                else:
                    neg[axis, 0, n] += 1
            n += 1
    print(pos.shape, pos)
    sens = pos[0] / (pos[1] + pos[0])
    spec = neg[0] / (neg[0] + neg[1])
    acc = (sens + spec) / 2
    print(acc.shape, acc)
    fig, ax = plt.subplots()
    plt.plot(np.arange(0, epochs, 10), acc, 'o-')
    ax.legend(labels)
    # plt.axhline(y=426, linewidth=1, color='r')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch of training')
    ax.set_title('Accuracy of test set, GCN model')
    ax.grid(True)

    plt.savefig("plots/Accuracy_epochs.jpg")
    plt.show()


def plot_abs(relevances, samples):
    x_pos = np.arange(len(relevances))
    width = 0.35
    print(relevances)
    fig, ax = plt.subplots()
    ax.bar(x_pos, relevances, width)
    ax.set_ylabel('Absolut Relevance')
    ax.set_title(r'Absolut Relevance values, $\epsilon =0.1$,$\gamma =0.1$')
    ax.set_xticks(x_pos, labels=samples)

    plt.savefig("plots/abs_r.jpg")
    plt.show()


def NN_res(R, sample):
    R = R.detach().numpy()
    keys = ['s2', 's1', 'src', 'tar', 't1', 't2']
    relevances = [R[0:128].sum(), R[128:256].sum(), R[256:384].sum(), R[382:512].sum(), R[512:640].sum(),
                  R[640:768].sum()]
    width = 0.35
    ind = np.arange(len(relevances))

    fig, ax = plt.subplots()
    for i in range(len(relevances)):
        if relevances[i] < 0:
            c = 'b'
        else:
            c = 'r'
        # p1 = ax.bar(ind[i], relevances[i], width, color=c)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Relevance')
    ax.set_title('Relevance per vector')
    ax.set_xticks(ind, labels=keys)

    plt.savefig("plots/barplot_" + str(sample) + ".png")
    plt.show()

    # scaled version


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
    plt.xlim([0, epochs + 5])

    for i in range(len(curves)):
        if i == len(curves) - 1 and combined:  # last elem
            ax2.plot(x, curves[i], label=labels[i])

        else:
            axs.plot(x, curves[i], label=labels[i])
            axs.legend()

    plt.legend()
    plt.savefig("plots/" + file_name)
    plt.show()


def accuracy(pos_preds, neg_preds):
    tresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    labels = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.99']
    pos = np.zeros((2, len(tresholds)))  # [true positiveves,false negatives]
    neg = np.zeros((2, len(tresholds)))  # [true negatives,false positives]
    n = 0
    for treshold in tresholds:
        for res in pos_preds:
            if res > treshold:
                pos[0, n] += 1
            else:
                pos[1, n] += 1
        for res in neg_preds:
            if res > treshold:
                neg[1, n] += 1
            else:
                neg[0, n] += 1
        n += 1
    sens = pos[0] / (pos[1] + pos[0])
    spec = neg[0] / (neg[0] + neg[1])
    acc = (sens + spec) / 2
    fig, ax = plt.subplots()
    plt.plot(tresholds, acc, 'o-')
    # plt.axhline(y=426, linewidth=1, color='r')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Treshold for positive classification')
    ax.set_title('Accuracy of test set, GNN Model')
    ax.grid(True)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig("plots/Accuracy.jpg")
    plt.show()


def plot_explain_nodes(walks, src, tar, one_hop_src, two_hop_src, one_hop_tar, two_hop_tar, relevances):
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
    lists = [one_hop_src, two_hop_src, one_hop_tar, two_hop_tar]
    relevances = [relevances[0:128].sum(), relevances[128:256].sum(), relevances[500:628].sum(),
                  relevances[628:756].sum()]
    for node in nodes:
        r = 0
        for i in range(len(lists)):
            if node in lists[i]:
                r += relevances[i]
        if r < 0.0:
            alpha = np.clip(-r.detach().numpy(), 0.1, 1)
            axs.plot(place[nodes.index(node), 0], place[nodes.index(node), 1], 'o', color='blue', ms=3, alpha=alpha)
        if r > 0.0:
            alpha = np.clip(r.detach().numpy(), 0.1, 1)
            axs.plot(place[nodes.index(node), 0], place[nodes.index(node), 1], 'o', color='red', ms=3, alpha=alpha)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o', color='green', ms=6, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o', color='yellow', ms=6, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], 'o', color='blue', label="negative relevance")
    axs.plot([], [], 'o', color='red', label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")

    node = str(src)
    name = "plots/LRP_plot_" + "_example_" + node + "baseline" + ".svg"
    fig.savefig(name)
    fig.show()


def plot_explain(relevances, src, tar, walks, pos, gamma, data):
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))
    n = 0

    for node in nodes:
        graph.add_vertices(str(node))

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))
        x.append(nodes.index(walk[2])), y.append(nodes.index(walk[1]))

    place = np.array(list(graph.layout_kamada_kawai()))
    # edges plotting

    fig, axs = plt.subplots()
    val_abs = 0
    max_abs = np.abs(max(map((lambda x: x.sum()), relevances)).detach().numpy())

    for walk in walks[:-1]:
        r = relevances[n].detach()

        r = r.sum()

        a = [place[nodes.index(walk[0]), 0], place[nodes.index(walk[1]), 0], place[nodes.index(walk[2]), 0],
             place[nodes.index(walk[3]), 0]]
        b = [place[nodes.index(walk[0]), 1], place[nodes.index(walk[1]), 1], place[nodes.index(walk[2]), 1],
             place[nodes.index(walk[3]), 1]]
        tx, ty = utils.shrink(a, b)

        axs.arrow(a[0], b[0], a[1] - a[0], b[1] - b[0], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.05)
        axs.arrow(a[1], b[1], a[2] - a[1], b[2] - b[1], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.05)
        axs.arrow(a[2], b[2], a[3] - a[2], b[3] - b[2], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.05)

        if r > 0.0:
            alpha = np.clip((2 / max_abs) * r.data.numpy(), 0, 1)
            axs.plot(tx, ty, alpha=alpha, color='red', lw=1.2)

        if r < -0.0:
            alpha = np.clip(-(2 / max_abs) * r.data.numpy(), 0, 1)
            axs.plot(tx, ty, alpha=alpha, color='blue', lw=1.2)

        n += 1

        val_abs += np.abs(r)

    # nodes plotting
    alpha_src = np.sqrt(((data[src].numpy() - data[nodes].numpy()) ** 2).sum(axis=1))
    alpha_src *= 1 / max(alpha_src)

    alpha_tar = np.sqrt(((data[tar].numpy() - data[nodes].numpy()) ** 2).sum(axis=1))
    alpha_tar *= 1 / max(alpha_tar)
    print(alpha_tar)
    print(alpha_src)
    for i in range(len(nodes)):
        """
        axs.plot(place[i, 0], place[i, 1], 'o', color='green', ms=9,
                 fillstyle='bottom',alpha=alpha_src[i])
        axs.plot(place[i, 0], place[i, 1], 'o', color='yellow', ms=9,
                 fillstyle='top',alpha=alpha_tar[i])
        """
        axs.plot(place[i, 0], place[i, 1], 'o', color='black', ms=3)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o',
             color='green', ms=5, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o',
             color='yellow', ms=5, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], color='blue', label="negative relevance")
    axs.plot([], [], color='red', label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")

    gamma = str(gamma)
    gamma = gamma.replace('.', '')
    node = str(src)
    name = "plots/LRP_plot_" + pos + "_example_" + node + gamma + ".svg"
    fig.savefig(name)
    fig.show()
    return val_abs


def validation(relevances: list, node):
    relevances = np.asarray(relevances)
    print(relevances)
    fig, axs = plt.subplots()
    axs.fill_between(np.arange(0, 25, 1), relevances[:,1])
    axs.set_xticks([0, 5, 10, 15, 20, 25], labels=[0, 5, 10, 15, 20, 25])

    plt.savefig("plots/validation_pru_" + str(node.numpy()))
    plt.show()


