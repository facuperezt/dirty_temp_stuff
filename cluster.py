import pandas as pd
import numpy as np
import torch
import torch_sparse
import dataLoader
from matplotlib import pyplot as plt, cm
import dbscan
from collections import Counter

def transitionMatrix(data, start):
    tmp = data.adj_t.set_diag()
    tmp = (tmp / torch.unsqueeze(tmp.sum(dim=1), 1))

    x = torch.zeros((1, data.num_nodes))
    x[0, start] = 1
    x = torch_sparse.SparseTensor.from_dense(x)

    for i in range(5):
        x = torch_sparse.matmul(x, tmp)

    x = x.to_dense().numpy().flatten()
    counter = Counter(np.around(x,decimals=5)).items()

    keys,items = np.asarray([x[0] for x in counter]),np.asarray([x[1]+1  for x in counter])

    tmp = np.vstack((keys,items)).T
    tmp = tmp[tmp[:, 0].argsort()]
    tmp = tmp[::-1]
    y_pos = np.arange(len(keys))

    # Plot 1 amounts of probability
    fig, axs = plt.subplots()
    axs.set_title("Amount of Probabilitys for Random Walk initiated on Paper iD : "+str(start))
    axs.set_ylabel("occurence of probaility")
    axs.set_xlabel("Probabilitys sorted desending by value")

    axs.bar(y_pos,tmp[:,1],log=True)

    plt.setp(axs.get_xticklabels(), visible=False)
    plt.savefig("plots/bar_randomwalk_amount"+str(start))
    plt.show()

    #PLot 2 probabilitys
    fig, axs = plt.subplots()
    axs.set_title("Probabilitys of Random Walk rounded, initiated on Paper iD : "+str(start))
    axs.set_ylabel("Probaility")
    axs.set_xlabel("Probabilitys sorted desending by value")

    axs.bar(y_pos, tmp[:, 0], log=True)

    plt.setp(axs.get_xticklabels(), visible=False)
    plt.savefig("plots/bar_randomwalk_value" + str(start))
    plt.show()


def dbscan_plot(x):
    labels, core_samples_mask = dbscan.DBSCAN(x, eps=0.3, min_samples=10)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    colors = [cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def statistics(data):
    adj = data.adj_t
    in_deg = adj.sum(dim=0)
    out_deg = adj.sum(dim=1)
    deg = in_deg + out_deg

    avrg_in = in_deg.sum() / data.num_nodes
    avrg_out = out_deg.sum() / data.num_nodes
    avrg_deg = deg.sum() / data.num_nodes

    var_in = ((in_deg - avrg_in) ** 2).sum() / data.num_nodes
    var_out = ((out_deg - avrg_out) ** 2).sum() / data.num_nodes
    var_deg = ((deg - avrg_deg) ** 2).sum() / data.num_nodes

    max_in, max_out = max(in_deg), max(out_deg)
    min_in, min_out = min(in_deg), min(out_deg)
    max_deg, min_deg = max(deg), min(deg)

    print("Averages:    in: ", avrg_in, " out: ", avrg_out, "combined: ", avrg_deg)
    print("Variance:    in: ", var_in, " out: ", var_out, "combined: ", var_deg)
    print("Max:    in: ", max_in, " out: ", max_out, "combined: ", max_deg)
    print("Min:    in: ", min_in, " out: ", min_out, "combined: ", min_deg)

    in_deg, out_deg, deg = in_deg.numpy(), out_deg.numpy(), deg.numpy()
    print("in", np.where(np.isclose(max_in, in_deg)))
    print("out", np.where(np.isclose(max_out, out_deg)))
    print("deg", np.where(np.isclose(max_deg, deg)))


def main():
    # loading the data
    dataset = dataLoader.LinkPredData("data/", use_small=False)
    data = dataset.load()
    start = 1454242 #2205095 #716145#1454242  #most citing paper
    transitionMatrix(data, start)
    #dbscan_plot(data.x)
    # statistics(data)


if __name__ == "__main__":
    main()
