import sys
import numpy
# import random
import torch
import igraph
import utils
import matplotlib.pyplot as plt


# Function to generate a BA graph
def scalefreegraph(seed=0, embed=True, growth=None):
    random = numpy.random.mtrand.RandomState(seed)
    N = 10
    A = numpy.zeros([N, N])
    A[1, 0] = 1
    A[0, 1] = 1
    growth = growth if growth is not None else random.randint(1, 3)
    N0 = 2
    for i in range(N0, N):
        if growth == 1:
            tt = 1  # Barabasi-Albert 1
        elif growth == 2:
            tt = 2  # Barabasi-Albert 2
        else:
            tt = 1 + 1 * ((growth - 1) > random.uniform(0, 1))
        p = A.sum(axis=0) / A.sum()
        for j in random.choice(N, tt, p=p, replace=False):
            A[i, j] = 1
            A[j, i] = 1
    r = random.permutation(len(A))
    A = A[r][:, r] * 1.0

    # Add Self-Connections
    A = A + numpy.identity(len(A))

    # Build Data Structures
    D = A.sum(axis=1)
    L = torch.FloatTensor(A / (numpy.outer(D, D) ** .5 + 1e-9))  # laplacian

    return {
        'adjacency': torch.FloatTensor(A),
        'laplacian': L,
        'target': growth,
        'layout': utils.layout(A, seed) if embed else None,
        'walks': utils.walks(A)
    }



####################
# Function to visualise a graph
def vis_graph(g, ax):
    # Arange graph layout
    r = g['layout']
    r = r - r.min(axis=0)
    r = r / r.max(axis=0) * 2 - 1

    # Plot the graph
    N = len(g['adjacency'])
    for i in numpy.arange(N):
        for j in numpy.arange(N):
            if g['adjacency'][i, j] > 0 and i != j: plt.plot([r[i, 0], r[j, 0]], [r[i, 1], r[j, 1]], color='gray',
                                                             lw=0.5, ls='dotted')
    ax.plot(r[:, 0], r[:, 1], 'o', color='black', ms=3)


def train_scalefree():
    model = GraphNet(10, 64, 2)
    optimizer = torch.optim.SGD(model.params, lr=0.001, momentum=0.99)
    erravg = 0.5
    print('Train model:')
    print('   iter | err')
    print('   -----------')
    for it in range(0, 20001):
        optimizer.zero_grad()
        g = scalefreegraph(seed=it, embed=False)
        y = model.forward(g['laplacian'])
        err = (y[0] - (g['target'] == 1) * 1.0) ** 2 + (y[1] - (g['target'] == 2) * 1.0) ** 2
        erravg = 0.999 * erravg + 0.001 * err.data.numpy()
        err.backward()
        optimizer.step()
        if it % 1000 == 0:
            print('% 8d %.3f' % (it, erravg))
    return model


def explain(g, nn, t, gamma=None, ax=None):
    # Arrange graph layout
    r = g['layout']
    r = r - r.min(axis=0)
    r = r / r.max(axis=0) * 2 - 1

    # Plot the graph
    N = len(g['adjacency'])
    for i in numpy.arange(N):
        for j in numpy.arange(N):
            if g['adjacency'][i, j] > 0 and i != j: plt.plot([r[i, 0], r[j, 0]], [r[i, 1], r[j, 1]], color='gray',
                                                             lw=0.5, ls='dotted')
    ax.plot(r[:, 0], r[:, 1], 'o', color='black', ms=3)

    for (i, j, k) in g['walks']:
        R = nn.lrp(g['laplacian'], gamma, t, (j, k))[i].sum()
        tx, ty = utils.shrink([r[i, 0], r[j, 0], r[k, 0]], [r[i, 1], r[j, 1], r[k, 1]])

        if R > 0.0:
            alpha = numpy.clip(20 * R.data.numpy(), 0, 1)
            ax.plot(tx, ty, alpha=alpha, color='red', lw=1.2)

        if R < -0.0:
            alpha = numpy.clip(-20 * R.data.numpy(), 0, 1)
            ax.plot(tx, ty, alpha=alpha, color='blue', lw=1.2)

    gamma = 0.1

    for target in [0, 1]:
        plt.figure(figsize=(3 * len(sample_ids), 3))
        for ids, seed in enumerate(sample_ids):
            ax = plt.subplot(1, len(sample_ids), ids + 1)
            sfg = scalefreegraph(seed=seed, embed=True)

            # Explain
            explain(sfg, model, target, gamma=gamma, ax=ax)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

            plt.axis('off')
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)

        plt.suptitle('Evidence for growth={} with $\gamma={}$'.format(target + 1, gamma), size=14)
        plt.show()
        plt.close()


class GraphNet:
    def __init__(self, d, h, c):
        self.U = torch.nn.Parameter(torch.FloatTensor(numpy.random.normal(0, d ** -.5, [d, h])))
        self.W1 = torch.nn.Parameter(torch.FloatTensor(numpy.random.normal(0, h ** -.5, [h, h])))
        self.W2 = torch.nn.Parameter(torch.FloatTensor(numpy.random.normal(0, h ** -.5, [h, h])))
        self.V = torch.nn.Parameter(torch.FloatTensor(numpy.random.normal(0, h ** -.5, [h, c])))
        self.params = [self.U, self.W1, self.W2, self.V]

    def forward(self, A):
        H = torch.eye(len(A))
        H = H.matmul(self.U).clamp(min=0)
        H = (A.transpose(1, 0).matmul(H.matmul(self.W1))).clamp(min=0)
        H = (A.transpose(1, 0).matmul(H.matmul(self.W2))).clamp(min=0)
        H = H.matmul(self.V).clamp(min=0)
        return H.mean(dim=0)


    def lrp(self, A, gamma, l, inds):
        print(self.U.shape,self.W1.shape,self.W2.shape,self.V.shape )
        if inds is not None:
            j, k = inds
            Mj = torch.FloatTensor(numpy.eye(len(A))[j][:, numpy.newaxis])
            Mk = torch.FloatTensor(numpy.eye(len(A))[k][:, numpy.newaxis])


        W1p = self.W1 + gamma * self.W1.clamp(min=0)  # god damn gamma rule w^- + yw^+
        W2p = self.W2 + gamma * self.W2.clamp(min=0)
        Vp = self.V + gamma * self.V.clamp(min=0)
        print("Len adj ",len(A))
        X = torch.eye(len(A)) # diagonal matrix --> baisicly only selfloops
        X.requires_grad_(True)

        H = X.matmul(self.U).clamp(min=0)  # selfloops mal input weigths


        Z = A.transpose(1, 0).matmul(H.matmul(self.W1)) # Adjacency *  (H * W1)
        Zp = A.transpose(1, 0).matmul(H.matmul(W1p)) #
        H = (Zp * (Z / (Zp + 1e-6)).data).clamp(min=0)


        if inds is not None: H = H * Mj + (1 - Mj) * (H.data)
        K = H

        Z = A.transpose(1, 0).matmul(H.matmul(self.W2))
        Zp = A.transpose(1, 0).matmul(H.matmul(W2p))
        H = (Zp * (Z / (Zp + 1e-6)).data).clamp(min=0)

        if inds is not None: H = H * Mk + (1 - Mk) * (H.data)
        J=H

        Z = H.matmul(self.V)
        Zp = H.matmul(Vp)
        H = (Zp * (Z / (Zp + 1e-6)).data).clamp(min=0)
        L=H
        Y = H.mean(dim=0)[l]

        Y.backward()
        print("Shape res",(X.data * X.grad).shape)
        return X.data * X.grad

# Plotting
sample_ids = [1, 3, 4, 5]
plt.figure(figsize=(3 * len(sample_ids), 3))
for ids, seed in enumerate(sample_ids):
    ax = plt.subplot(1, len(sample_ids), ids + 1)
    sfg = scalefreegraph(seed=seed, embed=True)

    vis_graph(sfg, ax=ax)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.axis('off')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    ax.set_title('growth={}'.format(sfg['target']))

plt.show()
plt.close()

model = train_scalefree()
test_size = 200

num_false = 0
for it in range(20001, 20001 + test_size):
    g = scalefreegraph(seed=it, embed=True)
    y = model.forward(g['laplacian'])
    prediction = int(y.data.argmax()) + 1

    if prediction != g['target']: num_false += 1

print('For {} test samples, the model predict the growth parameter with an accuracy of {} %'.format(test_size, 100 * (
        test_size - num_false) / test_size))


for target in [0, 1]:
    plt.figure(figsize=(3 * len(sample_ids), 3))
    for ids, seed in enumerate(sample_ids):
        ax = plt.subplot(1, len(sample_ids), ids + 1)
        sfg = scalefreegraph(seed=seed, embed=True)

        # Explain
        explain(sfg, model, target, gamma=0.1, ax=ax)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.axis('off')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)

    plt.suptitle('Evidence for growth={} with $\gamma={}$'.format(target + 1, gamma), size=14)
    plt.show()
    plt.close()
