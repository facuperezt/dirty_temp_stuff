#%%
import torch
import dataLoader
from utils import utils_func, ainb
from XAI import find_good_samples

all_walks : torch.Tensor = torch.load("walks_attempt_one.th")
#%%

# loading the data
dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
# dataset = dataLoader.ToyData("data/", "mini_graph", use_subset=True)

data = dataset.load()
split = dataset.get_edge_split()
train_set, valid_set, test_set = split["train"], split["valid"], split["test"]
data = dataset.load()
split = dataset.get_edge_split()
train_set, valid_set, test_set = split["train"], split["valid"], split["test"]
tmp = data.adj_t.set_diag()
deg = tmp.sum(dim=0).pow(-0.5)
deg[deg == float('inf')] = 0
tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
data.adj_t = tmp
# to plot proper plot the the LRP-method we need all walks:
explain_data = dataset.load(transform=False, explain=False)
exp_adj = utils_func.adjMatrix(explain_data.edge_index,
                                explain_data.num_nodes)  # Transpose of adj Matrix for find walks
samples = find_good_samples(data.adj_t, valid_set, remove_connection= True, criterion= "indirect connections", load = "", save = "")
#%%
for index, (walks, interesting_indexes) in samples.items():
    print(index)
    src, tar = valid_set["source_node"][index], valid_set["target_node"][index]
    print_flag = True
    for i, int_indx in enumerate(interesting_indexes):
        int_indx = torch.tensor(walks[int_indx]).view(-1,1)
        sparse_mask = (all_walks._indices() == int_indx.flip(0)).all(dim= 0)
        rel = all_walks._values()[sparse_mask]
        if len(rel) > 0:
            if print_flag is True:
                print(src, tar)
                print_flag = False
            print(f"\t{i}/{len(interesting_indexes)}")
            print(torch.where(sparse_mask)[0])
            print(int_indx, rel)
        
# %%

def sp(tensor, indices, iteration= 0):
    if len(indices) == 0:
        return tensor
    if len(indices) == 1:
        return tensor.index_select(iteration, torch.tensor([indices[0]]))._values()
    else:
        return sp(tensor.index_select(iteration, torch.tensor([indices[0]])), indices[1:], iteration + 1)
    
sp(all_walks, walks[interesting_indexes[0]])

# %%
