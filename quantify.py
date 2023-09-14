#%%
import torch
import tqdm
import dataLoader
from utils import utils_func, ainb
import glob
import os
import pandas as pd

# all_walks : torch.Tensor = torch.load("walks_attempt_one.th")
# #%%

# # loading the data
# dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
# # dataset = dataLoader.ToyData("data/", "mini_graph", use_subset=True)

# data = dataset.load()
# split = dataset.get_edge_split()
# train_set, valid_set, test_set = split["train"], split["valid"], split["test"]
# data = dataset.load()
# split = dataset.get_edge_split()
# train_set, valid_set, test_set = split["train"], split["valid"], split["test"]
# tmp = data.adj_t.set_diag()
# deg = tmp.sum(dim=0).pow(-0.5)
# deg[deg == float('inf')] = 0
# tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
# data.adj_t = tmp
# # to plot proper plot the the LRP-method we need all walks:
# explain_data = dataset.load(transform=False, explain=False)
# exp_adj = utils_func.adjMatrix(explain_data.edge_index,
#                                 explain_data.num_nodes)  # Transpose of adj Matrix for find walks
# samples = find_good_samples(data.adj_t, valid_set, remove_connection= True, criterion= "indirect connections", load = "", save = "")
# #%%
# for index, (walks, interesting_indexes) in samples.items():
#     print(index)
#     src, tar = valid_set["source_node"][index], valid_set["target_node"][index]
#     print_flag = True
#     for i, int_indx in enumerate(interesting_indexes):
#         int_indx = torch.tensor(walks[int_indx]).view(-1,1)
#         sparse_mask = (all_walks._indices() == int_indx.flip(0)).all(dim= 0)
#         rel = all_walks._values()[sparse_mask]
#         if len(rel) > 0:
#             if print_flag is True:
#                 print(src, tar)
#                 print_flag = False
#             print(f"\t{i}/{len(interesting_indexes)}")
#             print(torch.where(sparse_mask)[0])
#             print(int_indx, rel)
        
# %%
def compare_relevances(a : torch.Tensor, b : torch.Tensor, comparisson : str):
    if comparisson == "sum":
        a,b = a.sum(), b.sum()
    elif comparisson == "mean":
        a,b = a.mean(), b.mean()
    elif comparisson == "median":
        a,b = a.median(), b.median()

    return a,b 
        


def filter_relevances_and_pool(rel_matrix : torch.Tensor, src : torch.Tensor, tar : torch.Tensor):
    all_walks = rel_matrix._indices()
    relevances = rel_matrix._values()

    contains_src = (all_walks == src).any(dim=0)
    contains_tar = (all_walks == tar).any(dim=0)

    is_indirect_path = contains_src & contains_tar

    indirect_path_relevance = relevances[is_indirect_path]
    no_path_relevance = relevances[~is_indirect_path]

    connected_dict = {}
    not_connected_dict = {}

    for comparisson in ["sum", "mean", "median"]:
        connected, no_path = compare_relevances(indirect_path_relevance, no_path_relevance, comparisson)
        connected_dict["connected_" + comparisson] = connected.item()
        not_connected_dict["not_connected_" + comparisson] = no_path.item()
    
    return connected_dict, not_connected_dict 

def get_pooled_relevances(path_to_folder : str, all_src : torch.Tensor, all_tar : torch.Tensor):
    out = []
    for src, tar in tqdm.tqdm(zip(all_src, all_tar), desc= "Source-Target pairs:", total= len(all_src)):
        files = glob.glob(os.path.join(path_to_folder,f"{src.item()}_{tar.item()}_*.th"))
        d = {}
        for file in files:
            rel_matrix : torch.Tensor = torch.load(file)
            filename = os.path.splitext(file)[0].split('/')[-1]
            _src, _tar, gamma, epsilon = filename.split('_')
            gamma, epsilon = [float(num.replace(',', '.')) for num in [gamma, epsilon]]
            connected_dict, not_connected_dict = filter_relevances_and_pool(rel_matrix, src, tar)
            d = {
                'src' : src.item(),
                'tar' : tar.item(),
                'gamma' : gamma,
                'epsilon' : epsilon,
                'total_rel' : rel_matrix.sum().item(),
            }
            d.update(connected_dict); d.update(not_connected_dict)
        if len(d) > 0:
            d = pd.Series(d)
            out.append(d)

    return pd.DataFrame(out)


def sp(tensor, indices, iteration= 0):
    if len(indices) == 0:
        return tensor
    if len(indices) == 1:
        return tensor.index_select(iteration, torch.tensor([indices[0]]))._values()
    else:
        return sp(tensor.index_select(iteration, torch.tensor([indices[0]])), indices[1:], iteration + 1)
# %%
