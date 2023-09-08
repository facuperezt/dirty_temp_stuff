import torch
import torch_sparse
from utils.graph_utils import find_index_of_connection

def _create_sparse_tensor(row : list[int], col : list[int], value : list[float]) -> torch_sparse.SparseTensor:
    st = torch_sparse.SparseTensor(
        row= torch.tensor(row),
        col= torch.tensor(col),
        value= torch.tensor(value),
        sparse_sizes= [max(len(row), len(col))] *2
    )
    return st

def _example_sparse_tensor() -> torch_sparse.SparseTensor:
    st = _create_sparse_tensor(
        [0,1,1,2,2,3,3],
        [1,0,1,0,1,0,1],
        [1,1,1,1,1,1,1],
        )
    return st

def test_no_connection() -> None:
    sp = _example_sparse_tensor()
    ind = find_index_of_connection(sp, 0, 0)
    assert len(ind) == 0

def test_one_way_connection() -> None:
    sp = _example_sparse_tensor()
    ind = find_index_of_connection(sp, 3, 1)
    assert len(ind) == 1
    assert ind[0] == 6

def test_two_way_connection() -> None:
    sp = _example_sparse_tensor()
    ind = find_index_of_connection(sp, 0, 1)
    assert len(ind) == 2
    assert ind == [0,1]
