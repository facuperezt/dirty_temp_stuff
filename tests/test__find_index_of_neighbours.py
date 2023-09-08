import torch
import torch_sparse
from utils.graph_utils import _find_index_of_neighbours

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
        [0,1,1,2,2,3,3,4],
        [1,0,1,0,1,0,1,4],
        [1,1,1,1,1,1,1,0],
        )
    return st

def test_no_neighbours() -> None:
    sp = _example_sparse_tensor()
    ind_forward = _find_index_of_neighbours(sp, 5, 'forward')
    ind_backward = _find_index_of_neighbours(sp, 5, 'backward')

    assert ind_forward.eq(ind_backward).all()
    assert ind_forward.numel() == 0


def test_forward_neighbours() -> None:
    """
    Assumed to be: prev_node = row -> col = next_node
    """
    sp = _example_sparse_tensor()
    ind_forward = _find_index_of_neighbours(sp, 1, 'forward')

    assert ind_forward.eq(torch.tensor([1,2])).all()

def test_backward_neighbours() -> None:
    sp = _example_sparse_tensor()
    ind_forward = _find_index_of_neighbours(sp, 1, 'backward')

    assert ind_forward.eq(torch.tensor([0, 2, 4, 6])).all()

def test_connection_is_zero() -> None:
    """
    If the value of the connection in zero, there is no connection 
    even though there's an index in the tensor
    """
    sp = _example_sparse_tensor()

    ind_forward = _find_index_of_neighbours(sp, 4, 'forward')
    ind_backward = _find_index_of_neighbours(sp, 4, 'backward')

    assert ind_forward.eq(ind_backward).all()
    assert ind_forward.numel() == 0