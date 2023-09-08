from utils.graph_utils import remove_connection_at_index
import torch
import torch_sparse

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
        [0,0,1,1,2,2,3,3],
        [0,1,0,1,0,1,0,1],
        [1,1,1,1,1,1,1,1],
        )
    return st

def test_remove_no_indexes() -> None:
    st = _example_sparse_tensor()
    assert st == remove_connection_at_index(st, [])

def test_remove_one_index() -> None:
    st = _example_sparse_tensor()
    row, col, val = st.coo()
    inds_to_remove = [2]
    new_st = remove_connection_at_index(st, inds_to_remove)
    new_row, new_col, new_val = new_st.coo()
    # Sparse sizes shouldn't change
    assert st.sparse_sizes() == new_st.sparse_sizes()
    # The number of indices present should be reduced by the number of removed indices
    # Unless the tensor wasn't coalesced...
    if st.is_coalesced():
        assert (len(row) - len(inds_to_remove)) == len(new_row)
    # The value at the removed index should be an empty tensor
    for ind_to_remove in inds_to_remove:
        removed_index = new_st.index_select(0, row[ind_to_remove].view(1)).index_select(1, col[ind_to_remove].view(1))
        assert all([t.numel() == 0 for t in removed_index.coo()])

def test_remove_two_indexes() -> None:
    st = _example_sparse_tensor()
    row, col, val = st.coo()
    inds_to_remove = [2, 4]
    new_st = remove_connection_at_index(st, inds_to_remove)
    new_row, new_col, new_val = new_st.coo()
    # Sparse sizes shouldn't change
    assert st.sparse_sizes() == new_st.sparse_sizes()
    # The number of indices present should be reduced by the number of removed indices
    # Unless the tensor wasn't coalesced...
    if st.is_coalesced():
        assert (len(row) - len(inds_to_remove)) == len(new_row)
    # The value at the removed index should be an empty tensor
    for ind_to_remove in inds_to_remove:
        removed_index = new_st.index_select(0, row[ind_to_remove].view(1)).index_select(1, col[ind_to_remove].view(1))
        assert all([t.numel() == 0 for t in removed_index.coo()])



def test_remove_too_many_indexes() -> None:
    st = _example_sparse_tensor()
    row, col, val = st.coo()
    inds_to_remove = list(range(50))
    new_st = remove_connection_at_index(st, inds_to_remove)
    new_row, new_col, new_val = new_st.coo()
    # Sparse sizes shouldn't change
    assert st.sparse_sizes() == new_st.sparse_sizes()
    # The number of indices present should be reduced by the number of removed indices
    # Unless the tensor wasn't coalesced...
    if st.is_coalesced():
        assert max(0, len(row) - len(inds_to_remove)) == len(new_row)
    # The value at the removed index should be an empty tensor
    for ind_to_remove in inds_to_remove:
        if ind_to_remove >= len(row): continue
        removed_index = new_st.index_select(0, row[ind_to_remove].view(1)).index_select(1, col[ind_to_remove].view(1))
        assert all([t.numel() == 0 for t in removed_index.coo()])