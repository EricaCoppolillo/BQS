import scipy.sparse as sp
import numpy as np
import torch


def convert_sparse_mat_to_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)


def create_sparse_bipartite_adjacency(data,
                                      user_num,
                                      item_num,
                                      self_connection=False):
    '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
    '''
    n_nodes = user_num + item_num
    row_idx = [pair[0] for pair in data]
    col_idx = [pair[1] for pair in data]
    user_np = np.array(row_idx)
    item_np = np.array(col_idx)
    ratings = np.ones_like(user_np, dtype=np.float32)
    tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + user_num)), shape=(n_nodes, n_nodes), dtype=np.float32)
    adj_mat = tmp_adj + tmp_adj.T
    if self_connection:
        adj_mat += sp.eye(n_nodes)
    return adj_mat


def normalize_graph_mat(adj_mat):
    shape = adj_mat.get_shape()
    rowsum = np.array(adj_mat.sum(1))
    if shape[0] == shape[1]:
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
    else:
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
    return norm_adj_mat
