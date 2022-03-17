import os
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import torch
from pyHSICLasso import HSICLasso
from sklearn.preprocessing import LabelEncoder


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def encode_onehot(labels):
    classes = []
    for label in labels:
        if label not in classes:
            classes.append(label)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mx = mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split_graph_data(data, train_index, test_index, node_pairs):
    """Load citation network dataset (cora only for now)"""
    features_df = data.iloc[:, :-1]
    labels_df = data.iloc[:, -1]
    features = sp.csr_matrix(features_df.values, dtype=np.float32)
    labels = encode_onehot(labels_df.values)

    # build graph
    idx = list(data.index)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = node_pairs.values
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    idx_train = train_index
    idx_test = test_index
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_test


def hsic_lasso(x: pd.DataFrame, y: pd.DataFrame):
    le = LabelEncoder()
    hsic_lasso = HSICLasso()
    x, y, _l = x.to_numpy(), y.to_numpy(), x.columns.tolist()
    y = le.fit_transform(y)
    hsic_lasso.input(x, y, featname=_l)
    hsic_lasso.classification(1000, B=0, n_jobs=20)
    genes = hsic_lasso.get_features()
    return genes


def feature_selection(x: pd.DataFrame, y: pd.DataFrame, cancer_name):
    path = 'data/' + cancer_name + '_hsic_lasso.csv'
    if os.path.exists(path):
        genes_tran_df = pd.read_csv(path, header=0, index_col=None)
    else:
        print(cancer_name, 'HSIC-Lasso processing...')
        genes = hsic_lasso(x, y)
        genes_tran_df = pd.DataFrame(genes, columns=['genes'])
        genes_tran_df.to_csv(path)
        print(cancer_name, 'HSIC-Lasso finished')
    return genes_tran_df


def pairs_graph(genes_data, cancer_name):
    path = 'data/' + cancer_name + '_' + '_pairs_graph.csv'
    if os.path.exists(path):
        return pd.read_csv(path, header=0, index_col=0)
    results = []
    print(cancer_name, 'Pairs graph generating...')
    for i in range(genes_data.shape[0]):
        for j in range(i + 1, genes_data.shape[0]):
            x1 = genes_data.iloc[i, :]
            x2 = genes_data.iloc[j, :]
            cor, pvalue = scipy.stats.pearsonr(x1, x2)
            if pvalue >= 0.05:
                continue
            results.append({'i': i + 1, 'j': j + 1, 'spearman': cor})
    res = pd.DataFrame(results)
    res.to_csv(path)
    print(cancer_name, 'Pairs graph generated..')
    return res
