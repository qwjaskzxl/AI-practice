import random
import torch
import numpy as np
import scipy.sparse as sp
from a3_graph.pygcn.utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor


def load_data(path="../data/cora/", dataset="cora"):  # citeseer
    """Load citation network dataset (cora only for now)"""
    print('加载 {} 数据集...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))  # 将matrix变成了str
    features = sp.csr_matrix(idx_features_labels[:, 1:-1],
                             dtype=np.float32)  # 取出feature matrix
    labels = encode_onehot(idx_features_labels[:, -1])

    # 构建graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)  # 取出所有的边 [e,2]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)  # 构造稀疏邻接矩阵 [n,n], 转为原矩阵可以用toarray())
    # print(adj)
    # print()
    # print(adj.T)
    # exit()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 构建对称矩阵

    # print(adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 划分训练集测试号
    N = len(labels)
    idxs = list(range(N))
    random.shuffle(idxs)
    print(idxs)
    rate = 0.2
    idx_train = idxs[:int(N * rate)]
    idx_val = idx_train
    idx_test = idxs[int(N * rate):]

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
