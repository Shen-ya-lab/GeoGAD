import numpy as np
from torch_geometric.utils import to_dense_adj


import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset

def load_torch_dataset(data, device):

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, 'data')

    if data == "mutag":
        dataset = TUDataset(root=data_root, name='Mutagenicity', use_node_attr=True,
                            transform=T.RemoveIsolatedNodes())
        BSize = 256  # Batch size

    elif data == "enzymes":
        dataset = TUDataset(root=data_root, name='ENZYMES', use_node_attr=True)
        CD = []
        for ie, d in enumerate(dataset):
            if d.edge_index.shape[1] >= 2:
                CD.append(ie)

        dataset = dataset[CD]
        BSize = 128  # Batch size

    elif data == "er_md":
        dataset = TUDataset(root=data_root, name='ER_MD', use_node_attr=True,
                            transform=T.RemoveIsolatedNodes())
        CD = []
        for ie, d in enumerate(dataset):
            if d.edge_index.shape[1] >= 2:
                CD.append(ie)

        dataset = dataset[CD]
        BSize = 128  # Batch size

    elif data == "proteins":
        dataset = TUDataset(root=data_root, name='PROTEINS', use_node_attr=True)
        BSize = 256

    elif data == "dhfr":
        dataset = TUDataset(root=data_root, name='DHFR', use_node_attr=True)
        BSize = 256

    elif data == "bzr":
        dataset = TUDataset(root=data_root, name='BZR', use_node_attr=True)
        BSize = 256

    elif data == "aids":
        dataset = TUDataset(root=data_root, name='AIDS', use_node_attr=True,
                            transform=T.RemoveIsolatedNodes())
        BSize = 256

    elif data == "dd":
        dataset = TUDataset(root=data_root, name='DD', use_node_attr=True)
        BSize = 128

    elif data == "imdb":
        dtype = 'bio'
        X_bucket = None
        dataset = TUDataset(root=data_root, name='IMDB-BINARY', use_node_attr=True)
        total_degree = set()

        for i in range(1000):
            EE1, EE2 = torch.unique(dataset[i].edge_index, return_counts=True)
            EE1 = EE1.numpy()
            EE2 = EE2.numpy()

            for v1, v2 in zip(EE1, EE2):
                total_degree.add(int(v2 / 2))

        dataset = TUDataset(root=data_root, name='IMDB-BINARY', use_node_attr=True,
                            transform=T.OneHotDegree(max_degree=max(total_degree)))
        BSize = 128

    else:
        raise TypeError("Check data name")

    return dataset, BSize


def create_train_valid_test(dataset, difficulty='easy', anom_type=0):
    np.random.seed(0)
    ratio = 1.0

    Y = []
    for d in dataset:
        Y.append(int(d.y.item()))

    if anom_type == 0:
        normD = np.where(np.array(Y) == 1)[0]
        orig_anomD = np.where(np.array(Y) == 0)[0]

    elif anom_type == 1:
        normD = np.where(np.array(Y) == 0)[0]
        orig_anomD = np.where(np.array(Y) == 1)[0]

    splits = []
    n_normal_train = round(normD.shape[0] * 0.7)
    n_normal_valid = round(normD.shape[0] * 0.1)
    n_normal_test = normD.shape[0] - n_normal_train - n_normal_valid

    anomD = np.random.choice(orig_anomD, round(orig_anomD.shape[0] * 0.1),
                             replace=False)

    n_anom_train = round(anomD.shape[0] * 0.0)
    n_anom_valid = round(anomD.shape[0] * 0.5)

    n_anom_test = anomD.shape[0] - n_anom_train - n_anom_valid

    if isinstance(difficulty, float):
        n_noisy = round(n_normal_train * difficulty)

        print("Anomalies are mixed up for: {0}".format(n_noisy))

    for i in range(5):

        np.random.seed(i)
        anomD = np.random.choice(orig_anomD, round(orig_anomD.shape[0] * 0.1), replace=False)


        np.random.shuffle(normD)
        np.random.shuffle(anomD)

        to_be_added = []

        if isinstance(difficulty, float):

            anom_candids = np.array(list(set(orig_anomD) - set(anomD)))

            if anom_candids.shape[0] <= n_noisy:
                n_noisy = anom_candids.shape[0]

            to_be_added = list(np.random.choice(a=anom_candids, size=n_noisy, replace=False))

        train_graphs = (list(normD[:round(n_normal_train * ratio)]) + to_be_added, [1] *
                    round(n_normal_train * ratio) + [0] * len(to_be_added))

        valid_graphs = (list(normD[n_normal_train:n_normal_train + n_normal_valid]) + list(
            anomD[n_anom_train:n_anom_train + n_anom_valid]), [1] * n_normal_valid + [0] * n_anom_valid)
        test_graphs = (list(normD[n_normal_train + n_normal_valid:]) + list(anomD[n_anom_train + n_anom_valid:]),
                       [1] * n_normal_test + [0] * n_anom_test)

        splits.append((train_graphs, valid_graphs, test_graphs))

    return splits


def prepare_data(dataset, device, gamma=1.0):
    labels = []
    labels_pos_weights = []

    for d in dataset:
        adj = to_dense_adj(d.edge_index)[0]
        n_nodes_x = d.x.shape[0]

        adj = to_dense_adj(d.edge_index)[0]
        n_nodes_edge = adj.shape[0]


        adj_aligned = torch.zeros((n_nodes_x, n_nodes_x), dtype=adj.dtype)
        take = min(n_nodes_edge, n_nodes_x)
        adj_aligned[:take, :take] = adj[:take, :take]
        n_nodes = n_nodes_x

        adj_aligned.fill_diagonal_(1.0)

        adj_flat = adj_aligned.flatten().to(device)

        adj_sum = adj_aligned.sum()
        if adj_sum == 0:
            pos_weight = 1.0
        else:
            pos_weight = (float(n_nodes * n_nodes - adj_sum) / adj_sum) ** gamma

        labels.append(adj_flat)
        labels_pos_weights.append(pos_weight)

    return labels, labels_pos_weights