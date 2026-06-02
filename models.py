import geoopt
import networkx as nx
import numpy as np
import torch
from torch_geometric import nn as gnn
from sklearn.metrics import roc_auc_score

from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool

class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1, bias_term=True):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes, bias=bias_term)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units, bias=bias_term),
                      # nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units, bias=bias_term),
                               # nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes, bias=bias_term))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_units=32, decoder_out_dim=128, num_layers=3, dropout=0.15,
                 mlp_layers=2, train_eps=False, is_encoder=True, use_kappa=True, kappa_value=-1.0):
        super(Encoder, self).__init__()
        convs, bns = [], []

        for i in range(num_layers):
            input_dim = num_features if i == 0 else hidden_units

            if is_encoder:
                hidden_dim = hidden_units
            else:
                hidden_dim = hidden_units if i != num_layers - 1 else decoder_out_dim

            if use_kappa:
                convs.append(kappaGCNConv(k=kappa_value, in_dim=input_dim, out_dim=hidden_dim, learnable=True))
                bns.append(nn.Identity())
            else:
                convs.append(gnn.GINConv(MLP(input_dim, hidden_dim, hidden_dim, mlp_layers),
                                         train_eps=train_eps))
                bns.append(nn.BatchNorm1d(hidden_dim))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
        self.is_encoder = is_encoder
        self.use_kappa = use_kappa

        if self.is_encoder != True:  # Add learnable mask parameters
            self.encoder_mask = torch.nn.Parameter(torch.zeros(decoder_out_dim))
            self.decoder_mask = torch.nn.Parameter(torch.zeros(int(num_features)))

        self.final_layers = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        if self.is_encoder:  # Use layerwise-concatenation

            h_list = [x]

            for conv, bn in zip(self.convs, self.bns):
                h = conv(h_list[-1], edge_index)

                if self.use_kappa:
                    h_tan = conv.manifold.logmap0(h)
                    h_tan = self.dropout_layer(h_tan)
                    h_tan = torch.relu(h_tan)
                    h = conv.manifold.expmap0(h_tan)

                else:
                    h = self.dropout_layer(h)
                    h = torch.relu(h)

                h_list.append(h)

            out = torch.cat(h_list[1:], 1)
            return out

class kappaLinear(nn.Module):
    def __init__(self, manifold, in_dim: int, out_dim: int, dropout: float = 0.0, use_bias: bool = True):
        super(kappaLinear, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, project=True)
        if self.use_bias:
            bias = self.manifold.proju(self.manifold.origin(self.bias.shape), self.bias)
            kappa_bias = self.manifold.expmap0(bias, project=True)
            res = self.manifold.mobius_add(res, kappa_bias, project=True)
        return res


class kappaGCNConv(MessagePassing):
    def __init__(self, k, in_dim: int, out_dim: int, learnable=True):
        super().__init__(aggr='add')
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.lin = kappaLinear(manifold=self.manifold, in_dim=in_dim, out_dim=out_dim, use_bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index)
        x = self.lin(x)

        x_tan0 = self.manifold.logmap0(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x_tan0, norm=norm)
        out = self.manifold.expmap0(out, project=True)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j