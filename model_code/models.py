import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGraphConv, GCNConv
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.utils import to_dense_adj, to_dense_batch

from sparse_mincut_pool import sparse_mincut_pool


class GCN_model(torch.nn.Module):
    def __init__(self, cell_feature, n_cell_type_groups, gcn_type, n_gcns, 
                 s_dim2, mincut_type, skip_type):
        super(GCN_model, self).__init__()

        self.cell_feature = cell_feature
        self.n_cell_type_groups = n_cell_type_groups
        self.gcn_type = gcn_type
        self.n_gcns = n_gcns
        self.s_dim2 = s_dim2
        self.mincut_type = mincut_type
        self.skip_type = skip_type
        
        if self.skip_type in ["no", "add", "add2"]:
            self.skip_dim = 64
        else:
            raise ValueError(f"Unsupported skip_type: {self.skip_type}")

        self.relu = nn.ReLU()

        if self.cell_feature == 'comp2nd':
            self.linear0 = Linear((3*self.n_cell_type_groups), 64)

        if self.n_gcns > 2:
            if self.gcn_type == "gat":
                self.conv1 = GATConv(64, 16, 4)
            elif self.gcn_type == "gat2":
                self.conv1 = GATv2Conv(64, 16, 4)
            else:
                self.conv1 = GCNConv(64, 64)

        # for the case of adding skip connection right after the first convolution
        # only the version for two graph convolution layers is prepared

        if self.n_gcns > 1:
            if self.gcn_type == "gat":
                self.conv2 = GATConv(64, 16, 4)
            elif self.gcn_type == "gat2":
                self.conv2 = GATv2Conv(64, 16, 4)
            else:
                self.conv2 = GCNConv(64, 64)

        if self.gcn_type == "gat":
            self.conv3 = GATConv(self.skip_dim, 16, 4)
        elif self.gcn_type == "gat2":
            self.conv3 = GATv2Conv(self.skip_dim, 16, 4)
        else:
            self.conv3 = GCNConv(self.skip_dim, 64)

        self.linear1 = Linear(self.skip_dim, self.s_dim2)

    def forward(self, x, edge_index, batch, n_cells):

        if self.cell_feature == 'comp2nd':
            x_01 = torch.nn.functional.one_hot(x[:,0].long(), num_classes=self.n_cell_type_groups)
            x_02 = x[:, 1:(2*self.n_cell_type_groups+1)]
            x = torch.cat((x_01, x_02), 1)

            x = self.linear0(x)
            x = self.relu(x)

        identity = x

        if self.n_gcns == 1:
            out = self.conv3(x, edge_index)

        if self.n_gcns == 2:
            out = self.conv2(x, edge_index)
            if self.skip_type in ["add2"]:
                out += identity
            out = self.relu(out)
            out = self.conv3(out, edge_index)    

        if self.n_gcns == 3:
            out = self.conv1(x, edge_index)
            out = self.relu(out)
            out = self.conv2(out, edge_index)
            out = self.relu(out)
            out = self.conv3(out, edge_index)    

        if self.skip_type in ["add", "add2"]:
            out += identity

        out = self.relu(out)

        s = self.linear1(out)

        if self.mincut_type == "sparse_mincut_pool":
            x_cluster, adj, mc1, o1 = sparse_mincut_pool(out, edge_index, batch, s)
            return mc1, o1
