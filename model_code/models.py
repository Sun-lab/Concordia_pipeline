import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DenseGraphConv, GCNConv
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.utils import to_dense_adj, to_dense_batch

from sparse_unsupervised_pooling import sparse_mincutpool


class GCN_model(torch.nn.Module):
    def __init__(self, cell_feature, n_cell_type_groups, gcn_type, n_gcns, 
                 num_clusters, mincut_type, predictor_type, skip_type):
        super(GCN_model, self).__init__()

        self.cell_feature = cell_feature
        self.n_cell_type_groups = n_cell_type_groups
        self.gcn_type = gcn_type
        self.n_gcns = n_gcns
        self.n_clusters = num_clusters
        self.mincut_type = mincut_type
        self.predictor_type = predictor_type
        self.skip_type = skip_type
        
        if self.skip_type in ["no", "add", "add2"]:
            self.skip_dim = 64
        elif self.skip_type in ["concat", "concat2"]:
            self.skip_dim = 2*64

        self.relu = nn.ReLU()

        if self.cell_feature == "ct":
            self.ct_embedding = torch.nn.Embedding(self.n_cell_type_groups, 64)
            torch.nn.init.xavier_uniform_(self.ct_embedding.weight.data)

        if self.cell_feature == 'comp':
            self.linear0 = Linear((2*self.n_cell_type_groups), 64)

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

        self.linear1 = Linear(self.skip_dim, self.n_clusters)

        if self.predictor_type == "prop":
            self.linear2 = Linear(self.n_clusters, 1)
        elif self.predictor_type == "atilde":
            self.conv4 = DenseGraphConv(64, 64)
            self.linear2 = Linear(64, 64)
            self.linear3 = Linear(64, 1)
        elif self.predictor_type in ["ave", "flat"]:
            self.linear2 = Linear(self.n_clusters * 64, 64)
            self.linear3 = Linear(64, 1)
        elif self.predictor_type in ["atildeflat", "centroidatildeflat"]:
            self.linear2 = Linear((self.n_clusters * (64 + self.n_clusters)), 64)
            self.linear3 = Linear(64, 1)

    def forward(self, x, edge_index, batch, n_cells):

        if self.cell_feature == "ct":
            x = self.ct_embedding(x[:,0].long())

        if self.cell_feature in ['comp', 'comp2nd']:
            x_01 = torch.nn.functional.one_hot(x[:,0].long(), num_classes=self.n_cell_type_groups)
            if self.cell_feature == "comp":
                x_02 = x[:, 1:(self.n_cell_type_groups+1)]
            if self.cell_feature == "comp2nd":
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
            elif self.skip_type in ["concat2"]:
                out = torch.cat((out, identity), 1)
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
        elif self.skip_type in ["concat", "concat2"]:
            out = torch.cat((out, identity), 1)

        out = self.relu(out)

        s = self.linear1(out)

        if self.mincut_type == "sparse_mincutpool":
            x_cluster, adj, s_batched, mc1, o1 = sparse_mincutpool(out, edge_index, batch, s)

        if self.predictor_type == "prop":
            x = s_batched.sum(dim=-2)
            x = torch.div(x, torch.unsqueeze(n_cells, 1))
            x = self.linear2(x)
        elif self.predictor_type=="atilde":
            x = self.conv4(x_cluster, adj)
            x = x.mean(dim=1)
            x = self.linear2(x).relu()
            x = self.linear3(x)
        elif self.predictor_type=="ave":
            s_sum = s_batched.sum(dim=-2)
            x = torch.div(x_cluster, s_sum.unsqueeze(-1))
            x = torch.flatten(x, start_dim=1)
            x = self.linear2(x).relu()
            x = self.linear3(x)
        elif self.predictor_type=="flat":
            x = torch.flatten(x_cluster, start_dim=1)
            x = self.linear2(x).relu()
            x = self.linear3(x)
        elif self.predictor_type=="centroidatildeflat":
            s_sum = s_batched.sum(dim=-2)
            x1 = torch.div(x_cluster, s_sum.unsqueeze(-1))
            x1 = torch.flatten(x1, start_dim=1)
            x2 = torch.flatten(adj, start_dim=1)
            x = torch.cat((x1, x2), 1)
            x = self.linear2(x).relu()
            x = self.linear3(x)
        elif self.predictor_type=="atildeflat":
            x1 = torch.flatten(x_cluster, start_dim=1)
            x2 = torch.flatten(adj, start_dim=1)
            x = torch.cat((x1, x2), 1)
            x = self.linear2(x).relu()
            x = self.linear3(x)

        if self.mincut_type in ["sparse_mincutpool"]:
            return x, s_batched, mc1, o1
