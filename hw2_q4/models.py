import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_scatter.scatter import scatter_sum

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'): # change for GAT
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        if args.model_type=='GAT' and args.num_heads > 1 and args.concat:
            hidden_dim = hidden_dim*args.num_heads

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ############################################################################
        # TODO: Your code here! 
        # Each layer in GNN should consist of a convolution (specified in model_type),
        # a non-linearity (use RELU), and dropout. 
        # HINT: the __init__ function contains parameters you will need. You may 
        # also find pyg_nn.global_max_pool useful for graph classification.
        # Our implementation is ~6 lines, but don't worry if you deviate from this.

        # x = None # TODO
        # x = self.convs[0](x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training) 
        if self.task == "graph":
            x = pyg_nn.global_max_pool(x, batch)

        ############################################################################

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

from torch_geometric.utils import add_remaining_self_loops

class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        ############################################################################
        # TODO: Your code here! 
        # Define the layers needed for the forward function. 
        # Our implementation is ~2 lines, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, out_channels) # it has been initialized at origin code None # TODO
        self.agg_lin = nn.Linear(in_channels, out_channels, bias=False) # None # TODO

        ############################################################################

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        ############################################################################
        # TODO: Your code here! 
        # Given x, perform the aggregation and pass it through a MLP with skip-
        # connection. Place the result in out. 
        # HINT: It may be useful to read the pyg_nn implementation of GCNConv,
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # Our implementation is ~4 lines, but don't worry if you deviate from this.
        # ?????????????????????????????????????????????????????????????????????
        # out = None # TODO
        x_message = F.relu(self.lin(x))
        edge_index, _ = add_remaining_self_loops(edge_index, None, 1., num_nodes) # gcn implementation add the self loops
        out = self.propagate(edge_index, size=(num_nodes, num_nodes), x=x_message, x_origin=x)
        ############################################################################

        return out

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        row, col = edge_index
        deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, x_origin):
        ############################################################################
        # TODO: Your code here! Perform the update step here. 
        # Our implementation is ~1 line, but don't worry if you deviate from this.
        aggr_out += self.agg_lin(x_origin)
        aggr_out = F.relu(aggr_out)
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2., dim=-1)# None # TODO

        ############################################################################

        return aggr_out


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=8, concat=True,
                 dropout=0, bias=True, node_dim=0, **kwargs):
        super(GAT, self).__init__(aggr='add', node_dim=node_dim, **kwargs)
        if in_channels == out_channels and concat:
            in_channels = out_channels*num_heads

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

        ############################################################################
        #  TODO: Your code here!
        # Define the layers needed for the forward function. 
        # Remember that the shape of the output depends the number of heads.
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, self.heads*self.out_channels, bias=bias)# None # TODO
        # self.node_dim = 0

        ############################################################################

        ############################################################################
        #  TODO: Your code here!
        # The attention mechanism is a single feed-forward neural network parametrized
        # by weight vector self.att. Define the nn.Parameter needed for the attention
        # mechanism here. Remember to consider number of heads for dimension!
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.att = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels*2)) #None # TODO

        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        ############################################################################

    def forward(self, x, edge_index, size=None):
        ############################################################################
        #  TODO: Your code here!
        # Apply your linear transformation to the node feature matrix before starting
        # to propagate messages.
        # Our implementation is ~1 line, but don't worry if you deviate from this.
        x = self.lin(x).view(-1, self.heads, self.out_channels)# None # TODO
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        ############################################################################
        #  TODO: Your code here! Compute the attention coefficients alpha as described
        # in equation (7). Remember to be careful of the number of heads with 
        # dimension!
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        # x_i, x_j = x_i.view(-1, self.heads, self.out_channels), x_j.view(-1, self.heads, self.out_channels)
        cat_feature = torch.cat([x_i, x_j], dim=-1) # None # TODO
        alpha_ij = F.leaky_relu((self.att*cat_feature).sum(dim=-1), negative_slope=0.2)
        # alpha = alpha_ij - alpha_ij.max()
        # alphai_tot = scatter_sum(src=alpha, index=edge_index_i, dim=0, dim_size=size_i)
        # alpha /= (alphai_tot[edge_index_i]+1e-16) # this implementation is alright.
        alpha = pyg_utils.softmax(alpha_ij, edge_index_i, num_nodes=size_i)

        ############################################################################

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * (alpha.view(-1, self.heads, 1))

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
