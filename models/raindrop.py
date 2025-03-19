import math
import numpy as np
from typing import Union, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import gather_csr, scatter, segment_csr


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{x}_i, \mathbf{m}_i, \mathbf{x}_i - \mathbf{m}_i ])`
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_weights=None,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor,edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if edge_weights is not None:
            alpha = edge_weights.unsqueeze(-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Observation_progation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,
                 n_nodes: int, ob_dim: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1],  heads * out_channels*8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.ob_dim = ob_dim
        self.index = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, edge_index: Adj, edge_weights=None, use_beta=False,
                edge_attr: OptTensor = None, return_attention_weights=None):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message_selfattention(self, x_i: Tensor, x_j: Tensor,edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if edge_weights is not None:
            alpha = edge_weights.unsqueeze(-1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out *= alpha.view(-1, self.heads, 1)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        use_beta = self.use_beta
        if use_beta == True:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)

            p_emb = self.p_t.unsqueeze(0)

            aa = torch.cat([w_v.repeat(1, n_step, 1,), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta == True:
                gamma = beta*(edge_weights.unsqueeze(-1))
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)

                # edge prune, prune out half of edges
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)

        self.index = index
        if use_beta == True:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma

        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        decompose = False
        if decompose == False:
            out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        else:
            source_nodes = self.edge_index[0]
            target_nodes = self.edge_index[1]
            w1 = self.nodewise_weights[source_nodes].unsqueeze(-1)
            w2 = self.nodewise_weights[target_nodes].unsqueeze(1)
            out = torch.bmm(x_i.view(-1, self.heads, self.out_channels), torch.bmm(w1, w2))
        if use_beta == True:
            out = out * gamma.view(-1, self.heads, out.shape[-1])
        else:
            out = out * gamma.view(-1, self.heads, 1)
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe


class Raindrop_v2(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, args):
        super().__init__()
        d_inp=args.input_dim
        d_model=d_inp * 4
        nhead=2
        nhid=2 * d_model
        nlayers=2
        dropout=0.2
        max_len=args.max_len
        d_static=9
        MAX=100
        aggreg='mean'
        n_classes=args.num_class
        global_structure=torch.ones(d_inp, d_inp)
        sensor_wise_mask=False
        static=False
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        d_pe = 16
        d_enc = d_inp

        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        self.d_ob = int(d_model/d_inp)

        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        self.max_len = max_len

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, x, lens, mask, **kwargs):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        if x.shape[1] < self.max_len:
            pad = torch.zeros([x.shape[0], self.max_len-x.shape[1], x.shape[2]], device=x.device)
            x = torch.cat([x, pad], dim=1)
            mask = x = torch.cat([mask, pad], dim=1)
        src = x.transpose(0, 1)
        times = torch.arange(src.shape[0]).unsqueeze(1).repeat([1, src.shape[1]]).cuda() + 1
        static = None
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = mask
        # src = src[:, :, :int(src.shape[2]/2)]

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lens.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(self.d_inp).bool()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda()

            use_beta = False
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]

                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lens.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        logits = self.mlp_static(output)
        if kwargs.get("return_hidden", False) == True:
            return logits, output
        return logits #, distance, None
