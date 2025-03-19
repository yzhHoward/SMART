import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # b h t d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k ** 0.5 
    if mask is not None:  # 1 1 t t
        scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
    p_attn = F.softmax(scores, dim=-1)  # b h t t
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.d_k * self.h) for _ in range(3)])
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # 1 1 t t
            
        nbatches = query.size(0)  # b

        # d_model => h * d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # b num_head d_input d_k

        x, self.attn = attention(
            query, key, value, mask=mask,
            dropout=self.dropout)  # b num_head d_input d_v (d_k)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.final_linear(x), self.attn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class PositionwiseFeedForward(nn.Module):  # new added
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None


class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, dropout=None):
        super(FinalAttentionQKV, self).__init__()

        self.attention_hidden_dim = attention_hidden_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size, time_step, input_dim = input.size()
        input_q = self.W_q(torch.mean(input, 1))  # b, h
        input_k = self.W_k(input)  # b, d, h
        input_v = self.W_v(input)  # b, d, h

        q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1))  # b, h, 1
        e = torch.matmul(input_k, q).squeeze(2)  # b, d

        a = self.softmax(e)  # b, d
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze(1)  # b, d
        return v, a


class Concare(nn.Module):
    def __init__(self, args):
        super().__init__()

        # hyperparameters
        self.input_dim = args.input_dim
        self.hidden_dim = args.d_model  # d_model
        self.MHD_num_head = args.d_model
        self.d_ff = args.d_model
        self.output_dim = args.num_class

        # layers

        self.GRUs = nn.ModuleList([nn.GRU(1, self.hidden_dim, batch_first=True) for _ in range(self.input_dim)])
        self.SublayerConnection = SublayerConnection(self.hidden_dim, 0.1)
        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.hidden_dim, 0.1)
        self.PositionwiseFeedForward = PositionwiseFeedForward(self.hidden_dim, self.d_ff, dropout=0.1)
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, 0.1)

        # self.demo_proj_main = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

    def forward(self, x, lens, mask, **kwargs):
        # input shape [batch_size, timestep, feature_dim]
        # demo_main = self.tanh(self.demo_proj_main(series_static)).unsqueeze(1)# b hidden_dim

        B, T, I = x.shape
        assert (I == self.input_dim)  # input Tensor : 256 * 48 * 76
        assert (self.hidden_dim % self.MHD_num_head == 0)

        hs = []
        # x = torch.stack((x, mask), dim=-1)
        for i in range(I):
            hs.append(self.GRUs[i](x[:, :, i].reshape(B, T, -1))[0])
        hs = torch.stack(hs, dim=1)  # B, I, T, H
        index = (lens.reshape(-1, 1, 1, 1) - 1)  # B, 1, 1, 1
        ht = torch.gather(hs, -2, index.repeat(1, self.input_dim, 1, self.hidden_dim))
        ht = ht.reshape(B, I, -1)

        # ht = torch.cat((ht, demo_main), 1)# b i+1 h
        posi_input = self.dropout(ht)  # batch_size * d_input * hidden_dim

        contexts = self.SublayerConnection(
            posi_input, lambda x: self.MultiHeadedAttention(
                posi_input, posi_input, posi_input, None
            ))  # # batch_size * d_input * hidden_dim

        contexts = contexts[0]

        contexts = self.SublayerConnection(
            contexts, lambda x: self.PositionwiseFeedForward(contexts))[0]  # batch_size * d_input * hidden_dim

        weighted_contexts, final_attn = self.FinalAttentionQKV(posi_input) # batch_size * hidden_dim

        logits = self.output(weighted_contexts)  # b 1
        if kwargs.get("return_hidden", False) == True:
            return logits, weighted_contexts
        return logits
        
