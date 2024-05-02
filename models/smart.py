import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from utils.utils import length_to_mask


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.mlp(self.norm2(x))
        return x


class SeqAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            proj_drop=0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask, lens, lens_mask):
        B, N, C = x.shape  # B*I, T, H
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        mask = mask.reshape(-1, N)
        mask = ((mask.unsqueeze(-1) + mask.unsqueeze(-2))).reshape(-1, 1, mask.shape[1], mask.shape[1]).repeat(1, self.num_heads, 1, 1)
        mask += (~(lens_mask.unsqueeze(-2) * lens_mask.unsqueeze(-1))).reshape(-1, 1, lens_mask.shape[1], lens_mask.shape[1]).repeat(1, self.num_heads, 1, 1).float() * -1e9
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqAttBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )

    def forward(self, x, mask, lens, lens_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, mask, lens, lens_mask)
        x = torch.reshape(x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + x
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            proj_drop=0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask, lens, lens_mask):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)

        q = q[:, 0]
        mask = mask.reshape(B, N, P, 1, 1).repeat(1, 1, 1, self.num_heads,
                                self.head_dim).permute(0, 2, 3, 1, 4)
        k = k.masked_fill(~mask.bool(), 0).sum(dim=1) / mask.sum(dim=1)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
        )

    def forward(self, x, mask, lens, lens_mask):
        x = x + self.attn_var(self.norm1(x), mask, lens, lens_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(-2)].clone().detach()


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, proj_drop=proj_drop,
                                         norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, proj_drop=proj_drop,
                                         norm_layer=norm_layer)

        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, 
                                    proj_drop=proj_drop, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, mask, lens, lens_mask):
        lens_mask = lens_mask.repeat_interleave(x.shape[1], dim=0)
        x = self.seq_att_block(x, mask, lens, lens_mask)
        x = self.var_att_block(x, mask, lens, lens_mask)
        x = self.mlp(x)
        return x


class MLPEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, mask):
        x = torch.stack((x, mask), dim=-1)
        x = self.embed(x)
        x = x.permute(0, 2, 1, 3)
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedder = MLPEmbedder(args.d_model)
        self.query = nn.Parameter(torch.zeros(args.input_dim, 1, args.d_model))
        self.query.data.normal_(mean=0.0, std=0.02)
        self.position_enc = PositionalEncoding(args.d_model, n_position=args.max_len + 1)
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False,
                        mlp_ratio=4., proj_drop=args.dropout) for l in range(args.e_layers)]
        )

    def forward(self, x, lens, mask, **kwargs):
        x = self.embedder(x, mask)
        x = torch.cat((self.query.repeat(x.shape[0], 1, 1, 1), x), dim=2)
        x = self.position_enc(x)
        lens_mask = length_to_mask(lens + 1)
        mask = torch.cat((torch.ones(mask.shape[0], 1, mask.shape[-1], device=mask.device, dtype=mask.dtype), mask), dim=1)
        mask = mask.transpose(1, 2).float()
        for block in self.blocks:
            x = block(x, mask, lens, lens_mask)
        return x


class EmbeddingDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = Mlp(
            in_features=args.d_model,
            hidden_features=int(args.d_model * 4),
            act_layer=nn.GELU,
            drop=args.dropout,
        )
        self.proj_out = nn.Linear(args.d_model, args.d_model)

    def forward(self, x):
        x = self.mlp(x)
        x = self.proj_out(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = MLPBlock(dim=args.d_model, mlp_ratio=4, 
                            proj_drop=args.dropout, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.out = nn.Linear(args.d_model * args.input_dim, args.num_class)
        
    def forward(self, h, **kwargs):
        B, I, T, H = h.shape
        cls_token = h[:, :, 0]
        cls_token = cls_token.reshape(B, I, H)
        cls_token = self.mlp(cls_token)
        logits = self.out(cls_token.reshape(B, -1))
        return logits
