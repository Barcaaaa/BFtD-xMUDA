import numpy as np
import torch
from torch import nn


class Mult_EA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super(Mult_EA, self).__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef

        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x)  # (B,N,4C)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B,8,N,C/2)

        attn = self.linear_0(x)  # (B,8,N,k)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mult_CA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super(Mult_CA, self).__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.coef = 4
        self.trans_dims_k = nn.Linear(dim, dim * self.coef)
        self.trans_dims_q = nn.Linear(dim, dim * self.coef)
        self.trans_dims_v = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, sem_inp, geo_inp):
        """
        sem_inp: (1,C,80)
        geo_inp: (1,N,80)
        """
        _, C, _ = sem_inp.shape
        _, N, _ = geo_inp.shape

        # key
        sem_inp_k = self.trans_dims_k(sem_inp)  # (1,C,4*80)
        sem_inp_k = sem_inp_k.view(1, C, self.num_heads, -1).permute(0, 2, 1, 3)  # (1,8,C,40)
        # query
        geo_inp_q = self.trans_dims_q(geo_inp)  # (1,N,4*80)
        geo_inp_q = geo_inp_q.view(1, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (1,8,N,40)
        # value
        sem_inp_v = self.trans_dims_v(sem_inp)  # (1,C,4*80)
        sem_inp_v = sem_inp_v.view(1, C, self.num_heads, -1).permute(0, 2, 1, 3)  # (1,8,C,40)

        attn_map = geo_inp_q.matmul(sem_inp_k.permute(0, 1, 3, 2))  # (1,8,N,C)
        attn_map = attn_map.softmax(dim=-2)
        attn_map = attn_map / (1e-9 + attn_map.sum(dim=-1, keepdim=True))
        attn_map = self.attn_drop(attn_map)

        x = (attn_map.matmul(sem_inp_v)).permute(0, 2, 1, 3).reshape(1, N, -1)  # (1,N,8*40)
        x = self.proj(x)  # (1,N,80)
        x = self.proj_drop(x)

        return x


class Mult_CA_v1(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., kv_proj_kernel_size=1, bias=True, matmul_norm=True, **kwargs):
        super().__init__()
        self.d_embed = embed_dim
        self.n_head = num_heads
        self.d_head = embed_dim // num_heads

        self.matmul_norm = matmul_norm
        if self.matmul_norm:
            self.mat_norm = self.d_head ** -.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kv_proj_kernel_size, bias=bias)
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, kv_proj_kernel_size, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, batch_idx, batch_size):
        """Forward function."""
        """
        query_feats: [N1+N2+..., E]
        key_feats: [L=#cls, B, E]
        val_feats: [L=#cls, B, E]
        batch_idx: [N1+N2+..., ]

        return:    
            atted: [N1+N2+..., E]
        """

        # [N1+N2+..., E] -> [N1+N2+..., E]
        q = self.q_proj(query)
        # [L=#cls, B, E] -> [B, E, L=#cls]
        k = self.k_proj(key.permute(1, 2, 0))
        v = self.v_proj(value.permute(1, 2, 0))

        # q: [N1+N2+..., E] -> [N1+N2+..., #H, E//#H]
        # k: [B, E, L=#cls] -> [L=B, #H, E//#H, #cls]
        # v: [B, E, L=#cls] -> [L=B, #H, E//#H, #cls]
        # multihead format
        q = q.reshape(-1, self.n_head, self.d_head)
        k = k.reshape(batch_size, self.n_head, self.d_head, -1)
        v = v.reshape(batch_size, self.n_head, self.d_head, -1)
        atted_list = []

        # MHA in each frame
        for i in range(batch_size):
            cur_mask = batch_idx == i
            # [Ni, #H, E//#H] -> [#H, Ni, E//#H]
            cur_q = q[cur_mask].permute(1, 0, 2).contiguous()
            # [#H, E//#H, #cls]
            cur_k = k[i]
            # [#H, E//#H, #cls] -> [#H, #cls, E//#H]
            cur_v = v[i].permute(0, 2, 1).contiguous()

            # [#H, Ni, E//#H] x [#H, E//#H, #cls] ->  [#H, Ni, #cls]
            cur_sim_map = torch.bmm(cur_q, cur_k)
            if self.matmul_norm:
                cur_sim_map = self.mat_norm * cur_sim_map
            cur_sim_map = F.softmax(cur_sim_map, dim=-1)

            # [#H, Ni, #cls] x [#H, #cls, E//#H] -> [#H, Ni, E//#H]
            cur_atted = torch.bmm(cur_sim_map, cur_v)

            # reshape back: [#H, Ni, E//#H] -> [Ni, #H, E//#H]
            cur_atted = cur_atted.permute(1, 0, 2)
            atted_list.append(cur_atted)

        # cat: [N1+N2+..., #H, E//#H]
        atted = torch.cat(atted_list, dim=0)
        # reshape back: [N1+N2+..., #H, E//#H] -> [N1+N2+..., E]
        atted = self.out_proj(atted.reshape(-1, self.n_head * self.d_head))

        return atted


# if __name__ == '__main__':
#     input=torch.randn(50,49,512)
#     ea = ExternalAttention(d_model=512,S=8)
#     output=ea(input)
#     print(output.shape)