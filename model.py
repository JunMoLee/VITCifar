# based on https://github.com/lucidrains/vit-pytorch

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from utils import get_rel_pos

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.,
                 rel_pos=None):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.rel_pos = rel_pos
        if self.rel_pos is not None:
            self.rel_pos_embedder = torch.nn.Embedding(self.rel_pos.max() + 1, dim)

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        
        # relative positional encoding [ADDED]
        if self.rel_pos is not None:
            rel_pos = self.rel_pos_embedder(self.rel_pos)  # i, j, (hxd)
            rel_pos = rearrange(rel_pos, 'i j (h d) -> h i j d', h=h)
            rel_pos_weights = torch.einsum('bhid,hijd->bhij', queries, rel_pos)
            dots = dots + rel_pos_weights

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., rel_pos=None, rel_pos_mul=False):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        # relative positional encoding [ADDED]
        self.rel_pos = rel_pos
        if self.rel_pos is not None:
            self.rel_pos_embedder = torch.nn.Embedding(self.rel_pos.max() + 1, dim)
        # relative positional multiplier [ADDED]
        self.rel_pos_mul = rel_pos_mul
        if self.rel_pos_mul:
            self.rel_pos_mul_embedder = torch.nn.Embedding(self.rel_pos.max() + 1, heads)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),  # [ADDED]
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # relative positional encoding [ADDED]
        if self.rel_pos is not None:
            rel_pos = self.rel_pos_embedder(self.rel_pos)  # i, j, (hxd)
            rel_pos = rearrange(rel_pos, 'i j (h d) -> h i j d', h=h)
            rel_pos_weights = torch.einsum('bhid,hijd->bhij', q, rel_pos)
            dots = dots + rel_pos_weights

        # relative positional multiplier [ADDED]
        if self.rel_pos_mul:
            rel_pos_mul = self.rel_pos_mul_embedder(self.rel_pos).permute(2, 0, 1)  # h, i, j
            dots = dots * rel_pos_mul.unsqueeze(0).expand_as(dots)

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


# cnn-attention dual path architecture [ADDED]
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, rel_pos, rel_pos_mul, seq_len, linformer=0,
                 conv_ratio=0.5, n_mid_convs=5, sep_conv=False):
        super().__init__()
        self.seq_len = seq_len
        self.conv_dim = int(dim*conv_ratio)
        self.attention_dim = dim - self.conv_dim
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if self.conv_dim > 0:
                if sep_conv:
                    conv_blocks = [Residual(nn.Sequential(
                        nn.BatchNorm2d(self.conv_dim),
                        nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1, groups=self.conv_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.BatchNorm2d(self.conv_dim),
                        nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=1, stride=1, padding=0),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )) for _ in range(n_mid_convs)]
                else:
                    conv_blocks = [Residual(nn.Sequential(
                        nn.BatchNorm2d(self.conv_dim),
                        nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    )) for _ in range(n_mid_convs)]
                conv_layer = nn.Sequential(*conv_blocks)
            else:
                conv_layer = None
            if self.attention_dim > 0:
                if linformer:
                    attention = LinformerSelfAttention(dim=self.attention_dim, seq_len=seq_len, heads=heads,
                                                       k=linformer, rel_pos=rel_pos, dropout=dropout)
                else:
                    attention = Attention(dim=self.attention_dim, heads=heads, dropout=dropout, rel_pos=rel_pos,
                                          rel_pos_mul=rel_pos_mul)
                attention_layer = Residual(PreNorm(self.attention_dim, attention))
            else:
                attention_layer = None
            self.layers.append(nn.ModuleList([
                attention_layer,
                conv_layer,
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        if (self.conv_dim > 0) & (self.attention_dim > 0):
            self.linear = PreNorm(dim, nn.Linear(dim, dim))

    def forward(self, x, mask=None):
        for attn, conv, ff in self.layers:
            if self.conv_dim == 0:
                x = attn(x, mask=mask)
                x = ff(x)
            elif self.attention_dim == 0:
                conv_in = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(self.seq_len)))
                conv_out = conv(conv_in)
                x = rearrange(conv_out, 'b c h w -> b (h w) c')
            else:
                conv_in = rearrange(x[:, :, :self.conv_dim], 'b (h w) c -> b c h w', h=int(math.sqrt(self.seq_len)))
                conv_out = conv(conv_in)
                conv_out = rearrange(conv_out, 'b c h w -> b (h w) c')
                attn_out = ff(attn(x[:, :, self.conv_dim:], mask=mask))
                x = torch.cat((conv_out, attn_out), dim=-1)
                x = self.linear(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.,
                 emb_dropout=0., rel_pos=None, rel_pos_mul=False, n_out_convs=0, squeeze_conv=False, linformer=0,
                 conv_ratio=0.5, n_mid_convs=5, sep_conv=False):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size
        self.n_row_patch = image_size // patch_size

        # relative position [ADDED]
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.rel_pos = get_rel_pos(self.n_row_patch).cuda()
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, self.rel_pos, rel_pos_mul,
                                       seq_len=self.n_row_patch ** 2, linformer=linformer, conv_ratio=conv_ratio,
                                       n_mid_convs=n_mid_convs, sep_conv=sep_conv)
        # out convs [ADDED]
        out_conv = []
        for _ in range(n_out_convs):
            out_conv.append(Residual(nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Dropout(dropout)
            )))
        self.out_conv = nn.Sequential(*out_conv)

        # squeeze conv [ADDED]
        self.squeeze_conv = nn.Conv2d(dim, dim, kernel_size=self.n_row_patch, groups=dim) if squeeze_conv else None

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        if self.rel_pos is None:
            x += self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.n_row_patch)
        x = self.out_conv(x)

        # replaced cls token classification to patch weighted average classification
        if self.squeeze_conv:
            x = self.squeeze_conv(x).squeeze(-1).squeeze(-1)
        else:
            x = x.mean(dim=(2, 3))

        x = self.mlp_head(x)
        return x
