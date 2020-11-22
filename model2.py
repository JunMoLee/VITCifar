import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from utils import get_rel_pos
from sparsemax import Sparsemax

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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., rel_pos=None, rel_pos_mul=False):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.rel_pos = rel_pos
        if self.rel_pos is not None:
            self.rel_pos_embedder = torch.nn.Embedding(self.rel_pos.max()+1, dim)
        self.rel_pos_mul = rel_pos_mul
        if self.rel_pos_mul:
            self.rel_pos_mul_embedder = torch.nn.Embedding(self.rel_pos.max()+1, heads)
            
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
#         self.sparsemax = Sparsemax()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if self.rel_pos is not None:
            rel_pos = self.rel_pos_embedder(self.rel_pos)  # i, j, (hxd)
            rel_pos = rearrange(rel_pos, 'i j (h d) -> h i j d', h=h)
            rel_pos_weights = torch.einsum('bhid,hijd->bhij', q, rel_pos)
            dots = dots + rel_pos_weights
            
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
#         attn = self.sparsemax(dots)
    
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, rel_pos, rel_pos_mul):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, rel_pos=rel_pos, rel_pos_mul=rel_pos_mul))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0., rel_pos=None, rel_pos_mul=False, n_out_convs=0, squeeze_conv=False):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size
        self.n_row_patch = image_size//patch_size
                                
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.rel_pos = get_rel_pos(self.n_row_patch).cuda()
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, self.rel_pos, rel_pos_mul)
        
        out_conv = []
        for _ in range(n_out_convs):
            out_conv.append(Residual(nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Dropout(dropout)
            )))
        self.out_conv = nn.Sequential(*out_conv)
        
       
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
        
        if self.squeeze_conv:
            x = self.squeeze_conv(x).squeeze(-1).squeeze(-1)
        else:
            x = x.mean(dim=(2,3))
        x = self.mlp_head(x)
        return x
