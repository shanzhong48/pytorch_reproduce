import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.act(self.l1(x))
        out = self.dropout(out)
        out = self.l2(out)
        out = self.dropout(out)
        return out


class MyAttention(nn.Module):
    def __init__(self, in_dim, dim_head, num_heads, dropout=0):
        super().__init__()
        self.dim = dim_head
        self.num_heads = num_heads
        self.proj_out = not (num_heads == 1 and in_dim == dim_head)

        self.dropout = nn.Dropout(dropout)
        self.attend = nn.Softmax(dim = -1)

        self.to_kqv = nn.Linear(in_dim, dim_head * 3 * num_heads, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim_head * num_heads, in_dim),
            self.dropout,
        ) if self.proj_out else nn.Identity()
    
    def forward(self, x):
        # x: bs, len, dim 
        kqv = self.to_kqv(x).chunk(3, dim=-1)
        k, q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), kqv)

        scaling = self.dim ** (-0.5)

        attn = self.attend(q @ k.transpose(-1,-2) * scaling) # (b, h, n, n)
        attn = self.dropout(attn)

        out = attn @ v # (b, h, n, d)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class MyTransformer(nn.Module):
    def __init__(self, depth, in_dim, dim_head, num_heads, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(in_dim, Attention(in_dim, dim_head, num_heads, dropout)),
                PreNorm(in_dim, FeedForward(in_dim, hidden_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class myViT(nn.Module):
    def __init__(self, imag_size, patch_size, num_classes, dim, depth, heads, dim_head, mlp_dim, pool = 'cls', num_channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        imag_h, imag_w = pair(imag_size)
        patch_h, patch_w = pair(patch_size)

        assert imag_h % patch_h == 0 and imag_w % patch_w == 0, "Image dimensions must be divisible by the patch size."
        assert pool in {'mean', 'cls'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool

        num_patches = (imag_h // patch_h) * (imag_w // patch_w)
        patch_dim = num_channels * patch_h * patch_w
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = patch_h, pw = patch_w),
            nn.LayerNorm(patch_dim), 
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, images):
        x = self.to_patch_embedding(images)
        b, n, _ = x.shape

        extra_cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((extra_cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = torch.mean(x, dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

## test comment