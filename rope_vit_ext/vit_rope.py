from itertools import combinations

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Axial1DRoPE(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)
    
    def apply_rotary_pos_emb(self, q, k, sinu_pos):
        sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
        sin, cos = sinu_pos.unbind(dim = -2)

        sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
        q, k = map(lambda t: (t * cos) + (self.swap_first_two(t) * sin), (q, k))
        return q, k

    def swap_first_two(self, x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x_clone = x.clone()
        x_clone[..., [0, 1]] = x_clone[..., [1, 0]]
        x_clone[..., 0] = -x_clone[..., 0]
        return rearrange(x_clone, '... d j -> ... (d j)')

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)


class Axial2DRoPE(nn.Module):
    def __init__(self, dim, max_seq_len, image_size, patch_size, N=2):
        super().__init__()
        theta = 100.0
        self.N = N
        factor = 2 * N # 2D axial x N-D subspace
        freqs_x = 1.0 / (theta ** (torch.arange(0, dim, factor)[: (dim // factor)].float() / dim))
        freqs_y = 1.0 / (theta ** (torch.arange(0, dim, factor)[: (dim // factor)].float() / dim))
        end_x, end_y = image_size // patch_size, image_size // patch_size
        t_x, t_y = Axial2DRoPE.init_t_xy(end_x, end_y)
        freqs_x = torch.outer(t_x, freqs_x)
        freqs_y = torch.outer(t_y, freqs_y)
        emb = torch.cat((freqs_x.sin(), freqs_x.cos(), freqs_y.sin(), freqs_y.cos()), dim=-1)
        self.register_buffer('emb', emb)

    @staticmethod
    def init_t_xy(end_x, end_y):
        t = torch.arange(end_x * end_y, dtype=torch.float32)
        t_x = (t % end_x).float()
        t_y = torch.div(t, end_x, rounding_mode='floor').float()
        return t_x, t_y

    def apply_rotary_pos_emb(self, q, k, sinu_pos):
        sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=4)
        sin_x, cos_x, sin_y, cos_y = sinu_pos.unbind(dim=-2)

        sin_x, cos_x, sin_y, cos_y = map(
            lambda t: repeat(t, 'b n -> b (n j)', j=self.N), 
            (sin_x, cos_x, sin_y, cos_y)
        )
        dummy = torch.zeros_like(sin_x)

        # mask every Nth column
        for i in range(3, self.N+1):
            sin_x[..., i-1::self.N] = 0.0
            cos_x[..., i-1::self.N] = 0.0
            sin_y[..., i-1::self.N] = 0.0
            cos_y[..., i-1::self.N] = 0.0
            dummy[..., i-1::self.N] = 1.0

        _q = rearrange(q, '... n (j d) -> ... n j d', j=2)
        _qx, _qy = _q.unbind(dim=-2)
        _k = rearrange(k, '... n (j d) -> ... n j d', j=2)
        _kx, _ky = _k.unbind(dim=-2)

        qx, kx = map(
            lambda t: (t * cos_x) + (self.swap_first_two(t) * sin_x) + (t * dummy), (_qx, _kx)
        )
        qy, ky = map(
            lambda t: (t * cos_y) + (self.swap_first_two(t) * sin_y) + (t * dummy), (_qy, _ky)
        )

        q = torch.cat((qx, qy), dim=-1)
        k = torch.cat((kx, ky), dim=-1)
        return q, k

    def swap_first_two(self, x):
        x = rearrange(x, '... (d j) -> ... d j', j=self.N)
        x_clone = x.clone()
        x_clone[..., [0, 1]] = x_clone[..., [1, 0]]
        x_clone[..., 0] = -x_clone[..., 0]
        return rearrange(x_clone, '... d j -> ... (d j)')

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)


class WeightedAxial2DRoPE(Axial2DRoPE):
    @staticmethod
    def generate_index_pairs(N):
        indices = list(range(N))
        index_pairs = list(combinations(indices, 2))
        return index_pairs
    
    def swap_the_two(self, x, m, n):
        x = rearrange(x, '... (d j) -> ... d j', j=self.N)
        x_clone = x.clone()
        x_clone[..., [m, n]] = x_clone[..., [n, m]]
        x_clone[..., m] = -x_clone[..., m]
        return rearrange(x_clone, '... d j -> ... (d j)')

    def apply_rotary_pos_emb(self, q, k, sinu_pos):
        sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=4)
        sin_x, cos_x, sin_y, cos_y = sinu_pos.unbind(dim=-2)

        sin_x, cos_x, sin_y, cos_y = map(
            lambda t: repeat(t, 'b n -> b (n j)', j=self.N), 
            (sin_x, cos_x, sin_y, cos_y)
        )
        dummy = torch.zeros_like(sin_x)

        weighted_q = torch.zeros_like(q)
        weighted_k = torch.zeros_like(k)
        pairs = WeightedAxial2DRoPE.generate_index_pairs(self.N)
        for m, n in pairs:
            # mask every Nth column
            for i in range(self.N):
                if i != m and i != n:
                    sin_x[..., i::self.N] = 0.0
                    cos_x[..., i::self.N] = 0.0
                    sin_y[..., i::self.N] = 0.0
                    cos_y[..., i::self.N] = 0.0
                    dummy[..., i::self.N] = 1.0

            _q = rearrange(q, '... n (j d) -> ... n j d', j=2)
            _qx, _qy = _q.unbind(dim=-2)
            _k = rearrange(k, '... n (j d) -> ... n j d', j=2)
            _kx, _ky = _k.unbind(dim=-2)

            qx, kx = map(
                lambda t: (t * cos_x) + (self.swap_the_two(t, m, n) * sin_x) + (t * dummy), (_qx, _kx)
            )
            qy, ky = map(
                lambda t: (t * cos_y) + (self.swap_the_two(t, m, n) * sin_y) + (t * dummy), (_qy, _ky)
            )

            q = torch.cat((qx, qy), dim=-1)
            k = torch.cat((kx, ky), dim=-1)
            weighted_q = weighted_q + q
            weighted_k = weighted_k + k
        weighted_q /= len(pairs)
        weighted_k /= len(pairs)
        return weighted_q, weighted_k


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, pos_emb=None, apply_pos_emb_fn=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        if pos_emb is not None and apply_pos_emb_fn is not None:
            q, k = apply_pos_emb_fn(q, k, pos_emb)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
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
    def forward(self, x, pos_emb, apply_pos_emb_fn=None):
        for attn, ff in self.layers:
            x = attn(x, pos_emb=pos_emb, apply_pos_emb_fn=apply_pos_emb_fn) + x
            x = ff(x) + x
        return x


class ViTRoPE(nn.Module):
    def __init__(
            self, 
            *, 
            image_size, 
            patch_size, 
            num_classes, 
            dim, 
            depth, 
            heads, 
            mlp_dim, 
            pool = 'cls', 
            channels = 3, 
            dim_head = 64, 
            dropout = 0., 
            emb_dropout = 0., 
            rotary_position_emb = "1D_axial", 
            rotation_matrix_dim = 2, 
            weighted_rope = False,
        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        max_seq_len = image_size * image_size
        if rotary_position_emb == "1D_axial":
            self.layer_pos_emb = Axial1DRoPE(dim_head, max_seq_len)
        elif rotary_position_emb == "2D_axial":
            if weighted_rope:
                self.layer_pos_emb = WeightedAxial2DRoPE(
                    dim_head, max_seq_len, image_size=image_size, patch_size=patch_size, N=rotation_matrix_dim, 
                )
            else:
                self.layer_pos_emb = Axial2DRoPE(
                    dim_head, max_seq_len, image_size=image_size, patch_size=patch_size, N=rotation_matrix_dim, 
                )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # keep or remove cls token??
        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        #x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(
            x, 
            pos_emb=layer_pos_emb, 
            apply_pos_emb_fn=self.layer_pos_emb.apply_rotary_pos_emb, 
        )

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

