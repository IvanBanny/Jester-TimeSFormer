import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat


class PatchEmbed(nn.Module):
    def __init__(self, frame_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.grid_size = frame_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) e h w -> b t (h w) e', b=B, t=T)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimeSformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()
        # Temporal attention
        self.norm1 = LayerNorm(dim)
        self.attn_temporal = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # Spatial attention
        self.norm2 = LayerNorm(dim)
        self.attn_spatial = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # MLP block
        self.norm3 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, T, P, C = x.shape

        # Temporal attention
        xt = rearrange(x, 'b t p c -> (b p) t c')
        xt = xt + self.attn_temporal(self.norm1(xt))
        xt = rearrange(xt, '(b p) t c -> b t p c', b=B)

        # Spatial attention
        xs = rearrange(xt, 'b t p c -> (b t) p c')
        xs = xs + self.attn_spatial(self.norm2(xs))
        xs = rearrange(xs, '(b t) p c -> b t p c', b=B)

        # MLP block
        xs = rearrange(xs, 'b t p c -> (b t p) c')
        xs = xs + self.mlp(self.norm3(xs))
        xs = rearrange(xs, '(b t p) c -> b t p c', b=B, t=T)

        return xs


class TimeSformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_frames=16,
            num_classes=27,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(
            frame_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TimeSformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # Final layers
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, T, N, D)

        cls_token = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=B, t=x.shape[1])
        x = torch.cat([cls_token, x], dim=2)

        x = x + self.temporal_embed + self.spatial_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling and classification
        x = self.norm(x)
        x = x[:, :, 0].mean(dim=1)  # Average over temporal dimension for cls token
        x = self.head(x)

        return x
