import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast
from einops import rearrange, repeat


class PatchEmbed(nn.Module):
    """Convert video frames into patches and embed them"""

    def __init__(self, frame_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.grid_size = frame_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Process each frame independently
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)  # (B*T, embed_dim, grid_size, grid_size)
        x = rearrange(x, '(b t) e h w -> b t (h w) e', b=B, t=T)
        return x


class MultiheadAttention(nn.Module):
    """Custom implementation of multihead attention with separate time and space attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for temporal and spatial attention
        self.qkv_temporal = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_spatial = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_temporal(self, x):
        with autocast("cuda"):
            B, T, N, C = x.shape
            qkv = self.qkv_temporal(x).reshape(B, T, N, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, T, N, C)
            return x

    def forward_spatial(self, x):
        with autocast("cuda"):
            B, T, N, C = x.shape
            qkv = self.qkv_spatial(x).reshape(B, T, N, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(2, 3).reshape(B, T, N, C)
            return x

    def forward(self, x, time_first=True):
        if time_first:
            x = self.forward_temporal(x)
            x = self.forward_spatial(x)
        else:
            x = self.forward_spatial(x)
            x = self.forward_temporal(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Simple MLP block"""

    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeSformerBlock(nn.Module):
    """Basic TimeSformer block combining attention and MLP"""

    def __init__(
            self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
            drop=0., attn_drop=0., time_first=True
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiheadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=drop
        )
        self.time_first = time_first
        self.use_checkpoint = False

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = x + checkpoint(lambda x: self.attn(self.norm1(x), self.time_first), x, use_reentrant=False)
            x = x + checkpoint(self.mlp, self.norm2(x), use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x), time_first=self.time_first)
            x = x + self.mlp(self.norm2(x))
        return x


class TimeSformer(nn.Module):
    """Main TimeSformer architecture optimized for Jester dataset"""

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
            time_first=True
    ):
        super().__init__()
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbed(
            frame_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
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
                time_first=time_first
            )
            for _ in range(depth)
        ])

        # Final classification head
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize network weights"""

        def _init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

    def set_gradient_checkpointing(self, enable=True):
        for block in self.blocks:
            block.use_checkpoint = enable

    def forward(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, T, N, D)

        # Expand cls_token for batch and time dimensions
        cls_token = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=B, t=x.shape[1])
        x = torch.cat([cls_token, x], dim=2)

        # Add positional embedding and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling over patches and time
        x = self.norm(x)
        x = x[:, :, 0, :].mean(dim=1)  # Average over time dimension for cls token

        # Classification head
        x = self.head(x)
        return x
