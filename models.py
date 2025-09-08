import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import timm

class MultiViewSwinModel(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224.ms_in22k",
                 pretrained=False, num_classes=3, attn_dim=1024, num_heads=4):
        super(MultiViewSwinModel, self).__init__()

        # Load backbone without classification head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features  # should be 1024 for swin_base

        # Self-attention across views (3 views → sequence length 3)
        self.attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)

        # Final classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_feat(self, x):
        # Get unpooled features (B, H*W, C) and then pooled vector (B, C)
        features = self.backbone.forward_features(x)         # (B, H*W, C)
        pooled = self.backbone.forward_head(features, pre_logits=True)  # (B, C)
        return pooled

    def forward(self, img1, img2, img3):
        # Per-view embeddings
        f1 = self.extract_feat(img1)  # [B, C]
        f2 = self.extract_feat(img2)
        f3 = self.extract_feat(img3)

        # Stack views to form a sequence [B, 3, C]
        views = torch.stack([f1, f2, f3], dim=1)

        # Self-attention across views
        attn_out, _ = self.attn(views, views, views)  # [B, 3, C]

        # Aggregate (mean pooling across 3 views)
        fused = attn_out.mean(dim=1)  # [B, C]

        return self.classifier(fused)


class MultiViewSwinCrossAttn(nn.Module):
    """
    Cross‑attention Swin for 3‑view breast‑MRI (pre / post / subtraction).
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224.ms_in22k",
        pretrained: bool = False,
        num_classes: int = 3,
        num_xattn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1) Swin backbone WITHOUT final linear head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features        # 1024 for swin‑base

        # 2) Learnable CLS token (like ViT) and view‑type embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.view_embed = nn.Parameter(torch.zeros(3, 1, self.embed_dim))  # pre/post/sub

        # 3) Cross‑attention encoder (TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.xattn = nn.TransformerEncoder(encoder_layer, num_layers=num_xattn_layers)

        # 4) Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    # ------------------------------------------------------------------ #
    def _img_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward a single image through Swin and return pooled vector (B, C).
        """
        feats = self.backbone.forward_features(x)          # (B, H*W, C)
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # (B, C)
        return pooled

    def forward(self, pre, post, sub):
        """
        Args:
            pre, post, sub: tensors (B, 3, 224, 224)
        Returns:
            logits (B, num_classes)
        """
        # 1) Per‑view embeddings
        z_pre = self._img_embed(pre)    # (B, C)
        z_post = self._img_embed(post)
        z_sub = self._img_embed(sub)

        # 2) Assemble sequence:  [CLS] + three view tokens
        B = z_pre.size(0)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, C)
        views = torch.stack([z_pre, z_post, z_sub], dim=1)  # (B, 3, C)

        # add view‑type embeddings (broadcast over batch)
        views = views + self.view_embed.transpose(0, 1)    # (B, 3, C)

        tokens = torch.cat([cls, views], dim=1)            # (B, 4, C)

        # 3) Cross‑attention
        tokens = self.xattn(tokens)                        # (B, 4, C)

        # 4) CLS pooling → head
        out = self.classifier(self.norm(tokens[:, 0]))     # (B, num_classes)
        return out




class CrossModalAttentionABMIL_Swin(nn.Module):
    """
    Swin‑based Multiple‑Instance model with per‑slice cross‑modal attention
    over (pre, post, sub) features, followed by ABMIL pooling.

    Input  : (B, 32, 3, 224, 224)  # 3 modalities stacked in channel dim
    Output : logits (B, num_classes), attention‑per‑slice (B, 32)
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224.ms_in22k",
        pretrained: bool = False,
        num_classes: int = 3,
        hidden_dim: int = 256,
        cross_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # 1) Shared Swin backbone (no classifier head)
        # ------------------------------------------------------------
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features  # 768 for swin_tiny

        # ------------------------------------------------------------
        # 2) Per‑slice cross‑modal fusion
        #    Input: three embeddings (pre, post, sub)  -> fused embedding
        # ------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cross_attn_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # One small transformer (2 layers) that attends across 3 tokens
        self.slice_fuser = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Token type embedding to differentiate modalities
        self.mod_embed = nn.Parameter(torch.randn(3, 1, self.embed_dim))

        # ------------------------------------------------------------
        # 3) ABMIL Attention across 32 fused slices
        # ------------------------------------------------------------
        self.attn_V = nn.Linear(self.embed_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, 1)

        # ------------------------------------------------------------
        # 4) Classifier head
        # ------------------------------------------------------------
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Init
        nn.init.trunc_normal_(self.mod_embed, std=0.02)
        nn.init.trunc_normal_(self.attn_V.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_U.weight, std=0.02)

    # ------------------------------------------------------------
    def _encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B*N, 1, 224, 224)  -> Swin pooled feats (B*N, embed_dim)
        """
        feats = self.backbone.forward_features(x)
        return self.backbone.forward_head(feats, pre_logits=True)

    # ------------------------------------------------------------
    def forward(self, volume: torch.Tensor):
        """
        volume : (B, 32, 3, 224, 224)
        """
        B, N, C_img, H, W = volume.shape        # C_img = 3 modalities
        assert C_img == 3, "Expect channel dim = 3 (pre, post, sub)"

        # --------------------------------------------------------
        # 1) Split modalities, flatten, and encode with Swin
        # --------------------------------------------------------
        # Each is (B*N, 1, H, W)
        def expand3(x): return x.expand(-1, 3, -1, -1)

        pre  = expand3(volume[:, :, 0, :, :].contiguous().view(B * N, 1, H, W))
        post = expand3(volume[:, :, 1, :, :].contiguous().view(B * N, 1, H, W))
        sub  = expand3(volume[:, :, 2, :, :].contiguous().view(B * N, 1, H, W))

        feat_pre  = self._encode_slices(pre).view(B, N, -1)   # (B, N, C)
        feat_post = self._encode_slices(post).view(B, N, -1)
        feat_sub  = self._encode_slices(sub).view(B, N, -1)

        # --------------------------------------------------------
        # 2) Cross‑modal attention fusion (per slice)
        # --------------------------------------------------------
        # Build token sequence [pre, post, sub] for each slice
        # Shape before fuser: (B, N, 3, C)  -> we fuse along dim=2
        slice_tokens = torch.stack([feat_pre, feat_post, feat_sub], dim=2)

        # Add modality embeddings
        slice_tokens = slice_tokens + self.mod_embed.transpose(0, 1)  # (B, N, 3, C)
        slice_tokens = slice_tokens.view(B * N, 3, self.embed_dim)    # (B*N, 3, C)

        # Transformer encoder attends across the 3 tokens
        fused = self.slice_fuser(slice_tokens)[:, 0]                  # take CLS‑like first token
        fused = fused.view(B, N, self.embed_dim)                      # (B, 32, C)

        # --------------------------------------------------------
        # 3) ABMIL attention over 32 fused slices
        # --------------------------------------------------------
        A = torch.tanh(self.attn_V(fused))                # (B, N, hidden)
        A = self.attn_U(A)                                # (B, N, 1)
        A = torch.softmax(A, dim=1)                       # (B, N, 1)
        patient_feat = (A * fused).sum(dim=1)             # (B, C)

        # --------------------------------------------------------
        # 4) Head
        # --------------------------------------------------------
        logits = self.classifier(self.dropout(self.norm(patient_feat)))
        return logits, A.squeeze(-1)                      # (B, num_classes), (B, 32)


class ABMIL_Swin(nn.Module):
    """
    Attention‑based Multiple‑Instance Learning (ABMIL) model
    for 3‑channel breast MRI slice triplets (pre / post / sub).

    Input shape : (B, 32, 3, 224, 224)
    Output      : logits (B, num_classes)  +  attention weights (B, 32)
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224.ms_in22k",
        pretrained: bool = False,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1) Swin backbone WITHOUT final linear head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features  # e.g. 768 (tiny), 1024 (base)

        # 2) Attention network (ABMIL)
        self.attn_V = nn.Linear(self.embed_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, 1)

        # 3) Patient‑level classifier
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Optional init
        nn.init.trunc_normal_(self.attn_V.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_U.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def _slice_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward one or more images through Swin and return pooled features.
        Args:
            x : (N, 3, 224, 224)
        Returns:
            pooled : (N, C)
        """
        feats = self.backbone.forward_features(x)                 # (N, H*W, C)
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # (N, C)
        return pooled

    # ------------------------------------------------------------------ #
    def forward(self, volume: torch.Tensor):
        """
        Args:
            volume : (B, 32, 3, 224, 224)
        Returns:
            logits : (B, num_classes)
            attn_w : (B, 32)   -- attention per slice (sums to 1)
        """
        B, N, C, H, W = volume.shape
        x = volume.view(B * N, C, H, W)                # flatten slices
        
        # (B*N, 3, 224, 224) → (B*N, embed_dim) → reshape
        slice_feats = self._slice_embed(x).view(B, N, -1)  # (B, 32, embed_dim)

        # ----------------  ABMIL attention  ---------------- #
        A = torch.tanh(self.attn_V(slice_feats))            # (B, 32, hidden)
        A = self.attn_U(A)                                  # (B, 32, 1)
        A = torch.softmax(A, dim=1)                         # attention weights

        # Weighted sum → patient embedding
        patient_feat = (A * slice_feats).sum(dim=1)         # (B, embed_dim)

        # ----------------  Head  ---------------- #
        out = self.classifier(self.dropout(self.norm(patient_feat)))  # (B, num_classes)

        return out, A.squeeze(-1)  # logits, attention weights