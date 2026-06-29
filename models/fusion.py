import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatFusion(nn.Module):
    """
    Baseline Fusion: Simple concatenation of text and image features.
    """
    def __init__(self, text_dim: int, img_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(text_dim + img_dim, out_dim)
        
    def forward(self, text_feat, img_feat):
        fused = torch.cat((text_feat, img_feat), dim=1)
        return self.fc(fused)

class GatedAttentionFusion(nn.Module):
    """
    Advanced Fusion: Gated Attention.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.text_gate = nn.Linear(dim, dim)
        self.img_gate = nn.Linear(dim, dim)
        self.fusion_gate = nn.Linear(dim * 2, dim)
        
    def forward(self, text_feat, img_feat):
        # Compute attention weights
        g_t = torch.sigmoid(self.text_gate(text_feat))
        g_i = torch.sigmoid(self.img_gate(img_feat))
        
        # Modulate features
        t_mod = text_feat * g_t
        i_mod = img_feat * g_i
        
        # Final gate over concatenated modulated features
        concat = torch.cat((t_mod, i_mod), dim=1)
        g_f = torch.sigmoid(self.fusion_gate(concat))
        
        # Combine
        fused = (t_mod + i_mod) * g_f
        return fused

class ConditionalGatedAttentionFusion(nn.Module):
    """
    Proposed Fusion: Conditional Attention combined with Gated Attention.
    Text feature conditions the image feature extraction and vice-versa before gating.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Cross conditioning
        self.cond_t2i = nn.Linear(dim, dim)
        self.cond_i2t = nn.Linear(dim, dim)
        
        self.text_gate = nn.Linear(dim, dim)
        self.img_gate = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim * 2, dim)
        
    def forward(self, text_feat, img_feat):
        # Conditional shifts
        i_cond = img_feat * torch.sigmoid(self.cond_t2i(text_feat))
        t_cond = text_feat * torch.sigmoid(self.cond_i2t(img_feat))
        
        # Gating on conditioned features
        g_t = torch.sigmoid(self.text_gate(t_cond))
        g_i = torch.sigmoid(self.img_gate(i_cond))
        
        t_out = t_cond * g_t
        i_out = i_cond * g_i
        
        # Concat and project
        fused = self.out_proj(torch.cat((t_out, i_out), dim=1))
        return fused
