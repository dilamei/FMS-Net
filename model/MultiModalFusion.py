import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    def __init__(self, in_channel, text_dim=32, reduction_factor=4):
        super(MultiModalFusion, self).__init__()
        self.in_channel = in_channel
        self.text_dim = text_dim
        # Increase reduction factor to decrease intermediate dimensions
        self.reduced_dim = max(text_dim // reduction_factor, 8)
        self.groups = 2  # Use more groups for efficiency
        
        # Vision projection with grouped convolutions
        self.v_conv_proj = nn.Conv2d(in_channel, self.reduced_dim, kernel_size=1, 
                                     groups=min(self.groups, in_channel))
        
        # Streamlined vision back projection
        self.v_conv_back = nn.Sequential(
            nn.Conv2d(self.reduced_dim, in_channel, kernel_size=1, 
                      groups=min(self.groups, self.reduced_dim)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        
        # Text projections
        self.t_proj = nn.Linear(text_dim, self.reduced_dim)
        self.t_back = nn.Linear(self.reduced_dim, text_dim)
        
        # Simplified global attention
        self.global_attn = nn.Sequential(
            nn.Linear(self.reduced_dim, 1),
            nn.Sigmoid()
        )
        
        # Simplified local attention
        self.local_attn = nn.Sequential(
            nn.Conv2d(self.reduced_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attn_fusion = nn.Parameter(torch.tensor([0.5]))
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.v_norm = nn.InstanceNorm2d(in_channel, affine=True)
        
        self.v_scale = nn.Parameter(torch.ones(1))
        self.t_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, vision_feat, text_feat):
        v_res = vision_feat
        t_res = text_feat
        
        # Apply instance normalization
        v = self.v_norm(v_res)
        
        # Vision projection
        v_proj = self.v_conv_proj(v)  # [B, reduced_dim, H, W]
        
        # Text projection
        t_proj = self.t_proj(text_feat)  # [B, reduced_dim]
        
        B, C, H, W = v_proj.shape
        
        # Global attention - simplified
        v_global = F.adaptive_avg_pool2d(v_proj, 1).view(B, C)  # [B, reduced_dim]
        global_attn = self.global_attn(v_global * t_proj).view(B, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Local attention - simplified
        t_spatial = t_proj.view(B, C, 1, 1).expand_as(v_proj)  # [B, reduced_dim, H, W]
        v_t_concat = torch.cat([v_proj, t_spatial], dim=1)  # [B, reduced_dim*2, H, W]
        local_attn = self.local_attn(v_t_concat)  # [B, 1, H, W]
        
        # Fusion of attentions
        fused_attn = self.attn_fusion * global_attn + (1 - self.attn_fusion) * local_attn
        
        # Apply attention to vision features
        v_enhanced = v_proj * fused_attn
        v_out = self.v_conv_back(v_enhanced) * self.alpha + v_res * self.v_scale
        
        # Text enhancement
        v_enhanced_pool = F.adaptive_avg_pool2d(v_enhanced, 1).view(B, C)
        t_enhanced = t_proj + v_enhanced_pool * 0.1
        t_out = self.t_back(t_enhanced) * self.beta + t_res * self.t_scale
        
        return v_out, t_out