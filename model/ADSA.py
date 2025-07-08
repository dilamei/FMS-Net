import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ADSA(nn.Module):
    def __init__(self, channels, num_directions=8):
        super(ADSA, self).__init__()
        
        self.channels = channels
        self.num_directions = num_directions

        groups = min(4, channels)
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.spectral_decomp = SpectralDecomposition(channels, num_directions)

        self.direction_fusion = nn.Sequential(
            nn.Conv2d(channels * num_directions, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.direction_weights = nn.Parameter(torch.ones(num_directions) / num_directions)

        mid_channels = max(channels // 2, 16)
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.residual_scale = nn.Parameter(torch.ones(1))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        residual = x

        features = self.init_conv(x)

        batch_size, c, h, w = features.size()

        with torch.cuda.amp.autocast(enabled=True):
            x_freq = torch.fft.rfft2(features)
            direction_features = self.spectral_decomp(x_freq, h, w)

        weights = F.softmax(self.direction_weights, dim=0)

        weighted_features = []
        for i, feat in enumerate(direction_features):
            weighted_features.append(feat * weights[i])

        concatenated = torch.cat(weighted_features, dim=1)
        fused = self.direction_fusion(concatenated)

        enhanced = self.feature_enhance(fused)

        gate_value = self.gate(features)

        output = gate_value * enhanced + residual * self.residual_scale
        
        return output

class SpectralDecomposition(nn.Module):
    def __init__(self, channels, num_directions=8):
        super(SpectralDecomposition, self).__init__()
        self.channels = channels
        self.num_directions = num_directions
        self.freq_enhance = nn.Parameter(torch.ones(num_directions))

        self.angles = torch.tensor(
            [i * (np.pi / num_directions) for i in range(num_directions)]
        )
        self.angle_range = np.pi / num_directions
        
    def forward(self, x_freq, h, w):
        direction_features = []
        for i in range(self.num_directions):
            mask = self.create_direction_mask(h, w//2+1, i).to(x_freq.device)
            enhanced_mask = mask * self.freq_enhance[i]

            masked_freq = x_freq * enhanced_mask.unsqueeze(0).unsqueeze(0)

            dir_feature = torch.fft.irfft2(masked_freq, s=(h, w))
            direction_features.append(dir_feature)
            
        return direction_features
    
    def create_direction_mask(self, h, w, direction):
        device = self.freq_enhance.device
        y_indices, x_indices = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij'
        )

        cy = h // 2
        cx = 0  

        angle = self.angles[direction]
        angle_range = self.angle_range

        y_centered = cy - y_indices
        x_centered = x_indices
        angles = torch.atan2(y_centered, x_centered.clamp(min=1e-5))
        angles = torch.remainder(angles + np.pi, 2 * np.pi) - np.pi

        lower_bound = angle - angle_range / 2
        upper_bound = angle + angle_range / 2

        if lower_bound < -np.pi:
            mask = ((angles >= lower_bound + 2*np.pi) | (angles <= upper_bound)).float()
        elif upper_bound > np.pi:
            mask = ((angles >= lower_bound) | (angles <= upper_bound - 2*np.pi)).float()
        else:
            mask = ((angles >= lower_bound) & (angles <= upper_bound)).float()

        dist = torch.sqrt(((y_indices - cy)**2 + x_indices**2) / (cy**2 + w**2))

        freq_response = torch.ones_like(dist)
        freq_response = torch.where(dist < 0.2, 0.5 + 2.5 * dist,
                       torch.where(dist < 0.6, 1.0, 
                       1.0 - 0.5 * (dist - 0.6) / 0.4))

        return mask * freq_response