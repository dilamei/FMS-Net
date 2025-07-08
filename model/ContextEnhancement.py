import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEnhancement(nn.Module):
    def __init__(self, channel):
        super(ContextEnhancement, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel//4, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True)
            ) for rate in [1,2,3,4]
        ])

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, 1),
            nn.ReLU(inplace=True)
        )

        total_channels = channel + 4 * (channel//4) + channel//4

        self.fusion = nn.Conv2d(total_channels, channel, 1)
        
    def forward(self, x):

        context_feats = [branch(x) for branch in self.branches]
        global_feat = self.global_context(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:])

        concat_feat = torch.cat([x] + context_feats + [global_feat], dim=1)

        return self.fusion(concat_feat)