import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFeaturePyramid(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.non_local = NonLocalBlock(channel*4)
        self.global_context = nn.Sequential(
            nn.Conv2d(channel*4, channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel*4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2, x3, x4):
        size2, size3, size4 = x2.shape[2:], x3.shape[2:], x4.shape[2:]
        
        x2_up = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x3_up = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x4_up = F.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=True)

        global_features = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        global_features = self.non_local(global_features)
        scale_weights = self.global_context(global_features)

        w1, w2, w3, w4 = torch.split(scale_weights, 32, dim=1)

        x1_out = x1 * (1+w1)
        
        w2 = F.interpolate(w2, size=size2, mode='bilinear', align_corners=True)
        x2_out = x2 * (1+w2)
        
        w3 = F.interpolate(w3, size=size3, mode='bilinear', align_corners=True)
        x3_out = x3 * (1+w3)
        
        w4 = F.interpolate(w4, size=size4, mode='bilinear', align_corners=True)
        x4_out = x4 * (1+w4)
        
        return x1_out, x2_out, x3_out, x4_out

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.inter_channel = channel // 2
        self.query = nn.Conv2d(channel, self.inter_channel, 1)
        self.key = nn.Conv2d(channel, self.inter_channel, 1)
        self.value = nn.Conv2d(channel, self.inter_channel, 1)
        self.out = nn.Conv2d(self.inter_channel, channel, 1)
        
    def forward(self, x):
        batch, channel, height, width = x.size()

        query = self.query(x).view(batch, -1, height*width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height*width)
        value = self.value(x).view(batch, -1, height*width)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, self.inter_channel, height, width)
        out = self.out(out)
        
        return out