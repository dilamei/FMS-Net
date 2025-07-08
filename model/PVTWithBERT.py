import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from model.pvtv2 import pvt_v2_b2
from model.MultiModalFusion import MultiModalFusion
from model.ADSA import ADSA

from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer

class BasicConv2d(nn.Module):
    """Basic 2D Convolution Module"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256):
        super(Decoder, self).__init__()

        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv3 = BasicConv2d(channel_2, channel_3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return x3
        
class decoder(nn.Module):
    def __init__(self, channel=800):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.decoder4 = nn.Sequential(
            Decoder(32, 32, 32),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.S4 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            Decoder(64, 32, 32),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.S3 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            Decoder(64, 32, 32),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.S2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            Decoder(64, 32, 32),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)


    def forward(self, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        
        x4_up = self.decoder4(x4)
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat([x3, x4_up], dim=1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat([x2, x3_up], dim=1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat([x1, x2_up], dim=1))
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4 

class DecoderPyramid(nn.Module):
    """Pyramid Decoder with ADSA Attention"""
    def __init__(self, channel=32):
        super(DecoderPyramid, self).__init__()

        self.Spectral_4 = ADSA(channel)  
        self.Spectral_3 = ADSA(channel*2)
        self.Spectral_2 = ADSA(channel*2)
        self.Spectral_1 = ADSA(channel*2) 

        self.decoder4 = nn.Sequential(
            BasicConv2d(channel*2, channel, 1, padding=0),  
            BasicConv2d(channel, channel, 3, padding=1),  
            #nn.Dropout(0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.decoder3 = nn.Sequential(
            BasicConv2d(channel*3, channel, 1, padding=0),  
            BasicConv2d(channel, channel, 3, padding=1),   
            #nn.Dropout(0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder2 = nn.Sequential(
            BasicConv2d(channel*3, channel, 1, padding=0),  
            BasicConv2d(channel, channel, 3, padding=1),  
            #nn.Dropout(0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder1 = nn.Sequential(
            BasicConv2d(channel*3, channel, 1, padding=0), 
            BasicConv2d(channel, channel, 3, padding=1),  
            #nn.Dropout(0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.pred4 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.pred3 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.pred2 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.pred1 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        
    def forward(self, x4_fused, x3_fused, x2_fused, x1_fused):

        d4_spectral = self.Spectral_4(x4_fused)
        d4_concat = torch.cat([x4_fused, d4_spectral], dim=1)  
        d4 = self.decoder4(d4_concat) 
        p4 = self.pred4(d4)

        d3_input = torch.cat([x3_fused, d4], dim=1) 
        d3_spectral = self.Spectral_3(d3_input) 
        d3_concat = torch.cat([x3_fused, d3_spectral], dim=1) 
        d3 = self.decoder3(d3_concat)
        p3 = self.pred3(d3)

        d2_input = torch.cat([x2_fused, d3], dim=1)  
        d2_spectral = self.Spectral_2(d2_input)
        d2_concat = torch.cat([x2_fused, d2_spectral], dim=1) 
        d2 = self.decoder2(d2_concat)
        p2 = self.pred2(d2)

        d1_input = torch.cat([x1_fused, d2], dim=1) 
        d1_spectral = self.Spectral_1(d1_input)  
        d1_concat = torch.cat([x1_fused, d1_spectral], dim=1) 
        d1 = self.decoder1(d1_concat) 
        p1 = self.pred1(d1)
        
        return p1, p2, p3, p4
        
class PVTwithBERT(nn.Module):
    def __init__(self, channel=32, bert_path='./model/huggingface/bert-base-uncased'):
        super(PVTwithBERT, self).__init__()

        self.backbone = pvt_v2_b2()
        self.init_backbone()

        self.bert = BertModel.from_pretrained(bert_path)
        
        modules = list(self.bert.encoder.layer)
        for module in modules[-3:]:
            for param in module.parameters():
                param.requires_grad = True
        
#        for param in self.bert.parameters():
#            param.requires_grad = False
            
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, channel),
            nn.LayerNorm(channel)
        )

        self.norm1 = BasicConv2d(64, channel, 3, padding=1)
        self.norm2 = BasicConv2d(128, channel, 3, padding=1)
        self.norm3 = BasicConv2d(320, channel, 3, padding=1)  
        self.norm4 = BasicConv2d(512, channel, 3, padding=1)

        self.fusion1 = MultiModalFusion(channel)   
        self.fusion2 = MultiModalFusion(channel)  
        self.fusion3 = MultiModalFusion(channel)
        self.fusion4 = MultiModalFusion(channel)
        
        self.dimen1 = BasicConv2d(64, channel, 1, padding=0)
        self.dimen2 = BasicConv2d(64, channel, 1, padding=0)
        self.dimen3 = BasicConv2d(64, channel, 1, padding=0)  
        self.dimen4 = BasicConv2d(64, channel, 1, padding=0)

        self.Decoder = DecoderPyramid(64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    
    def init_backbone(self):
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict() 
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
    
    def forward(self, x_rgb, text_input):
        attention_mask = (text_input != 0).float()
        text_output = self.bert(text_input, attention_mask=attention_mask)
        text_feat = self.text_proj(text_output[0].mean(dim=1))

        x1, x2, x3, x4 = self.backbone(x_rgb)

        x1_norm = self.norm1(x1)
        x2_norm = self.norm2(x2)
        x3_norm = self.norm3(x3)
        x4_norm = self.norm4(x4)

        x1_enhanced, text1 = self.fusion1(x1_norm, text_feat)
        x2_enhanced, text2 = self.fusion2(x2_norm, text1)
        x3_enhanced, text3 = self.fusion3(x3_norm, text2)
        x4_enhanced, text4 = self.fusion4(x4_norm, text3)
        
        x1_enhanced = torch.cat([x1_norm, x1_enhanced], dim=1) 
        x2_enhanced = torch.cat([x2_norm, x2_enhanced], dim=1) 
        x3_enhanced = torch.cat([x3_norm, x3_enhanced], dim=1) 
        x4_enhanced = torch.cat([x4_norm, x4_enhanced], dim=1) 
        
#        x1_enhanced = self.dimen1(x1_enhanced)
#        x2_enhanced = self.dimen2(x2_enhanced) 
#        x3_enhanced = self.dimen3(x3_enhanced)
#        x4_enhanced = self.dimen4(x4_enhanced)

        p1, p2, p3, p4 = self.Decoder(x4_enhanced, x3_enhanced, x2_enhanced, x1_enhanced)
        #p1, p2, p3, p4 = self.Decoder(x4_norm, x3_norm, x2_norm, x1_norm)

        s1 = self.upsample2(p1)
        s2 = self.upsample4(p2)
        s3 = self.upsample8(p3)
        s4 = self.upsample16(p4)
        
        return s1, s2, s3, s4