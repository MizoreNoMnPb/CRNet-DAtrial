import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import conv

class AttentionModule(nn.Module):
    def __init__(self, in_channels=8, mid_channels=64):
        super().__init__()
        
        # 特征提取
        self.feat_extract = nn.Sequential(
            conv(in_channels, mid_channels, 3, 1, 1, mode='CBR'),
            conv(mid_channels, mid_channels, 3, 1, 1, mode='CBR'),
            conv(mid_channels, mid_channels, 3, 2, 1, mode='CBR'),
            conv(mid_channels, mid_channels * 2, 3, 1, 1, mode='CBR'),
            conv(mid_channels * 2, mid_channels * 2, 3, 1, 1, mode='CBR'),
            conv(mid_channels * 2, mid_channels * 2, 3, 2, 1, mode='CBR'),
        )
        
        # 特征融合和注意力图生成
        self.attention_gen = nn.Sequential(
            conv(mid_channels * 2, mid_channels, 3, 1, 1, mode='CBR'),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv(mid_channels, mid_channels // 2, 3, 1, 1, mode='CBR'),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv(mid_channels // 2, 2, 3, 1, 1, mode='CB'),  # 2 channels: fog and dark
            nn.Sigmoid()  # 输出0-1之间的注意力值
        )
    
    def forward(self, x):
        # x: [N, C, H, W] 输入图像
        
        # 提取特征
        feats = self.feat_extract(x)
        
        # 生成注意力图
        attention = self.attention_gen(feats)
        
        # 雾区注意力图和暗区注意力图
        fog_attention = attention[:, 0:1, :, :]
        dark_attention = attention[:, 1:2, :, :]
        
        # 融合注意力图
        combined_attention = torch.max(fog_attention, dark_attention)
        
        return combined_attention, fog_attention, dark_attention

class AdaptiveEnhancementModule(nn.Module):
    def __init__(self, in_channels=64, mid_channels=64):
        super().__init__()
        
        # 特征提取
        self.feat_extract = nn.Sequential(
            conv(in_channels, mid_channels, 3, 1, 1, mode='CBR'),
            conv(mid_channels, mid_channels, 3, 1, 1, mode='CBR'),
        )
        
        # 增强参数预测
        self.enhance_params = nn.Sequential(
            conv(mid_channels + 1, mid_channels, 3, 1, 1, mode='CBR'),  # +1 for attention map
            conv(mid_channels, mid_channels, 3, 1, 1, mode='CBR'),
            conv(mid_channels, 3, 3, 1, 1, mode='CB'),  # 3 channels: gain, gamma, offset
        )
    
    def forward(self, x, attention):
        # x: [N, C, H, W] 特征图
        # attention: [N, 1, H, W] 注意力图
        
        # 提取特征
        feats = self.feat_extract(x)
        
        # 融合注意力图
        feats_with_attention = torch.cat([feats, attention], dim=1)
        
        # 预测增强参数
        params = self.enhance_params(feats_with_attention)
        
        # 分离参数
        gain = torch.sigmoid(params[:, 0:1, :, :]) * 2  # 增益因子 0-2
        gamma = torch.sigmoid(params[:, 1:2, :, :]) * 2  # 伽马因子 0-2
        offset = torch.tanh(params[:, 2:3, :, :]) * 0.1  # 偏移量 -0.1 to 0.1
        
        # 应用增强
        enhanced = torch.pow(torch.clamp(x, min=0), gamma) * gain + offset
        
        # 根据注意力图进行自适应融合
        output = x * (1 - attention) + enhanced * attention
        
        return output, gain, gamma, offset
