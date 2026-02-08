import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import conv

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        
        self.feature_extract = nn.Sequential(
            conv(in_channels, in_channels * 2, 3, 2, 1, mode='CBR'),
            conv(in_channels * 2, in_channels * 4, 3, 2, 1, mode='CBR'),
            conv(in_channels * 4, in_channels * 8, 3, 2, 1, mode='CBR'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        self.classify = nn.Sequential(
            nn.Linear(in_channels * 8, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x: [N, C, H, W] 特征图
        
        # 提取特征
        feats = self.feature_extract(x)
        
        # 域分类
        domain_pred = self.classify(feats)
        
        return domain_pred

class DomainAdaptationModule(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        
        # 特征提取器
        self.feat_extract = nn.Sequential(
            conv(in_channels, in_channels, 3, 1, 1, mode='CBR'),
            conv(in_channels, in_channels, 3, 1, 1, mode='CBR'),
        )
        
        # 域判别器
        self.domain_discriminator = DomainDiscriminator(in_channels)
    
    def forward(self, x, domain_label=None):
        # x: [N, C, H, W] 特征图
        # domain_label: [N, 1] 域标签 (0: source, 1: target)
        
        # 提取特征
        feats = self.feat_extract(x)
        
        # 域分类
        domain_pred = self.domain_discriminator(feats)
        
        if domain_label is not None:
            # 计算域分类损失
            domain_loss = F.binary_cross_entropy(domain_pred, domain_label)
            return feats, domain_pred, domain_loss
        else:
            return feats, domain_pred

class CRNetWithDomainAdaptation(nn.Module):
    def __init__(self, opt, mid_channels=64):
        super().__init__()
        
        from .cat_model import TMRNet
        from .attention_module import AttentionModule, AdaptiveEnhancementModule
        
        # 原始TMRNet
        self.tmrnet = TMRNet(opt)
        
        # 注意力模块
        self.attention_module = AttentionModule(in_channels=4, mid_channels=mid_channels)
        
        # 自适应增强模块
        self.adaptive_enhancement = AdaptiveEnhancementModule(in_channels=mid_channels, mid_channels=mid_channels)
        
        # 域适应模块
        self.domain_adaptation = DomainAdaptationModule(in_channels=mid_channels)
        
        # 最终输出模块
        self.final_output = nn.Sequential(
            conv(mid_channels, mid_channels, 3, 1, 1, mode='CBR'),
            conv(mid_channels, 4, 3, 1, 1, mode='CB'),
        )
    
    def forward(self, lqs, domain_label=None):
        n, t, c, h, w = lqs.size()
        
        # 原始TMRNet前向传播
        # 这里简化处理，实际需要根据TMRNet的具体实现进行调整
        # 提取第一个输入图像作为参考
        ref_img = lqs[:, 0, :, :, :]
        
        # 生成注意力图
        combined_attention, fog_attention, dark_attention = self.attention_module(ref_img)
        
        # 调整注意力图大小以匹配特征图
        combined_attention = F.interpolate(combined_attention, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        # 假设这里获取了TMRNet的特征图
        # 实际实现中需要修改TMRNet以返回中间特征
        # 这里使用简化的特征提取
        from .cat_model import ResidualBlocksWithInputConv
        feat_extract = ResidualBlocksWithInputConv(2 * 4, mid_channels=mid_channels, num_blocks=5)
        lqs_view = lqs.view(-1, c, h, w)
        lqs_in = torch.zeros([n * t, 2 * c, h, w], dtype=lqs_view.dtype, device=lqs_view.device)
        lqs_in[:, 0::2, :, :] = lqs_view
        lqs_in[:, 1::2, :, :] = torch.pow(torch.clamp(lqs_view, min=0), 1 / 2.2)
        feats_ = feat_extract(lqs_in).view(n, t, -1, h, w)
        base_feat = feats_[:, 0, :, :, :]
        
        # 自适应增强
        enhanced_feat, gain, gamma, offset = self.adaptive_enhancement(base_feat, combined_attention)
        
        # 域适应
        if domain_label is not None:
            adapted_feat, domain_pred, domain_loss = self.domain_adaptation(enhanced_feat, domain_label)
        else:
            adapted_feat, domain_pred = self.domain_adaptation(enhanced_feat)
            domain_loss = None
        
        # 最终输出
        output = self.final_output(adapted_feat)
        
        if domain_loss is not None:
            return output, combined_attention, fog_attention, dark_attention, domain_loss
        else:
            return output, combined_attention, fog_attention, dark_attention
