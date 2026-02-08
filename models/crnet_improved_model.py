import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
from . import losses as L
import torch.nn.functional as F
from util.util import mu_tonemap
from .attention_module import AttentionModule, AdaptiveEnhancementModule
from .domain_adaptation import DomainAdaptationModule

class CRNetImprovedModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(CRNetImprovedModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = ['TMRNet_l1', 'attention', 'domain', 'Total']
        self.visual_names = ['data_gt', 'data_in', 'data_out', 'attention_map'] 
        self.model_names = ['TMRNet', 'Attention', 'Enhancement', 'Domain']
        self.optimizer_names = ['TMRNet_optimizer_%s' % opt.optimizer, 'Attention_optimizer_%s' % opt.optimizer, 'Enhancement_optimizer_%s' % opt.optimizer, 'Domain_optimizer_%s' % opt.optimizer]

        from .cat_model import TMRNet
        tmrnet = TMRNet(opt)
        self.netTMRNet = N.init_net(tmrnet, opt.init_type, opt.init_gain, opt.gpu_ids)

        # 注意力模块
        self.netAttention = N.init_net(AttentionModule(in_channels=4, mid_channels=64), opt.init_type, opt.init_gain, opt.gpu_ids)

        # 自适应增强模块
        self.netEnhancement = N.init_net(AdaptiveEnhancementModule(in_channels=64, mid_channels=64), opt.init_type, opt.init_gain, opt.gpu_ids)

        # 域适应模块
        self.netDomain = N.init_net(DomainAdaptationModule(in_channels=64), opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            self.optimizer_TMRNet = optim.AdamW(self.netTMRNet.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            self.optimizer_Attention = optim.AdamW(self.netAttention.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            self.optimizer_Enhancement = optim.AdamW(self.netEnhancement.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            self.optimizer_Domain = optim.AdamW(self.netDomain.parameters(),
                                              lr=opt.lr * 0.1,  # 域判别器学习率较低
                                              betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer_TMRNet, self.optimizer_Attention, self.optimizer_Enhancement, self.optimizer_Domain]

            self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
            self.criterionAttention = nn.MSELoss()
            self.criterionDomain = nn.BCELoss()

    def set_input(self, input):
        self.data_gt = input['gt'].to(self.device)
        self.data_raws = input['raws'].to(self.device)
        self.image_paths = input['fname']

        expo = torch.stack([torch.pow(torch.tensor(4, dtype=torch.float32, device=self.data_raws.device), 2-x) 
                            for x in range(0, self.data_raws.shape[1])])
        self.expo = expo[None,:,None,None,None]

        # 域标签 (0: source, 1: target)
        if 'domain_label' in input:
            self.domain_label = input['domain_label'].to(self.device)
        else:
            # 默认假设是目标域
            self.domain_label = torch.ones((self.data_raws.shape[0], 1), device=self.data_raws.device)

    def forward(self):
        self.data_raws = self.data_raws * self.expo 
        self.data_in = self.data_raws[:,0,...].squeeze(1)

        if self.isTrain or (not self.isTrain and not self.opt.chop):
            # 生成注意力图
            combined_attention, fog_attention, dark_attention = self.netAttention(self.data_in)
            self.attention_map = combined_attention

            # 原始TMRNet前向传播
            self.data_out = self.netTMRNet(self.data_raws)

            # 提取TMRNet的中间特征（需要修改TMRNet以返回中间特征）
            # 这里使用简化的方法，直接使用输入图像生成特征
            from .cat_model import ResidualBlocksWithInputConv
            feat_extract = ResidualBlocksWithInputConv(2 * 4, mid_channels=64, num_blocks=5).to(self.device)
            lqs_view = self.data_raws.view(-1, 4, self.data_raws.shape[3], self.data_raws.shape[4])
            lqs_in = torch.zeros([lqs_view.shape[0], 2 * 4, lqs_view.shape[2], lqs_view.shape[3]], dtype=lqs_view.dtype, device=lqs_view.device)
            lqs_in[:, 0::2, :, :] = lqs_view
            lqs_in[:, 1::2, :, :] = torch.pow(torch.clamp(lqs_view, min=0), 1 / 2.2)
            feats_ = feat_extract(lqs_in).view(self.data_raws.shape[0], self.data_raws.shape[1], 64, self.data_raws.shape[3], self.data_raws.shape[4])
            base_feat = feats_[:, 0, :, :, :]

            # 调整注意力图大小
            combined_attention = F.interpolate(combined_attention, size=(base_feat.shape[2], base_feat.shape[3]), mode='bilinear', align_corners=True)

            # 自适应增强
            enhanced_feat, gain, gamma, offset = self.netEnhancement(base_feat, combined_attention)

            # 域适应
            adapted_feat, domain_pred, domain_loss = self.netDomain(enhanced_feat, self.domain_label)
            self.domain_pred = domain_pred
            self.domain_loss = domain_loss

    def backward(self, epoch):
        self.loss_TMRNet_l1 = self.criterionL1(
            mu_tonemap(torch.clamp(self.data_out / 4**2, min=0)), 
            mu_tonemap(torch.clamp(self.data_gt / 4**2, 0, 1))).mean()

        # 注意力损失：鼓励注意力图关注雾区和暗区
        # 这里使用简化的损失，实际应用中可以使用更复杂的监督信号
        self.loss_attention = torch.mean(self.attention_map) * 0.1

        # 域适应损失
        self.loss_domain = self.domain_loss * 0.01

        self.loss_Total = self.loss_TMRNet_l1 + self.loss_attention + self.loss_domain
        self.loss_Total.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        
        # 优化生成器（TMRNet + 注意力 + 增强）
        self.set_requires_grad([self.netTMRNet, self.netAttention, self.netEnhancement], True)
        self.set_requires_grad([self.netDomain], False)
        
        self.optimizer_TMRNet.zero_grad()
        self.optimizer_Attention.zero_grad()
        self.optimizer_Enhancement.zero_grad()
        self.backward(epoch)
        self.optimizer_TMRNet.step()
        self.optimizer_Attention.step()
        self.optimizer_Enhancement.step()

        # 优化域判别器
        self.set_requires_grad([self.netTMRNet, self.netAttention, self.netEnhancement], False)
        self.set_requires_grad([self.netDomain], True)
        
        # 再次前向传播获取特征
        self.forward()
        
        # 计算域判别器损失
        domain_loss = self.domain_loss
        self.optimizer_Domain.zero_grad()
        domain_loss.backward()
        self.optimizer_Domain.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
