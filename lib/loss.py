from typing import List
import torch.nn as nn
import torch
from torch.nn import functional as F

from config import Config

class loss_cls(nn.Module):
    def __init__(self):
        super(loss_cls, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])

        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0-pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * ((1.0-pos_label[:, 0, :, :])**4.0) * (pos_pred[:, 0, :, :]**2.0)

        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])

        cls_loss = 0.01 * torch.sum(focal_weight*log_loss) / max(1.0, assigned_box)

        return cls_loss


class loss_reg(nn.Module):
    def __init__(self):
        super(loss_reg, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :]*self.smoothl1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss


class loss_offset(nn.Module):
    def __init__(self):
        super(loss_offset, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss

class loss_pseudo_score(nn.Module):
    def __init__(self):
        super().__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self, score_map:torch.Tensor, pseudo_map):
        return self.smoothl1(score_map, pseudo_map)

class loss_proto_contrast(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.temp = 7e-2
        self.ce = nn.CrossEntropyLoss()
        self.human_index = config.classnames.index('human')

    def forward(self, feature:torch.Tensor, center_map: torch.Tensor, score_map: torch.Tensor):
        B, E, H, W = feature.shape
        # [B, C, h, w]
        C = score_map.shape[1]

        score_map = F.interpolate(score_map, (H, W))
        
        # [B, hw, C] 
        score_map = score_map.contiguous().view(B, C, -1).transpose(1, 2).detach()

        # [B, hw, 1] 
        center_map = center_map[:, 0, :, :].contiguous().view(B, -1).unsqueeze(-1).detach()

        # scale up the gap between logits of different classes
        score_map = (score_map / 1e-3).softmax(dim=-1)

        # [B, E, hw]
        feature = feature.contiguous().view(B, E, -1)
        # [B, E, C]
        cate_protos = feature @ score_map
        # Normalize
        cate_protos:torch.Tensor = cate_protos / torch.clamp_min(torch.norm(cate_protos, p=2, dim=1, keepdim=True), 1e-5)

        # Non Ped idxs
        non_ped_idxs = torch.arange(cate_protos.shape[-1])!=self.human_index
        
        # Positive Proto [B, E, 1]
        pos_ped_proto = feature @ center_map
        # Normalize
        pos_ped_proto:torch.Tensor = pos_ped_proto / torch.clamp_min(torch.norm(pos_ped_proto, p=2, dim=1, keepdim=True), 1e-5)

        # Negative Proto [E, B*(C-1)]
        neg_protos = cate_protos[:, :, non_ped_idxs].transpose(0, 1).contiguous().view(E, B*(C-1))

        # Normalize [B, E, hw]
        feat_norm:torch.Tensor = feature / torch.clamp_min(torch.norm(feature, p=2, dim=1, keepdim=True), 1e-5)

        # distance between features at all pixels and neg protos in batch dim [B, hw, B*(C-1)]
        feat_neg_score = feat_norm.transpose(1, 2) @ neg_protos

        # select the ped positions [K, 1, B*(C-1)]
        l_neg = feat_neg_score[center_map.squeeze(-1) > 0].unsqueeze(1)

        # distance between features at all pixels and ped proto in batch dim [B, hw, 1]
        feat_ped_score = feat_norm.transpose(1, 2) @ pos_ped_proto

        # select the ped positions [K, 1, 1]
        l_pos = feat_ped_score[center_map.squeeze(-1) > 0].unsqueeze(1)

        # Contrast Logits, from Pos & Neg [K, 1+B*(C-1)]
        contrast_logits = torch.cat([l_pos, l_neg], dim=-1).squeeze(1)

        assert contrast_logits.shape[-1] == 1+B*(C-1)

        K = contrast_logits.shape[0]
        # Label
        labels = torch.zeros(K).long().cuda()

        if contrast_logits.shape[0] <= 0:
            return torch.Tensor([0.0]).cuda()
        else:
            return self.ce(contrast_logits/self.temp, labels)
