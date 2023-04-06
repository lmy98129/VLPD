import torch
import torch.nn as nn

from config import Config

from .backbone import ResNet_concat, ResNet50_CLIP, ResNet50_CLIP_VLPD
from .detection_head import  CSP_Head

class CSP(nn.Module):
    """CSP framework"""
    def __init__(self, cfg: Config):
        super(CSP, self).__init__()
        backbone = cfg.backbone
        assert backbone is not None, 'Backbone must be specified.'
        if backbone == 'ResNet50-CLIP':
            self.backbone = ResNet50_CLIP(cfg)
        elif backbone == 'ResNet50-CLIP-VLPD':
            self.backbone = ResNet50_CLIP_VLPD(cfg)
        else:
            self.backbone = ResNet_concat()

        self.head = CSP_Head()

        self.init_weights()
        self.config = cfg

    def init_weights(self):
        self.head.init_weights()

    def forward(self, x, is_train=True):
        if self.config.score_map:
            if isinstance(self.backbone, ResNet50_CLIP):
                features, score_map = self.backbone(x)
                cls_map, reg_map, off_map = self.head(features)
                return cls_map, reg_map, off_map, score_map
            if isinstance(self.backbone, ResNet50_CLIP_VLPD):
                features, score_map, contrast_logits = self.backbone(x, is_train)
                cls_map, reg_map, off_map = self.head(features)
                return cls_map, reg_map, off_map, score_map, contrast_logits
        else:
            features = self.backbone(x)
            cls_map, reg_map, off_map = self.head(features)
            return cls_map, reg_map, off_map
        