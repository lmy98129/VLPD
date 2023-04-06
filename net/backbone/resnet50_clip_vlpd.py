from collections import OrderedDict
from typing import Iterator, Union, List

from clip.model import CLIP
from clip.model import Bottleneck as BottleneckCLIP
import clip
import torch
import torch.nn as nn
from config import Config
from lib.l2norm import L2Norm
from torch.nn import functional as F

class BottleneckDilated(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilate=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                                dilation=dilate, padding=dilate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * BottleneckDilated.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet50_CLIP_VLPD(nn.Module):
    _AVAILABLE_MODELS = ['RN50']

    def __init__(self,
                cfg: Config,
                name: str = None,
                classnames: List[str] = None,
                templates: List[str] = None,
                device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                jit: bool = False, download_root: str = None):
        super(ResNet50_CLIP_VLPD, self).__init__()

        self.config = cfg
        if name is None:
            name = self.config.clip_weight

        if classnames is None:
            self.classnames = self.config.classnames

        if templates is None:
            self.templates = self.config.templates

        tmp_device = 'cpu'
        clip_model, preprocess = clip.load(name, tmp_device, jit, download_root)
        self.visual = clip_model.visual

        self.human_index = self.classnames.index('human')

        self._init_concat(device)
        self._make_dilated_layer4(device)
        self._init_visual(device)
        self._init_zeroshot_classifier(clip_model, device)

    def _init_visual(self, device):
        self.visual.to(device).to(self.dtype)
        self.conv1 = nn.Conv2d(self.visual.attnpool.v_proj.in_features,
                               self.visual.attnpool.v_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        self.conv2 = nn.Conv2d(self.visual.attnpool.c_proj.in_features,
                               self.visual.attnpool.c_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        conv1_weight_shape = (*self.visual.attnpool.v_proj.weight.shape, 1, 1)
        conv2_weight_shape = (*self.visual.attnpool.c_proj.weight.shape, 1, 1)
        self.conv1.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.v_proj.weight.reshape(conv1_weight_shape),
                        bias=self.visual.attnpool.v_proj.bias))
        self.conv2.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.c_proj.weight.reshape(conv2_weight_shape),
                        bias=self.visual.attnpool.c_proj.bias))
        
        del self.visual.layer4
        del self.visual.attnpool

    def _make_dilated_block(self, block:BottleneckCLIP, stride=1, dilate=1):
        block_inplanes = block.conv1.in_channels
        block_planes = block.conv1.out_channels
        block_dilated = BottleneckDilated(block_inplanes, block_planes, stride=stride, dilate=dilate)
        block_dilated.load_state_dict(block.state_dict())
        return block_dilated

    def _make_dilated_layer4(self, device):
        layers = [
            self._make_dilated_block(self.visual.layer4[0], stride=1, dilate=2),
            self._make_dilated_block(self.visual.layer4[1], stride=1, dilate=2),
            self._make_dilated_block(self.visual.layer4[2], stride=1, dilate=2),
        ]
        self.dilated_layer4 = nn.Sequential(*layers).to(device)

    def _init_concat(self, device):
        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1).to(device)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0).to(device)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0).to(device)

        self.p3_l2 = L2Norm(256, 10).to(device)
        self.p4_l2 = L2Norm(256, 10).to(device)
        self.p5_l2 = L2Norm(256, 10).to(device)

        self.feat = nn.Conv2d(768 + len(self.classnames), 256, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.feat_bn = nn.BatchNorm2d(256, momentum=0.01).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)

        self.init_concat_weights()

    def init_concat_weights(self):
        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)

        nn.init.xavier_normal_(self.feat.weight)

    @torch.no_grad()
    def _init_zeroshot_classifier(self, clip_model:CLIP, device):
        # refer to: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
        zeroshot_weights = []
        for classname in self.classnames:
            texts = [template.format(classname) for template in self.templates]  # format with class
            texts = clip.tokenize(texts)  # tokenize
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        # shape: [E, C]
        # where E is the dimension of an embedding and C is the number of classes.
        self.zeroshot_weights = nn.Parameter(torch.stack(zeroshot_weights, dim=1), requires_grad=False).to(device)

    @property
    def dtype(self):
        return torch.float32

    def _stem(self, x):
        if hasattr(self.visual, 'relu'):
            for conv, bn in [(self.visual.conv1, self.visual.bn1),
                            (self.visual.conv2, self.visual.bn2),
                            (self.visual.conv3, self.visual.bn3)]:
                x = self.visual.relu(bn(conv(x)))
        else:
            x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
            x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
            x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))

        x = self.visual.avgpool(x)
        return x

    def forward(self, image, is_train=True):
        image = image.type(self.dtype)
        feature = self._stem(image)
        feature = self.visual.layer1(feature)
        feature = self.visual.layer2(feature)

        p3 = self.p3(feature)
        p3 = self.p3_l2(p3)

        feature_layer3 = self.visual.layer3(feature)

        p4 = self.p4(feature_layer3)
        p4 = self.p4_l2(p4)
        
        dilated_feat = self.dilated_layer4(feature_layer3)
        p5 = self.p5(dilated_feat)
        p5 = self.p5_l2(p5)

        if not self.config.score_map:
            # Vanilla CSP
            cat = torch.cat([p3, p4, p5], dim=1)

            feat = self.feat(cat)
            feat = self.feat_bn(feat)
            feat = self.relu(feat)
            return feat
        else:
            # Score Map supervised by CLIP
            B, C, H, W = p5.shape

            feature_proj: torch.Tensor = self.conv1(dilated_feat)
            feature_proj = self.conv2(feature_proj)
            feature_proj_norm = feature_proj / torch.norm(feature_proj, p=2, dim=1, keepdim=True)
            # [B, w, h, E]
            feature_proj_t = feature_proj_norm.transpose(1, 3)
            # [B, w, h, C]
            score_map = feature_proj_t @ self.zeroshot_weights
            # [B, C, h, w]
            score_map = score_map.transpose(1, 3)
            score_map_p5: torch.Tensor = F.interpolate(score_map, (H, W))

            if is_train:
                cat_ori = torch.cat([p3, p4, p5], dim=1)
                contrast_logits = cat_ori
            else:
                contrast_logits = None
            
            cat = torch.cat([p3, p4, p5, score_map_p5], dim=1)
            feat = self.feat(cat)
            feat = self.feat_bn(feat)
            feat = self.relu(feat)

            return feat, score_map, contrast_logits

    @staticmethod
    def available_models():
        return ResNet50_CLIP_VLPD._AVAILABLE_MODELS