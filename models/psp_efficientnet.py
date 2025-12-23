"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models.efficientnet import *


__all__ = ['get_efficientnet_b0',
           'get_efficientnet_b1',
           'get_efficientnet_b2',
           'get_efficientnet_b3',
           'get_efficientnet_b4',
           'get_efficientnet_b5',
           'get_efficientnet_b6',
           'get_efficientnet_b7',
           'get_psp_efficientnet']





class PSPNet(nn.Module):

    def __init__(self, nclass, backbone='efficientnet_b0', pretrained_base=True, out_indices=[7], **kwargs):
        super(PSPNet, self).__init__()

        if backbone == 'efficientnet_b0':
            self.pretrained = get_efficientnet_b0(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b1':
            self.pretrained = get_efficientnet_b1(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b2':
            self.pretrained = get_efficientnet_b2(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b3':
            self.pretrained = get_efficientnet_b3(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b1':
            self.pretrained = get_efficientnet_b4(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b2':
            self.pretrained = get_efficientnet_b5(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b3':
            self.pretrained = get_efficientnet_b6(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        elif backbone == 'efficientnet_b3':
            self.pretrained = get_efficientnet_b7(pretrained_base, out_indices=out_indices)
            in_channels = self.pretrained.out_channels_last
            self.head = _PSPHead(in_channels, nclass, **kwargs)
        else:
            raise KeyError('no such network')

    def forward(self, x):
        feats = self.pretrained(x)   # list[Tensor]
        c4 = feats[-1]               # usa l’ultimo out_indices
        x, features = self.head(c4)
        return [x, features]


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(in_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        print('PSPNet input channels to classifier:', in_channels)
        
        #compute the output channels after PPM
        out_channels = int(in_channels / 4)
        
        #clamp to minimum 128 channels
        if out_channels < 128:
            out_channels = 128
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(out_channels, nclass, 1)

    def forward(self, x):
        #get the output features from the PPM
        x = self.psp(x)

        #pass them through the final classifier
        x = self.block(x)
        feature = x

        #feed the classifier
        x = self.classifier(x)
        return x, feature


def get_psp_efficientnet(backbone='efficientnet_b0', local_rank=None,  pretrained=None,
            pretrained_base=True, num_class=19, out_indices=[7], **kwargs):

    model = PSPNet(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, out_indices=out_indices, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
    return model



if __name__ == '__main__':
    net = get_psp_efficientnet('efficientnet_b0', 'citys')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 512, 512)) / 1e6))

