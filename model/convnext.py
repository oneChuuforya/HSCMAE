# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is basically a copy of: https://github.com/facebookresearch/ConvNeXt/blob/06f7b05f922e21914916406141f50f82b4a15852/models/convnext.py
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from args import args
from .encoder import SparseConvNeXtBlock, SparseConvNeXtLayerNorm


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    
    def __init__(self,unet_layers=2,bigkernel=True, in_chans=9, num_classes=1000,
                 depths=[1, 1, 1], dims=[96, 192, 384], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., global_pool='avg',
                 sparse=True,
                 ):
        super().__init__()
        self.unet = unet_layers
        self.dims: List[int] = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(#BCH经过layernorm-> BHC  yuanlai4
            nn.Conv1d(in_chans, dims[0], kernel_size=1, stride=1),
            SparseConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first", sparse=sparse)
        )
        self.downsample_layers.append(stem)
        for i in range(self.unet-1):#yuanlai2
            downsample_layer = nn.Sequential(
                SparseConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first", sparse=sparse),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.unet):#yuanlai3
            stage = nn.Sequential(
                *[SparseConvNeXtBlock(dim=dims[i],bigkernel = bigkernel, drop_path=dp_rates[cur + j],
                                      layer_scale_init_value=layer_scale_init_value, sparse=sparse) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.depths = depths
        
        self.apply(self._init_weights)
        if num_classes > 0:
            self.norm = SparseConvNeXtLayerNorm(dims[-1], eps=1e-6, sparse=False)  # final norm layer for LE/FT; should not be sparse
            self.fc = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = nn.Identity()
            self.fc = nn.Identity()

    
    def get_downsample_ratio(self) -> int:
        return  2**(self.unet-1)#yuanlai 16
    
    def get_feature_map_channels(self) -> List[int]:
        return self.dims
    
    def forward(self, x, hierarchical=False):
        if hierarchical:
            ls = []
            for i in range(self.unet):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                ls.append(x)
            return ls
        else:
            x = x.permute(0,2,1)
            for i in range(self.unet):
                # print(self.downsample_layers[i])
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x = x.mean(-1)
            x = self.norm(x)
            x = self.fc(x)
            return x
            # return self.fc(self.norm(x.mean(-1))) # (B, C, H, W) =mean=> (B, C) =norm&fc=> (B, NumCls)
    
    def get_classifier(self):
        return self.fc
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    unet_layers = kwargs['unet_layers']
    depths = kwargs['depths']
    dims = kwargs['dims']
    bigkernel = kwargs['bigkernel']
    kwargs.pop('unet_layers')
    kwargs.pop('depths')
    kwargs.pop('dims')
    kwargs.pop('bigkernel')
    model = ConvNeXt(unet_layers=unet_layers,bigkernel=bigkernel,in_chans=args.data_shape[1],
                     depths=depths, dims=dims, **kwargs)#dims=[96, 192, 384]
    kwargs['unet_layers'] = unet_layers
    kwargs['depths'] = depths
    kwargs['dims'] = dims
    kwargs['bigkernel'] = bigkernel
    return model

@register_model
def convnext_tiny1(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop('pretrained_cfg')
    unet_layers = kwargs['unet_layers']
    depths = kwargs['depths']
    dims = kwargs['dims']
    bigkernel = kwargs['bigkernel']
    kwargs.pop('unet_layers')
    kwargs.pop('depths')
    kwargs.pop('dims')
    kwargs.pop('bigkernel')
    model = ConvNeXt(unet_layers=unet_layers,bigkernel=bigkernel,in_chans=args.data_shape[1],depths=depths, dims=dims, sparse= False,**kwargs)#dims=[96, 192, 384],depths=[3,6,3]
    kwargs['unet_layers'] = unet_layers
    kwargs['depths'] = depths
    kwargs['dims'] = dims
    kwargs['bigkernel'] = bigkernel
    return model