# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List
from args import args
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_



class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn1d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose1d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        # self.conv1 = nn.Conv1d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False)
        # self.act1 = bn1d(cin)
        # self.act2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        # self.act3 = bn1d(cout)
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn1d(cin), nn.ReLU6(inplace=True),
            nn.Conv1d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn1d(cout)
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv(x)
        return x

#768
class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=384, sbn=False):   # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(n+1)] # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn1d = nn.BatchNorm1d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn1d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv1d(channels[-1],args.data_shape[1] , kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec: List[torch.Tensor]): #BCH
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)
    
    def extra_repr(self) -> str:
        return f'width={self.width}'
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
