import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.drop import DropPath, drop_path
from torch.nn.init import xavier_normal_, uniform_, constant_
from typing import List
from model import encoder
from .decoder import LightDecoder
from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock, MultiHeadAttention, Attention
from timm.models.layers import trunc_normal_
from timm import create_model




class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)


class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


class HSCMAE(nn.Module):
    def __init__(self, args,unet_layers,depths,dims,sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,global_decoder: LightDecoder,densify_norm='ln', sbn=False):
        super(HSCMAE, self).__init__()
        # d_model = args.d_model
        input_size, downsample_ratio = sparse_encoder.input_size, sparse_encoder.downsample_ratio
        self.downsample_ratio = downsample_ratio
        self.fmap_h = input_size // downsample_ratio
        self.mask_ratio = args.mask_ratio
        self.len_keep = round(self.fmap_h * (1 - self.mask_ratio))
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        self.global_decoder = global_decoder

        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        self.denseEnc = create_model('convnext_tiny1',unet_layers=unet_layers, depths=depths, dims=dims,bigkernel = args.bigker,
                                     num_classes = 0,drop_path_rate=0.1).to('cuda')
        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(
                self.hierarchy):  # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)

            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm1d if self.sbn else encoder.SparseBatchNorm1d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)

            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()  # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i + 1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv1d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         bias=True)
                print(
                    f'[SparK.__init__, densify {i + 1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)

            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2

        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
    def copy_weight(self,net1,net2):
        with torch.no_grad():
            for (param_a, param_b) in zip(net1.sp_cnn.parameters(), net2.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, inp_bchw: torch.Tensor ,active_b1ff=None, vis=False):
        # step1. Mask
        inp_bchw = inp_bchw.permute(0,2,1)
        if active_b1ff is None:  # rand mask
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        encoder._cur_active = active_b1ff  # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_ratio, 2)# (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest  BCH

        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff  # (B, 1, f, f)
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff,
                                   mask_tokens)  # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        attention = Attention()
        att_to_dec = []
        for vec in to_dec:
            rep,_ = attention(vec.transpose(-1,-2),vec.transpose(-1,-2),vec.transpose(-1,-2))
            rep = rep.transpose(-1,-2)
            att_to_dec.append(rep)
        # step4. Decode and reconstruct
        rec_global_bchw = self.global_decoder(att_to_dec)
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = self.patchify(inp_bchw), self.patchify(
            rec_bchw)  # inp and rec: (B, L = f*f, N = C*downsample_ratio**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .4
        inp = (inp - mean) / var    #计算有多少个最小图，最小图中计算均值方差，考虑尺度变换，方差还原回原尺度，所以5次方
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)

        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (
                    non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        loss = nn.MSELoss()
        global_loss= loss(inp_bchw, rec_global_bchw)
        if vis:
            masked_bchw = inp_bchw * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
            return inp_bchw, masked_bchw, rec_or_inp
        else:
            return recon_loss,global_loss



    def patchify(self, bchw):
        p = self.downsample_ratio
        h = self.fmap_h
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p))
        bchw = torch.einsum('bchp->bhpc', bchw)
        bln = bchw.reshape(shape=(B, h, C * p))  # (B, f*f, 3*downsample_raito**2)
        return bln

    def unpatchify(self, bln):
        p = self.downsample_ratio
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw
    def forward(self, x,hierarchical=False):
        self.linear_proba = True
        if self.linear_proba:
            with torch.no_grad():
                self.copy_weight(self.sparse_encoder,self.denseEnc)
                ls = self.denseEnc(x.transpose(1, 2),hierarchical = True)
                x = torch.mean(ls[-1], dim=1)
                return x


    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens

    def mask(self, B: int, device, generator=None):
        h = self.fmap_h
        idx = torch.rand(B, h , generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, h, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B,1,h)
