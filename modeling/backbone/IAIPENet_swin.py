#!/usr/bin/env Python
# coding=utf-8
from functools import partial

from torchvision.models import resnet50

from modeling.backbone.textureattention import EdgeAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils import model_zoo

from utils.checkpoint import load_checkpoint
from utils.logger import get_root_logger



class msfrmv(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(msfrmv, self).__init__()
        self.scale_aware_proj = scale_aware_proj
        self.scene_encoder1_list = nn.ModuleList()
        for i in range(4):
            scene_encoder2 = nn.Sequential(
                    nn.Conv2d(channel_list[i], in_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels, in_channels, 1),
                )
            self.scene_encoder1_list.append(scene_encoder2)
        self.scene_encoder1 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels, in_channels, 1),
                )
        self.content_encoders1 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True)
                )
        self.feature_reencoders1 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True)
                )

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, i, 1),
                    nn.ReLU(True),
                    nn.Conv2d(i, i, 1),
                ) for i in [96, 192, 384, 768]]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, i, 1),
                    nn.ReLU(True),
                    nn.Conv2d(i, i, 1),
                ) for i in [96, 192, 384, 768]]
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c, i  in zip (channel_list, [96, 192, 384, 768]):
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, i, 1),
                    nn.BatchNorm2d(i),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, i, 1),
                    nn.BatchNorm2d(i),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature,feature_x_xonv,xg,j):
        content_feats = self.content_encoders1(feature_x_xonv)
        scene_feats = self.scene_encoder1_list[j](scene_feature)
        relations = self.normalizer(scene_feats*content_feats).sum(dim=1, keepdim=True)
        p_feats = self.feature_reencoders1(feature_x_xonv)
        xg = xg.view(xg.shape[0], 1, 1, 1)
        refined_feats = relations*p_feats*xg



        return refined_feats



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp_1(nn.Module):
    def __init__(self, config, i_layer):
        super(Mlp_1, self).__init__()
        self.fc1 = Linear(config.hidden_size[i_layer], config.hidden_size[i_layer]*4)
        self.fc2 = Linear(config.hidden_size[i_layer]*4, config.hidden_size[i_layer])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class CatMerging(nn.Module):
    def __init__(self, dim, conv_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.dim)
        self.norm = norm_layer(dim + conv_dim)
        self.reduction = nn.Linear(dim + conv_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

class CatMerging1(nn.Module):
    def __init__(self, dim, conv_dim):
        super().__init__()
        self.conv = nn.Conv2d(dim + dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        # self.norm = nn.LayerNorm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x





def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim #输入序列的通道数
        self.window_size = window_size  # Wh, Ww （w, h）
        self.num_heads = num_heads #多头的数量
        head_dim = dim // num_heads #每个投的通道数维度
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        #定义相对位置偏差参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # 获取窗口内每个标记的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # k = k.transpose(-2,-1)
        attn = (q @ k.transpose(-2, -1)) #b, h, n, c   b h c n -> b h n n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) #  b h n n * b h n c-  b h n c   b n h c
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
# 输入特征和卷积特征之间的上下文融合操作
def configgggg1():
    pass


class ContextFuseBlock(nn.Module):
    """ Swin Transformer Block.

       Args:
           dim (int): Number of input channels.
           num_heads (int): Number of attention heads.
           window_size (int): Window size.
           shift_size (int): Shift size for SW-MSA.
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
           qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
           qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
           drop (float, optional): Dropout rate. Default: 0.0
           attn_drop (float, optional): Attention dropout rate. Default: 0.0
           drop_path (float, optional): Stochastic depth rate. Default: 0.0
           act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
           norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
       """

    def __init__(self, i_layer, dim, conv_dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.i_layer = i_layer
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.lk = nn.Linear(conv_dim, dim, bias=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.redution = nn.Linear(2* dim, dim, bias=False)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.encoder = Encoder(configgggg1(), vis=False, i_layer=self.i_layer)

    def forward(self, x, y, i):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        y = self.lk(y)
        atten_x, atten_y = self.encoder(self.norm1(x), self.norm2(y), i)
        x = x + self.drop_path(atten_x)
        y = y + self.drop_path(atten_y)
        x = self.redution(torch.cat((x, y), dim=-1))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x



class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,agent_num=49, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # self.attn = AgentAttention(dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #                            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        #                            agent_num=agent_num)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


#将输入特征图进行下采样
class Merge_Block(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride= 2, padding=1)
        self.norm = nn.BatchNorm2d(1)
        # self.norm = nn.LayerNorm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.idx = idx
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, q,k,v):
        """
        x: B L C
        """
        # q, k = qk[0], qk[1]
        # conv_v = v[0]


        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        conv_v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        if self.idx == 0:
            x = (attn @ conv_v)
        else:
            x = (attn @ conv_v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

class CSWinBlock(nn.Module):

    def __init__(self,conv_dim, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.lk = nn.Linear(conv_dim, dim, bias=False)

    def forward(self, x, conv_out):
        """
        x: B, H*W, C
        """

        # H = W = self.patches_resolution
        B, L, C = x.shape
        # assert L == H * W, "flatten img_tokens has wrong size"
        conv_out = self.lk(conv_out)
        img = self.norm1(x)
        conv_out_img = self.norm4(conv_out)
        qk = self.qkv(img).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        v = conv_out_img.reshape(B, -1, 1, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qk[:, :, :, :C // 2],v[:, :, :, :C // 2])
            x2 = self.attns[1](qk[:, :, :, C // 2:],v[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qk,v)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import ml_collections

import math
class Attention(nn.Module):
    def __init__(self, config, vis, i_layer, mode=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.mode = mode
        self.num_attention_heads = config.transformer["num_heads"][i_layer]
        self.attention_head_size = int(config.hidden_size[i_layer] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.key = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.value = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.out = Linear(config.hidden_size[i_layer], config.hidden_size[i_layer])

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.queryd = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.keyd = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.valued = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.outd = Linear(config.hidden_size[i_layer], config.hidden_size[i_layer])

        self.valued1 = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.outd1 = Linear(config.hidden_size[i_layer], config.hidden_size[i_layer])

        self.valued2 = Linear(config.hidden_size[i_layer], self.all_head_size)
        self.outd2 = Linear(config.hidden_size[i_layer], config.hidden_size[i_layer])

        if self.mode == 'mba':
            self.w11 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)



        self.softmax = Softmax(dim=-1)
        self.resolution = [64,32,16,8]
        self.split_size = [1,2,8,8]
        self.num_heads = [2, 4, 8, 16]
        self.attns = nn.ModuleList([
            LePEAttention(
                config.hidden_size[i_layer], resolution=self.resolution[i_layer], idx=i,
                split_size=self.split_size[i_layer], num_heads=self.num_heads[i_layer] //2, dim_out=config.hidden_size[i_layer],
                qk_scale=None, attn_drop=0.1, proj_drop=0.1)
            for i in range(2)])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_statesx, hidden_statesy):
        mixed_query_layer = self.query(hidden_statesx)  # 256 768
        mixed_key_layer = self.key(hidden_statesx)
        mixed_value_layer = self.value(hidden_statesx)

        mixed_queryd_layer = self.queryd(hidden_statesy)  # 256 768
        mixed_keyd_layer = self.keyd(hidden_statesy)
        mixed_valued_layer = self.valued(hidden_statesy)

        attention_sx = self.attns[0](mixed_query_layer, mixed_keyd_layer, mixed_valued_layer)
        #
        attention_sy = self.attns[1](mixed_queryd_layer, mixed_key_layer, mixed_value_layer)




        query_layer = self.transpose_for_scores(mixed_query_layer)  # 12 256 64
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        queryd_layer = self.transpose_for_scores(mixed_queryd_layer)  # 12 256 64
        keyd_layer = self.transpose_for_scores(mixed_keyd_layer)
        valued_layer = self.transpose_for_scores(mixed_valued_layer)




        # return attention_sx, attention_sy, weights
        if self.mode == 'mba':
            # ## Cross Attention x: Qx, Ky, Vy
            attention_scores = torch.matmul(query_layer, keyd_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, valued_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cx = self.out(context_layer)
            attention_cx = self.proj_dropout(attention_cx)

            ## Cross Attention y: Qy, Kx, Vx
            attention_scores = torch.matmul(queryd_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cy = self.outd(context_layer)
            attention_cy = self.proj_dropout(attention_cy)


            attention_sx = self.w11 * attention_sx + self.w12 * attention_cx
            attention_sy = self.w21 * attention_sy + self.w22 * attention_cy


        return attention_sx, attention_sy
class Block(nn.Module):
    def __init__(self, config, vis, i_layer, mode=None,):
        super(Block, self).__init__()
        self.i_layer = i_layer
        self.hidden_size = config.hidden_size[i_layer]
        self.attention_norm = LayerNorm(config.hidden_size[i_layer], eps=1e-6)
        self.attention_normd = LayerNorm(config.hidden_size[i_layer], eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size[i_layer], eps=1e-6)
        self.ffn_normd = LayerNorm(config.hidden_size[i_layer], eps=1e-6)
        self.ffn = Mlp_1(config,i_layer = self.i_layer)
        self.ffnd = Mlp_1(config,i_layer = self.i_layer)
        self.attn = Attention(config, vis, mode=mode,i_layer = self.i_layer)

    def forward(self, x, y):
        hx = x
        hy = y
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y = self.attn(x, y)
        x = x + hx
        y = y + hy

        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        return x, y

class Encoder(nn.Module):
    def __init__(self, config, vis, i_layer):
        super(Encoder, self).__init__()
        self.vis = vis
        self.i_layer = i_layer
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size[i_layer], eps=1e-6)
        self.encoder_normd = LayerNorm(config.hidden_size[i_layer], eps=1e-6)

        self.layer = Block(config, vis, mode='mba', i_layer = self.i_layer)

    def forward(self, hidden_statesx, hidden_statesy,i):

        hidden_statesx, hidden_statesy = self.layer(hidden_statesx, hidden_statesy)
        encodedx = self.encoder_norm(hidden_statesx)
        encodedy = self.encoder_normd(hidden_statesy)
        return encodedx, encodedy





class BasicLayer(nn.Module):

    def __init__(self,
                 i_layer,
                 dim,
                 depth,
                 conv_channel,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 use_attens=1,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_attens = use_attens
        self.i_layer = i_layer
        # self.split_size = [1, 2, 8, 8]
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.conv_channel = conv_channel

        self.catmerge = CatMerging(dim=dim, conv_dim=conv_channel)




        self.fuseblock = ContextFuseBlock(
            i_layer = i_layer,
            dim=dim,
            conv_dim=conv_channel,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[depth - 1] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer
        )


        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, conv_out, H, W,i):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        # if i == 4:
        #     x = torch.cat((x, conv_out), dim=-1)
        #     x = self.catmerge(x)
        # else:
        #     print(111111111111111)
        x = self.fuseblock(x, conv_out,i)
            # x = self.fuseblock(x, conv_out)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224,patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patches_resolution = patches_resolution

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        #对输入图像进行了零填充（zero-padding），以确保图像尺寸能够被图块大小整除。这是因为图块嵌入需要确保输入图像能够被等分为若干图块。
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        #使用卷积操作 nn.Conv2d 对输入图像进行线性投影，将每个图块映射到一个较高维度的表示，
        #以生成图块嵌入。投影的卷积核大小由 patch_size 决定，步幅与 patch_size 相同。
        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x #返回图块嵌入作为输出


class ResNetBlock(nn.Module):
    def __init__(self, in_channels = 3, pretrained=True, ):
        """Declare all needed layers."""
        super(ResNetBlock, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook
        layers_cfg = [4, 5, 6, 7]
        self.blocks = nn.ModuleList()
        self.attens = nn.ModuleList()
        self.out_channels = [256, 512, 1024, 2048]
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    def forward(self, x, i):
        B, C = x.size(0), x.size(1)
        if i < 1:
            x = self.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

        block = self.blocks[i]
        x = block(x)
        # print("____________________________________block______________________________________________")
        # print(block)
        if i == 3:
            x = self.maxpool(x)

        return x, x.permute(0, 2, 3, 1).view(B, -1, self.out_channels[i])

# 实现了一个基于 Swin Transformer 模型的图像分类的神经网络模型。
class ISwinTransformerV3(nn.Module):


    def __init__(self,
                 pretrain_img_size=512,  #指定用于训练预训练模型的输入图像尺寸
                 patch_size=4,  #指定图像划分为图块的大小。可以是单个整数或元组，以指定宽度和高度不同的图
                 in_chans=3,  #指定输入图像的通道数
                 embed_dim=96,  #指定线性投影输出通道的数量，即图块嵌入的维度。
                 depths=[2, 2, 6, 2],  #指定每个 Swin Transformer 阶段的层数。
                 num_heads=[3, 6, 12, 24],  #指定每个阶段的注意力头数。
                 strides = [2, 2, 2, 1],  #指定每个阶段的步幅，用于决定图块嵌入的输出大小。
                 window_size=7,  #窗口大小，影响 Swin Transformer 的注意力机制中的局部窗口大小。
                 mlp_ratio=4.,  #MLP（多层感知器）的隐藏维度与嵌入维度之间的比率。
                 qkv_bias=True,  #如果为 True，则为查询（query）、键（key）和值（value）添加可学习的偏置项。
                 qk_scale=None,  #用于覆盖默认的查询和键的缩放因子，如果设置为 None，则使用默认值。
                 drop_rate=0.,  #全局的 dropout 比率，用于在模型的不同层中进行随机失活。
                 attn_drop_rate=0.,  #注意力机制中的 dropout 比率。
                 drop_path_rate=0.2,  #用于实现随机深度（stochastic depth）的比率，控制是否丢弃某些层。
                 norm_layer=nn.LayerNorm,  #指定规范化层的类型，用于在模型中应用规范化操作。
                 ape=False,  #如果为 True，则添加绝对位置嵌入到图块嵌入中。
                 patch_norm=True,  #如果为 True，则在图块嵌入后添加规范化。
                 out_indices=(0, 1, 2, 3),  #指定从哪些阶段输出中获取中间结果。用于多尺度特征融合等任务。
                 frozen_stages=-1,  #指定应该冻结的阶段数，即停止更新参数。-1 表示不冻结任何参数。
                 use_checkpoint=False,  #是否使用检查点技术以节省内存
                 use_attens=1,  #指定使用哪种类型的注意力机制，用于控制哪些位置相互作用。
                 pretrained=True,  #否加载预训练权重。
                 layer_name="tiny", scm=None):#指定加载哪个版本的预训练权重，可以是 "tiny"、"small" 或 "large"。
        super().__init__()

        self.out_channels = [256, 512, 1024, 2048]

        self.gaaa1 = scm.GlobalAvgPool2D()
        self.sr = msfrmv(in_channels=2048,
                         channel_list=(256, 512, 1024, 2048),
                         out_channels=256,
                         scale_aware_proj=True, )


        self.pretrain_img_size = pretrain_img_size#指定用于训练预训练模型的输入图像尺寸
        self.num_layers = len(depths)#指定每个 Swin Transformer 阶段的层数。
        # print("self.num_layers:",self.num_layers)
        self.embed_dim = embed_dim#指定线性投影输出通道的数量，即图块嵌入的维度。
        self.strides = strides#指定每个阶段的步幅，用于决定图块嵌入的输出大小。
        self.ape = ape#如果为 True，则添加绝对位置嵌入到图块嵌入中
        self.patch_norm = patch_norm#如果为 True，则在图块嵌入后添加规范化
        self.out_indices = out_indices#指定从哪些阶段输出中获取中间结果。用于多尺度特征融合等任务。
        self.frozen_stages = frozen_stages#指定应该冻结的阶段数，即停止更新参数。-1 表示不冻结任何参数。

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=pretrain_img_size,patch_size=patch_size, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        # 创建并初始化绝对位置嵌入张量 self.absolute_pos_embed，
        if self.ape:
            #pretrain_img_size 和 patch_size 变量都被转换为元组形式，以确保它们是长度为 2 的元组。这是为了确保能够进行后续的计算。
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            #计算了输入图像可以划分成多少个图块，分别在宽度（patches_resolution[0]）
            #和高度（patches_resolution[1]）方向上。这个计算通过将输入图像的尺寸 pretrain_img_size 除以图块大小 patch_size 得到。
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        patches_resolution1 = self.patch_embed.patches_resolution
        self.conv_layers = nn.ModuleList()
        self.conv_channels = [256, 512, 1024, 2048]
        self.in_dim = [96, 192, 384, 768]
        self.dim = [96, 192, 384, 768]
        self.input_resolution = [64,32,16,8]
        self.edge_attentions_layers = nn.ModuleList()
        self.catmerge_list = nn.ModuleList()

        for i_layer in range(self.num_layers):

            layer = BasicLayer(
                i_layer = i_layer,
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                conv_channel=self.conv_channels[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_attens = use_attens)
            self.edge_attentions_layers.append(
                EdgeAttention(dim=int(embed_dim * 2 ** i_layer),
                              input_resolution=(self.input_resolution[i_layer],
                                                self.input_resolution[i_layer]),
                              conv_channel=self.in_dim[i_layer], ))
            self.catmerge_list.append(CatMerging1(dim=self.dim[i_layer], conv_dim=self.conv_channels[i_layer]))

            self.layers.append(layer)
            #print(layer)


        self.gaaa1 = scm.GlobalAvgPool2D()
        self.sss1_list = nn.ModuleList()
        for i in range(4):
            ss1 = msfrmv(in_channels=self.in_dim[i],
                        channel_list=(96, 192, 384, 768),
                        out_channels=96,
                        scale_aware_proj=True, )
            self.sss1_list.append(ss1)

        self.gate_conv_beta_list = nn.ModuleList()
        self.raw_beta = nn.Parameter(data=torch.Tensor(1), requires_grad=True)  # 偏置参数声明

        for i in range(4):
            gate_conv_beta = nn.Sequential(
                nn.Conv2d(self.in_dim[i], 4, kernel_size=1, stride=1, padding=0, bias=False, ),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=True))
            self.gate_conv_beta_list.append(gate_conv_beta)
        self.conv0_list = nn.ModuleList()
        for i in [96, 192, 384, 768]:
            conv0 = nn.Conv2d(i, 1, kernel_size=1, bias=True)
            self.conv0_list.append(conv0)
        self.maxpool = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.w11 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w12 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.w11.data.fill_(0.5)
        self.w12.data.fill_(0.5)

        self.blocks = ResNetBlock()
        self.downsample = Merge_Block()

        self.conv3_list = nn.ModuleList()
        self.maxpool3x3_list = nn.ModuleList()
        self.maxpool5x5_list = nn.ModuleList()
        for i in range(4):
            conv3 = nn.Conv2d(self.in_dim[i], self.in_dim[i], kernel_size=1, bias=False)
            maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            maxpool5x5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            self.conv3_list.append(conv3)
            self.maxpool3x3_list.append(maxpool3x3)
            self.maxpool5x5_list.append(maxpool5x5)


        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            # print(num_features[i_layer])
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)



        self._freeze_stages()
        if pretrained:
            if layer_name == "tiny":
                self._load_pretrained_model()
            elif layer_name == "small":
                self._load_pretrained_model1()
            elif layer_name == "large":
                self._load_pretrained_model2()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self, x):
        """Forward function."""
        # 获取输入张量 x 的通道数
        c = x.size(1)
        B = x.size(0)
        # 用于确保输入张量 x 具有3个通道
        if c == 3:
            x_conv = x
        else:
            x_homogeneity = x[:, 9:12, :, :]
            x_entropy = x[:, 6:9, :, :]
            x_conv = x[:, 3:6, :, :]
            x = x[:, 0:3, :, :]

        # 分成小块
        x = self.patch_embed(x)
        #
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        # print(x.size())
        x = self.pos_drop(x)
        # print(x.size())

        outs = []
        for i in range(self.num_layers):#self.num_layers=4
            # print(i)
            layer = self.layers[i]
            x_conv, conv_out = self.blocks(x_conv, i)

            x_out, H, W, x, Wh, Ww = layer(x,conv_out, Wh, Ww,i)
            x_out = x_out if (i < self.num_layers - 1) else x

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)


        outs = self.forward_MSRFM(outs,B)

        outs = self.forward_TEEM(outs,x_homogeneity,x_entropy)



        return tuple(outs)
    def forward_MSRFM(self, feature_x,B):
        feature = []
        belta = torch.sigmoid(self.raw_beta)
        for i in range(4):
            x_getEntropy = self.conv0_list[i](feature_x[i])
            xe = self.getEntropy(x_getEntropy.view(B, -1), B)
            out = xe.view(B, 1, 1, 1) * feature_x[i]
            xg = self.getGate(out, i) + belta
            # out_multi = []
            n, c, h, w = feature_x[i].shape
            feature_out = torch.zeros(n, c, h, w)
            # 定义目标设备
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # 假设image是你的输入张量，将其移动到目标设备上
            feature_out = feature_out.to(device)
            for j in range(4):
                scene_feature = self.gaaa1(feature_x[j])
                x_out = self.sss1_list[i](scene_feature, feature_x[i], xg[:, j:(j + 1)], j)
                # out_multi.append(x_out)
                feature_out = feature_out + x_out
            feature.append(feature_out + feature_x[i])
        return feature

    def getGate(self, x, i):

        n, c, h, w = x.shape
        x = torch.relu(torch.tanh(self.gate_conv_beta_list[i](x).view(n, 4)))
        return x

    @torch.no_grad()
    def getEntropy(self, x, B):
        '''
        信息熵计算
        '''
        res = []
        # print(x.size()) # 4 3136
        dvc = x.device
        for i in range(B):
            tt = x[i]
            output, inverse_indices = torch.unique(tt, sorted=True, return_inverse=True)
            z = torch.bincount(inverse_indices)
            z = z.float() / z.sum()
            # e = -sum(pi * log2(pi))
            res.append(z.view(1, -1).mm(-torch.log2(z.view(-1, 1))).item())

        return torch.tensor(res).view(-1, 1).to(dvc)

    def forward_TEEM(self,feature,x_homogeneity,x_entropy):
        x_homogeneity = self.maxpool(x_homogeneity)
        x_entropy = self.maxpool(x_entropy)

        outs = []
        for i,x in zip([0,1,2,3],feature):
            # x, _ = self.blocks(x, i)
            B, C, H, W = x.shape

            x_homogeneity = self.downsample(x_homogeneity)
            x_entropy = self.downsample(x_entropy)

            B1, C1, H1, W1 = x_homogeneity.shape

            x_homogeneity1 = x_homogeneity.reshape(B1, C1, H1 * W1).permute(0, 2, 1).contiguous()
            x_entropy1 = x_entropy.reshape(B1, C1, H1 * W1).permute(0, 2, 1).contiguous()

            # layer = self.layers[i]
            qkv_x = self.forward_x_qkv(x, i)

            x_edge_homogeneity = x_homogeneity1.repeat(1, 1, C) + qkv_x[0]
            x_edge_entropy = x_entropy1.repeat(1, 1, C) + qkv_x[0]

            x_out_homogeneity = self.edge_attentions_layers[i](qkv_x[1], x_edge_homogeneity)
            x_out_entropy = self.edge_attentions_layers[i](qkv_x[1], x_edge_entropy)
            # outs.append(x_out)
            x_out = self.w11 * x_out_entropy + self.w12 * x_out_homogeneity
            # x_out, H, W, x, Wh, Ww = layer(x,conv_out, Wh, Ww)
            # x_out = x_out if (i < self.num_layers - 1) else x
            #
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                x_out = torch.cat((x, out), dim=1)
                out = self.catmerge_list[i](x_out)
                outs.append(out)

        return outs
    def forward_x_qkv(self,x,i):
        B, C, H, W = x.shape
        q_x = self.conv3_list[i](x)
        q_x = q_x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
        x1 = self.conv3_list[i](x)
        x2 = self.maxpool3x3_list[i](x)
        x3 = self.maxpool5x5_list[i](x)
        kv_x = torch.cat((x1, x2, x3), dim=1)
        kv_x = kv_x.reshape(B, C*3, H * W).permute(0, 2, 1).contiguous()
        return [q_x,kv_x]


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ISwinTransformerV3, self).train(mode)
        self._freeze_stages()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict['model'].items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _load_pretrained_model1(self):
        pretrain_dict = model_zoo.load_url(
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict['model'].items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _load_pretrained_model2(self):
        pretrain_dict = model_zoo.load_url(
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict['model'].items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
