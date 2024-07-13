
import torch.nn as nn
import torch

import torch.nn.functional as F


from modeling.backbone.IAIPENet_swin import ISwinTransformerV3
from modeling.sseg.uperhead import UperNetHead

class IAIPENet(nn.Module):
    def __init__(self, pretrain_img_size=224, num_classes=2, in_chans = 3, use_attens=1):
        super(IAIPENet, self).__init__()
        self.backbone = ISwinTransformerV3(
            pretrain_img_size=pretrain_img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
            use_attens=use_attens,
            layer_name="tiny")
        self.decode_head = UperNetHead(
            in_channels=[96, 192, 384, 768],
            channels=512,
            num_classes=num_classes,
        )


    def forward(self, input):

        size = input.size()[2:]

        x = self.backbone(input)

        main_ = self.decode_head(x)
        # print(main_.shape)

        main_ = F.interpolate(main_, size, mode='bilinear', align_corners=True)
        # print(main_.shape)
        return main_  # 主分类器，辅助分类器




if __name__ == '__main__':
    # 定义目标设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = IAIPENet(pretrain_img_size=512)
    model = model.to(device)

    images = torch.rand(size=(16, 12, 256, 256))
    images = images.to(device)

    ret1 = model(images)

