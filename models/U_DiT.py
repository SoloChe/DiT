# import os
# import sys
# sys.path.append(os.path.realpath('./'))
# from models import DiT_models

import torch
import torch.nn as nn
from models.models import DiT_models

from collections.abc import Sequence
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep


class U_DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 2,
        # for diffusion
        num_classes: int = 2,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.1,
        # for DiT
        dit: str = 'DiT-XL/16',
        pos_embed_dim: int = 4,
    ):
        super().__init__()
        in_channels = in_channels
        out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.pos_embed_dim = pos_embed_dim
        
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims) # patch size for DiT
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
     
        
        # DiT_B_16: 12 layers, 12 heads, 768 hidden size, 3072 mlp size, 16 patch size
        # self.dit = DiT(
        #     input_size=img_size[0],
        #     patch_size=self.patch_size[0],
        #     in_channels=in_channels,
        #     hidden_size=hidden_size,
        #     depth=num_layers,
        #     num_heads=num_heads,
        #     mlp_ratio=4.0,
        #     class_dropout_prob=self.class_dropout_prob,
        #     num_classes=self.num_classes,
        #     learn_sigma=self.learn_sigma,
        #     dim=spatial_dims,
        #     pos_embed_dim=pos_embed_dim,
        #     return_hidden_states=True,
        # )
        self.dit = DiT_models[dit](
            in_channels=in_channels,
            input_size=img_size[0],
            num_classes=self.num_classes,
            dim=spatial_dims,
            pos_embed_dim=pos_embed_dim,
            learn_sigma=self.learn_sigma,
            return_hidden_states=True,
        )
        hidden_size = self.dit.hidden_size
        print(f'hidden_size: {hidden_size}')
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        # self.encoder5 = UnetrPrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=hidden_size,
        #     out_channels=feature_size * 16,
        #     num_layer=0,
        #     kernel_size=3,
        #     stride=1,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     conv_block=conv_block,
        #     res_block=res_block,
        # )
        # self.decoder6 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 16,
        #     out_channels=feature_size * 8,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims)) # (0, 3, 1, 2)
        self.proj_view_shape = list(self.feat_size) + [hidden_size] # 224/16 x 224/16 x 768
        
    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape # n x 14 x 14 x 768
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous() # n x 768 x 14 x 14
        return x

    # TODO
    def forward(self, x_in, t, y):
        hidden_states_out = self.dit(x_in, t, y) # ViT-B 12 layers; ViT-L 24 layers; ViT-XL 28 layers 7, 14, 21, 28
        # print(f'x_in: {x_in.shape}') # n x 1 x 224 x 224 for example
        enc1 = self.encoder1(x_in)
        # print(f'enc1: {enc1.shape}') # n x 16 x 224 x 224 where channels x 16
        x2 = hidden_states_out[6]
        # print(f'x2: {x2.shape}') # n x 196 x 768
        enc2 = self.encoder2(self.proj_feat(x2))
        # print(f'enc2: {enc2.shape}') # n x 32 x 112 x 112
        x3 = hidden_states_out[13]
        # print(f'x3: {x3.shape}') # n x 196 x 768
        enc3 = self.encoder3(self.proj_feat(x3))
        # print(f'enc3: {enc3.shape}') # n x 64 x 56 x 56
        x4 = hidden_states_out[20]
        # print(f'x4: {x4.shape}') # n x 196 x 768
        enc4 = self.encoder4(self.proj_feat(x4))
        # print(f'enc4: {enc4.shape}') # n x 128 x 28 x 28
        
        
        x = hidden_states_out[-1]
        # print(f'x: {x.shape}') # n x 196 x 768 if use ViT-B/16 for example
        dec4 = self.proj_feat(x)
        # print(f'dec4: {dec4.shape}') # n x 768 x 14 x 14
        dec3 = self.decoder5(dec4, enc4)
        # print(f'dec3: {dec3.shape}') # n x 128 x 28 x 28
        dec2 = self.decoder4(dec3, enc3)
        # print(f'dec2: {dec2.shape}') # n x 64 x 56 x 56
        dec1 = self.decoder3(dec2, enc2)
        # print(f'dec1: {dec1.shape}') # n x 32 x 112 x 112
        out = self.decoder2(dec1, enc1)
        # print(f'out: {out.shape}') # n x 16 x 224 x 224
        return self.out(out)  # n x 2 x 224 x 224 if learn_sigma else n x 1 x 224 x 224

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        half_eps = (1 + cfg_scale) * cond_eps - cfg_scale * uncond_eps
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


if __name__ == "__main__":
    size = 224
    x = torch.randn(3, 1, size, size)
    t = torch.randint(0, 1000, (3,))
    y = torch.tensor([0, 1, 1])
    model = U_DiT(
        img_size=size, in_channels=1, learn_sigma=True, spatial_dims=2, dit='DiT-XL/16'
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params/1024/1024:.2f}M")
    out = model(x, t, y)
    print(out.shape)
