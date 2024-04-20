import os
import sys
sys.path.append(os.path.realpath('./'))
import torch
import torch.nn as nn
from monai.networks.nets import UNETR
from models import DiT, TimestepEmbedder, LabelEmbedder
from collections.abc import Sequence



    
class U_DiT(UNETR):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                img_size: Sequence[int] | int,
                feature_size: int = 16,
                hidden_size: int = 768,
                mlp_dim: int = 3072,
                num_heads: int = 12,
                pos_embed: str = "conv",
                proj_type: str = "conv",
                norm_name: tuple | str = "instance",
                conv_block: bool = True,
                res_block: bool = True,
                dropout_rate: float = 0.0,
                spatial_dims: int = 3,
                qkv_bias: bool = False,
                save_attn: bool = False,
                # for diffusion
                num_classes: int = 2,
                learn_sigma: bool = False,
                class_dropout_prob: float = 0.1,
                pos_embed_dim: int = 1):
        
        
        super().__init__(in_channels=in_channels,
                        out_channels=out_channels,
                        img_size=img_size,
                        feature_size=feature_size,
                        hidden_size=hidden_size,
                        mlp_dim=mlp_dim,
                        num_heads=num_heads,
                        pos_embed=pos_embed,
                        proj_type=proj_type,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,
                        dropout_rate=dropout_rate,
                        spatial_dims=spatial_dims,
                        qkv_bias=qkv_bias,
                        save_attn=save_attn)
        
        self.learn_sigma = learn_sigma
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.t_embedder = TimestepEmbedder(feature_size)
        self.y_embedder = LabelEmbedder(num_classes, feature_size, class_dropout_prob)
        
        self.dit = DiT(input_size=img_size[0] if isinstance(img_size, tuple) else img_size,
                        patch_size=self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size,
                        in_channels=in_channels,
                        hidden_size=self.hidden_size,
                        depth=self.num_layers,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        class_dropout_prob=self.class_dropout_prob,
                        num_classes=self.num_classes,
                        learn_sigma=True,
                        dim=spatial_dims,
                        pos_embed_dim=pos_embed_dim,
                        return_hidden_states=True)
    
    # TODO
    def forward(self, x_in, t, y):
        x, hidden_states_out = self.dit(x_in, t, y)
        print(f'x_in: {x_in.shape}')
        enc1 = self.encoder1(x_in)
        print(f'enc1: {enc1.shape}')
        x2 = hidden_states_out[3]
        print(f'x2: {x2.shape}')
        enc2 = self.encoder2(self.proj_feat(x2))
        print(f'enc2: {enc2.shape}')
        x3 = hidden_states_out[6]
        print(f'x3: {x3.shape}')
        enc3 = self.encoder3(self.proj_feat(x3))
        print(f'enc3: {enc3.shape}')
        x4 = hidden_states_out[9]
        print(f'x4: {x4.shape}')
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)
        
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
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
       
        # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        half_eps = (1+cfg_scale) * cond_eps - cfg_scale * uncond_eps
        
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
if __name__ == '__main__':
    size = 224
    x = torch.randn(2, 1, size, size, size)
    t = torch.randint(0, 1000, (2,))
    y = torch.tensor([0, 1])
    model = U_DiT(img_size=(size, size, size), 
                  in_channels=1, 
                  out_channels=1,
                  learn_sigma=False)
    out = model(x, t, y)
    print(out.shape)
        
        
    
        