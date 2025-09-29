# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
try:
    # from MultiScaleDeformableAttention.modules import MSDeformAttn
    from ops.deformable_attention.modules import MSDeformAttn
except:
    print("Please install MSDeformAttn if you want to use ViT-Adapter")
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs, LayerNorm
from .intern_vit import InternViT

_logger = logging.getLogger(__name__)



from .base.vit import TIMMVisionTransformer
# from .base.vit import ResBottleneckBlock


@BACKBONES.register_module()
class ViTAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=16, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False,with_last_norm=False,only_feat_out=False, *args, **kwargs):
        
        super().__init__(num_heads=num_heads, pretrained=pretrained,
                         with_cp=with_cp, *args, **kwargs)
        
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        self.with_last_norm=with_last_norm
        self.only_feat_out=only_feat_out
        if self.with_last_norm == False:
            del self.norm
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=with_cp)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def _get_pos_embed(self, pos_embed, H, W):
    #     pos_embed = pos_embed.reshape(
    #         1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
    #     pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
    #         reshape(1, -1, H * W).permute(0, 2, 1)
    #     return pos_embed
    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.img_size // 16, self.img_size // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
        
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        
        if self.with_last_norm:
            if self.only_feat_out:
                return [f1, f2, f3, f4]
            else:
                return [f1, f2, f3, f4],self.norm(x)
        else:
            return [f1, f2, f3, f4],x

# from .base.vit import TIMMVisionTransformer
# from .base.vit import ResBottleneckBlock
@BACKBONES.register_module()
class InternViTAdapter(InternViT):
    def __init__(self, pretrain_size=224, num_heads=16, conv_inplane=64, n_points=4, deform_num_heads=16,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=0.5, add_vit_feature=True, use_extra_extractor=True, with_cp=True,
                 out_indices=(0, 1, 2, 3), use_final_norm=True, only_feat_out=False, *args, **kwargs):

        super().__init__(num_heads=num_heads, with_cp=with_cp, *args, **kwargs)

        # self.num_classes = 80
        self.cls_token = None
        self.num_layers = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.out_indices = out_indices
        self.use_final_norm = use_final_norm
        embed_dim = self.embed_dim
        self.only_feat_out = only_feat_out

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim,
                                      out_indices=out_indices , with_cp = with_cp)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(
                                 interaction_indexes) - 1 else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        if len(out_indices) == 4:
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            if self.use_final_norm:
                self.norm1 = LayerNorm(embed_dim)
            self.up.apply(self._init_weights)

        if self.use_final_norm:
            self.norm2 = LayerNorm(embed_dim)
            self.norm3 = LayerNorm(embed_dim)
            self.norm4 = LayerNorm(embed_dim)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        # _freeze_params(self.blocks)
        # _freeze_params(self.patch_embed)
        # self.pos_embed.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        x = x.to(self.dtype)
        # SPM forward
        if len(self.out_indices) == 4:
            c1, c2, c3, c4 = self.spm(x)
        else:
            c2, c3, c4 = self.spm(x)

        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        # without class token
        # _, _, H, W=x.shape
        
        # x = self.embeddings(x)
        # bs, n, dim = x.shape
        # # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        # x = self.pos_drop(x + self.pos_embed[:, 1:])
        # x = self.embeddings(x)
        # x, H, W = self.patch_embed(x)

        x, H, W, bs, n, dim = self.embeddings.forward_adapter(x)
        
        # x, H, W, bs, n, dim = self.embeddings.forward(x)
        
        # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        # x = self.pos_drop(x + self.pos_embed[:, 1:])
        # x= x[:,1:,:]
        
        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        if len(self.out_indices) == 4:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            if len(self.out_indices) == 4:
                x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
                # 处理BFloat16兼容性问题
                orig_dtype = x3.dtype
                if orig_dtype == torch.bfloat16:
                    x3_cast = x3.float()
                    x1 = F.interpolate(x3_cast, scale_factor=4, mode='bilinear', align_corners=False).to(orig_dtype)
                    x2 = F.interpolate(x3_cast, scale_factor=2, mode='bilinear', align_corners=False).to(orig_dtype)
                    x4 = F.interpolate(x3_cast, scale_factor=0.5, mode='bilinear', align_corners=False).to(orig_dtype)
                else:
                    x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
                    x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            else:
                x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
                # 处理BFloat16兼容性问题
                orig_dtype = x3.dtype
                if orig_dtype == torch.bfloat16:
                    x3_cast = x3.float()
                    x2 = F.interpolate(x3_cast, scale_factor=2, mode='bilinear', align_corners=False).to(orig_dtype)
                    x4 = F.interpolate(x3_cast, scale_factor=0.5, mode='bilinear', align_corners=False).to(orig_dtype)
                else:
                    x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
                    x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
                c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        # print('shape_f1: ',c1.shape)
        # print('shape_f2: ',c2.shape)
        # print('shape_f3: ',c3.shape)
        # print('shape_f4: ',c4.shape)
        # print('x.shape: ',x.shape)
        if self.use_final_norm:
            if len(self.out_indices) == 4:
                f1 = self.norm1(c1.float()).contiguous()
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                if self.only_feat_out:
                    return [f1, f2, f3, f4]
                return [f1, f2, f3, f4],x
            else:
                f2 = self.norm2(c2.float()).contiguous()
                f3 = self.norm3(c3.float()).contiguous()
                f4 = self.norm4(c4.float()).contiguous()
                return [f2, f3, f4]
        else:
            return [c1.float().contiguous(),
                    c2.float().contiguous(),
                    c3.float().contiguous(),
                    c4.float().contiguous()]
