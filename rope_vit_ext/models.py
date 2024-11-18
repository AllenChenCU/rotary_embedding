from functools import partial

import torch
import torch.nn as nn
import torchvision
#from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

from vit import ViT
from vit_rope import ViTRoPE


__all__ = [
    "vit_cifar10", 
    "vit_rope_1D_axial_2x2_cifar10", 
    "vit_rope_2D_axial_2x2_cifar10", 
    "vit_rope_2D_axial_3x3_cifar10", 
    "vit_rope_2D_axial_4x4_cifar10", 
    "vit_rope_2D_axial_5x5_cifar10", 
    "vit_rope_2D_axial_6x6_cifar10", 
]

@register_model
def vit_cifar10(pretrained=False, **kwargs):
    model = ViT(
        image_size=32, 
        patch_size=1, 
        num_classes=10, 
        dim=128, 
        depth=4, 
        heads=4, 
        mlp_dim=256, 
        dropout=0.25, 
        dim_head=32, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_1D_axial_2x2_cifar10(pretrained=False, **kwargs):
    model = ViTRoPE(
        image_size=32, 
        patch_size=1, 
        num_classes=10, 
        dim=128, 
        depth=4, 
        heads=4, 
        mlp_dim=256, 
        dropout=0.25, 
        dim_head=32,
        rotary_position_emb="1D_axial", 
        rotation_matrix_dim=2, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_1D_axial_2x2_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_2x2_cifar10(pretrained=False, **kwargs):
    model = ViTRoPE(
        image_size=32, 
        patch_size=1, 
        num_classes=10, 
        dim=128, 
        depth=4, 
        heads=4, 
        mlp_dim=256, 
        dropout=0.25, 
        dim_head=32,
        rotary_position_emb="2D_axial", 
        rotation_matrix_dim=2, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_2x2_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_3x3_cifar10(pretrained=False, **kwargs):
    model = ViTRoPE(
        image_size=36, 
        patch_size=1, 
        num_classes=10, 
        dim=144, # should be dim_head * heads
        depth=4, 
        heads=4, 
        mlp_dim=288, # dim * 2
        dropout=0.25, 
        dim_head=36, # should be same as image_size
        rotary_position_emb="2D_axial", 
        rotation_matrix_dim=3, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_3x3_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_4x4_cifar10(pretrained=False, **kwargs):
    model = ViTRoPE(
        image_size=32, 
        patch_size=1, 
        num_classes=10, 
        dim=128, 
        depth=4, 
        heads=4, 
        mlp_dim=256, 
        dropout=0.25, 
        dim_head=32,
        rotary_position_emb="2D_axial", 
        rotation_matrix_dim=4, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_4x4_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_5x5_cifar10(pretrained=False, **kwargs):
    model = ViTRoPE(
        image_size=40, 
        patch_size=1, 
        num_classes=10, 
        dim=160, 
        depth=4, 
        heads=4, 
        mlp_dim=320, 
        dropout=0.25, 
        dim_head=40,
        rotary_position_emb="2D_axial", 
        rotation_matrix_dim=5, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_5x5_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_6x6_cifar10(pretrained=False, **kwargs):
    model = ViTRoPE(
        image_size=36, 
        patch_size=1, 
        num_classes=10, 
        dim=144, 
        depth=4, 
        heads=4, 
        mlp_dim=288, 
        dropout=0.25, 
        dim_head=36,
        rotary_position_emb="2D_axial", 
        rotation_matrix_dim=6, 
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_6x6_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


# @register_model
# def vit_small_patch16_224_scratch(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16,
#         embed_dim=384, 
#         depth=12, 
#         num_heads=6, 
#         mlp_ratio=4, 
#         qkv_bias=True, 
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     model.default_cfg = _cfg()
#     if pretrained:
#         curr_dir = os.path.dirname(__file__)
#         checkpoint = os.path.join(curr_dir, "model_registry/vit_small_path16_224.pth")
#         state_dict = torch.load(checkpoint)
#         model.load_state_dict(state_dict["net"])
#     return model
