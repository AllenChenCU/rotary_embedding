from functools import partial

import torch
import torch.nn as nn
import torchvision
#from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

from vit import ViT
from vit_rope import ViTRoPE


__all__ = [
    "vit_no_rope_cifar10", 
    "vit_rope_1D_axial_2x2_cifar10", 
    "vit_rope_2D_axial_2x2_cifar10", 
    "vit_rope_2D_axial_3x3_0_cifar10", 
    "vit_rope_2D_axial_4x4_0_cifar10", 
    "vit_rope_2D_axial_5x5_0_cifar10", 
    "vit_rope_2D_axial_6x6_0_cifar10", 
    "vit_weighted_rope_2D_axial_3x3_cifar10", 
    "vit_weighted_rope_2D_axial_4x4_cifar10", 
    "vit_rope_2D_axial_3x3_1_cifar10", 
    "vit_rope_2D_axial_3x3_2_cifar10", 
]

@register_model
def vit_no_rope_cifar10(pretrained=False, **kwargs):
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
        checkpoint = os.path.join(curr_dir, "model_registry/vit_no_rope_cifar10.pth")
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
def vit_rope_2D_axial_3x3_0_cifar10(pretrained=False, **kwargs):
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
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_3x3_0_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_3x3_1_cifar10(pretrained=False, **kwargs):
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
        m=0,
        n=2,
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_3x3_1_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_3x3_2_cifar10(pretrained=False, **kwargs):
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
        m=1,
        n=2,
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_3x3_2_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_4x4_0_cifar10(pretrained=False, **kwargs):
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
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_4x4_0_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_5x5_0_cifar10(pretrained=False, **kwargs):
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
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_5x5_0_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_rope_2D_axial_6x6_0_cifar10(pretrained=False, **kwargs):
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
        checkpoint = os.path.join(curr_dir, "model_registry/vit_rope_2D_axial_6x6_0_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_weighted_rope_2D_axial_3x3_cifar10(pretrained=False, **kwargs):
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
        weighted_rope=True,
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_weighted_rope_2D_axial_3x3_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model


@register_model
def vit_weighted_rope_2D_axial_4x4_cifar10(pretrained=False, **kwargs):
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
        weighted_rope=True,
    )
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_weighted_rope_2D_axial_4x4_cifar10.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model
