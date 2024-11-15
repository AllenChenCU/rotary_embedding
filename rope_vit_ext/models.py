from functools import partial

import torch
import torchvision
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


__all__ = [
    "vit_small_patch16_224", 
]


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        mlp_ratio=4, 
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        curr_dir = os.path.dirname(__file__)
        checkpoint = os.path.join(curr_dir, "model_registry/vit_small_path16_224.pth")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["net"])
    return model
    
