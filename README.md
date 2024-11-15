# Extending RoPE mechanism from [Heo et al., 2024]

Author: Allen Chen

Problem statement: The goal of this task is to extend regular Rotary Positional Encodings (or RoPE), presented in particular in [Heo et al., 2024]. The regular Rotary Positional Encoding mechanism applies rotation matrices with 2 × 2 blocks constituting learnable Givens rotations (2-dimensional rotations in spaces spanned by canonical directions). Can you design and test (see: Problem 1) an extension of the RoPE mechanism that applies blocks of arbitrary sizes ? Remember that you would like to preserve the relative positional encoding property: the resulting rotation matrix R1R⊤ should depend only on the delta-vectors between the coordinate vectors associated with the tokens.

## Design

Pre-trained model: Vision Transformer

Benchmark dataset: CIFAR10

For each of the following tasks, 
1. fine-tune ViT without RoPE
    - vit_b_16
2. fine-tune ViT with RoPE with 2x2 axial frequency (non-efficient way with rotational matrix)
3. fine-tune ViT with RoPE with 2x2 axial frequency (efficient way with complex number)
4. fine-tune ViT with RoPE with 3x3 axial frequency (non-efficient way with rotational matrix)

Measure accuracy and speed for training and inferencing. 

export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python3 rope_vit_ext/main.py --cuda --epochs 1 --run_id vit_small_patch16_224 --dataset cifar10 --distributed --world_size 4 --optimizer adam --batch_size 64



