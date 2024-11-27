# Arbitrary Dimension of Rotation Matrix in Rotary Position Embedding

Author: Allen Chen

Problem statement (Extending RoPE mechanism): The goal of this task is to extend regular Rotary Positional Encodings (or RoPE), presented in particular in [Heo et al., 2024]. The regular Rotary Positional Encoding mechanism applies rotation matrices with 2x2 blocks constituting learnable Givens rotations (2-dimensional rotations in spaces spanned by canonical directions). Can you design and test (see: Problem 1) an extension of the RoPE mechanism that applies blocks of arbitrary sizes? Remember that you would like to preserve the relative positional encoding property: the resulting rotation matrix $R_{1}R^T$ should depend only on the delta-vectors between the coordinate vectors associated with the tokens.

## Design 

Model: Vision Transformer

Benchmark dataset: CIFAR10

## Repository
```
├── outputs: outputs for all the experiments in the paper
├── rope_vit_ext
│   ├── model_registry: saved model files from all the experiments
│   ├── data.py: helper function to load the data
│   ├── main.py: entry point to running the experiments
│   ├── models.py: configured models with defined names
│   ├── train.py: model training helper functions
│   ├── utils.py: utility functions
│   ├── vit_rope.py: an implementation of ViT with RoPE
│   ├── vit.py: an implementation of ViT
├── results.ipynb: An analysis for the outputs from all the experiments
```

## Commands

Train models with 4 Nvidia T4 GPUs

```
# baseline models
python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_no_rope_cifar10 --model vit_no_rope_cifar10 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_1D_axial_2x2_cifar10 --model vit_rope_1D_axial_2x2_cifar10 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_2x2_cifar10 --model vit_rope_2D_axial_2x2_cifar10 --input_size 32 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001


# Models with higher-dimension rotation matrix
python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_3x3_0_cifar10 --model vit_rope_2D_axial_3x3_0_cifar10 --input_size 36 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_3x3_1_cifar10 --model vit_rope_2D_axial_3x3_1_cifar10 --input_size 36 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_3x3_2_cifar10 --model vit_rope_2D_axial_3x3_2_cifar10 --input_size 36 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_4x4_0_cifar10 --model vit_rope_2D_axial_4x4_0_cifar10 --input_size 32 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_5x5_0_cifar10 --model vit_rope_2D_axial_5x5_0_cifar10 --input_size 40 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_rope_2D_axial_6x6_0_cifar10 --model vit_rope_2D_axial_6x6_0_cifar10 --input_size 36 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001


# Weighted embedding
python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_weighted_rope_2D_axial_3x3_cifar10 --model vit_weighted_rope_2D_axial_3x3_cifar10 --input_size 36 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001 

python3 rope_vit_ext/main.py --optimizer adamw --epochs 25 --run_id vit_weighted_rope_2D_axial_4x4_cifar10 --model vit_weighted_rope_2D_axial_4x4_cifar10 --input_size 36 --cuda --distributed --world_size 4 --batch_size 64 --lr 0.001 
```

Train locally

```
python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_no_rope_cifar10 --model vit_no_rope_cifar10 --batch_size 64 --lr 0.001

python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_rope_1D_axial_2x2_cifar10 --model vit_rope_1D_axial_2x2_cifar10

python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_rope_2D_axial_2x2_cifar10 --model vit_rope_2D_axial_2x2_cifar10 --input_size 32

python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_rope_2D_axial_3x3_0_cifar10 --model vit_rope_2D_axial_3x3_0_cifar10 --input_size 36

python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_rope_2D_axial_4x4_0_cifar10 --model vit_rope_2D_axial_4x4_0_cifar10 --input_size 32

python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_rope_2D_axial_5x5_0_cifar10 --model vit_rope_2D_axial_5x5_0_cifar10 --input_size 40

python3 rope_vit_ext/main.py --optimizer adamw --epochs 5 --run_id vit_rope_2D_axial_6x6_0_cifar10 --model vit_rope_2D_axial_6x6_0_cifar10 --input_size 36
```


Notes:
```
# Run this after launching compute instance
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Run this if launching a compute instance for the first time
pip install -r requirements.txt
```