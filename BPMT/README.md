# 🌊 CascadeFormer: Two-stage Cascading Transformer for Human Action Recognition

## Paper (under review at AAAI 2026 Main Technical Track)

- [main paper](papers/CascadeFormer__main_paper_.pdf)
- [supplementary material](papers/CascadeFormer__supplementary_material_.pdf)

## Evaluation Results

We open source the following model checkpoints on HuggingFace: [YusenPeng/CascadeFormerCheckpoints](https://huggingface.co/YusenPeng/CascadeFormerCheckpoints)

| dataset | #videos | #joints | CF 1.0 | CF 1.1 | CF 1.2 |
| ------- | ------- | ---------- | ------ | ------- | ------ |
| Penn Action | 2,326 | 13, 2D | **94.66%** | **94.10%** | **94.10%** |
| N-UCLA | 1,494 | 20, 3D | **89.66%** | **91.16%** | **90.73%** |
| NTU/CS | 56,880 | 25, 3D | **81.01%** | **79.62%** | **80.48%** |
| NTU/CV | 56,880 | 25, 3D | **88.17%** | **86.86%** | **87.24%** |


## Environment Setup

```bash
conda env create -f environment.yml
conda activate CascadeFormer
```

## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Alper Yilmaz (yilmaz.15@osu.edu)

Or describe it in Issues.