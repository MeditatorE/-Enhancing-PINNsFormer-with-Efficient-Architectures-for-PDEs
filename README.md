# Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs
This project is used to provide code support for Enhancing PINNsFormer with Efficient Architectures for PDEs

## Introduction
This project focuses on enhancing PINNsFormer, a Transformer-based architecture designed for solving time-dependent partial differential equations (PDEs) using physics-informed neural networks (PINNs). Building on the original implementation, we conduct ablation studies on three key components—attention mechanisms, activation functions, and positional encodings. Our results demonstrate that replacing the default activation with GELU significantly improves performance, while standard multi-head attention remains the most stable for low-dimensional PDEs. These findings contribute to the development of efficient and accurate deep learning models for physical systems.

### Folder Structure
```
./main
├── checkpoint
│   ├── 1dreaction_pinnsformer.pt
│   ├── README.md
│   ├── convection_pinnsformer.pt
│   └── environment.yml
├── demo
│   ├── 1d_reaction_pinnsformer.ipynb
│   ├── activation
│   │   ├── GeLu_1d_reaction_pinnsformer.ipynb
│   │   ├── Mish_1d_reaction_pinnsformer.ipynb
│   │   └── Swish_1d_reaction_pinnsformer.ipynb
│   ├── attention
│   │   ├── MultiPhysicsCouplingAttention_1d_reaction_pinnsformer.ipynb
│   │   ├── MultiScaleAttention_1d_reaction_pinnsformer.ipynb
│   │   ├── PhysicsAwareAttention_1d_reaction_pinnsformer.ipynb
│   │   └── convection
│   │       ├── MultiScaleAttention
│   │       │   └── convection_pinnsformer_MultiScale.ipynb
│   │       ├── PhysicsAwareAttention
│   │       │   └── convection_pinnsformer_PhysicsAwareAttention.ipynb
│   │       └── default
│   │           └── convection_pinnsformer.ipynb
│   └── encoding
│       └── Fourier_1d_reaction_pinnsformer.ipynb
├── model
│   ├── att.py
│   ├── pinn.py
│   ├── pinnsformer.py
│   ├── pinnsformer_Fourier_.py
│   ├── pinnsformer_MultiPhysicsCouplingAttention.py
│   ├── pinnsformer_MultiScaleAttention.py
│   ├── pinnsformer_PhysicsAwareAttention.py
│   ├── pinnsformer_gelu.py
│   ├── pinnsformer_mish.py
│   └── pinnsformer_swish.py
├── pyhessian.py
├── util.py
└── vis_landscape.py
```
