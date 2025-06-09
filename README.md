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

## Quick Start
1. Clone this repo
```
git clone https://github.com/MeditatorE/Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs.git
cd Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs-main
```

2. Download **[Jupyter notebook](https://jupyter.org)** and run
```
jupyter notebook
```

3. See **[demo Folder](https://github.com/MeditatorE/Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs/tree/main/demo)** and run

### File Description
- `model`: This folder contains all the original and modified model code.

- `demo/attention`: This folder contains test code and demonstration files for testing the replacement attention mechanism.

- `demo/activition`: This folder contains test code and demonstration files for testing the replacement activation function.

- `demo/encoding`:This folder contains test code and demonstration files for testing the replacement encoding method.

## Method
To enhance the original PINNsFormer architecture for solving time-dependent PDEs, we propose improvements in three key areas:


Activation Function:
We replace the original wavelet-based activation with GELU, Swish, and Mish. Among them, GELU achieves the best performance, significantly reducing training loss and generalization error.


Attention Mechanism:
We evaluate four attention modules: standard multi-head attention, physics-aware attention, multi-scale attention, and multi-physics coupling. Standard multi-head attention proves to be the most stable and accurate for 1D PDEs.


Positional Encoding:
We experiment with Fourier Feature Encoding to embed spatial-temporal position information. While it enriches input representations, the original pseudo-sequence approach remains more effective for simple PDEs.


All models are trained on a 1D reaction-diffusion equation using the L-BFGS optimizer. Each component is studied via ablation to isolate its impact on model performance.

