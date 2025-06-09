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


**Activation Function:**
We replace the original wavelet-based activation with GELU, Swish, and Mish. Among them, GELU achieves the best performance, significantly reducing training loss and generalization error.


**Attention Mechanism:**
We evaluate four attention modules: standard multi-head attention, physics-aware attention, multi-scale attention, and multi-physics coupling. Standard multi-head attention proves to be the most stable and accurate for 1D PDEs.


**Positional Encoding:**
We experiment with Fourier Feature Encoding to embed spatial-temporal position information. While it enriches input representations, the original pseudo-sequence approach remains more effective for simple PDEs.


All models are trained on a 1D reaction-diffusion equation using the L-BFGS optimizer. Each component is studied via ablation to isolate its impact on model performance.

## Result

### Activition Function
![](https://github.com/MeditatorE/Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs/blob/main/activation_error.png)
**Mish and Swish (Top Row):**
Large yellow regions indicate high absolute error across the spatial-temporal domain, suggesting poor performance and failure to accurately model the solution.


**GELU (Bottom Left):**
Shows the lowest error, with small localized regions of mild error near boundaries. It achieves the best overall accuracy and generalization.


**Wavelet (Bottom Right):**
Performs moderately well, better than Mish/Swish but not as accurate as GELU. It retains useful inductive bias for periodic patterns.

### Attention
![](https://github.com/MeditatorE/Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs/blob/main/attention_error.png)
**Multi-Physics Coupling / Multi-Scale / Physics Aware Attention (Top Left, Top Right, Bottom Left):**
All show large error regions (bright yellow), especially around the central domain. These complex attention designs introduce overfitting or instability in this low-dimensional PDE task.


**Multi-head Attention (Bottom Right):**
Achieves the lowest error, with errors mostly below 0.05 and concentrated only near the boundaries. It provides the best accuracy and generalization among all variants.

### Encoding
![](https://github.com/MeditatorE/Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs/blob/main/position_encoding.png)

**Fourier Features (Left):**
Produces moderate to large errors across the domain (values >1.2 in some regions), especially near the center. Although it introduces richer frequency components, it may overfit or misalign with the PDE's structure.


**Non-Positional Encoding (Right):**
Achieves significantly lower error, mostly under 0.05 except near the boundaries. Despite lacking explicit position encoding, it performs better in this simple 1D PDE due to its implicit temporal unfolding.

## Conclusion
In this project, we enhance the PINNsFormer framework for solving time-dependent PDEs by evaluating and refining its core components. Our results show that using GELU activation and standard multi-head attention significantly improves accuracy and stability, while non-positional encoding surprisingly outperforms Fourier features in simple 1D problems. These findings suggest that simpler, well-tuned modules can be more effective than complex designs in physics-informed learning. Future work will extend these improvements to higher-dimensional PDEs, explore adaptive positional encodings, and test scalability in multi-physics or chaotic systems.

## Reference
**Paper**: [https://arxiv.org/abs/2307.11833](https://arxiv.org/abs/2307.11833)

**Source Code**: [https://github.com/AdityaLab/pinnsformer](https://github.com/AdityaLab/pinnsformer)


