# -Enhancing-PINNsFormer-with-Efficient-Architectures-for-PDEs
This project is used to provide code support for Enhancing PINNsFormer with Efficient Architectures for PDEs


./pinnsformer-main
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
