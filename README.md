# TemporalMoE-ViT: A Heterogeneous Mixture-of-Experts Architecture for Unified Video Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX) <!-- TODO: Replace with your actual arXiv link -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official PyTorch implementation for the paper: **"TemporalMoE-ViT: A Heterogeneous Mixture-of-Experts Architecture for Unified Video Reasoning"**.

Our work introduces **TemporalMoE-ViT**, a unified architecture that embeds a Heterogeneous Mixture-of-Experts (MoE) directly into the vision skeleton of a Transformer. Unlike conventional MoE systems with identical, "blank-slate" experts, TemporalMoE-ViT leverages a committee of architecturally specialized computational pathways. This includes a `MotionExpert` fused with optical flow, a `TextureExpert` with a convolutional stem, and a `QA-AlignedExpert` explicitly conditioned on the textual query. A dynamic gating network learns to route spatio-temporal tokens to these specialists, enabling the model to allocate computation based on content, leading to significant gains in both efficiency and performance on video reasoning tasks.

## Architecture Overview

Our proposed architecture integrates a heterogeneous Mixture-of-Experts block directly into a Vision Transformer backbone, allowing for dynamic, content-aware computation.

<!-- TODO: Insert your architecture diagram image here -->
<!-- Example: ![Architecture Diagram](assets/architecture_diagram.png) -->
<p align="center">
  <img src="path/to/your/diagram.png" width="800">
</p>

**Figure 1: High-level overview of the TemporalMoE-ViT framework.** The model takes joint video and text input, passes it through a stack of our novel `TemporalMoEBlock` layers, and produces a final prediction. The key novelty is the routing of tokens to a diverse pool of architecturally specialized experts.

## Key Contributions

-   **A Unified Heterogeneous MoE Transformer:** We present a novel framework that integrates diverse, specialized expert modules directly into the spatio-temporal encoder for token-wise, content-adaptive computation.
-   **Specialized Expert and Router Design:** We develop a lightweight gating network for top-K expert selection and a unique committee of experts, each with a strong inductive bias for a specific visual phenomenon (e.g., motion, texture, query relevance).
-   **Efficient & Scalable Training:** We leverage a training strategy with auxiliary load-balancing losses to ensure stable convergence and efficient utilization of all experts.
-   **State-of-the-Art Performance:** We demonstrate that TemporalMoE-ViT can outperform a standard dense Vision Transformer baseline in both accuracy and computational efficiency on video-language benchmarks.

