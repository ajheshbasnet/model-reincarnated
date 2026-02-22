# GPT-OSS (Mini) — A Reincarnation

> A lightweight reimplementation of OpenAI's GPT-OSS architecture, built from scratch in PyTorch with modular components for learning and experimentation.  
> credit: Dataset is inspired by **Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)** (Tiny Shakespeare dataset).

---

## Features

This project reimagines the GPT-OSS style architecture with modern improvements:

- **Mixture of Experts (MoE)**  
  - Implements `n_experts` experts with top-k routing.  
  - Supports `n_shared_experts` for shared computation.  

- **Grouped Query Attention (GQA)**  
  - More memory-efficient than vanilla multi-head attention.  
  - Supports `n_groups` grouping for QKV projections.  

- **Rotary Position Embeddings (RoPE)**  
  - Replaces vanilla sinusoidal positional encoding.  
  - Enables extrapolation to longer contexts.  

- **SwiGLU Activation**  
  - Uses SiLU (Swish) instead of ReLU.  
  - Gated feedforward design for richer representations.  

- **RMSNorm Normalization**  
  - Lightweight and numerically stable compared to LayerNorm.  

- **Configurable Architecture**  
  - Model dimensions, experts, groups, and context length are all tweakable via a simple `config` class.

---

## Current Configuration

```python
class config:
    d_model: int = 256
    hidden_states: int = 1024
    n_experts: int = 6
    top_k_experts: int = 3
    n_shared_experts: int = 2
    n_heads: int = 8
    n_decoder_block: int = 8
    n_groups: int = 4
    vocab_size: int = 65
    max_len: int = 180
    max_input_length: int = 500

config = config()
