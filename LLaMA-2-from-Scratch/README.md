# 🦙 LLaMA-2 From Scratch

This repository contains a clean, educational implementation of **LLaMA-2 architecture from scratch**, focusing on modern transformer improvements.  
You can run the full code on **Google Colab** and modify the dataset as you wish.  

👉 [Open in Google Colab](https://colab.research.google.com/drive/1zcszVTUOSWXVYtoV9MTjJOfV3fVxjSWV?usp=sharing)

---

## 🚀 Features Implemented

This notebook implements LLaMA-style transformers with the following enhancements:

### 🔹 1. Grouped Query Attention (GQA)
- Traditional transformers use **multi-head attention**, where each head has its own queries, keys, and values.  
- **GQA** reduces computation and memory usage by grouping **multiple query heads to share the same key & value heads**.  
- ✅ Benefits:
  - Speeds up training & inference.  
  - Reduces memory cost without hurting model quality.  
  - Already used in LLaMA-2 and other cutting-edge LLMs.  

---

### 🔹 2. Rotary Position Embeddings (RoPE)
- Instead of adding learned or sinusoidal positional embeddings to token embeddings, **RoPE** encodes positional information **directly in the attention mechanism**.  
- It applies a rotation in complex space to queries and keys, which naturally encodes relative positions.  
- ✅ Benefits:
  - Handles **longer sequences** better.  
  - More generalizable than vanilla positional embedding addition.  

---

### 🔹 3. Key-Value (KV) Cache
- During autoregressive generation, recomputing attention over the entire sequence is inefficient.  
- With **KV caching**, past keys and values are stored and reused instead of being recomputed.  
- ✅ Benefits:
  - Greatly improves **inference speed**.  
  - Reduces redundant computations for long sequences.  

---

### 🔹 4. SwiGLU Activation
- Standard transformers often use **ReLU** or **GELU** in feed-forward networks.  
- LLaMA uses **SwiGLU**, a gated linear unit with a swish nonlinearity:  


- ✅ Benefits:
  - Improves model expressiveness.  
  - Empirically shown to perform better than GELU/ReLU in large models.  

---

## 📂 How to Use

1. Clone this repo or just open the notebook in Colab:  
   👉 [Run on Google Colab](https://colab.research.google.com/drive/1zcszVTUOSWXVYtoV9MTjJOfV3fVxjSWV?usp=sharing)

2. Replace the dataset with your own:  
   - The notebook is modular.  
   - Simply update the data loading cell with your custom dataset.  

3. Train and experiment with:  
   - Different datasets  
   - Model hyperparameters (layers, heads, hidden size)  
   - Sampling strategies during text generation  

---

## 📖 Learning Goals

This project is not just code — it’s designed to **teach modern transformer design choices**.  
You’ll gain hands-on understanding of:  
- Why **Grouped Query Attention** is faster and efficient.  
- Why **RoPE** is used instead of vanilla positional embeddings.  
- How **KV Cache** accelerates inference.  
- Why **SwiGLU** improves over GELU/ReLU in practice.  

---

## 🤝 Contributing

Feel free to fork the repo, play with the architecture, and submit PRs with improvements or experiments.  
Sharing benchmarks with different datasets would be especially valuable!  

---

## 📜 License

This project is for **educational and research purposes only**.  
Check the original [LLaMA-2 license](https://ai.meta.com/llama/) for commercial usage restrictions.  

---

⭐ If you find this useful, don’t forget to **star the repo**!
