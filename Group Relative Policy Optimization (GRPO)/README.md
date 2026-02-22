
# 🧠 GPT-2 RLHF Pipeline: SFT → Reward Model → GRPO

> A complete, research-grade implementation of Reinforcement Learning from Human Feedback (RLHF) on GPT-2, covering Supervised Fine-Tuning, Reward Modeling, and Group Relative Policy Optimization.

---

## 📌 Overview

This repository implements the full RLHF alignment pipeline on **GPT-2** (117M parameters) using Reddit post data as the human preference corpus. The pipeline proceeds in three distinct stages:

```
GPT-2 (Pretrained)
      │
      ▼
[Stage 1] Supervised Fine-Tuning (SFT)
      │  → Language-aligned base model
      ▼
[Stage 2] Reward Model Training
      │  → Scalar preference scorer
      ▼
[Stage 3] GRPO (RL Fine-Tuning)
      │  → Human-aligned policy
      ▼
GPT-2 (RLHF Aligned)
```

---

## 🗂️ Repository Structure

```
├── SUPERVISED FINE TUNING.ipynb   # Stage 1: SFT on Reddit corpus
├── REWARD_MODEL.ipynb             # Stage 2: Bradley-Terry reward model
├── GRPO.ipynb                     # Stage 3: RL alignment via GRPO
└── README.md
```

---

## Stage 1 — Supervised Fine-Tuning (SFT)

### Objective

Train GPT-2 to adopt the **distribution of human-written Reddit responses** and learn to properly terminate generation with the `<EOS>` token.

### Training Scheme

The model is trained on concatenated `[prompt + chosen_response]` sequences in a standard causal language modeling setup:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

where $x = [\text{prompt} \oplus \text{chosen\_response}]$ is the full sequence. No masking is applied — the model learns from both the prompt tokens and the response tokens jointly.

### Why Train on the Full Sequence?

Masking only response tokens is more common in instruction-tuning, but training on the full sequence forces the model to learn **contextual coherence** — how a good response relates to its prompt. This matters for Reddit-style data where the prompt sets strong stylistic priors.

### Key Details

- **Base Model:** `gpt2` (HuggingFace)
- **Data:** Reddit `(prompt, chosen_response)` pairs
- **EOS Learning:** The model explicitly learns to emit `<EOS>` at the correct position, which is critical for downstream RL stability — a model that doesn't know when to stop will produce runaway sequences during GRPO rollouts.
- **Output:** `sft_model` — a language model checkpoint used to initialize both the reward model and the RL policy.

---

## Stage 2 — Reward Model

### Objective

Train a scalar-valued function $r_\phi(x, y) \in \mathbb{R}$ that scores a response $y$ given a prompt $x$, reflecting human preference.

### Architecture

The reward model reuses the **transformer backbone from the SFT model** and appends a linear reward head:

```
SFT Transformer Blocks (frozen at low lr)
          │
          ▼  hidden_state ∈ ℝ^768  (last token)
    Linear(768 → 1)
          │
          ▼  scalar reward r ∈ ℝ
```

The reward is extracted from the **last token's hidden state**, since GPT-2 is a causal model and the final token aggregates full-sequence context.

### Bradley-Terry Preference Loss

Given a prompt $x$ with a chosen response $y_w$ and a rejected response $y_l$, the model is trained with the **Bradley-Terry ranking objective**:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]$$

Intuitively: the model is penalized when it assigns a higher score to the rejected response. The sigmoid squashes the margin into a probability, and we maximize the log-likelihood that the chosen response wins.

### Differential Learning Rates

Two parameter groups are used deliberately:

| Component | Learning Rate | Rationale |
|---|---|---|
| Reward Head (Linear 768→1) | `1e-4` | Random init, needs fast convergence |
| Transformer Blocks | `2e-5` | Pre-trained weights; slow drift preserves representations |

This is a form of **layer-wise learning rate decay** — applying a much higher LR to the head prevents the problem where the head stays random while the backbone unnecessarily destabilizes.

### Training Data

Pairs of `(prompt, chosen)` and `(prompt, rejected)` are used. The reward model learns the **relative** notion of quality, not an absolute one — this is key. It never sees an absolute score; it only learns "this response is better than that one."

---

## Stage 3 — GRPO (Group Relative Policy Optimization)

### What is GRPO?

GRPO is a **PPO-family RL algorithm** designed for LLMs that eliminates the need for a separate value/critic network. Instead, it estimates the **baseline reward** by sampling a *group* of responses for each prompt and computing their relative advantage in-place.

This makes GRPO significantly more memory-efficient than standard PPO while remaining theoretically grounded.

### The Core Idea: Group Sampling

For each prompt $x$, sample $G$ responses from the current policy:

$$\{y_1, y_2, \ldots, y_G\} \sim \pi_\theta(\cdot \mid x)$$

Score each with the reward model:

$$r_i = r_\phi(x, y_i), \quad i = 1, \ldots, G$$

Compute the **group-normalized advantage** (this replaces the critic):

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

This tells each response: *"Were you better or worse than your siblings sampled from the same prompt?"* No separate value network is needed.

### GRPO Objective

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\ \{y_i\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i \right) - \beta \cdot \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]$$

where:

$$\rho_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$$

is the **importance sampling ratio** between the current and old policy.

### Breaking Down Each Term

**① Clipped Surrogate (PPO core)**

$$\min\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right)$$

The clip prevents the policy from making excessively large updates in a single step. If $\rho_i$ drifts far from 1 (i.e., the new policy diverges too much from the old), the gradient is cut off. This ensures **monotonic improvement** without catastrophic policy collapse.

**② KL Penalty**

$$-\beta \cdot \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right]$$

The KL divergence from the **SFT reference model** $\pi_{\text{ref}}$ acts as a regularizer. Without it, the RL policy would **reward-hack** — finding degenerate outputs that score highly on the reward model but are gibberish as language. The $\beta$ coefficient controls this trade-off between alignment and linguistic coherence.

$$\mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \sum_t \pi_\theta(y_t \mid x, y_{<t}) \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$

### Why GRPO Over PPO?

| Property | PPO | GRPO |
|---|---|---|
| Critic network | ✅ Required | ❌ Not needed |
| Memory overhead | High | Low |
| Variance reduction | Via value baseline | Via group mean baseline |
| Suitable for LLMs | Marginal | ✅ Yes |

The group mean $\mathbb{E}[r_i]$ is an unbiased estimate of the value function $V(x)$ under the current policy when $G$ is large enough — GRPO exploits this statistical shortcut.

---

## Pipeline Summary

```
Stage       Input                    Output              Loss
─────────────────────────────────────────────────────────────────────
SFT         prompt + chosen          Next token logits   Cross-Entropy
Reward      prompt + chosen/rejected Scalar r ∈ ℝ       Bradley-Terry
GRPO        prompt                   Aligned policy π_θ  Clipped PPO + KL
```

---

## Key Design Decisions

- **No critic in GRPO:** Group sampling provides a low-variance baseline without a value network, keeping the implementation lean on GPT-2's modest scale.
- **Differential LR in RM:** Protects pre-trained representations while allowing the reward head to converge quickly.
- **EOS supervision in SFT:** Ensures clean generation termination during RL rollouts — a subtle but important detail that prevents infinite-loop degeneration.
- **KL to SFT reference:** The SFT checkpoint serves dual purpose — reward model backbone *and* RL reference policy — a common and elegant reuse strategy.

---

## References

- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017
- Stiennon et al., [*Learning to summarize from human feedback*](https://arxiv.org/abs/2009.01325), OpenAI 2020
- Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300), 2024 — GRPO origin
- Bradley & Terry, *Rank Analysis of Incomplete Block Designs*, 1952
