# GPT-2 RLHF Pipeline: SFT → Reward Model → GRPO

A complete implementation of Reinforcement Learning from Human Feedback (RLHF) on GPT-2, spanning Supervised Fine-Tuning, Reward Modeling, and Group Relative Policy Optimization.

---

## Pipeline Overview

```
GPT-2 (Pretrained)
        |
        v
[Stage 1]  Supervised Fine-Tuning       ->  Language-aligned base model
        |
        v
[Stage 2]  Reward Model Training        ->  Scalar preference scorer
        |
        v
[Stage 3]  GRPO (RL Fine-Tuning)        ->  Human-aligned policy
```

---

## Repository Structure

```
.
├── SUPERVISED FINE TUNING.ipynb    # Stage 1: SFT on Reddit corpus
├── REWARD_MODEL.ipynb              # Stage 2: Bradley-Terry reward model
├── GRPO.ipynb                      # Stage 3: RL alignment via GRPO
└── README.md
```

---

## Qualitative Results: Before vs After GRPO

The following outputs demonstrate the effect of the full SFT → Reward Model → GRPO pipeline on GPT-2's generation quality. The model before mid-training produces verbose, incoherent, and structurally inconsistent text with no preference alignment. After GRPO fine-tuning, the policy generates responses that are contextually grounded, concise, and aligned with human-preferred outputs — a direct consequence of the reward signal shaping the policy through group-relative advantage estimation.

### Before GRPO Fine-Tuning

![Before GRPO Fine-Tuning](https://raw.githubusercontent.com/ajheshbasnet/model-reincarnated/main/Group%20Relative%20Policy%20Optimization%20(GRPO)/before-mid-training.png)

### After GRPO Fine-Tuning

![After GRPO Fine-Tuning](https://raw.githubusercontent.com/ajheshbasnet/model-reincarnated/main/Group%20Relative%20Policy%20Optimization%20(GRPO)/AFTER-MID-TRAINING.png)

---

## Stage 1 — Supervised Fine-Tuning (SFT)

### Objective

Train GPT-2 to model the distribution of human-written Reddit responses and learn to terminate generation correctly with the `<EOS>` token.

### Training Formulation

The model is trained on concatenated `[prompt + chosen_response]` sequences under a standard causal language modeling objective:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

where $x = [\text{prompt} \oplus \text{chosen\_response}]$ is the full token sequence. No response-masking is applied — the model learns from both prompt and response tokens jointly, which encourages it to internalize how a well-formed response follows its prompt contextually.

### Why Train on the Full Sequence

Masking only response tokens is common in instruction-tuning, but training on the full sequence forces the model to learn contextual coherence — the stylistic and semantic relationship between a prompt and its response. For Reddit-style data where prompts set strong distributional priors, this joint training produces a more coherent SFT checkpoint.

### EOS Supervision

Explicit `<EOS>` prediction is trained here and is critical for downstream RL stability. A policy that does not know when to stop produces runaway sequences during GRPO rollouts, inflating sequence length and destabilizing reward computation.

### Configuration

| Parameter | Value |
|---|---|
| Base model | `gpt2` (HuggingFace, 117M) |
| Training data | Reddit `(prompt, chosen_response)` pairs |
| Objective | Causal LM on full sequence |
| Output | `sft_model` checkpoint |

---

## Stage 2 — Reward Model

### Objective

Learn a scalar-valued function $r_\phi(x, y) \in \mathbb{R}$ that scores a response $y$ given prompt $x$, reflecting human preference without requiring absolute annotations.

### Architecture

The reward model reuses the transformer backbone from the SFT checkpoint and appends a linear projection head:

```
SFT Transformer Blocks
        |
        v   hidden state h in R^768  (last token position)
  Linear(768 -> 1)
        |
        v   scalar reward  r in R
```

The reward is read from the last token's hidden state. Since GPT-2 is a causal (left-to-right) model, the final token position attends over the full sequence context — making it the natural extraction point for a sequence-level signal.

### Bradley-Terry Preference Loss

Given a prompt $x$ with a chosen response $y_w$ and a rejected response $y_l$, the model is trained with the Bradley-Terry pairwise ranking objective:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,\, y_w,\, y_l)\,\sim\,\mathcal{D}} \left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

The model is penalized when the rejected response scores higher than the chosen one. The sigmoid converts the score margin into a probability, and maximizing the log-likelihood is equivalent to maximizing the probability that the chosen response wins. No absolute reward labels are required — only relative pairwise comparisons.

### Differential Learning Rates

| Component | Learning Rate | Rationale |
|---|---|---|
| Reward head (Linear 768→1) | `1e-4` | Random initialization; needs fast adaptation |
| Transformer blocks | `2e-5` | Pre-trained weights; slow drift preserves representations |

This is layer-wise learning rate decay in practice. A higher rate on the reward head lets it converge quickly without forcing the backbone — which already encodes rich language structure — to drift far from the SFT distribution.

---

## Stage 3 — Group Relative Policy Optimization (GRPO)

### What is GRPO

GRPO is a PPO-family algorithm designed for LLMs that eliminates the need for a separate critic/value network. Instead of learning a value function, it estimates the baseline by sampling a group of responses per prompt and computing relative advantages in-place — making it significantly more memory-efficient than standard PPO while remaining theoretically grounded.

### Group Sampling and Advantage Estimation

For each prompt $x$, sample $G$ responses from the current policy:

$$\{y_1, y_2, \ldots, y_G\} \sim \pi_\theta(\cdot \mid x)$$

Score each with the reward model:

$$r_i = r_\phi(x, y_i), \quad i = 1, \ldots, G$$

Compute the group-normalized advantage:

$$\hat{A}_i = \frac{r_i - \text{mean}\!\left(\{r_j\}_{j=1}^{G}\right)}{\text{std}\!\left(\{r_j\}_{j=1}^{G}\right)}$$

This replaces the critic entirely. Each response is evaluated relative to its siblings sampled from the same prompt. The group mean is an unbiased estimate of the state value $V(x)$ under the current policy — GRPO exploits this statistical fact to avoid training a value network altogether.

### GRPO Objective

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\!\left(\rho_i\,\hat{A}_i,\ \text{clip}(\rho_i,\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_i\right) - \beta\cdot\mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right]\right]$$

where the importance sampling ratio is:

$$\rho_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$$

### Term-by-Term Breakdown

**Clipped Surrogate**

$$\min\!\left(\rho_i\,\hat{A}_i,\ \text{clip}(\rho_i,\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_i\right)$$

The importance ratio $\rho_i$ corrects for the distributional shift between the old policy (used for rollouts) and the updated policy (used for gradient computation). The clip prevents the new policy from drifting too far in a single update — if $\rho_i$ exits $[1-\varepsilon,\;1+\varepsilon]$, the gradient contribution is truncated. This is the mechanism that enforces monotonic improvement and prevents policy collapse.

**KL Divergence Penalty**

$$-\beta\cdot\mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] = -\beta\sum_t \pi_\theta(y_t \mid x, y_{<t})\log\frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$

Without this term, the policy reward-hacks — finding degenerate outputs that score highly on the reward model but are linguistically incoherent. The KL penalty keeps the RL policy anchored to the SFT reference model $\pi_{\text{ref}}$, and the coefficient $\beta$ controls the trade-off between reward maximization and language quality preservation. The SFT checkpoint serves here as both the reward model backbone and the RL reference policy.

### GRPO vs PPO

| Property | PPO | GRPO |
|---|---|---|
| Critic network | Required | Not needed |
| Memory overhead | High | Low |
| Baseline estimation | Learned value function | Group mean reward |
| Designed for LLMs | No | Yes |

---

## Full Pipeline Summary

| Stage | Input | Output | Loss |
|---|---|---|---|
| SFT | `prompt + chosen` | Token logits | Cross-entropy |
| Reward Model | `prompt + chosen / rejected` | Scalar $r \in \mathbb{R}$ | Bradley-Terry |
| GRPO | `prompt` | Aligned policy $\pi_\theta$ | Clipped surrogate + KL |

---

## References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. https://arxiv.org/abs/1707.06347
- Stiennon et al., *Learning to summarize from human feedback*, OpenAI 2020. https://arxiv.org/abs/2009.01325
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024. https://arxiv.org/abs/2402.03300
- Bradley & Terry, *Rank Analysis of Incomplete Block Designs*, Biometrika 1952
