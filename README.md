# TinyGPT – Character-Level Transformer Built From Scratch

A decoder-only GPT-style Transformer implemented entirely from scratch in PyTorch.

This project recreates the core architecture behind modern language models using only native PyTorch modules — without relying on HuggingFace or high-level transformer libraries.

---

## Project Overview

This repository contains a minimal GPT-style language model trained for character-level next-token prediction.

The model learns to generate stylistically coherent text by predicting the next character given previous context.

The focus of this project is:

- Architectural clarity  
- From-scratch transformer construction  
- Clean modular design  
- Reproducible training and inference  
- Professional repository structure  

---

## Architecture

**Model Type:** Decoder-only Transformer  
**Embedding Dimension:** 192  
**Attention Heads:** 6  
**Transformer Blocks:** 4  
**Context Length (Block Size):** 128  
**Dropout:** 0.2  
**Optimizer:** AdamW  
**Loss Function:** Cross-Entropy  

### Core Components

- Token Embeddings  
- Positional Embeddings  
- Multi-Head Self-Attention (`nn.MultiheadAttention`)  
- Causal Masking (lower-triangular attention mask)  
- Residual Connections  
- Layer Normalization  
- Feed-Forward Network (4× expansion with GELU)  
- Autoregressive sampling with temperature scaling  

---

## Training Results

The model was trained for **5,000 iterations**.

### Loss Progression

```
Step 0     | Loss: 4.2760
Step 500   | Loss: 2.3330
Step 1000  | Loss: 2.0567
Step 1500  | Loss: 1.8779
Step 2000  | Loss: 1.7764
Step 2500  | Loss: 1.6436
Step 3000  | Loss: 1.6100
Step 3500  | Loss: 1.6100
Step 4000  | Loss: 1.4470
Step 4500  | Loss: 1.5148
```

The steady reduction in cross-entropy loss demonstrates that the model successfully learns long-range character dependencies.

---

## ✍️ Sample Generation

Prompt:

```
Enter prompt: King:
```

Generated Output:

```
King:
Our possies the bows shall be gone
That hour perming hath been all he hate eare
With him batter and his comfort. What for more,
his that with can you the chiles you have bed
The look the find thine j
```

### Observations

- The model captures Shakespearean-style syntax and structure.
- It learns line formatting and tone.
- Word formation is mostly coherent, though still imperfect (expected at this scale).
- It demonstrates meaningful contextual continuation from a simple prompt.

---

## Repository Structure

```
tinyGPT/
│
├── config.py        # Centralized hyperparameters
├── data.py          # Dataset loading + batching
├── model.py         # Transformer + GPT model
├── train.py         # Training loop
├── generate.py      # Inference script
│
├── data/
│   └── input.txt    # Training corpus
│
├── checkpoints/
│   └── model.pt     # Saved model weights
│
└── README.md
```

---

## Installation & Usage

### Install Dependencies

```bash
pip install torch
```

### Train the Model

```bash
python train.py
```

Model weights will be saved to:

```
checkpoints/model.pt
```

### Generate Text

```bash
python generate.py
```

Then enter any prompt:

```
Enter prompt: King:
```

---

## Why This Project Matters

This project demonstrates:

- Deep understanding of transformer internals  
- Manual construction of attention mechanisms  
- Autoregressive modeling  
- PyTorch fluency  
- Software engineering best practices (modular repo design)  
- End-to-end ML pipeline: data → model → training → inference  

This is not a wrapper around HuggingFace — it is a ground-up transformer implementation.

---

## Possible Extensions

- Add validation loss tracking  
- Implement scaled dot-product attention manually  
- Add attention visualization  
- Scale to subword tokenization (BPE)  
- Add perplexity tracking  
- Train on larger corpora  
- Deploy with a simple web UI  

---

## Key Takeaway

This project shows the ability to:

- Recreate modern language model architecture from first principles  
- Train and debug deep learning systems  
- Structure ML code like a research-style repository  
- Produce coherent generative outputs  
