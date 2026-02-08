# RNNs to Transformers: From Recurrence to Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**A pedagogical deep dive into the evolution of sequence models — from vanilla RNNs, through LSTMs, to the Transformer architecture.**

This repository contains a self-contained tutorial (with math, NumPy code, and explanations) that walks through why RNNs were revolutionary, why they struggle with long dependencies (vanishing gradients, fixed hidden-state bottleneck), how LSTMs helped, and why attention + self-attention (Transformers) ultimately took over in modern NLP and sequence modeling.

**Status:** Work in progress — core RNN + BPTT is solid; Transformer/attention sections are being expanded.

## 📖 What You'll Find Here

- Detailed mathematical derivations
  - RNN forward pass & recurrence
  - Backpropagation Through Time (BPTT) with recursive gradients
  - Vanishing & exploding gradient problems
- Clean NumPy implementation of a character-level RNN (forward pass + loss)
- Conceptual explanations (hidden-state bottleneck, teacher forcing vs autoregressive generation, etc.)
- Upcoming: LSTM gates & cell state, full self-attention (QKV, scaled dot-product), positional encodings, parallelism advantages, and RNN vs Transformer comparison table

Perfect for:
- ML students/intermediates wanting to understand **why** Transformers replaced RNNs
- People implementing sequence models from scratch
- Preparing for interviews (RNN limitations, attention mechanism)

## 🚀 Quick Start

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rnns-to-transformers.git
   cd rnns-to-transformers
