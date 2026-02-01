## RNNS RNNs to Transformers: A Technical Evolution Survey
A comprehensive, mathematically rigorous survey and implementation guide exploring the transition from sequential recurrence to parallelized self-attention.

## Objective
To provide a deep-dive architectural reference for sequence modeling, bridging the gap between foundational Recurrent Neural Networks (RNNs) and modern Large Language Models (LLMs). This project serves as both a theoretical framework and a practical implementation guide for engineers and researchers.

## Technical Highlights
 1. **The Sequential Challenge & FoundationsVanishing Gradient Analysis:** A rigorous mathematical look at the Jacobian of hidden-state transitions and why standard BPTT fails over long sequences.
    
 2. **State Management:** Understanding the necessity of internal "memory" in feedforward-resistant temporal data.
  
 3 **Advanced Recurrent ArchitecturesLSTM**
      **(Long Short-Term Memory):** Detailed breakdown of the "Constant Error Carousel" and additive gating mechanisms.
       **GRU (Gated Recurrent Units):** Analysis of parameter efficiency through Reset and Update gates.
   ## Tech Stack
**Deep Learning Framework:** PyTorch (CharRNN and CharLSTM implementations).

**Documentation Engine:** Docusaurus 3.0.

**Math Rendering:** KaTeX / LaTeX.

**Visualizations:** Mermaid.js.
