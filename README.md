# fMRI2TextAligner

**Aligning brain activity (fMRI) with natural language representations using contrastive learning**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python" alt="Python version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License: MIT">
</p>

## Overview

fMRI2TextAligner is a lightweight framework for learning joint representations between **functional Magnetic Resonance Imaging (fMRI)** brain activity and **natural language text** embeddings.

The core idea is to train a **contrastive alignment model** that maps fMRI voxel patterns (typically from naturalistic story-listening or movie-watching paradigms) into the same semantic space as modern large language model hidden states or sentence transformers.

This enables downstream applications such as:

- Zero-shot / few-shot **brain-to-text decoding**
- Semantic reconstruction from brain signals
- Neuro-semantic decoding & encoding analysis
- Brain-language interpretability research

## Key Features

- Simple & modular contrastive training pipeline
- Support for popular fMRI preprocessing formats (e.g., `.npy`, `.nii.gz`, NSD / Harry Potter / Friends datasets)
- Compatible with frozen text encoders: Sentence-BERT, E5, GTE, OpenAI text-embedding-*, Llama-2/3 hidden states, etc.
- Multiple loss variants: InfoNCE, SigLIP-style, etc.
- Evaluation metrics: retrieval accuracy, semantic correlation, BLEU/ROUGE/BERTScore (when paired with a frozen LLM decoder)

## Installation

```bash
# clone repository
git clone https://github.com/CaiZongyuan/fMRI2TextAligner.git
cd fMRI2TextAligner

# recommended: use conda / uv / pixi
conda create -n fmri2text python=3.10
conda activate fmri2text

pip install -r requirements.txt
# or
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA
pip install -e .
