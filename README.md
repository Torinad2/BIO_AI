# BIO_AI — Learning AI for Biology & Medicine

A hands-on learning project: applying modern AI (protein language models, deep learning) to biology and medicine, built step by step on a local GPU workstation.

## What this is

I am studying how protein language models work in practice — starting from environment setup and a first ESM-2 experiment, moving toward a small but honest ML project (an enzyme classifier trained on real Swiss-Prot data with leakage-aware data splitting).

## Stack

- Python 3.11 (Miniconda env `bioai`)
- PyTorch 2.6 (CUDA 12.4) on NVIDIA RTX 3090 (24 GB)
- Hugging Face Transformers (ESM-2)
- pandas, scikit-learn

## Progress

- **Phase 0 — Setup ✅** Miniconda env, GPU verified (`check_env.py`: torch, CUDA, RTX 3090, smoke test).
- **Phase 1 — First ESM-2 run ✅** `esm2_first_steps.py` downloads human and mouse p53 from UniProt, computes ESM-2 (35M) embeddings — cosine similarity **0.992** — and tries masked-amino-acid prediction at position 50.
- **Phase 2 — Foundations** Rosalind problems + Hugging Face course (in progress).
- **Phase 3 — ML basics** train/test splitting, overfitting, metrics on p53 embeddings.
- **Phase 4 — Project** Enzyme classifier: Swiss-Prot EC classes → MMseqs2 cluster split → ESM-2 650M embeddings → logistic regression vs. small MLP, with honest evaluation.

## Files

- `check_env.py` — verifies PyTorch/CUDA/GPU
- `esm2_first_steps.py` — first ESM-2 experiment (embeddings + masked prediction)
- `esm2-enzyme-classifier-plan.md` — plan for the Phase 4 project

## Run it

conda create -n bioai python=3.11 -y
conda activate bioai
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas scikit-learn requests
python check_env.py
python esm2_first_steps.py
