# BIO_AI — Learning AI for Biology & Medicine

A hands-on learning project: applying modern AI (protein language models, deep learning) to biology and medicine, built step by step on a local GPU workstation.

## What this is

I am studying how protein language models work in practice — starting from environment setup and a first ESM-2 experiment, moving toward a small but honest ML project (an enzyme classifier trained on real Swiss-Prot data with leakage-aware data splitting).

Code comments and docstrings are bilingual (`#RU:` / `#EN:` in code, `RU:` / `EN:` in docstrings).

## Stack

- Python 3.11 (Miniconda env `bioai`)
- PyTorch 2.6 (CUDA 12.4) on NVIDIA RTX 3090 (24 GB)
- Hugging Face Transformers (ESM-2)
- pandas, scikit-learn
- Biopython, matplotlib (sequence alignment, plots, optional cross-checks)

## Progress

- **Phase 0 — Setup ✅** Miniconda env, GPU verified (`check_env.py`: torch, CUDA, RTX 3090, smoke test).
- **Phase 1 — First ESM-2 runs ✅**
  - **v1** (`esm2_first_steps_v1.py`): downloads human and mouse p53 from UniProt and computes ESM-2 (35M) embeddings. Cosine similarity human vs mouse p53 is **0.992**, versus **0.942** for an unrelated control (human hemoglobin beta) — a reminder that mean-pooled ESM embeddings are high even for unrelated proteins, so only the gap is meaningful. Masked-amino-acid prediction at position 50 (0-based): the true residue (E) was not in the top-5 and the distribution was flat (best guess L, p=0.247), which is plausible for a disordered N-terminal region.
  - **v2** (`esm2_first_steps_v2.py`): extends v1 with (1) a comparison of the N-terminus vs the DNA-binding domain, (2) masked marginals over every position with a confidence plot (`p53_confidence.png`), and (3) ESM-style scoring of real human→mouse substitutions against random substitutions, using a Biopython alignment.
- **Phase 2 — Foundations** (in progress): Rosalind problems (`rosalind_starter.py`: first 8 problems with a self-test and Rosalind-format output) + Hugging Face course.
- **Phase 3 — ML basics** train/test splitting, overfitting, metrics on p53 embeddings.
- **Phase 4 — Project** Enzyme classifier: Swiss-Prot EC classes → MMseqs2 cluster split → ESM-2 650M embeddings → logistic regression vs. small MLP, with honest evaluation.

## Files

- `check_env.py` — verifies PyTorch/CUDA/GPU
- `esm2_first_steps_v1.py` — first ESM-2 experiment (embeddings, hemoglobin control, masked prediction)
- `esm2_first_steps_v2.py` — extended p53 analysis (domain comparison, confidence plot, human vs mouse substitution scoring)
- `rosalind_starter.py` — first 8 Rosalind problems (DNA, RNA, REVC, FIB, GC, HAMM, PROT, SUBS) with edge-case tests, optional Biopython cross-check and CLI output in Rosalind format
- `esm2-enzyme-classifier-plan.md` — plan for the Phase 4 project

## Run it

```bash
conda create -n bioai python=3.11 -y
conda activate bioai
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas scikit-learn requests
pip install biopython matplotlib
python check_env.py
python esm2_first_steps_v1.py
python esm2_first_steps_v2.py
```

### Rosalind

```bash
# self-test on official samples and edge cases (PASS / TODO / FAIL / ERROR)
python rosalind_starter.py

# print the answer for a downloaded dataset in Rosalind format
python rosalind_starter.py gc rosalind_gc.txt
```
