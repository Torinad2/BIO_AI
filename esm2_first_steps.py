"""
esm2_first_steps.py — вся Фаза 1 одним запуском: скачает p53 человека и мыши, загрузит ESM-2, 
посчитает эмбеддинги и сходство, потом замаскирует аминокислоту и покажет топ-5 догадок.
"""

"""Phase 1 of the bio-AI start plan: first contact with a real bio model.

Does four things, in order (mirrors bio-ai-start-plan-2026-11-12.md, Phase 1):
  1. Downloads human p53 (UniProt P04637, reviewed Swiss-Prot entry) as FASTA.
  2. Loads the small ESM-2 (facebook/esm2_t12_35M_UR50D) on the GPU.
  3. Computes per-amino-acid embeddings and one mean-pooled vector per protein,
     then compares human p53 vs mouse p53 (P02340) by cosine similarity.
  4. Masks one amino acid and asks the masked-LM head to guess it (top-5).

Run inside the `bioai` conda environment:
    python esm2_first_steps.py
First run downloads ~150 MB of model weights from Hugging Face.
"""
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

MODEL = "facebook/esm2_t12_35M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fetch_fasta(accession: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    text = requests.get(url, timeout=30).text
    return "".join(text.splitlines()[1:])


def protein_vector(model, tok, seq: str) -> torch.Tensor:
    inputs = tok(seq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[0]  # [length+2, 480]
    return emb[1:-1].mean(0)  # drop BOS/EOS, mean-pool over amino acids


def main() -> None:
    seq_human = fetch_fasta("P04637")
    seq_mouse = fetch_fasta("P02340")
    print(f"human p53: {len(seq_human)} aa, starts {seq_human[:60]}")
    print(f"mouse p53: {len(seq_mouse)} aa")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

    vec_h = protein_vector(model, tok, seq_human)
    vec_m = protein_vector(model, tok, seq_mouse)
    sim = F.cosine_similarity(vec_h, vec_m, dim=0).item()
    print(f"embedding shape per protein: {tuple(vec_h.shape)}")
    print(f"cosine similarity human vs mouse p53: {sim:.3f}  (close to 1 = related)")

    # Step 4: masked amino-acid guessing needs the MLM head, not AutoModel.
    mlm = AutoModelForMaskedLM.from_pretrained(MODEL).to(DEVICE).eval()
    pos = 50
    masked = seq_human[:pos] + "<mask>" + seq_human[pos + 1:]
    inputs = tok(masked, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = mlm(**inputs).logits
    mask_idx = inputs["input_ids"][0].tolist().index(tok.mask_token_id)
    top = logits[0, mask_idx].topk(5)
    print(f"\nmodel guesses for masked position {pos}:")
    for prob, idx in zip(top.values.softmax(-1), top.indices):
        print(f"  {tok.decode(idx.item())}  p={prob.item():.3f}")
    print(f"true amino acid: {seq_human[pos]}")


if __name__ == "__main__":
    main()
