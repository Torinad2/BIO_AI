import requests

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

MODEL = "facebook/esm2_t12_35M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_fasta(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

    text = requests.get(url, timeout=30).text

    return "".join(text.splitlines()[1:])

def protein_vector(model, tok, seq: str) -> torch.Tensor:
    inputs = tok(seq, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[0]

    return emb[1:-1].mean(0)


def main() -> None:
   
    seq_human = fetch_fasta("P04637")
    seq_mouse = fetch_fasta("P02340")

    print()
    print(f"Human p53: {len(seq_human)} aa, starts {seq_human[:60]}")
    print(f"Mouse p53: {len(seq_mouse)} aa, starts {seq_mouse[:60]}")
    print()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

   
    vec_h = protein_vector(model, tok, seq_human)
    vec_m = protein_vector(model, tok, seq_mouse)

    sim = F.cosine_similarity(vec_h, vec_m, dim=0).item()

    print()
    print(f"embedding shape per protein: {tuple(vec_h.shape)}")
    print(f"cosine similarity human vs mouse p53: {sim:.3f}  (close to 1 = related)")
    print()

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

    print()
    print(f"true amino acid: {seq_human[pos]}")
    print()

if __name__ == "__main__":
    main()
