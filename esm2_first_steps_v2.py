#RU: Требуются: pip install biopython matplotlib
#EN: Requires: pip install biopython matplotlib

import random

import numpy as np
import requests
import torch

#RU: Бэкенд без окна: график просто сохраняется в файл
#EN: Non-interactive backend: the plot is simply saved to a file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bio import Align
from Bio.Align import substitution_matrices
from transformers import AutoModelForMaskedLM, AutoTokenizer

#RU: Название предобученной модели (ESM-2, 12 слоёв, 35M параметров)
#EN: Name of the pretrained model (ESM-2, 12 layers, 35M parameters)
MODEL = "facebook/esm2_t12_35M_UR50D"

#RU: Используем GPU, если он доступен, иначе CPU
#EN: Use the GPU if available, otherwise fall back to the CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#RU: 20 стандартных аминокислот (порядок задаёт колонки матрицы вероятностей)
#EN: The 20 standard amino acids (the order defines the columns of the probability matrix)
AA = "ACDEFGHIKLMNPQRSTVWY"

#RU: Границы доменов p53 человека (нумерация с 1, по UniProt P04637)
#EN: Human p53 domain boundaries (1-based numbering, per UniProt P04637)
DBD_START, DBD_END = 102, 292

#RU: Выходной файл с графиком
#EN: Output file for the plot
PLOT_PATH = "p53_confidence.png"


def fetch_fasta(accession: str) -> str:
    """
    RU: Скачивает белок из UniProt по accession-номеру и возвращает
        его аминокислотную последовательность одной строкой (без заголовка FASTA).
    EN: Downloads a protein from UniProt by accession number and returns
        its amino acid sequence as a single string (without the FASTA header).
    """
    #RU: URL к REST API UniProt в формате FASTA
    #EN: UniProt REST API URL in FASTA format
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

    #RU: Запрос с таймаутом и проверкой на ошибку сервера
    #EN: Request with a timeout and a server error check
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    #RU: Пропускаем заголовок и склеиваем строки последовательности
    #EN: Skip the header and join the sequence lines
    return "".join(resp.text.splitlines()[1:])


def masked_log_probs(mlm, tok, seq: str, batch_size: int = 16) -> torch.Tensor:
    """
    RU: Для каждой позиции белка по очереди скрывает остаток (masked marginals)
        и возвращает матрицу log-вероятностей [длина, 20] по 20 аминокислотам.
        Softmax считается по всему словарю модели, затем берутся 20 колонок.
    EN: Masks each protein position in turn (masked marginals) and returns
        a [length, 20] matrix of log-probabilities over the 20 amino acids.
        Softmax is computed over the whole model vocabulary, then 20 columns are kept.
    """
    length = len(seq)

    #RU: Токены исходной последовательности: <cls> + остатки + <eos>
    #EN: Tokens of the original sequence: <cls> + residues + <eos>
    ids = tok(seq, return_tensors="pt")["input_ids"][0].to(DEVICE)

    #RU: id токенов 20 аминокислот в словаре модели
    #EN: Vocabulary ids of the 20 amino acid tokens
    aa_ids = torch.tensor(tok.convert_tokens_to_ids(list(AA)), device=DEVICE)

    chunks = []
    for start in range(0, length, batch_size):
        positions = list(range(start, min(start + batch_size, length)))

        #RU: Копируем последовательность по числу позиций в батче и маскируем по одной позиции в каждой копии
        #EN: Copy the sequence once per position in the batch and mask one position in each copy
        batch = ids.unsqueeze(0).repeat(len(positions), 1)
        for row, p in enumerate(positions):
            batch[row, p + 1] = tok.mask_token_id  #RU: +1 из-за <cls> #EN: +1 because of <cls>

        with torch.no_grad():
            logits = mlm(input_ids=batch).logits

        #RU: Берём логиты только в замаскированной позиции каждой копии
        #EN: Take the logits only at the masked position of each copy
        rows = torch.arange(len(positions), device=DEVICE)
        cols = torch.tensor(positions, device=DEVICE) + 1
        pos_logits = logits[rows, cols]

        #RU: log-softmax по всему словарю, затем оставляем 20 аминокислот
        #EN: log-softmax over the whole vocabulary, then keep the 20 amino acids
        chunks.append(pos_logits.log_softmax(-1)[:, aa_ids].cpu())

    return torch.cat(chunks)


def report_position(logp: torch.Tensor, seq: str, pos: int, label: str) -> None:
    """
    RU: Печатает топ-5 предсказаний модели для одной позиции, вероятность и ранг настоящей аминокислоты.
    EN: Prints the model's top-5 predictions for one position, plus the probability and rank of the true residue.
    """
    probs = logp[pos].exp()
    top = probs.topk(5)
    guesses = ", ".join(f"{AA[i]}={p:.3f}" for p, i in zip(top.values, top.indices))

    true_aa = seq[pos]
    p_true = probs[AA.index(true_aa)].item()

    #RU: Ранг настоящей аминокислоты среди 20 (1 = лучший)
    #EN: Rank of the true residue among the 20 (1 = best)
    rank = int((probs > probs[AA.index(true_aa)]).sum().item()) + 1

    print(f"{label} (residue {pos + 1}, true {true_aa}): p_true={p_true:.3f}, rank={rank}/20")
    print(f"    top-5: {guesses}")


def residue_stats(logp: torch.Tensor, seq: str):
    """
    RU: Считает для каждой позиции вероятность настоящей аминокислоты и энтропию распределения
        (энтропия считается по 20 аминокислотам после перенормировки).
    EN: Computes, for each position, the probability of the true residue and the entropy of the distribution
        (entropy is computed over the 20 amino acids after renormalization).
    """
    idx = torch.tensor([AA.index(a) for a in seq])
    p_true = logp[torch.arange(len(seq)), idx].exp().numpy()

    p = logp.exp()
    p = p / p.sum(-1, keepdim=True)
    entropy = -(p * p.log()).sum(-1).numpy()
    return p_true, entropy


def moving_average(x: np.ndarray, window: int = 10) -> np.ndarray:
    """
    RU: Скользящее среднее для сглаживания графика (края считаются по неполному окну).
    EN: Moving average for smoothing the plot (edges use a partial window).
    """
    #RU: Нормируем на число реально попавших в окно точек, чтобы края не проваливались к нулю
    #EN: Normalize by the number of points actually in the window so edges do not sink toward zero
    kernel = np.ones(window)
    return np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")


def plot_confidence(p_true: np.ndarray, entropy: np.ndarray, path: str) -> None:
    """
    RU: Строит график уверенности модели по длине белка: вероятность настоящей аминокислоты
        и энтропию предсказания; ДНК-связывающий домен выделен цветом.
    EN: Plots the model's confidence along the protein: probability of the true residue
        and prediction entropy; the DNA-binding domain is shaded.
    """
    x = np.arange(1, len(p_true) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    #RU: Верхняя панель: вероятность настоящей аминокислоты
    #EN: Top panel: probability of the true residue
    axes[0].plot(x, p_true, alpha=0.3, label="per residue")
    axes[0].plot(x, moving_average(p_true), label="moving average (10)")
    axes[0].set_ylabel("p(true aa)")
    axes[0].legend(loc="upper right")

    #RU: Нижняя панель: энтропия (низкая = модель уверена)
    #EN: Bottom panel: entropy (low = model is confident)
    axes[1].plot(x, entropy, alpha=0.3)
    axes[1].plot(x, moving_average(entropy))
    axes[1].set_ylabel("entropy (nats)")
    axes[1].set_xlabel("residue position (1-based)")

    #RU: Подсвечиваем ДНК-связывающий домен на обеих панелях
    #EN: Highlight the DNA-binding domain on both panels
    for ax in axes:
        ax.axvspan(DBD_START, DBD_END, color="orange", alpha=0.15)
    axes[0].set_title(f"ESM-2 confidence along human p53 (shaded: DBD {DBD_START}-{DBD_END})")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aligned_pairs(seq_a: str, seq_b: str):
    """
    RU: Глобально выравнивает две последовательности (BLOSUM62) и возвращает список пар индексов
        (i, j) для позиций без гэпов; индексы с нуля.
    EN: Globally aligns two sequences (BLOSUM62) and returns a list of index pairs
        (i, j) for gap-free positions; indices are zero-based.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    #RU: Берём лучшее выравнивание
    #EN: Take the best alignment
    aln = aligner.align(seq_a, seq_b)[0]

    pairs = []
    #RU: aln.aligned — блоки совпадающих по позициям участков без гэпов
    #EN: aln.aligned holds the gap-free aligned blocks
    for (a_start, a_end), (b_start, b_end) in zip(*aln.aligned):
        for k in range(a_end - a_start):
            pairs.append((a_start + k, b_start + k))
    return pairs


def substitution_score(logp: torch.Tensor, pos: int, wt: str, mut: str) -> float:
    """
    RU: Оценка замены в стиле ESM: log p(mut) - log p(wt) в контексте последовательности.
        Чем ниже значение, тем менее вероятной модель считает замену.
    EN: ESM-style substitution score: log p(mut) - log p(wt) in sequence context.
        The lower the value, the less likely the model considers the substitution.
    """
    return (logp[pos, AA.index(mut)] - logp[pos, AA.index(wt)]).item()


def main() -> None:
    """
    RU: Основной сценарий: три анализа p53 с помощью ESM-2 (регионы белка, график уверенности,
        оценка межвидовых замен человек vs мышь).
    EN: Main routine: three p53 analyses with ESM-2 (protein regions, confidence plot,
        scoring of human vs mouse substitutions).
    """
    random.seed(0)

    #RU: Скачиваем последовательности p53 человека и мыши
    #EN: Download human and mouse p53 sequences
    seq_human = fetch_fasta("P04637")
    seq_mouse = fetch_fasta("P02340")
    print(f"Human p53: {len(seq_human)} aa | Mouse p53: {len(seq_mouse)} aa\n")

    #RU: Загружаем модель и токенизатор
    #EN: Load the model and tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mlm = AutoModelForMaskedLM.from_pretrained(MODEL).to(DEVICE).eval()

    #RU: Одна матрица masked marginals для человеческого p53 используется во всех трёх анализах
    #EN: One masked-marginals matrix for human p53 is reused in all three analyses
    print("Computing masked marginals for every position of human p53...")
    logp_h = masked_log_probs(mlm, tok, seq_human)
    print("Done.\n")

    # ------------------------------------------------------------------
    #RU: АНАЛИЗ 1: сравнение позиции в N-конце и в ДНК-связывающем домене
    #EN: ANALYSIS 1: compare a position in the N-terminus with one in the DNA-binding domain
    # ------------------------------------------------------------------
    print("=== 1. N-terminus vs DNA-binding domain ===")
    #RU: Остаток 51 (как в первом скрипте) и остаток 175 (известная "горячая" точка мутаций R175H)
    #EN: Residue 51 (as in the first script) and residue 175 (the known R175H mutation hotspot)
    report_position(logp_h, seq_human, 50, "N-terminal, disordered")
    report_position(logp_h, seq_human, 174, "DBD, conserved")

    #RU: Средняя уверенность по регионам: количественная версия того же сравнения
    #EN: Mean confidence per region: a quantitative version of the same comparison
    p_true, entropy = residue_stats(logp_h, seq_human)
    regions = {
        f"N-term (1-{DBD_START - 1})": (0, DBD_START - 1),
        f"DBD ({DBD_START}-{DBD_END})": (DBD_START - 1, DBD_END),
        f"C-term ({DBD_END + 1}-{len(seq_human)})": (DBD_END, len(seq_human)),
    }
    print("\nMean over region:")
    for name, (a, b) in regions.items():
        print(f"  {name:<18} p_true={p_true[a:b].mean():.3f}  entropy={entropy[a:b].mean():.3f}")
    print()

    # ------------------------------------------------------------------
    #RU: АНАЛИЗ 2: график уверенности по всей длине белка
    #EN: ANALYSIS 2: confidence plot along the whole protein
    # ------------------------------------------------------------------
    print("=== 2. Confidence plot ===")
    plot_confidence(p_true, entropy, PLOT_PATH)
    print(f"Saved: {PLOT_PATH}\n")

    # ------------------------------------------------------------------
    #RU: АНАЛИЗ 3: реальные замены человек -> мышь против случайных замен
    #EN: ANALYSIS 3: real human -> mouse substitutions versus random substitutions
    # ------------------------------------------------------------------
    print("=== 3. Human vs mouse substitutions ===")
    pairs = aligned_pairs(seq_human, seq_mouse)
    same = sum(seq_human[i] == seq_mouse[j] for i, j in pairs)
    print(f"Aligned (gap-free) positions: {len(pairs)}, identical: {same} ({100 * same / len(pairs):.1f}%)")

    real = []    #RU: (оценка, позиция, wt, mut) для реальных замен #EN: (score, position, wt, mut) for real substitutions
    random_scores = []  #RU: оценки случайных замен в тех же позициях #EN: scores of random substitutions at the same positions
    for i, j in pairs:
        wt, mut = seq_human[i], seq_mouse[j]
        if wt == mut or wt not in AA or mut not in AA:
            continue
        real.append((substitution_score(logp_h, i, wt, mut), i + 1, wt, mut))

        #RU: Контроль: случайная аминокислота (не wt) в той же позиции
        #EN: Control: a random amino acid (not wt) at the same position
        rnd = random.choice([a for a in AA if a != wt])
        random_scores.append(substitution_score(logp_h, i, wt, rnd))

    real_scores = np.array([r[0] for r in real])
    random_scores = np.array(random_scores)
    print(f"Real substitutions: {len(real)}")
    print(f"  mean score, real species substitutions: {real_scores.mean():+.3f}")
    print(f"  mean score, random substitutions:       {random_scores.mean():+.3f}")
    print("  (real should be higher: evolution keeps substitutions the model finds acceptable)\n")

    real.sort()
    print("Least likely real substitutions (most negative):")
    for score, pos, wt, mut in real[:10]:
        print(f"  {wt}{pos}{mut}  {score:+.3f}")
    print("Most likely real substitutions (most positive):")
    for score, pos, wt, mut in real[-10:][::-1]:
        print(f"  {wt}{pos}{mut}  {score:+.3f}")


#RU: Запускаем main() только при прямом запуске файла, а не при импорте
#EN: Run main() only when the file is executed directly, not when imported
if __name__ == "__main__":
    main()

