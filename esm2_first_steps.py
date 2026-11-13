"""
RU: Фаза 1 плана bio-AI: первый контакт с настоящей био-моделью.
EN: Phase 1 of the bio-AI start plan: first contact with a real bio model.


EN: What the script does, in order:
  1. Downloads human p53 (UniProt P04637, reviewed Swiss-Prot entry) as FASTA.
  2. Loads the small ESM-2 (facebook/esm2_t12_35M_UR50D) on the GPU.
  3. Computes per-amino-acid embeddings and one mean-pooled vector per protein,
     then compares human p53 vs mouse p53 (P02340) by cosine similarity.
  4. Masks one amino acid and asks the masked-LM head to guess it (top-5).

RU: Запуск внутри conda-окружения `bioai`:
EN: Run inside the `bioai` conda environment:
    python esm2_first_steps.py

RU: Первый запуск скачивает ~150 MB весов модели с Hugging Face.
EN: First run downloads ~150 MB of model weights from Hugging Face.
"""

# RU: requests — HTTP-запросы: качаем последовательности белков с UniProt.
# EN: requests — HTTP calls: downloads the protein sequences from UniProt.
import requests

# RU: torch — PyTorch: тензоры, вычисления на GPU, косинусное сходство.
# EN: torch — PyTorch: tensors, GPU compute, cosine similarity.
import torch
import torch.nn.functional as F

# RU: Классы Transformers: AutoModel — сама модель (эмбеддинги),
#     AutoModelForMaskedLM — модель с «головой» для угадывания маски,
#     AutoTokenizer — превращает строку белка в числа для модели.
# EN: Transformers classes: AutoModel — the base model (embeddings),
#     AutoModelForMaskedLM — the model with a head that guesses masked tokens,
#     AutoTokenizer — turns a protein string into numbers for the model.
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

# RU: Какая именно модель ESM-2: t12 = 12 слоёв трансформера,
#     35M = 35 миллионов параметров (самая маленькая, быстрая для учёбы),
#     UR50D = обучалась на базе UniRef50.
# EN: Which ESM-2 model: t12 = 12 transformer layers,
#     35M = 35 million parameters (the smallest one, fast for learning),
#     UR50D = trained on the UniRef50 dataset.
MODEL = "facebook/esm2_t12_35M_UR50D"

# RU: Если есть CUDA-видеокарта — работаем на ней ("cuda"), иначе на CPU.
# EN: If a CUDA GPU is present, run on it ("cuda"), otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fetch_fasta(accession: str) -> str:
    # RU: Качает белок по его коду UniProt (например, P04637) и возвращает
    #     только саму последовательность одной строкой, без заголовка.
    # EN: Downloads a protein by its UniProt accession (e.g. P04637) and
    #     returns only the sequence as one string, without the header.
    #
    # RU: Формат FASTA: первая строка — заголовок (начинается с ">"),
    #     дальше идёт сама последовательность, разбитая на строки по 60 букв.
    # EN: FASTA format: the first line is a header (starts with ">"),
    #     the sequence itself follows, wrapped into lines of 60 letters.

    # RU: REST-адрес UniProt: по этому URL сервер отдаёт FASTA-файл белка.
    # EN: The UniProt REST URL: the server returns the protein's FASTA at this address.
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

    # RU: GET-запрос; timeout=30 — не висеть дольше 30 секунд без ответа.
    # EN: A GET request; timeout=30 — never hang longer than 30 seconds.
    text = requests.get(url, timeout=30).text

    # RU: splitlines() режет текст на строки; [1:] отбрасывает первую
    #     (заголовок); "".join(...) склеивает остальное в одну строку.
    # EN: splitlines() splits the text into lines; [1:] drops the first one
    #     (the header); "".join(...) glues the rest into a single string.
    return "".join(text.splitlines()[1:])


def protein_vector(model, tok, seq: str) -> torch.Tensor:
    # RU: Превращает одну последовательность белка в один вектор-«отпечаток»
    #     из 480 чисел: прогоняет белок через модель и усредняет векторы
    #     всех аминокислот.
    # EN: Turns one protein sequence into a single 480-number "fingerprint"
    #     vector: runs the protein through the model and averages the
    #     per-amino-acid vectors.

    # RU: Токенизатор превращает строку букв в тензор чисел (input_ids).
    #     return_tensors="pt" — вернуть именно PyTorch-тензор.
    #     .to(DEVICE) — перенести тензор на видеокарту.
    #     ВАЖНО: токенизатор сам добавляет служебные токены в начало и
    #     конец (BOS/EOS), поэтому длина становится на 2 больше.
    # EN: The tokenizer turns the letter string into a tensor of numbers
    #     (input_ids). return_tensors="pt" — return a PyTorch tensor.
    #     .to(DEVICE) — move the tensor onto the GPU.
    #     NOTE: the tokenizer itself adds special start/end tokens (BOS/EOS),
    #     so the length grows by 2.
    inputs = tok(seq, return_tensors="pt").to(DEVICE)

    # RU: torch.no_grad() — обучение не нужно, только прямой проход:
    #     PyTorch не строит графы градиентов, экономит память и ускоряет.
    # EN: torch.no_grad() — no training here, only a forward pass:
    #     PyTorch skips gradient bookkeeping, saving memory and time.
    with torch.no_grad():
        # RU: **inputs распаковывает словарь токенизатора в аргументы модели.
        #     last_hidden_state — выход последнего слоя: тензор формы
        #     [1, длина+2, 480]; [0] убирает размерность пакета (у нас 1 белок).
        #     480 — размер вектора каждой аминокислоты у этой модели.
        # EN: **inputs unpacks the tokenizer's dict into model arguments.
        #     last_hidden_state — the last layer's output, a tensor of shape
        #     [1, length+2, 480]; [0] drops the batch dimension (we have 1 protein).
        #     480 — the size of each amino acid's vector in this model.
        emb = model(**inputs).last_hidden_state[0]  # [length+2, 480]

    # RU: [1:-1] отрезает первый и последний векторы — это служебные
    #     токены BOS/EOS, не аминокислоты. mean(0) усредняет векторы по
    #     всем позициям (ось 0) — получается один вектор на весь белок.
    # EN: [1:-1] cuts off the first and last vectors — those are the BOS/EOS
    #     special tokens, not amino acids. mean(0) averages the vectors over
    #     all positions (axis 0) — one vector for the whole protein.
    return emb[1:-1].mean(0)  # drop BOS/EOS, mean-pool over amino acids


def main() -> None:
    # RU: Скачиваем оба белка: p53 человека (P04637) и мыши (P02340).
    #     Это ортологи — «один и тот же» белок у двух видов.
    # EN: Download both proteins: human p53 (P04637) and mouse p53 (P02340).
    #     They are orthologs — the "same" protein in two species.
    seq_human = fetch_fasta("P04637")
    seq_mouse = fetch_fasta("P02340")

    print()
    # RU: Печатаем длину и первые 60 аминокислот для контроля, что
    #     скачалось то, что нужно (у человека должно быть 393).
    # EN: Print the length and the first 60 amino acids as a sanity check
    #     that the right thing downloaded (human must be 393).
    print(f"Human p53: {len(seq_human)} aa, starts {seq_human[:60]}")
    print(f"Mouse p53: {len(seq_mouse)} aa")
    print()

    # RU: Загружаем токенизатор и базовую модель; .to(DEVICE) — на GPU,
    #     .eval() — режим предсказания (выключает dropout и т.п.).
    #     Веса берутся из локального кэша, если уже скачаны.
    # EN: Load the tokenizer and the base model; .to(DEVICE) — onto the GPU,
    #     .eval() — inference mode (turns off dropout etc.).
    #     Weights come from the local cache if already downloaded.
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

    # RU: Считаем вектор-«отпечаток» для каждого белка.
    # EN: Compute the fingerprint vector for each protein.
    vec_h = protein_vector(model, tok, seq_human)
    vec_m = protein_vector(model, tok, seq_mouse)

    # RU: Косинусное сходство двух векторов: 1.0 — направлены одинаково
    #     (белки «одинаковые»), 0 — никак не связаны. dim=0 — векторы одномерные.
    #     .item() достаёт обычное число Python из тензора.
    # EN: Cosine similarity of the two vectors: 1.0 — same direction
    #     (the proteins are "the same"), 0 — unrelated. dim=0 — the vectors
    #     are one-dimensional. .item() pulls a plain Python number out of the tensor.
    sim = F.cosine_similarity(vec_h, vec_m, dim=0).item()

    print()
    # RU: tuple(vec_h.shape) — форма вектора, должно быть (480,).
    # EN: tuple(vec_h.shape) — the vector's shape, expected (480,).
    print(f"embedding shape per protein: {tuple(vec_h.shape)}")
    print(f"cosine similarity human vs mouse p53: {sim:.3f}  (close to 1 = related)")
    print()

    # RU: Шаг 4: угадывание замаскированной аминокислоты. Для этого нужна
    #     ОТДЕЛЬНАЯ «голова» — AutoModelForMaskedLM: базовая модель даёт
    #     только эмбеддинги, а эта умеет предсказывать, какая аминокислота
    #     прячется под маской. Веса внутри те же, добавляется выходной слой.
    # EN: Step 4: guessing a masked amino acid. This needs a SEPARATE head —
    #     AutoModelForMaskedLM: the base model only produces embeddings,
    #     while this one can predict which amino acid hides under the mask.
    #     The inner weights are the same; an output layer is added.
    mlm = AutoModelForMaskedLM.from_pretrained(MODEL).to(DEVICE).eval()

    # RU: Какую позицию прячем: индекс 50 (счёт с нуля — это 51-я
    #     аминокислота в последовательности).
    # EN: Which position to hide: index 50 (zero-based — the 51st amino
    #     acid in the sequence).
    pos = 50

    # RU: Собираем копию последовательности, где буква на позиции pos
    #     заменена на служебный токен <mask>: всё до pos + <mask> + всё после.
    # EN: Build a copy of the sequence where the letter at pos is replaced
    #     by the special <mask> token: everything before pos + <mask> + the rest.
    masked = seq_human[:pos] + "<mask>" + seq_human[pos + 1:]

    # RU: Токенизируем замаскированную последовательность.
    # EN: Tokenize the masked sequence.
    inputs = tok(masked, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # RU: logits — «сырые баллы» модели по КАЖДОЙ позиции и КАЖДОЙ
        #     возможной аминокислоте: форма [1, длина+2, словарь].
        # EN: logits — the model's raw scores for EVERY position and EVERY
        #     possible amino acid: shape [1, length+2, vocabulary].
        logits = mlm(**inputs).logits

    # RU: Ищем, на каком месте в input_ids стоит id токена маски, —
    #     именно для этой позиции нам нужны предсказания.
    # EN: Find where the mask token's id sits inside input_ids —
    #     that is the position whose predictions we need.
    mask_idx = inputs["input_ids"][0].tolist().index(tok.mask_token_id)

    # RU: Берём баллы только позиции маски: logits[0, mask_idx] — вектор
    #     по всему словарю; .topk(5) — пять самых больших баллов и их индексы.
    # EN: Take the scores of the mask position only: logits[0, mask_idx] is a
    #     vector over the whole vocabulary; .topk(5) — the five largest
    #     scores and their indices.
    top = logits[0, mask_idx].topk(5)

    print(f"\nmodel guesses for masked position {pos}:")

    # RU: Идём по пяти лучшим вариантам. top.values — баллы, top.indices —
    #     id токенов. softmax(-1) превращает баллы в вероятности (сумма = 1),
    #     поэтому пары «вероятность, индекс» идут синхронно через zip.
    #     tok.decode(idx) превращает id обратно в букву аминокислоты.
    # EN: Walk the five best guesses. top.values — the scores, top.indices —
    #     the token ids. softmax(-1) turns scores into probabilities
    #     (they sum to 1), so probability/index pairs move together via zip.
    #     tok.decode(idx) turns an id back into an amino-acid letter.
    for prob, idx in zip(top.values.softmax(-1), top.indices):
        print(f"  {tok.decode(idx.item())}  p={prob.item():.3f}")

    print()
    # RU: Настоящая аминокислота на этой позиции — для сравнения с догадками.
    # EN: The true amino acid at this position — to compare with the guesses.
    print(f"true amino acid: {seq_human[pos]}")
    print()


# RU: Стандартный страж: main() запускается, только если файл выполнен напрямую (python esm2_first_steps.py), а не импортирован другим кодом.
# EN: The standard guard: main() runs only when the file is executed directly (python esm2_first_steps.py), not when imported by other code.
if __name__ == "__main__":
    main()

