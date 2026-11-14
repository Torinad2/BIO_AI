#RU: Библиотека для HTTP-запросов (скачивание последовательностей из UniProt)
#EN: HTTP library used to download sequences from UniProt
import requests

#RU: PyTorch: тензоры и вычисления на CPU/GPU
#EN: PyTorch: tensors and computation on CPU/GPU
import torch
import torch.nn.functional as F

#RU: Hugging Face Transformers: токенизатор и модель ESM-2 с головой маскированного языкового моделирования
#EN: Hugging Face Transformers: tokenizer and ESM-2 model with a masked language modeling head
from transformers import AutoModelForMaskedLM, AutoTokenizer

#RU: Название предобученной модели (ESM-2, 12 слоёв, 35M параметров)
#EN: Name of the pretrained model (ESM-2, 12 layers, 35M parameters)
MODEL = "facebook/esm2_t12_35M_UR50D"

#RU: Используем GPU, если он доступен, иначе CPU
#EN: Use the GPU if available, otherwise fall back to the CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fetch_fasta(accession: str) -> str:
    """
    RU: Скачивает белок из UniProt по accession-номеру и возвращает
        его аминокислотную последовательность одной строкой (без заголовка FASTA).
    EN: Downloads a protein from UniProt by accession number and returns
        its amino acid sequence as a single string (without the FASTA header).
    """
    #RU: Формируем URL к REST API UniProt в формате FASTA
    #EN: Build the UniProt REST API URL in FASTA format
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

    #RU: Делаем запрос с таймаутом и проверяем, что сервер ответил без ошибки
    #EN: Send the request with a timeout and make sure the server replied without an error
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    #RU: Пропускаем первую строку (заголовок) и склеиваем остальные строки в одну последовательность
    #EN: Skip the first line (header) and join the remaining lines into one sequence
    return "".join(resp.text.splitlines()[1:])


def protein_vector(mlm, tok, seq: str) -> torch.Tensor:
    """
    RU: Превращает последовательность белка в один вектор фиксированной длины
        (480 чисел для esm2_t12_35M): эмбеддинги остатков усредняются по длине белка.
    EN: Converts a protein sequence into a single fixed-size vector
        (480 numbers for esm2_t12_35M): residue embeddings are averaged over the protein length.
    """
    #RU: Токенизируем последовательность и переносим тензоры на нужное устройство
    #EN: Tokenize the sequence and move the tensors to the chosen device
    inputs = tok(seq, return_tensors="pt").to(DEVICE)

    #RU: Градиенты не нужны, так как мы только делаем предсказание (инференс)
    #EN: Gradients are not needed because we only run inference
    with torch.no_grad():
        #RU: mlm.esm — базовая модель внутри EsmForMaskedLM, отдельно грузить AutoModel не нужно
        #EN: mlm.esm is the base model inside EsmForMaskedLM, no need to load AutoModel separately
        emb = mlm.esm(**inputs).last_hidden_state[0]

    #RU: Отбрасываем служебные токены <cls> и <eos> и усредняем эмбеддинги по остаткам
    #EN: Drop the special <cls> and <eos> tokens and average the embeddings over residues
    return emb[1:-1].mean(0)


def main() -> None:
    """
    RU: Основной сценарий: скачивает три белка, сравнивает их эмбеддинги
        и проверяет, как модель угадывает скрытую аминокислоту.
    EN: Main routine: downloads three proteins, compares their embeddings
        and checks how well the model guesses a masked amino acid.
    """
    #RU: Скачиваем последовательности белков
    #EN: Download the protein sequences
    seq_human = fetch_fasta("P04637")  #RU: p53 человека #EN: human p53
    seq_mouse = fetch_fasta("P02340")  #RU: p53 мыши #EN: mouse p53
    #RU: Контроль: бета-цепь гемоглобина человека, не родственная p53
    #EN: Control: human hemoglobin beta chain, unrelated to p53
    seq_control = fetch_fasta("P68871")

    #RU: Печатаем длины и начало каждой последовательности для проверки
    #EN: Print the length and the beginning of each sequence as a sanity check
    print()
    print(f"Human p53: {len(seq_human)} aa, starts {seq_human[:60]}")
    print(f"Mouse p53: {len(seq_mouse)} aa, starts {seq_mouse[:60]}")
    print(f"Control (HBB): {len(seq_control)} aa, starts {seq_control[:60]}")
    print()

    #RU: Загружаем токенизатор и одну модель (MLM), переводим в режим оценки (без dropout)
    #EN: Load the tokenizer and a single (MLM) model, switch to eval mode (no dropout)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mlm = AutoModelForMaskedLM.from_pretrained(MODEL).to(DEVICE).eval()

    #RU: Получаем по одному вектору на каждый белок
    #EN: Get one vector per protein
    vec_h = protein_vector(mlm, tok, seq_human)
    vec_m = protein_vector(mlm, tok, seq_mouse)
    vec_c = protein_vector(mlm, tok, seq_control)

    #RU: Косинусное сходство: человек vs мышь и человек vs контроль
    #EN: Cosine similarity: human vs mouse and human vs control
    sim_hm = F.cosine_similarity(vec_h, vec_m, dim=0).item()
    sim_hc = F.cosine_similarity(vec_h, vec_c, dim=0).item()

    print(f"embedding shape per protein: {tuple(vec_h.shape)}")
    print(f"cosine similarity human p53 vs mouse p53:  {sim_hm:.3f}")
    print(f"cosine similarity human p53 vs control:    {sim_hc:.3f}")
    #RU: Важна разница между значениями, а не абсолютное число
    #EN: What matters is the gap between the values, not the absolute number
    print("(значимо только разница между ними: средний эмбеддинг у ESM")
    print(" высокий даже для неродственных белков)")
    print()

    #RU: Позиция для маскирования (индекс с нуля, то есть 51-й остаток)
    #EN: Position to mask (zero-based index, i.e. the 51st residue)
    pos = 50

    #RU: Заменяем аминокислоту на маску-токен
    #EN: Replace the amino acid with the mask token
    masked = seq_human[:pos] + tok.mask_token + seq_human[pos + 1:]

    #RU: Токенизируем последовательность с маской
    #EN: Tokenize the sequence containing the mask
    inputs = tok(masked, return_tensors="pt").to(DEVICE)

    #RU: Прогоняем через модель и получаем логиты по всему словарю для каждой позиции
    #EN: Run the model and get logits over the whole vocabulary for every position
    with torch.no_grad():
        logits = mlm(**inputs).logits

    #RU: Находим индекс маски среди токенов (с учётом <cls> в начале)
    #EN: Find the index of the mask among the tokens (accounting for the leading <cls>)
    mask_idx = inputs["input_ids"][0].tolist().index(tok.mask_token_id)

    #RU: Softmax по всему словарю, и только потом top-k, чтобы вероятности были честными
    #EN: Softmax over the whole vocabulary, and only then top-k, so probabilities are honest
    probs = logits[0, mask_idx].softmax(-1)
    top = probs.topk(5)

    #RU: Выводим пять самых вероятных аминокислот
    #EN: Print the five most likely amino acids
    print(f"model guesses for masked position {pos}:")
    for prob, idx in zip(top.values, top.indices):
        print(f"  {tok.decode(idx.item())}  p={prob.item():.3f}")

    #RU: Для сравнения печатаем настоящую аминокислоту
    #EN: For comparison, print the true amino acid
    print()
    print(f"true amino acid: {seq_human[pos]}")
    print()


#RU: Запускаем main() только при прямом запуске файла, а не при импорте
#EN: Run main() only when the file is executed directly, not when imported
if __name__ == "__main__":
    main()

