# Rosalind — первые 8 задач (Фаза 2 плана Bio AI)
# Порядок по плану: DNA, RNA, REVC, FIB, GC, HAMM, PROT, SUBS
#
# Как работать:
#   1. DNA уже решена — это образец оформления.
#   2. В каждой следующей функции замени строку `raise NotImplementedError`
#      своим кодом (подсказка — в комментарии над функцией).
#   3. Запуск:  python rosalind_starter.py
#      Самопроверка прогонит все задачи на официальных примерах Rosalind
#      и покажет PASS / TODO / FAIL по каждой.
#   4. Когда самопроверка зелёная — неси тот же код на rosalind.info:
#      сайт даст свой набор данных (dataset) и проверит ответ за ~5 минут.
#
# Никаких внешних библиотек не нужно — чистый Python из окружения bioai.


# --- 1. DNA: подсчёт нуклеотидов -------------------------------------------
# Пример: AGCT... -> A=20 C=12 G=17 T=21 (порядок именно A C G T)
def dna_counts(seq: str) -> tuple[int, int, int, int]:
    return (seq.count("A"), seq.count("C"), seq.count("G"), seq.count("T"))


# --- 2. RNA: транскрипция ДНК в РНК ----------------------------------------
# Подсказка: в РНК вместо тимина (T) стоит урацил (U). Одна замена.
def transcribe(seq: str) -> str:
    raise NotImplementedError


# --- 3. REVC: обратная комплементарная цепь --------------------------------
# Подсказка: A<->T, C<->G, и всю цепь развернуть в обратном порядке.
# Поможет словарь замен + срез [::-1] (или str.maketrans).
def reverse_complement(seq: str) -> str:
    raise NotImplementedError


# --- 4. FIB: кролики (рекурсия / динамика) ----------------------------------
# Каждый месяц зрелая пара даёт k новых пар; кролики не умирают.
# n=5, k=3 -> 19. Подсказка: это Фибоначчи, где следующий = взрослые + k*дети;
# цикл из двух переменных быстрее рекурсии.
def rabbits(n: int, k: int) -> int:
    raise NotImplementedError


# --- 5. GC: GC-состав --------------------------------------------------------
# На вход — словарь {имя: последовательность} (на сайте это FASTA-файл).
# Вернуть (имя, процент GC) записи с максимальным GC-составом.
# Подсказка: процент = 100 * (число G + число C) / длина.
def best_gc(records: dict[str, str]) -> tuple[str, float]:
    raise NotImplementedError


# --- 6. HAMM: расстояние Хэмминга --------------------------------------------
# Число позиций, где две цепи одинаковой длины различаются (точечные мутации).
# Подсказка: zip(s, t) + генератор с sum().
def hamming(s: str, t: str) -> int:
    raise NotImplementedError


# --- 7. PROT: трансляция РНК в белок -----------------------------------------
# РНК режется на тройки (кодоны); каждый кодон -> аминокислота по таблице;
# стоп-кодоны (UAA, UAG, UGA) заканчивают белок и в него не входят.
# Подсказка: таблицу кодонов не пиши руками — возьми готовую из Biopython:
#   from Bio.Data import CodonTable
#   table = CodonTable.standard_rna_table.forward_table  # {'UUU': 'F', ...}
# (biopython ставился вместе с окружением по плану Фазы 0.)
# Либо держи таблицу словарём прямо в коде — Rosalind даёт её в условии.
def translate(rna: str) -> str:
    raise NotImplementedError


# --- 8. SUBS: поиск мотива ----------------------------------------------------
# Все стартовые позиции (нумерация с 1!), где мотив t встречается в цепи s.
# Вхождения могут перекрываться: ATAT в GATATAT... находится дважды подряд.
# Подсказка: цикл по всем позициям + срез s[i:i+len(t)] == t.
def find_motif(s: str, t: str) -> list[int]:
    raise NotImplementedError


# --- Самопроверка на официальных примерах Rosalind ---------------------------
GC_SAMPLE = {
    "Rosalind_6404": "CCTGCGGAAGATCGGCACTAGAATAGCCAGAACCGTTTCTCTGAGGCTTCCGGCCTTCC"
                     "CTCCCACTAATAATTCTGAGG",
    "Rosalind_5959": "CCATCGGTAGCGCATCCTTAGTCCAATTAAGTCCCTATCCAGGCGCTCCGCCGAAGGTCT"
                     "ATATCCATTTGTCAGCAGACACGC",
    "Rosalind_0808": "CCACCCTCGTGGTATGGCTAGGCATTCAGGAACCGGAGAACGCTTCAGACCAGCCCGGAC"
                     "TGGGAACCTGCGGGCAGTAGGTGGAAT",
}

TESTS = [
    ("DNA",  lambda: dna_counts("AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC"),
     (20, 12, 17, 21)),
    ("RNA",  lambda: transcribe("GATGGAACTTGACTACGTAAATT"),
     "GAUGGAACUUGACUACGUAAAUU"),
    ("REVC", lambda: reverse_complement("AAAACCCGGT"),
     "ACCGGGTTTT"),
    ("FIB",  lambda: rabbits(5, 3),
     19),
    ("GC",   lambda: (lambda r: (r[0], round(r[1], 4)))(best_gc(GC_SAMPLE)),
     ("Rosalind_0808", 60.9195)),
    ("HAMM", lambda: hamming("GAGCCTACTAACGGGAT", "CATCGTAATGACGGCCT"),
     7),
    ("PROT", lambda: translate("AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA"),
     "MAMAPRTEINSTRING"),
    ("SUBS", lambda: find_motif("GATATATGCATATACTT", "ATAT"),
     [2, 4, 10]),
]

if __name__ == "__main__":
    done = 0
    for name, fn, expected in TESTS:
        try:
            got = fn()
        except NotImplementedError:
            print(f"{name:5} TODO   — ещё не решена")
            continue
        if got == expected:
            print(f"{name:5} PASS")
            done += 1
        else:
            print(f"{name:5} FAIL   — получено {got!r}, ожидалось {expected!r}")
    print(f"\nРешено: {done} из {len(TESTS)}. Зелёные задачи можно сдавать на rosalind.info.")
