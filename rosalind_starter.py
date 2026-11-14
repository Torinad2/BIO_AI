#RU: Rosalind — первые 8 задач (Фаза 2 плана Bio AI). Порядок: DNA, RNA, REVC, FIB, GC, HAMM, PROT, SUBS.
#EN: Rosalind — first 8 problems (Phase 2 of the Bio AI plan). Order: DNA, RNA, REVC, FIB, GC, HAMM, PROT, SUBS.
#
#RU: Как работать:
#EN: How to work:
#RU:   1. DNA уже решена — это образец оформления.
#EN:   1. DNA is already solved — it is a style example.
#RU:   2. В каждой следующей функции замените `raise NotImplementedError` своим кодом (подсказка в docstring).
#EN:   2. In each following function replace `raise NotImplementedError` with your code (hint in the docstring).
#RU:   3. Самопроверка:  python rosalind_starter.py
#RU:      Покажет PASS / TODO / FAIL / ERROR по официальным примерам, крайним случаям и (если есть Biopython) случайным тестам.
#EN:   3. Self-test:     python rosalind_starter.py
#EN:      Shows PASS / TODO / FAIL / ERROR for official samples, edge cases and (if Biopython is installed) random tests.
#RU:   4. Ответ для сайта:  python rosalind_starter.py <задача> <файл_датасета>   например: python rosalind_starter.py gc rosalind_gc.txt
#EN:   4. Answer for the site:  python rosalind_starter.py <task> <dataset_file>   e.g.: python rosalind_starter.py gc rosalind_gc.txt
#RU:      Скрипт печатает ответ в формате Rosalind — его можно сразу вставлять на rosalind.info.
#EN:      The script prints the answer in Rosalind format — paste it straight into rosalind.info.
#
#RU: Внешние библиотеки не нужны; Biopython используется только для необязательной перекрёстной проверки.
#EN: No external libraries are required; Biopython is used only for the optional cross-check.

import math
import random
import sys
from pathlib import Path


# =============================================================================
#RU: РЕШЕНИЯ
#EN: SOLUTIONS
# =============================================================================

# --- 1. DNA ------------------------------------------------------------------
def dna_counts(seq: str) -> tuple[int, int, int, int]:
    """
    RU: DNA. Подсчёт нуклеотидов. Возвращает числа A, C, G, T именно в таком порядке.
        Пример: AGCT... -> (20, 12, 17, 21).
    EN: DNA. Count nucleotides. Returns the counts of A, C, G, T in exactly this order.
        Example: AGCT... -> (20, 12, 17, 21).
    """
    return (seq.count("A"), seq.count("C"), seq.count("G"), seq.count("T"))


# --- 2. RNA ------------------------------------------------------------------
def transcribe(seq: str) -> str:
    """
    RU: RNA. Транскрипция ДНК в РНК: в РНК вместо тимина (T) стоит урацил (U).
        Подсказка: одна замена.
    EN: RNA. Transcribe DNA into RNA: RNA has uracil (U) instead of thymine (T).
        Hint: a single replacement.
    """
    raise NotImplementedError


# --- 3. REVC -----------------------------------------------------------------
def reverse_complement(seq: str) -> str:
    """
    RU: REVC. Обратная комплементарная цепь: A<->T, C<->G и цепь разворачивается.
        Подсказка: словарь замен + срез [::-1] (или str.maketrans).
    EN: REVC. Reverse complement: A<->T, C<->G and the strand is reversed.
        Hint: a replacement dictionary + the [::-1] slice (or str.maketrans).
    """
    raise NotImplementedError


# --- 4. FIB ------------------------------------------------------------------
def rabbits(n: int, k: int) -> int:
    """
    RU: FIB. Кролики: каждый месяц зрелая пара даёт k новых пар, кролики не умирают.
        n=5, k=3 -> 19. Рекуррентность: F(n) = F(n-1) + k * F(n-2), F(1) = F(2) = 1.
        Подсказка: цикл с двумя переменными; наивная рекурсия экспоненциально медленная.
    EN: FIB. Rabbits: every month a mature pair produces k new pairs, rabbits never die.
        n=5, k=3 -> 19. Recurrence: F(n) = F(n-1) + k * F(n-2), F(1) = F(2) = 1.
        Hint: a loop with two variables; naive recursion is exponentially slow.
    """
    raise NotImplementedError


# --- 5. GC -------------------------------------------------------------------
def best_gc(records: dict[str, str]) -> tuple[str, float]:
    """
    RU: GC. На вход словарь {имя: последовательность}. Вернуть (имя, процент GC) записи
        с максимальным GC-составом. Процент = 100 * (G + C) / длина.
        Rosalind принимает ответ с ошибкой до 0.001.
    EN: GC. Input is a {name: sequence} dictionary. Return (name, GC percent) of the record
        with the highest GC content. Percent = 100 * (G + C) / length.
        Rosalind accepts an answer with an error of up to 0.001.
    """
    raise NotImplementedError


# --- 6. HAMM -----------------------------------------------------------------
def hamming(s: str, t: str) -> int:
    """
    RU: HAMM. Расстояние Хэмминга: число позиций, где цепи одинаковой длины различаются.
        Если длины разные — выбросить ValueError (zip без проверки молча обрежет длинную цепь).
        Подсказка: zip(s, t) + sum(...); в Python 3.10+ есть zip(..., strict=True).
    EN: HAMM. Hamming distance: the number of positions where two equal-length strands differ.
        If the lengths differ, raise ValueError (a plain zip silently truncates the longer strand).
        Hint: zip(s, t) + sum(...); Python 3.10+ has zip(..., strict=True).
    """
    raise NotImplementedError


# --- 7. PROT -----------------------------------------------------------------
def translate(rna: str) -> str:
    """
    RU: PROT. Трансляция РНК в белок: РНК режется на кодоны (тройки), каждый кодон -> аминокислота.
        Стоп-кодоны (UAA, UAG, UGA) заканчивают белок и в него не входят.
        Подсказка: таблица из Biopython:
            from Bio.Data import CodonTable
            table = CodonTable.standard_rna_table.forward_table   # {'UUU': 'F', ...}
        ВАЖНО: в forward_table нет стоп-кодонов, table["UAA"] даст KeyError — стоп проверяйте отдельно.
        Незаконченный последний кодон игнорируется.
    EN: PROT. Translate RNA into a protein: RNA is split into codons (triplets), each codon -> an amino acid.
        Stop codons (UAA, UAG, UGA) end the protein and are not included in it.
        Hint: a table from Biopython:
            from Bio.Data import CodonTable
            table = CodonTable.standard_rna_table.forward_table   # {'UUU': 'F', ...}
        NOTE: forward_table has no stop codons, table["UAA"] raises KeyError — check stops separately.
        An incomplete final codon is ignored.
    """
    raise NotImplementedError


# --- 8. SUBS -----------------------------------------------------------------
def find_motif(s: str, t: str) -> list[int]:
    """
    RU: SUBS. Все стартовые позиции (нумерация с 1!), где мотив t встречается в цепи s.
        Вхождения могут перекрываться: ATAT в GATATAT находится дважды.
        Подсказка: цикл по позициям + s[i:i+len(t)] == t; альтернативы: s.find(t, start)
        в цикле со start = i + 1 или регулярное выражение с lookahead (?=...).
    EN: SUBS. All start positions (1-based numbering!) where motif t occurs in strand s.
        Occurrences may overlap: ATAT occurs twice in GATATAT.
        Hint: loop over positions + s[i:i+len(t)] == t; alternatives: s.find(t, start)
        in a loop with start = i + 1, or a regular expression with lookahead (?=...).
    """
    raise NotImplementedError


# =============================================================================
#RU: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: чтение FASTA и форматирование ответа
#EN: HELPERS: FASTA reading and answer formatting
# =============================================================================

def parse_fasta_text(text: str) -> dict[str, str]:
    """
    RU: Разбирает текст FASTA в словарь {имя: последовательность}. Многострочные
        последовательности склеиваются, пустые строки игнорируются.
    EN: Parses FASTA text into a {name: sequence} dictionary. Multi-line sequences
        are joined, blank lines are ignored.
    """
    records: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            #RU: Новая запись: имя без символа ">"
            #EN: A new record: the name without the ">" character
            name = line[1:].strip()
            records[name] = []
        elif name is None:
            raise ValueError("sequence line before the first FASTA header")
        else:
            records[name].append(line)
    return {k: "".join(v) for k, v in records.items()}


def parse_fasta(path: str) -> dict[str, str]:
    """
    RU: Читает FASTA-файл с диска (см. parse_fasta_text).
    EN: Reads a FASTA file from disk (see parse_fasta_text).
    """
    return parse_fasta_text(Path(path).read_text(encoding="utf-8"))


#RU: Названия задач для командной строки
#EN: Task names for the command line
TASK_NAMES = ("dna", "rna", "revc", "fib", "gc", "hamm", "prot", "subs")


def run_task(task: str, text: str) -> str:
    """
    RU: Решает задачу по тексту датасета и возвращает ответ строго в формате Rosalind.
    EN: Solves a task from the dataset text and returns the answer strictly in Rosalind format.
    """
    tokens = text.split()
    if task == "dna":
        return " ".join(map(str, dna_counts(tokens[0])))
    if task == "rna":
        return transcribe(tokens[0])
    if task == "revc":
        return reverse_complement(tokens[0])
    if task == "fib":
        n, k = int(tokens[0]), int(tokens[1])
        return str(rabbits(n, k))
    if task == "gc":
        name, percent = best_gc(parse_fasta_text(text))
        #RU: Rosalind показывает процент с 6 знаками после запятой
        #EN: Rosalind shows the percent with 6 digits after the decimal point
        return f"{name}\n{percent:.6f}"
    if task == "hamm":
        return str(hamming(tokens[0], tokens[1]))
    if task == "prot":
        return translate(tokens[0])
    if task == "subs":
        return " ".join(map(str, find_motif(tokens[0], tokens[1])))
    raise ValueError(f"unknown task: {task}")


# =============================================================================
#RU: САМОПРОВЕРКА
#EN: SELF-TEST
# =============================================================================

#RU: Официальные примеры Rosalind для задачи GC
#EN: Official Rosalind samples for the GC problem
GC_SAMPLE = {
    "Rosalind_6404": "CCTGCGGAAGATCGGCACTAGAATAGCCAGAACCGTTTCTCTGAGGCTTCCGGCCTTCC"
                     "CTCCCACTAATAATTCTGAGG",
    "Rosalind_5959": "CCATCGGTAGCGCATCCTTAGTCCAATTAAGTCCCTATCCAGGCGCTCCGCCGAAGGTCT"
                     "ATATCCATTTGTCAGCAGACACGC",
    "Rosalind_0808": "CCACCCTCGTGGTATGGCTAGGCATTCAGGAACCGGAGAACGCTTCAGACCAGCCCGGAC"
                     "TGGGAACCTGCGGGCAGTAGGTGGAAT",
}

#RU: Тот же пример в виде текста FASTA (как на сайте), с многострочными последовательностями
#EN: The same sample as FASTA text (as on the site), with multi-line sequences
GC_FASTA = "\n".join(f">{name}\n{seq[:60]}\n{seq[60:]}" for name, seq in GC_SAMPLE.items()) + "\n"

DNA_SAMPLE = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC"

#RU: Тесты: (задача, вид, функция, ожидаемое). Если ожидаемое — класс исключения, функция обязана его выбросить.
#EN: Tests: (task, kind, function, expected). If expected is an exception class, the function must raise it.
TESTS = [
    #RU: --- официальные примеры ---
    #EN: --- official samples ---
    ("DNA",  "sample", lambda: dna_counts(DNA_SAMPLE), (20, 12, 17, 21)),
    ("RNA",  "sample", lambda: transcribe("GATGGAACTTGACTACGTAAATT"), "GAUGGAACUUGACUACGUAAAUU"),
    ("REVC", "sample", lambda: reverse_complement("AAAACCCGGT"), "ACCGGGTTTT"),
    ("FIB",  "sample", lambda: rabbits(5, 3), 19),
    ("GC",   "sample", lambda: best_gc(GC_SAMPLE), ("Rosalind_0808", 60.919540)),
    ("HAMM", "sample", lambda: hamming("GAGCCTACTAACGGGAT", "CATCGTAATGACGGCCT"), 7),
    ("PROT", "sample", lambda: translate("AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA"), "MAMAPRTEINSTRING"),
    ("SUBS", "sample", lambda: find_motif("GATATATGCATATACTT", "ATAT"), [2, 4, 10]),

    #RU: --- крайние случаи ---
    #EN: --- edge cases ---
    ("DNA",  "edge", lambda: dna_counts(""), (0, 0, 0, 0)),
    ("RNA",  "edge", lambda: transcribe(""), ""),
    ("RNA",  "edge", lambda: transcribe("TTT"), "UUU"),
    ("REVC", "edge", lambda: reverse_complement(""), ""),
    ("REVC", "edge", lambda: reverse_complement("A"), "T"),
    ("FIB",  "edge", lambda: rabbits(1, 3), 1),
    ("FIB",  "edge", lambda: rabbits(2, 3), 1),
    ("FIB",  "edge", lambda: rabbits(6, 1), 8),
    ("GC",   "edge", lambda: best_gc({"x": "GGCC"}), ("x", 100.0)),
    ("GC",   "edge", lambda: best_gc({"a": "ATAT", "b": "GCAT"}), ("b", 50.0)),
    ("HAMM", "edge", lambda: hamming("ACGT", "ACGT"), 0),
    ("HAMM", "edge", lambda: hamming("ACGT", "ACG"), ValueError),
    ("PROT", "edge", lambda: translate("AUGGCC"), "MA"),          #RU: нет стоп-кодона #EN: no stop codon
    ("PROT", "edge", lambda: translate("AUGUAAGCC"), "M"),        #RU: стоп посередине #EN: stop in the middle
    ("PROT", "edge", lambda: translate(""), ""),
    ("SUBS", "edge", lambda: find_motif("AAAA", "AA"), [1, 2, 3]),  #RU: перекрытия #EN: overlaps
    ("SUBS", "edge", lambda: find_motif("ACGT", "TT"), []),

    #RU: --- проверка вспомогательных функций (они уже готовы, должны быть PASS) ---
    #EN: --- helper checks (already implemented, should be PASS) ---
    ("FASTA", "helper", lambda: parse_fasta_text(GC_FASTA), GC_SAMPLE),
    ("DNA",   "format", lambda: run_task("dna", DNA_SAMPLE), "20 12 17 21"),
]


def same(got, expected) -> bool:
    """
    RU: Сравнение результата с ожидаемым: числа с плавающей точкой сравниваются с допуском 0.001
        (как на Rosalind), список и кортеж считаются одинаковыми, если равны элементы.
    EN: Compares a result with the expected value: floats are compared with a 0.001 tolerance
        (as on Rosalind); a list and a tuple are considered equal if their elements are equal.
    """
    if isinstance(expected, float):
        return isinstance(got, (int, float)) and math.isclose(got, expected, abs_tol=1e-3)
    if isinstance(expected, (tuple, list)):
        return (isinstance(got, (tuple, list))
                and len(got) == len(expected)
                and all(same(g, e) for g, e in zip(got, expected)))
    return got == expected


def crosscheck() -> int:
    """
    RU: Необязательная перекрёстная проверка с Biopython на 200 случайных входах для RNA, REVC, PROT и GC.
        Возвращает число задач с расхождениями.
    EN: Optional cross-check against Biopython on 200 random inputs for RNA, REVC, PROT and GC.
        Returns the number of tasks with mismatches.
    """
    try:
        from Bio.Seq import Seq
        from Bio.SeqUtils import gc_fraction
    except ImportError:
        print("\nCROSS  пропущена: нужен Biopython >= 1.80 / skipped: Biopython >= 1.80 is required")
        return 0

    print("\n--- Cross-check vs Biopython (200 random inputs each) ---")
    rng = random.Random(0)  #RU: фиксированное зерно: результат воспроизводим #EN: fixed seed: reproducible results

    def rand_seq(alphabet: str, n: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(n))

    def run(name, mine, ref, make) -> int:
        try:
            for _ in range(200):
                args = make()
                if not same(mine(*args), ref(*args)):
                    print(f"{name:5} FAIL   — расхождение / mismatch on {args!r}")
                    return 1
        except NotImplementedError:
            print(f"{name:5} TODO   — ещё не решена / not solved yet")
            return 0
        except Exception as exc:
            print(f"{name:5} ERROR  — {type(exc).__name__}: {exc}")
            return 1
        print(f"{name:5} PASS")
        return 0

    bad = 0
    bad += run("RNA", transcribe,
               lambda s: str(Seq(s).transcribe()),
               lambda: (rand_seq("ACGT", rng.randint(0, 60)),))
    bad += run("REVC", reverse_complement,
               lambda s: str(Seq(s).reverse_complement()),
               lambda: (rand_seq("ACGT", rng.randint(0, 60)),))
    bad += run("PROT", translate,
               lambda r: str(Seq(r).translate(to_stop=True)),
               lambda: (rand_seq("ACGU", 3 * rng.randint(0, 40)),))
    bad += run("GC", lambda recs: best_gc(recs)[1],
               lambda recs: 100 * max(gc_fraction(s) for s in recs.values()),
               lambda: ({f"r{i}": rand_seq("ACGT", rng.randint(20, 80)) for i in range(5)},))
    return bad


def selftest() -> int:
    """
    RU: Прогоняет все тесты и печатает PASS / TODO / FAIL / ERROR. Ошибка в одном решении
        не останавливает остальные. Возвращает 1, если есть FAIL или ERROR, иначе 0.
    EN: Runs all tests and prints PASS / TODO / FAIL / ERROR. An error in one solution
        does not stop the others. Returns 1 if there is any FAIL or ERROR, otherwise 0.
    """
    ok = bad = todo = 0
    for name, kind, fn, expected in TESTS:
        tag = f"{name:5} {kind:6}"
        expects_exception = isinstance(expected, type)
        try:
            got = fn()
        except NotImplementedError:
            print(f"{tag} TODO   — ещё не решена / not solved yet")
            todo += 1
            continue
        except Exception as exc:
            #RU: Исключение — это PASS только если именно оно и ожидалось
            #EN: An exception is a PASS only if exactly this one was expected
            if expects_exception and isinstance(exc, expected):
                print(f"{tag} PASS")
                ok += 1
            else:
                print(f"{tag} ERROR  — {type(exc).__name__}: {exc}")
                bad += 1
            continue

        if expects_exception:
            print(f"{tag} FAIL   — ожидалось исключение / expected {expected.__name__}")
            bad += 1
        elif same(got, expected):
            print(f"{tag} PASS")
            ok += 1
        else:
            print(f"{tag} FAIL   — получено / got {got!r}, ожидалось / expected {expected!r}")
            bad += 1

    print(f"\nPASS {ok} | FAIL/ERROR {bad} | TODO {todo} | всего / total {len(TESTS)}")
    bad += crosscheck()
    if bad == 0 and todo == 0:
        print("Всё зелёное — можно сдавать на rosalind.info. / All green — ready for rosalind.info.")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    """
    RU: Без аргументов запускает самопроверку; с аргументами <задача> <файл> печатает ответ для сайта.
    EN: With no arguments runs the self-test; with <task> <file> arguments prints the answer for the site.
    """
    if len(argv) == 1:
        return selftest()

    if len(argv) == 3 and argv[1].lower() in TASK_NAMES:
        text = Path(argv[2]).read_text(encoding="utf-8")
        try:
            print(run_task(argv[1].lower(), text))
        except NotImplementedError:
            print("Эта задача ещё не решена / This task is not solved yet")
            return 1
        return 0

    print(f"Usage: python {Path(argv[0]).name} [<task> <dataset_file>]")
    print(f"Tasks: {', '.join(TASK_NAMES)}")
    return 2


#RU: Запускаем только при прямом запуске файла, а не при импорте
#EN: Run only when the file is executed directly, not when imported
if __name__ == "__main__":
    sys.exit(main(sys.argv))
