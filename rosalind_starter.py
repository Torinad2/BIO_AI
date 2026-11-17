
dna = "GCAGTTGCGATCTGGCACCCATCTTAGGGGGTTCCCCGCTACAATCAGGCACCGTTGATGGGTCGCCCGAGCTACTCTTCAATCCTCCCAAGGAGGTACAATGTCGCGCCATAGGCTCGTAGAATACGACCACCTCGTCCACGACGGTAGACGAGCCGCAACTTCGGCGGTATAATGGATAATCCTCATTCTTATCCGCCTTCAACTTATCGTGAATACTTGAAGCTCCGTTTCTACCCAGTGTTCAAAAAACGGTCATAGTGCGGATTATCACGCAACCGTGCAAAAACTACCTGACGCTCCCCCCTCTCAGGAGGGTAATTTAGCGAAGCATGACTCCCCTCAACCCTAGCCACATTCTATTAGCATAGCAGGCGGCAGACAGCGTGAAGCCATACGGATACGTGCTCAGTGCTAGGTAAATCACTGGCTGCGAGTGGAAAAATTGCTAAGAGCTCCTTCACCACCCAATCTCCCCTCTAAAAAGCGCGCTCTAATCAATCCGGCTTTCCGTCTTCGGAGATCTACCGAAAGAACGCGTGCTTCGGCGAAAACTCGGAACGAATGGATAATGAAATTTTTCCGGTCGACCCAAATTTAATATCGAGCTTGCGGCGCGACGGAGTATAGAGACAAGTCACCTATGTATGACAAAGGCTGTTCTCGCATTGTGTCTGTCGACACGTCTTAACCAACCACCAGATCATTAAATTAGAGGCGTACGGTGTCGCGTATTCGGCAGACTCGCCGATCTCCCCCGGGTACGGAGTTGCGCTGTCAGCAGAGCGAGTTCAGACCCT"
# --- 1. Counting DNA Nucleotides
def nucleotids_count(seq) -> tuple[int, int, int, int]:
    return (seq.count("A"), seq.count("C"), seq.count("G"), seq.count("T"))


# --- 2. Transcribing DNA into RNA
def transcribe_dna(seq) -> str:
    return seq.replace("T", "U")

# --- 3. Complementing a Strand of DNA

def complement_dna(seq) -> str:
    complement = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C"
    }

    reverse = dna[::-1]

    return "".join(complement[n] for n in reverse)

# --- 4. Rabbits and Recurrence Relations
def rabbits(n, k) -> int:
    young_rabbits = 1
    adult_rabbits = 0

    for _ in range(n-1):
        new_born = adult_rabbits * k
        adult_rabbits += young_rabbits
        young_rabbits = new_born

    return adult_rabbits + young_rabbits

# --- 5. Computing GC Content
def compute_gc_content(seq):
    pass

print(rabbits(5, 3))
print(list(range(5-1)))