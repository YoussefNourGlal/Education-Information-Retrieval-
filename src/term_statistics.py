import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "outputs" / "positional_index.json"
TF_FILE = PROJECT_DIR / "outputs" / "tf_matrix.csv"
WTF_FILE = PROJECT_DIR / "outputs" / "weighted_tf_matrix.csv"
DF_IDF_FILE = PROJECT_DIR / "outputs" / "df_idf.csv"

N = 10
DOC_IDS = [f"d{i}" for i in range(1, N + 1)]


def load_index():
    with open(INDEX_FILE, encoding="utf-8") as f:
        return json.load(f)


def compute_term_statistics(index):
    terms = sorted(index)
    tf, wtf, df, idf = {}, {}, {}, {}

    for term in terms:
        postings = index[term]

        
        tf[term] = {doc: len(postings.get(doc, [])) for doc in DOC_IDS}

        
        wtf[term] = {
            doc: (1 + math.log10(c) if c > 0 else 0)
            for doc, c in tf[term].items()
        }

        
        df[term] = sum(1 for doc in DOC_IDS if tf[term][doc] > 0)
        idf[term] = math.log10(N / df[term])

    return terms, tf, wtf, df, idf


def display_results(terms, tf, df, idf):
    print("\n========== RAW TF ==========")
    print("term".ljust(15) + "".join(d.rjust(6) for d in DOC_IDS))
    for term in terms:
        print(term.ljust(15) + "".join(str(tf[term][d]).rjust(6) for d in DOC_IDS))

    print("\n========== DF / IDF ==========")
    print("term".ljust(15) + "DF".rjust(5) + "IDF".rjust(10))
    for term in terms:
        print(term.ljust(15) + str(df[term]).rjust(5) + f"{idf[term]:.4f}".rjust(10))


def save_results(terms, tf, wtf, df, idf):
    with open(TF_FILE, "w", encoding="utf-8") as f:
        f.write("term," + ",".join(DOC_IDS) + "\n")
        for term in terms:
            f.write(term + "," + ",".join(str(tf[term][d]) for d in DOC_IDS) + "\n")

    with open(WTF_FILE, "w", encoding="utf-8") as f:
        f.write("term," + ",".join(DOC_IDS) + "\n")
        for term in terms:
            f.write(term + "," + ",".join(f"{wtf[term][d]:.4f}" for d in DOC_IDS) + "\n")

    with open(DF_IDF_FILE, "w", encoding="utf-8") as f:
        f.write("term,DF,IDF\n")
        for term in terms:
            f.write(f"{term},{df[term]},{idf[term]:.4f}\n")


if __name__ == "__main__":
    index = load_index()
    terms, tf, wtf, df, idf = compute_term_statistics(index)
    display_results(terms, tf, df, idf)
    save_results(terms, tf, wtf, df, idf)
    print("\nSaved tf_matrix.csv, weighted_tf_matrix.csv, df_idf.csv")
