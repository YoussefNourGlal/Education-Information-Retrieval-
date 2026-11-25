
import csv
import math
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

WTF_FILE = OUTPUT_DIR / "weighted_tf_matrix.csv"
DF_IDF_FILE = OUTPUT_DIR / "df_idf.csv"

TFIDF_FILE = OUTPUT_DIR / "tf_idf_matrix.csv"
LENGTH_FILE = OUTPUT_DIR / "document_vector_lengths.csv"
NORMALIZED_FILE = OUTPUT_DIR / "normalized_tf_idf.csv"

DOC_IDS = [f"d{i}" for i in range(1, 11)]


def load_weighted_tf():
    weighted_tf = {}

    with open(WTF_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            term = row["term"]

            weighted_tf[term] = {
                doc: float(row[doc])
                for doc in DOC_IDS
            }

    return weighted_tf


def load_idf():
    idf = {}

    with open(DF_IDF_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            idf[row["term"]] = float(row["IDF"])

    return idf


def calculate_tfidf(weighted_tf, idf):

    tfidf = {}

    for term in weighted_tf:

        tfidf[term] = {}

        for doc in DOC_IDS:

            tfidf[term][doc] = (
                weighted_tf[term][doc] * idf[term]
            )

    return tfidf


def calculate_document_lengths(tfidf):

    lengths = {}

    for doc in DOC_IDS:

        total = sum(
            tfidf[term][doc] ** 2
            for term in tfidf
        )

        lengths[doc] = math.sqrt(total)

    return lengths


def normalize_tfidf(tfidf, lengths):

    normalized = {}

    for term in tfidf:

        normalized[term] = {}

        for doc in DOC_IDS:

            if lengths[doc] != 0:
                normalized[term][doc] = (
                    tfidf[term][doc] / lengths[doc]
                )
            else:
                normalized[term][doc] = 0

    return normalized


def save_matrix(file_path, matrix):

    with open(file_path, "w", encoding="utf-8", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["term"] + DOC_IDS)

        for term in sorted(matrix):

            writer.writerow(
                [term] + [
                    f"{matrix[term][doc]:.6f}"
                    for doc in DOC_IDS
                ]
            )


def save_lengths(lengths):

    with open(LENGTH_FILE, "w", encoding="utf-8", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["document", "vector_length"])

        for doc in DOC_IDS:

            writer.writerow([
                doc,
                f"{lengths[doc]:.6f}"
            ])


def display_results(tfidf, lengths, normalized):

    print("\n========== TF-IDF MATRIX ==========")

    print("term".ljust(15) + "".join(
        doc.rjust(11) for doc in DOC_IDS
    ))

    for term in sorted(tfidf):

        print(term.ljust(15) + "".join(
            f"{tfidf[term][doc]:.4f}".rjust(11)
            for doc in DOC_IDS
        ))

    print("\n========== DOCUMENT VECTOR LENGTHS ==========")

    for doc in DOC_IDS:
        print(f"{doc}: {lengths[doc]:.6f}")

    print("\n========== NORMALIZED TF-IDF MATRIX ==========")

    print("term".ljust(15) + "".join(
        doc.rjust(11) for doc in DOC_IDS
    ))

    for term in sorted(normalized):

        print(term.ljust(15) + "".join(
            f"{normalized[term][doc]:.4f}".rjust(11)
            for doc in DOC_IDS
        ))


def main():

    print("Starting TF-IDF and Normalization...")

    weighted_tf = load_weighted_tf()
    idf = load_idf()

    tfidf = calculate_tfidf(weighted_tf, idf)

    lengths = calculate_document_lengths(tfidf)

    normalized = normalize_tfidf(tfidf, lengths)

    display_results(tfidf, lengths, normalized)

    save_matrix(TFIDF_FILE, tfidf)
    save_lengths(lengths)
    save_matrix(NORMALIZED_FILE, normalized)

    print("\nAll results saved successfully!")


if __name__ == "__main__":
    main()
