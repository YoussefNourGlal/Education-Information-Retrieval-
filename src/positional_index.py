
import re
import json
from pathlib import Path
from pyspark.sql import SparkSession


# ==========================================
# 1. Project Paths
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "positional_index.json"


# ==========================================
# 2. Tokenization Function
# ==========================================

def tokenize(text):
    """
    Convert text into lowercase tokens.
    """

    tokens = re.findall(r"[a-z0-9]+", text.lower())

    return tokens


# ==========================================
# 3. Build Positional Index
# ==========================================

def build_positional_index(spark):

    sc = spark.sparkContext

    # Read all text documents using Spark
    dataset_path = DATASET_DIR.resolve().as_uri() + "/*.txt"

    documents = sc.wholeTextFiles(dataset_path)

    # Generate (term, document_id, position)
    term_positions = documents.flatMap(
        lambda document: [
            (
                (term, re.search(r"(d\d+)\.txt$", document[0], re.I).group(1)),
                position
            )
            for position, term in enumerate(
                tokenize(document[1]), start=1
            )
        ]
    )

    # Group positions by term and document
    grouped_positions = term_positions.groupByKey()

    # Convert positions into lists
    term_document_positions = grouped_positions.map(
        lambda item: (
            item[0][0],
            (
                item[0][1],
                sorted(list(item[1]))
            )
        )
    )

    # Group documents under each term
    positional_index_rdd = term_document_positions.groupByKey()

    # Collect final result
    positional_index = {}

    for term, document_positions in positional_index_rdd.collect():

        positional_index[term] = {
            doc_id: positions
            for doc_id, positions in sorted(
                document_positions,
                key=lambda item: int(item[0][1:])
            )
        }

    return dict(sorted(positional_index.items()))


# ==========================================
# 4. Display Positional Index
# ==========================================

def display_positional_index(positional_index):

    print("\n========== POSITIONAL INDEX ==========\n")

    for term, documents in positional_index.items():

        print(term)

        for doc_id, positions in documents.items():

            print(f"    {doc_id}: {positions}")

        print()


# ==========================================
# 5. Save Positional Index as JSON
# ==========================================

def save_positional_index(positional_index):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            positional_index,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nPositional Index saved to: {OUTPUT_FILE}")


# ==========================================
# 6. Main Function
# ==========================================

def main():

    spark = (
        SparkSession.builder
        .appName("Education_Positional_Index")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        print("Starting Positional Index Construction...")

        positional_index = build_positional_index(spark)

        display_positional_index(positional_index)

        save_positional_index(positional_index)

        print("\nPositional Index Built Successfully!")

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
