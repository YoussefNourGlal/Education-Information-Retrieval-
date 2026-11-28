import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "outputs" / "positional_index.json"

PHRASE_QUERIES = ["online learning", "student assessment"]

BOOLEAN_QUERIES = [
    "online learning AND student assessment",
    "online learning AND NOT student assessment",
]


#  Load Positional Index

def load_index():
    with open(INDEX_FILE, encoding="utf-8") as file:
        return json.load(file)


def doc_sort_key(doc_id):
    return int(doc_id[1:])


#  Phrase Query (exact, contiguous)

def phrase_query(index, phrase):

    terms = phrase.lower().split()

    if not terms or any(term not in index for term in terms):
        return {}

    # Documents that contain every term of the phrase
    candidate_docs = set(index[terms[0]])
    for term in terms[1:]:
        candidate_docs &= set(index[term])

    matches = {}

    for doc_id in sorted(candidate_docs, key=doc_sort_key):
        starts = [
            position
            for position in index[terms[0]][doc_id]
            if all(
                (position + offset) in index[terms[offset]][doc_id]
                for offset in range(1, len(terms))
            )
        ]

        if starts:
            matches[doc_id] = starts

    return matches


#  Boolean Query (AND / AND NOT)

def parse_boolean_query(query):
    """
    Splits 'A AND B' or 'A AND NOT B' into (first, operator, second).
    """
    if " AND NOT " in query:
        first, second = query.split(" AND NOT ", 1)
        return first.strip(), "AND NOT", second.strip()

    if " AND " in query:
        first, second = query.split(" AND ", 1)
        return first.strip(), "AND", second.strip()

    raise ValueError(f"Unsupported Boolean query: {query}")


def boolean_query(index, query):

    first, operator, second = parse_boolean_query(query)

    first_docs = set(phrase_query(index, first))
    second_docs = set(phrase_query(index, second))

    if operator == "AND":
        result = first_docs & second_docs
    else:
        result = first_docs - second_docs

    return sorted(result, key=doc_sort_key), first, second, operator


# Display

def display_phrase_results(phrase, matches, phrase_length):
    print(f'\nPhrase query: "{phrase}"')

    if not matches:
        print("    No matching documents.")
        return

    print("    Document   Starting positions   Phrase token ranges")

    for doc_id, starts in matches.items():
        ranges = ", ".join(f"{s}-{s + phrase_length - 1}" for s in starts)
        print(
            f"    {doc_id.ljust(10)} "
            f"{', '.join(map(str, starts)).ljust(20)} {ranges}"
        )


def display_boolean_results(query, docs):
    print(f'\nBoolean query: "{query}"')
    print("    Matching documents:", ", ".join(docs) if docs else "None")

# Main

def main():
    index = load_index()

    print("\n========== PHRASE QUERIES ==========")
    for phrase in PHRASE_QUERIES:
        matches = phrase_query(index, phrase)
        display_phrase_results(phrase, matches, len(phrase.split()))

    print("\n========== BOOLEAN QUERIES ==========")
    for query in BOOLEAN_QUERIES:
        docs, *_ = boolean_query(index, query)
        display_boolean_results(query, docs)


if __name__ == "__main__":
    main()