import gc
import json

from torch import Value
from data_sanitizer import sanitize_arxiv_record
from faiss_index import VectorIndex
from rank_fuser import reciprocal_rank_fusion
from tokenizer import tokenizer
from inverted_index import InvertedIndex
from vectored_index import CustomVectorIndex
from semantic_searcher import SemanticSearcher
from lexical_searcher import LexicalSearcher


def run_sanitization_stage(raw_data_path, sanitized_path):
    processed_count = 0
    print("Starting data sanitization...")
    with open(raw_data_path, "r") as infile, open(sanitized_path, "w") as outfile:
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip broken lines

            clean_record = sanitize_arxiv_record(record)
            outfile.write(json.dumps(clean_record) + "\n")
            processed_count += 1
            if processed_count % 50000 == 0:
                print(f"  ...sanitized {processed_count} documents.")
    print(f"Success! Processed and cleaned {processed_count} papers.")


def run_tokenization_stage(sanitized_path, tokenized_path):
    processed_count = 0
    print("Starting Tokenization...")
    with open(sanitized_path, "r") as infile, open(tokenized_path, "w") as outfile:
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip broken lines

            clean_title = tokenizer(record.get("clean_title", ""))
            clean_abstract = tokenizer(record.get("clean_abstract", ""))
            clean_record = {
                "id": record.get("id", ""),
                "categories": record.get("categories", ""),
                "clean_title": clean_title,
                "clean_abstract": clean_abstract,
            }
            outfile.write(json.dumps(clean_record) + "\n")

            processed_count += 1
            if processed_count % 50000 == 0:
                print(f"  ...tokenized {processed_count} documents.")
    print(f"Success! Output File Path:-{tokenized_path} ")


def run_indexation_stage(tokenized_path, indexed_path):
    print("Starting Indexing.....")
    inverted = InvertedIndex()

    processed_count = 0

    with open(tokenized_path, "r") as infile:
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = record.get("id")
            title_tokens = record.get("clean_title", [])
            abstract_tokens = record.get("clean_abstract", [])

            if doc_id:
                inverted.add_document(doc_id, "title", title_tokens)
                inverted.add_document(doc_id, "abstract", abstract_tokens)
            processed_count += 1
            if processed_count % 50000 == 0:
                gc.collect()
                print(f"  ...indexed {processed_count} documents.")
    inverted.save_to_disk(indexed_path)
    print(f"Successfully Indexed:{processed_count} documents")


def run_vector_stage_custom(sanitized_path, vectored_path):
    vector = CustomVectorIndex()
    vector.create_indexes_from_file(sanitized_path)
    vector.save_to_disk(vectored_path)


def run_vector_stage_faiss(sanitized_path, vectored_path):
    vector = VectorIndex()
    vector.train_index_on_file(
        sanitized_path, d=384, m=8, nbits=8, nlist=1024, chunk_size=150000
    )
    vector.save_to_disk(vectored_path)


#   def run_lsh_algo(input_path):
#       b = 20
#       k = 8
#       shingle_set = []
#       doc_ids = []
#       lsh = LSH(b)
#       with open(input_path,'r') as infile:
#           for line in infile:
#               try:
#                   record = json.loads(line)
#               except json.JSONDecodeError :
#                   continue
#               clean_text = record.get("clean_title","")+ " " + record.get("clean_abstract","")
#               doc_ids.append(record.get("id"))
#               shingle_set.append(hsh.shingle(clean_text,k))
#       vocab = hsh.vocab(shingle_set)
#       shingle_1hot = []
#       for shingle in shingle_set:
#           shingle_1hot.append(hsh.one_hot_encoder(shingle,vocab))
#       shingle_1hot = np.stack(shingle_1hot)
#       arr = hsh.min_hash(vocab,100)
#       signatures = []
#       for vector in shingle_1hot:
#           signatures.append(hsh.get_signature(arr,vector))
#       signatures = np.stack(signatures)
#       for signature in signatures:
#           lsh.add_hash(signature)
#       candidate_pairs = lsh.check_candidates(doc_ids)
#       return candidate_pairs


def load_document_with_abstract(metadata_path):
    metadata = {}
    with open(metadata_path, "r") as f:
        for line in f:
            record = json.loads(line)

            metadata[record["id"]] = {
                "title": record.get("clean_title", "No Title"),
                "abstract": record.get("clean_abstract", "No Abstract"),
            }
    return metadata


def load_document_metadata(metadata_path):
    metadata = {}
    with open(metadata_path, "r") as f:
        for line in f:
            record = json.loads(line)

            metadata[record["id"]] = record.get("clean_title", "No Title")
    return metadata


def run_search_lexical(index_path, query, top_k=10):
    searcher = LexicalSearcher(index_path)
    results = searcher.search(query, top_k)
    if not results:
        return None
    else:
        return results


def run_search_cli_lexical(index_path, sanitized_path):
    print("Loading Lexical Search Engine...")

    searcher = LexicalSearcher(index_path)
    doc_metadata = load_document_metadata(sanitized_path)
    print("Search Engine Loaded. Type ':q' to quit/end")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == ":q":
            print("Adios Nuclear Reactor")
            break

        if not query:
            continue

        results = searcher.search(query)

        if not results:
            print("No results found.")
        else:
            print(f"\n--- Top {len(results)} results for '{query}' ---")
            for i, (doc_id, score) in enumerate(results):
                title = doc_metadata.get(doc_id, "Title not found.")
                print(f"{i + 1}. [Score: {score:.4f}] {title} (ID: {doc_id})")


def run_search_cli_semantic(vectored_path, sanitized_path, type="faiss"):
    print("Loading Semantic Search Engine...")

    searcher = SemanticSearcher(vectored_path, type)
    doc_metadata = load_document_metadata(sanitized_path)
    print("Search Engine Loaded. Type ':q' to quit/end")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == ":q":
            print("Adios Nuclear Reactor")
            break

        if not query:
            continue

        results = searcher.search(query)

        if not results:
            print("No results found.")
        else:
            print(f"\n--- Top {len(results)} results for '{query}' ---")
            for i, (doc_id, score) in enumerate(results):
                title = doc_metadata.get(doc_id, "Title not found.")
                print(f"{i + 1}. [Score: {score:.4f}] {title} (ID: {doc_id})")


def run_search_combined_cli(sanitized_path, vectored_path, indexed_path, top_k: int):
    searcher_semantic = SemanticSearcher(vectored_path)
    searcher_lexical = LexicalSearcher(indexed_path)
    doc_metadata = load_document_metadata(sanitized_path)
    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == ":q":
            print("Adios Nuclear Reactor")
            break

        if not query:
            continue

        results_semantic = searcher_semantic.search(query, top_k)
        results_lexical = searcher_lexical.search(query, top_k)
        results = reciprocal_rank_fusion(results_semantic, results_lexical)
        if not results:
            print("No results found.")
        else:
            print(f"\n--- Top {len(results)} results for '{query}' ---")
            for i, (doc_id, score) in enumerate(results):
                title = doc_metadata.get(doc_id, "Title not found.")
                print(f"{i + 1}. [Score: {score:.4f}] {title} (ID: {doc_id})")


def run_search_combined(
    query, top_k, doc_metadata, searcher_semantic, searcher_lexical
):
    results_semantic = searcher_semantic.search(query, top_k)
    results_lexical = searcher_lexical.search(query, top_k)
    results = reciprocal_rank_fusion(results_semantic, results_lexical)

    if not results:
        return "No results found."
    else:
        new_results = []
        for doc_id, score in results:
            doc = doc_metadata.get(doc_id, "Title not found")
            title = doc.get("title")
            abstract = doc.get("abstract")
            new_results.append(
                {"doc_id": doc_id, "title": title, "abstract": abstract, "score": score}
            )
        return new_results


if __name__ == "__main__":
    raw_data_path = "./archive/arxiv-metadata-oai-snapshot.json"
    sanitized_path = "./archive/output-oai.json"
    tokenized_path = "./archive/output-tokenized.json"
    indexed_path = "./archive/output-indexed.pkl"
    vectored_path = "./archive/output-vectored"
    vectored_path_faiss = "./archive/searchis_index"

    # run_sanitization_stage(raw_data_path, sanitized_path)
    # run_tokenization_stage(sanitized_path, tokenized_path)
    # run_indexation_stage(tokenized_path, indexed_path)

    # run_vector_stage(sanitized_path,vectored_path)
    # run_vector_stage_faiss(sanitized_path, vectored_path_faiss)
    # run_search_cli_semantic(vectored_path_faiss, sanitized_path)
    # cand_pairs = run_lsh_algo(sanitized_path)
    # run_search_combined_cli(query, 30)
    while True:
        print("1.Search CLI Lexical")
        print("2.Search CLI Semantic(Faiss)")
        print("3.Search CLI Semantic(Custom)")
        print("4.Search RRF(Combined)")
        print("5.Quit")
        try:
            choice = int(input("Enter Choice:"))
        except ValueError:
            print("Enter Valid Number")
            continue
        match choice:
            case 1:
                run_search_cli_lexical(indexed_path, sanitized_path)
            case 2:
                run_search_cli_semantic(vectored_path_faiss, sanitized_path)
            case 3:
                run_search_cli_semantic(vectored_path, sanitized_path, type="Custom")
            case 4:
                break
            case _:
                print("Enter from given Choices(1-4)")
                continue
