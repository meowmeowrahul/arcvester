import json
import numpy as np
import concurrent.futures
from itertools import product
import sys
import os

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core_engine.lexical_searcher import LexicalSearcher 
worker_queries = None
worker_doc_ids = None
worker_truth_dict = None
worker_searcher = None

def init_worker(q, d, td, index_path):
    global worker_queries, worker_doc_ids, worker_truth_dict, worker_searcher
    worker_queries = q
    worker_doc_ids = d
    worker_truth_dict = td
    worker_searcher = LexicalSearcher(index_path)

def evaluate_parameters(params):
    b, k1 = params
    local_recall_list = []

    for query_doc_id, query_text in zip(worker_doc_ids, worker_queries):
        ground_truth = set(worker_truth_dict[query_doc_id])
        if not ground_truth:
            continue

        result = worker_searcher.search(query_text, top_k=100, b=b, k1=k1)
        retrieved = {doc_id for doc_id, score in result}
        true_positives = len(ground_truth & retrieved)

        recall = true_positives / len(ground_truth)
        local_recall_list.append((recall, k1, b))
    return local_recall_list

if __name__ == '__main__':
    ground_truth_path = "/home/rahul/searchis/benchmarks/dataset/ground_truth.json"
    inverted_indexed_path = "/home/rahul/searchis/benchmarks/dataset/inverted-indices.json"
    raw_data = "/home/rahul/searchis/benchmarks/dataset/raw-data.json"
    queries = []
    doc_ids = []
                    
    print("Loading Ground Truth...")
    with open(ground_truth_path, "r") as f:
        truth_dict = json.load(f)
    doc_id_set = set(truth_dict.keys())
    with open(raw_data,"r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = record.get("id")
            if doc_id in doc_id_set:
                doc_ids.append(doc_id)
                queries.append(record.get("abstract"))
                if len(queries) == len(doc_id_set):
                    break
    b_values = np.arange(0.2, 0.9, 0.1)
    k1_values = np.arange(0.5, 2.5, 0.5)
    parameter_grid = list(product(b_values, k1_values))

    print("Starting grid search on multiple CPU cores...")
    final_recall_list = []

    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_worker,
        initargs=(queries, doc_ids, truth_dict, inverted_indexed_path)
    ) as executor:
        results = executor.map(evaluate_parameters, parameter_grid)
        
        for res in results:
            final_recall_list.extend(res)

    print(f"Finished evaluating {len(final_recall_list)} combinations/queries.")
    
    with open("/home/rahul/searchis/benchmarks/dataset/final_recall_lexical.json","w") as f:
        json.dump(final_recall_list, f)