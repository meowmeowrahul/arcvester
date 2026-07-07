import sys
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Get the absolute path of the parent directory
parent_dir = os.path.abspath("../..")

# Add the parent directory to Python's module search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import core_engine.main as SearchEngine
from core_engine.semantic_searcher import SemanticSearcher
from core_engine.lexical_searcher import LexicalSearcher

app = FastAPI(title="arXiv Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
engine = SearchEngine  # Loads FAISS, BM25, and Data
indexed_path = "/home/rahul/searchis/core_engine/archive/output-indexed.pkl"
vectored_path_faiss = "/home/rahul/searchis/core_engine/archive/searchis_index"
sanitized_path = "/home/rahul/searchis/core_engine/archive/output-oai.json"
searcher_semantic = SemanticSearcher(vectored_path_faiss)
searcher_lexical = LexicalSearcher(indexed_path)

doc_metadata = engine.load_document_with_abstract(sanitized_path)


@app.get("/search")
async def search_arxiv(query: str, request: Request):
    body = await request.json()
    top_k = body.get("top_k") if body.get("top_k") is not None else 10
    # This calls RRF fusion method
    results = engine.run_search_combined(
        query, top_k, doc_metadata, searcher_semantic, searcher_lexical
    )
    return {"query": query, "results": results}
