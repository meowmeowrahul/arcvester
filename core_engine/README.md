# Core Engine

Low-level indexing and retrieval algorithms powering hybrid search over **3.06 million** arXiv documents. This directory contains no API or UI code — only the raw data structures, scoring functions, and serialization logic that make sub-second search possible on **16 GB RAM**.

---

## Architecture & Data Ingestion

```
Query
  │
  ├──► Lexical Pipeline ──► Inverted Index ──► BM25 Scoring ──┐
  │                                                            ├──► Rank Fusion (RRF) ──► Results
  └──► Semantic Pipeline ──► FAISS / LSH Index ──► ANN Search ─┘
```

Two independent retrieval pipelines — lexical (BM25) and semantic (dense vector) — produce candidate sets that are merged via Reciprocal Rank Fusion.

### Data Ingestion Pipeline
The document indexing and vector ingestion flow follows this structure:

![Data Pipeline](../images/data-pipeline.png)

---

## Inverted Index

**File:** `inverted_index.py`

Memory-optimized BM25 inverted index engineered to index **~3M documents** within a **16 GB RAM** budget.

### Binary-Packed Postings

Each posting is packed into **7 bytes** using `struct.Struct('<IBH')`:

| Field       | Type     | Size    | Range          |
|-------------|----------|---------|----------------|
| `doc_id`    | `uint32` | 4 bytes | 0 – 4.29B docs |
| `field_id`  | `uint8`  | 1 byte  | 0 = title, 1 = abstract |
| `count`     | `uint16` | 2 bytes | 0 – 65,535 TF  |

- **7 bytes/posting** vs **~80 bytes** as native Python tuples — an **~11x memory reduction**
- String `doc_id`s stored **once** in a flat list; postings reference integer offsets

### Disk-Spilling Shard Management

- Monitors process RSS via `psutil` against a configurable ceiling (default: **65% of total RAM**, capped at 10 GB)
- When pressure is detected, packs in-memory postings to binary and spills to temporary `.shard.pkl` files
- At save time, shards are merged as **raw byte concatenation** — no deserialization into Python objects
- Garbage collection triggered every **50,000** documents

### Crash-Safe Persistence

- Writes to a `.tmp` file first, then atomically replaces via `os.replace`
- Shard files are deleted **only after** the final save succeeds
- Pre-save disk space check prevents partial writes

---

## Custom LSH Engine

**File:** `lsh_custom_class.py`

A from-scratch Locality-Sensitive Hashing implementation — **zero external ANN library dependencies**.

### How It Works

1. **Random Hyperplane Projection** — generates random hyperplanes in **384-dimensional** space to produce binary signatures
2. **Banding Strategy** — splits each binary signature into `b` bands of `r` rows
   - Two documents collide in a band if their sub-signature matches exactly
   - Probability of candidate match: `P = 1 - (1 - s^r)^b`, where `s` = cosine similarity
3. **Candidate Retrieval** — query signature is hashed into the same band structure; union of all bucket collisions forms the candidate set
4. **Cosine Verification** — candidates are re-scored using exact cosine similarity on original dense embeddings

### Design Tradeoffs

- **Pros:** Full control over the similarity/recall curve via `b` and `r`; no native library compilation
- **Cons:** O(n) bucket scan at query time; suited for research/benchmarking, not production-scale latency targets

---

## Vector Index (Custom)

**File:** `vectored_index.py`

Dense embedding index built on `sentence-transformers` and the custom LSH engine above.

- **Model:** `all-MiniLM-L6-v2` — **384-dimensional** embeddings
- **Device:** auto-detects GPU via `torch.cuda.is_available()`; falls back to CPU
- **Hashing:** random projection signatures (`hasher.get_signature`) fed into the LSH banding structure
- **Search flow:** encode query → hash to binary signature → retrieve LSH candidates → re-rank by cosine similarity on original embeddings

### Serialization

| Artifact         | Format | Contents                                      |
|------------------|--------|-----------------------------------------------|
| `*_data.npz`     | NumPy  | `doc_ids`, `embeddings`, `hyperplanes`        |
| `*_lsh.pkl`      | Pickle | Full LSH state (buckets, counter)             |

---

## FAISS Index (Production)

**File:** `faiss_index.py`

Production-grade approximate nearest neighbor search using FAISS with Voronoi cell clustering.

### Index Configuration

| Parameter | Value | Purpose                            |
|-----------|-------|------------------------------------|
| `d`       | 384   | Embedding dimensionality           |
| `nlist`   | 1024  | Number of Voronoi cells (clusters) |
| `m`       | 8     | PQ sub-quantizer count             |
| `nbits`   | 8     | Bits per sub-quantizer             |
| `nprobe`  | 10    | Cells searched at query time       |

### Memory-Optimized Streaming Ingestion

1. **Train** on the first chunk (**150K** documents) — learns the Voronoi partitioning
2. **Stream** remaining documents in batches of **150K** — encode, add to index, free memory
3. `doc_ids` converted to `numpy` array (`dtype=U20`) post-ingestion for **~3x RAM savings** over Python lists

### GPU Acceleration

- Auto-detects CUDA; moves trained index to GPU via `faiss.StandardGpuResources`
- Transparently falls back to CPU index on save (`index_gpu_to_cpu`)
- Native FAISS serialization (`faiss.write_index` / `faiss.read_index`) for fast cold-start loads

---

## Lexical Searcher

**File:** `lexical_searcher.py`

Field-weighted BM25 ranking over the binary-packed inverted index.

### Scoring Formula

```
Score(D, Q) = Σ  IDF(qᵢ)  ·  Σ  wⱼ · [ tf · (k₁ + 1) / (tf + k₁ · (1 - b + b · |Fⱼ| / avgFⱼ)) ]
             qᵢ∈Q           j∈fields
```

| Parameter    | Default | Description                        |
|--------------|--------:|-------------------------------------|
| `k1`         |   1.5   | Term frequency saturation          |
| `b`          |   0.75  | Document length normalization      |
| Title weight |   1.5x  | Boosted relevance for title matches|
| Abstract weight | 1.0x | Baseline field weight              |

- **IDF:** `log((N - n_q + 0.5) / (n_q + 0.5) + 1)` — smoothed inverse document frequency
- **Runtime configurable** — `b` and `k1` can be overridden per query

---

## Rank Fusion

**File:** `rank_fuser.py`

Reciprocal Rank Fusion (RRF) merges ranked lists from the semantic and lexical pipelines into a single unified ranking.

```
RRF_Score(d) = Σ  1 / (k + rankᵢ(d))
              i∈{semantic, lexical}
```

- **k = 60** (standard RRF constant) — dampens the influence of high-ranked outliers
- Rank-based (not score-based) — naturally handles score incompatibility between BM25 and cosine similarity
- Returns the top-k documents sorted by fused RRF score

---

## Module Map

| File                    | Purpose                                                    |
|-------------------------|-------------------------------------------------------------|
| `inverted_index.py`     | Binary-packed BM25 inverted index with disk-spilling       |
| `lsh_custom_class.py`   | From-scratch LSH via random hyperplane projections         |
| `vectored_index.py`     | Dense vector index with custom LSH backend                 |
| `faiss_index.py`        | Production FAISS index with streaming ingestion            |
| `lexical_searcher.py`   | Field-weighted BM25 scorer                                 |
| `semantic_searcher.py`  | Unified semantic search interface (FAISS or custom)        |
| `rank_fuser.py`         | Reciprocal Rank Fusion across retrieval pipelines          |
| `tokenizer.py`          | Custom tokenizer with stop-word removal                    |
| `hasher.py`             | Random hyperplane signature generation for LSH             |
| `data_sanitizer.py`     | Raw arXiv JSON cleaning and normalization                  |
| `stopwords.py`          | Stop-word dictionary                                       |
| `main.py`               | Pipeline orchestrator and CLI entry point                  |

---

## Interactive CLI Search Interface

A command-line search interface is built directly into `core_engine/main.py`. This interface allows you to communicate and query each retrieval engine individually (Lexical BM25, Custom Semantic LSH, or production FAISS index).

To run the interactive CLI search utility:
```bash
python3 -m core_engine.main
```
*(Note: to run the said cli, `python3 -m core_engine.py` is also used depending on module loading preferences).*

---

## Key Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Binary struct packing over Python dicts | **~11x** memory reduction; enables 3M-doc index on commodity hardware |
| Shard-and-merge on RSS pressure | Avoids OOM kills during indexing without requiring external databases |
| Atomic file replacement (`os.replace`) | Prevents corrupted index files from partial writes or crashes |
| Custom LSH alongside FAISS | Demonstrates first-principles ANN understanding; used for benchmarking |
| RRF over learned fusion | Zero training data required; robust across heterogeneous score distributions |
