from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import json
import faiss
import gc


class VectorIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading sentence transformer model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device, backend="onnx")
        self.doc_ids = []
        self.embeddings = None
        self.index = None
        print("Model loaded.")

    def train_index_on_file(
        self,
        input_path: str,
        d: int,
        m: int,
        nbits: int,
        nlist: int,
        chunk_size: int = 50000,
    ):
        assert d % m == 0
        print(f"Starting memory-optimized vector index creation from {input_path}...")

        # --- STAGE 1: Gather a sampling chunk to TRAIN the IVFPQ Quantizer ---
        train_texts = []
        sample_count = 0

        with open(input_path, "r") as infile:
            for line in infile:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                full_text = (
                    record.get("clean_title", "")
                    + ". "
                    + record.get("clean_abstract", "")
                )
                train_texts.append(full_text.strip())
                sample_count += 1
                if sample_count >= chunk_size:  # Use first chunk for training
                    break

        print(f"Encoding training sample of {len(train_texts)} documents...")
        train_embeddings = self.model.encode(
            train_texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=256,
        ).astype(np.float32)

        # Initialize FAISS Index
        quantizer = faiss.IndexFlatIP(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

        print("Training FAISS index on sample...")
        cpu_index.train(train_embeddings)

        # Clean up training arrays from memory immediately
        del train_texts
        del train_embeddings
        gc.collect()

        # Move to GPU if available
        if self.device == "cuda":
            print("Moving trained FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        # --- STAGE 2: Stream through the file in blocks and ADD to index ---
        print("Streaming remaining documents incrementally into FAISS...")
        current_chunk_texts = []

        with open(input_path, "r") as infile:
            for line in infile:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.doc_ids.append(record.get("id"))
                full_text = (
                    record.get("clean_title", "")
                    + ". "
                    + record.get("clean_abstract", "")
                )
                current_chunk_texts.append(full_text.strip())

                # Once our chunk buffer fills up, process it and wipe it
                if len(current_chunk_texts) == chunk_size:
                    chunk_embeddings = self.model.encode(
                        current_chunk_texts,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                        batch_size=256,
                    ).astype(np.float32)

                    self.index.add(chunk_embeddings)
                    print(
                        f"  ...indexed {len(self.doc_ids)} documents total (RAM Safe)."
                    )

                    # Clear memory buffers
                    current_chunk_texts = []
                    del chunk_embeddings
                    gc.collect()

            # Process any remaining leftovers
            if current_chunk_texts:
                chunk_embeddings = self.model.encode(
                    current_chunk_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=256,
                ).astype(np.float32)
                self.index.add(chunk_embeddings)
                del chunk_embeddings
                gc.collect()

        # Convert doc_ids to numpy array for memory efficiency (~3x less RAM than a Python list)
        self.doc_ids = np.array(self.doc_ids, dtype="U20")

        # Set nprobe for decent recall on IVF indexes
        self.index.nprobe = 10

        print(f"Index successfully prepared with {len(self.doc_ids)} elements!")

    def search(self, query: str, k: int = 10):
        # Ensure nprobe is set for IVF indexes
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 10

        query_embedding = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)

        # D = Distances (Scores), I = Indices
        D, I = self.index.search(query_embedding, k)

        results = []

        # Iterate through both the indices and the distances
        for i, idx in enumerate(I[0]):
            if idx != -1:  # -1 means FAISS didn't find enough neighbors
                doc_id = self.doc_ids[idx]
                score = D[0][i]  # Extract the corresponding score
                results.append((doc_id, score))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def save_to_disk(self, path: str):
        """
        Saves the doc_ids/embeddings to .npz and the FAISS index to a native .index file.
        """
        print(f"Saving vector index to {path}...")
        if self.index is None:
            raise RuntimeError("No FAISS index to save. Run train_index_on_file first.")

        # 1. Save standard Python/NumPy data
        np.savez_compressed(f"{path}_data.npz", doc_ids=np.array(self.doc_ids))

        # 2. Save the FAISS index natively
        if self.device == "cuda":
            print("Moving index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, f"{path}_faiss.index")
        else:
            faiss.write_index(self.index, f"{path}_faiss.index")

        print("Successfully saved index.")

    def load_from_disk(self, path: str):
        """
        Loads the doc_ids/embeddings and reconstructs the FAISS index.
        """
        print(f"Loading vector index from {path}...")

        # 1. Load standard Python/NumPy data
        data = np.load(f"{path}_data.npz", allow_pickle=True)
        self.doc_ids = data["doc_ids"].tolist()

        # 2. Load the FAISS index natively
        cpu_index = faiss.read_index(f"{path}_faiss.index")

        # 3. Push it back to the GPU if available
        if self.device == "cuda":
            print("Moving loaded index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        print(f"Successfully loaded index with {len(self.doc_ids)} documents.")
