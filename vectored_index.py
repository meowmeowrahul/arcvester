from sentence_transformers import SentenceTransformer, util
import hasher
from lsh_custom_class import LSH
import json
import numpy as np
import torch
import pickle

class VectorIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the VectorIndex.
        
        Args:
            model_name (str): The name of the sentence-transformer model to use.
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.doc_ids = []
        self.embeddings = None
        self.lsh = LSH(b=10)
        self.candidates = ()
        self.num_dimensions = 384
        self.num_bits = 100
        self.hyperplanes = np.random.randn(self.num_bits,self.num_dimensions)
        print("Model loaded.")

    def create_indexes_from_file(self, input_path: str):
        """
        Creates vector embeddings for documents from a JSONL file.
        Each document is a combination of its title and abstract.
        """
        print(f"Starting vector index creation from {input_path}...")
        documents_to_encode = []
        processed_count = 0
        
        with open(input_path, 'r') as infile:
            for line in infile:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # We combine title and abstract for a richer embedding
                title = record.get('clean_title', '')
                abstract = record.get('clean_abstract', '')
                # A common technique is to add a separator
                full_text = title + ". " + abstract
                
                self.doc_ids.append(record.get('id'))
                documents_to_encode.append(full_text.strip())
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  ...prepared {processed_count} documents for encoding.")

        # The model's encode method is highly optimized for batch processing
        print(f"Encoding {len(documents_to_encode)} documents in a single batch...")
        self.embeddings = self.model.encode(
            documents_to_encode, 
            show_progress_bar=True,
            convert_to_tensor=True # Optimized for util.semantic_search
        )
        #minhash = hasher.min_hash(np.arange(len(self.embeddings)),resolution=100)  
        
        signatures = [hasher.get_signature(self.hyperplanes,embedding) for embedding in self.embeddings ]

        for i,sig in enumerate(signatures):
            self.lsh.add_hash(sig)

        self.candidates = self.lsh.check_candidates(self.doc_ids)
        print(self.candidates)

                
        print("\n--- Testing Candidate Similarities ---")
        # Ensure we don't go out of bounds if there are fewer than 3 candidates
        num_to_check = min(3, len(self.candidates)) 
        candidate_list = list(self.candidates)
        
        for i in range(num_to_check-1):
            # A candidate pair is a tuple, so we unpack it directly
            doc_id_1 = candidate_list[i][0]
            doc_id_2 = candidate_list[i][1]
            
            print(f"DOC_IDs:\n{doc_id_1} & {doc_id_2}")
           #print(self.doc_ids[0])
           #print(type(self.doc_ids[0]))
            # SAFE INDEXING: Find the actual index of the doc_id in your list
            idx_1 = self.doc_ids.index(doc_id_1)
            idx_2 = self.doc_ids.index(doc_id_2)
            
            # USE ORIGINAL EMBEDDINGS, NOT SIGNATURES
            # We pull them from the GPU/CPU tensor and convert to standard numpy arrays
            vec_1 = self.embeddings[idx_1].cpu().numpy()
            vec_2 = self.embeddings[idx_2].cpu().numpy()
            
            print("cosine_similarity:")
            print(self.lsh.cosine_similarity(vec_1, vec_2))
            print("-" * 30)

    def search(self, query: str, top_k: int = 10):
        """
        Performs a semantic search for a given query.

        Args:
            query (str): The user's search query.
            top_k (int): The number of top results to return.

        Returns:
            A list of tuples, where each tuple is (doc_id, score).
        """
        if self.embeddings is None:
            raise RuntimeError("Index is not built or loaded. Cannot perform search.")

        # 1. Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        signature = hasher.get_signature(self.hyperplanes,query_embedding)
        
        # 2. Perform the search
        # util.semantic_search will find the top_k most similar documents
       #hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)
        candidate_indices = self.lsh.get_candidates(signature)
        # The result is a list of lists, since we only have one query, we take the first element
       #hits = hits[0]
        # 3. Format the results
        if not candidate_indices:
            return []
        results = []
        embeddings_np = self.embeddings.cpu().numpy() if torch.is_tensor(self.embeddings)  else self.embeddings
       #print(embeddings_np)
        for idx in candidate_indices:
            doc_vector = embeddings_np[idx]
            
            score = self.lsh.cosine_similarity(query_embedding,doc_vector)
            results.append((self.doc_ids[idx],score))
        results.sort(key=lambda x:x[1],reverse = True)

        return results[:top_k]

    def save_to_disk(self, path: str):
        """
        Saves the embeddings/hyperplanes to .npz and the LSH state to .pkl
        """
        print(f"Saving vector index to {path}...")
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save.")
        
        # 1. Save tensors and arrays
        data_to_save = {
            'doc_ids': np.array(self.doc_ids),
            'embeddings': self.embeddings.cpu().numpy() if torch.is_tensor(self.embeddings) else self.embeddings,
            'hyperplanes': self.hyperplanes 
        }
        np.savez_compressed(f"{path}_data.npz", **data_to_save)
        
        # 2. Save the LSH object state (the buckets and counter)
        with open(f"{path}_lsh.pkl", "wb") as f:
            pickle.dump(self.lsh, f)
            
        print("Successfully saved index.")

    def load_from_disk(self, path: str):
        """
        Loads both the tensor data and the stateful LSH engine.
        """
        print(f"Loading vector index from {path}...")
        
        # 1. Load numpy arrays
        data = np.load(f"{path}_data.npz", allow_pickle=True)
        self.doc_ids = data['doc_ids'].tolist()
        self.embeddings = torch.from_numpy(data['embeddings']).to(self.device)
        self.hyperplanes = data['hyperplanes']
        
        # 2. Load LSH state
        with open(f"{path}_lsh.pkl", "rb") as f:
            self.lsh = pickle.load(f)
            
        print(f"Successfully loaded index with {len(self.doc_ids)} documents.")
