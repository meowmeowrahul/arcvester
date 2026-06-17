from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import torch

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
        print("Encoding complete.")

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
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # 2. Perform the search
        # util.semantic_search will find the top_k most similar documents
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)
        
        # The result is a list of lists, since we only have one query, we take the first element
        hits = hits[0]

        # 3. Format the results
        results = []
        for hit in hits:
            doc_id = self.doc_ids[hit['corpus_id']]
            score = hit['score']
            results.append((doc_id, score))
            
        return results

    def save_to_disk(self, path: str):
        """
        Saves the vector index to disk in a compressed .npz format.
        """
        print(f"Saving vector index to {path}...")
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Please run create_indexes_from_file first.")
        
        # We'll save as a dictionary for clarity
        data_to_save = {
            'doc_ids': np.array(self.doc_ids),
            'embeddings': self.embeddings.cpu().numpy() # Move to CPU and convert to numpy for saving
        }
        np.savez_compressed(path, **data_to_save)
        print("Successfully saved index.")

    def load_from_disk(self, path: str):
        """
        Loads the vector index from a .npz file.
        """
        print(f"Loading vector index from {path}...")
        data = np.load(path)
        self.doc_ids = data['doc_ids'].tolist()
        
        # Convert the loaded numpy array back to a tensor on the correct device
        loaded_embeddings = torch.from_numpy(data['embeddings']).to(self.device)
        self.embeddings = loaded_embeddings
        print(f"Successfully loaded index with {len(self.doc_ids)} documents.")