from vectored_index import VectorIndex

class SemanticSearcher:
    def __init__(self, path: str):
        """
        Initializes the SemanticSearcher.

        Args:
            path (str): The path to the saved vector index (.npz file).
        """
        self.vector_index = VectorIndex()
        self.vector_index.load_from_disk(path)
        
    def search(self, query: str, top_k: int = 10):
        """
        Performs a semantic search.

        Args:
            query (str): The user's search query.
            top_k (int): The number of top results to return.

        Returns:
            A list of tuples, where each tuple is (doc_id, score).
        """
        return self.vector_index.search(query, top_k=top_k)
