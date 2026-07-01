from core_engine.inverted_index import InvertedIndex, ID_TO_FIELD
from core_engine.tokenizer import tokenizer
import math


class LexicalSearcher:
    def __init__(self, index_path):
        self.index = InvertedIndex()
        self.index.load_from_disk(index_path)
        self.field_weights = {"title": 1.5, "abstract": 1.0}
        self.avg_field_lengths = self._calculate_avg_field_lengths()
        self.b = 0.75
        self.k1 = 1.5
    def _calculate_avg_field_lengths(self):
        total_lengths = {"title": 0, "abstract": 0}
        field_counts = {"title": 0, "abstract": 0}

        for doc_id, fields in self.index.document_lengths.items():
            for field_name, length in fields.items():
                total_lengths[field_name] += length
                field_counts[field_name] += 1
        avg_lengths = {}
        for field_name, total_len in total_lengths.items():
            if field_counts[field_name] > 0:
                avg_lengths[field_name] = total_len / field_counts[field_name]
            else:
                avg_lengths[field_name] = 0
        return avg_lengths

    def _calc_bm25_scores(self, candidate_docs, query):
        scores = {}

        idf_scores = {token: self._calc_idf(token) for token in query}

        for doc_id, tokens_in_doc in candidate_docs.items():
            doc_score = 0
            for token in query:
                if token in tokens_in_doc:
                    weighted_tf = 0
                    for field_name, tf in tokens_in_doc[token].items():
                        field_len = self.index.document_lengths[doc_id][field_name]
                        avg_len = self.avg_field_lengths[field_name]

                        tf_component = (tf * (self.k1 + 1)) / (
                            tf
                            + self.k1
                            * (1 - self.b + self.b * field_len / avg_len)
                        )

                        weighted_tf += (
                            self.field_weights.get(field_name, 1.0) * tf_component
                        )
                    doc_score += idf_scores[token] * weighted_tf

            if doc_score > 0:
                scores[doc_id] = doc_score
        return scores

    def _calc_idf(self, token):
        N = len(self.index.document_lengths)
        n_q = self.index.get_postings_count(token)
        if n_q == 0:
            return 0
        idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
        return idf

    def search(self, query, top_k=10,b=None,k1=None):
        self.b = self.b if b == None else b
        self.k1 = self.k1 if k1 == None else k1

        # Query to Tokens
        query_tokens = tokenizer(query)

        # Finding all Docs — postings are (doc_int_id, field_id, count) tuples
        candidate_docs = {}

        for token in query_tokens:
            postings = self.index.get_postings(token)
            for doc_int_id, field_id, count in postings:
                doc_id = self.index.int_to_doc_id[doc_int_id]
                field_name = ID_TO_FIELD[field_id]

                if doc_id not in candidate_docs:
                    candidate_docs[doc_id] = {}
                if token not in candidate_docs[doc_id]:
                    candidate_docs[doc_id][token] = {}
                candidate_docs[doc_id][token][field_name] = count

        scores = self._calc_bm25_scores(candidate_docs, query_tokens)

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        return sorted_docs[:top_k]