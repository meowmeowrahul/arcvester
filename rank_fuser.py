def reciprocal_rank_fusion(semantic_results, lexical_results, k=60):
    """
    Fuses results from multiple retrievers using RRF.
    Expects inputs to be lists of (doc_id, score) tuples, already sorted by score descending.
    """
    fused_scores = {}

    # Process Semantic Results
    for rank, (doc_id, raw_score) in enumerate(semantic_results, start=1):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0.0
        fused_scores[doc_id] += 1.0 / (k + rank)

    # Process Lexical (BM25) Results
    for rank, (doc_id, raw_score) in enumerate(lexical_results, start=1):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0.0
        fused_scores[doc_id] += 1.0 / (k + rank)

    # Sort the final dictionary by the fused RRF score in descending order
    final_scores = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    return final_scores[:k]
