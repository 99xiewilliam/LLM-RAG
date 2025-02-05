from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

class HybridSearcher:
    @staticmethod
    def combine_scores(
        vector_results: List[Tuple[str, float]],
        query: str,
        vector_weight: float = 0.7
    ) -> List[Tuple[str, float]]:
        if not vector_results:
            return []
            
        texts = [r[0] for r in vector_results]
        vector_scores = np.array([r[1] for r in vector_results])
        
        # BM25 scoring
        tokenized_corpus = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = np.array(bm25.get_scores(query.split()))
        
        # Normalize scores
        max_bm25 = np.max(bm25_scores) if bm25_scores.size > 0 else 1
        normalized_bm25 = bm25_scores / max_bm25 if max_bm25 != 0 else bm25_scores
        
        # Combine scores using numpy operations
        combined_scores = (
            vector_weight * (1 - vector_scores) + 
            (1 - vector_weight) * normalized_bm25
        )
        
        # Create results list
        combined_results = list(zip(texts, combined_scores))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results