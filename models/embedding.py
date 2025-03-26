from typing import List, Union, Optional
import torch
from transformers import AutoModel, AutoTokenizer
import os

class EmbeddingModel:
    def __init__(self, model_name_or_path):
        try:
            # Use sentence-transformers for all models
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name_or_path)
        except Exception as e:
            # Fall back to manual loading if sentence-transformers fails
            try:
                from transformers import AutoModel, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
            except Exception as e2:
                raise RuntimeError(f"Failed to load embedding model: {str(e)} and fallback also failed: {str(e2)}")

            self.device = device
            self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get embeddings
        outputs = self.model(**encoded)
        embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
        return embeddings.cpu().numpy()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)