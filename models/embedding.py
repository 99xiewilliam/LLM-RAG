from typing import List, Union, Optional
import torch
from transformers import AutoModel, AutoTokenizer
import os

class EmbeddingModel:
    def __init__(self, model_name_or_path=None, model_name=None):
        # Use model_name if provided, otherwise use model_name_or_path
        model_path = model_name if model_name is not None else model_name_or_path
        
        try:
            if "bge" in model_path.lower():
                # For BGE models, we need to explicitly set the model architecture
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                
                # Create a config with explicit model_type
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                config.model_type = "bert"  # BGE models are BERT-based
                
                # Load with the updated config
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_path,
                    config=config,
                    trust_remote_code=True
                )
            else:
                # Original code for other embedding models
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")

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