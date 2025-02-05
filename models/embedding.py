from typing import List, Union, Optional
import torch
from transformers import AutoModel, AutoTokenizer
import os

class EmbeddingModel:
    def __init__(
        self, 
        model_name: str, 
        device: str = "cuda",
        model_path: Optional[str] = None  # 添加本地模型路径参数
    ):
        """
        初始化嵌入模型
        Args:
            model_name: 模型名称（用于在线下载时的标识）
            device: 设备类型 ('cuda' 或 'cpu')
            model_path: 模型的具体本地路径，如果提供则优先使用
        """
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading embedding model from local path: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True
                ).to(device)
            else:
                print(f"Loading embedding model from HuggingFace: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")

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