import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import os

class Reranker:
    # 定义本地模型路径映射
    LOCAL_MODEL_PATHS = {
        "BAAI/bge-reranker-large": "/path/to/local/bge-reranker-large",
        "BAAI/bge-reranker-base": "/path/to/local/bge-reranker-base",
        # 可以添加更多模型路径映射
    }
    
    def __init__(self, model_name: str, device: str = "cuda", local_models_dir: str = None):
        """
        初始化重排序模型
        Args:
            model_name: 模型名称或本地路径
            device: 设备类型 ('cuda' 或 'cpu')
            local_models_dir: 本地模型根目录，如果提供，将在此目录下查找模型
        """
        # 确定模型路径
        if local_models_dir:
            # 如果提供了本地模型目录，优先使用本地路径
            # model_path = os.path.join(local_models_dir, os.path.basename(model_name))
            model_path = local_models_dir
        else:
            # 否则查找预定义的本地路径映射
            model_path = self.LOCAL_MODEL_PATHS.get(model_name, model_name)

        # 检查本地路径是否存在
        if os.path.exists(model_path):
            print(f"Loading model from local path: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True  # 强制使用本地文件
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            ).to(device)
        else:
            # 如果本地路径不存在，发出警告并尝试从在线加载
            print(f"Warning: Local model not found at {model_path}, attempting to download...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def compute_score(self, pairs: List[List[str]]) -> List[float]:
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        scores = self.model(**features).logits.squeeze(-1)
        scores = scores.cpu().numpy().tolist()
        
        return scores if isinstance(scores, list) else [scores]

    def rerank(self, query: str, texts: List[str], top_k: int) -> List[Tuple[str, float]]:
        pairs = [[query, text] for text in texts]
        scores = self.compute_score(pairs)
        
        scored_texts = list(zip(texts, scores))
        scored_texts.sort(key=lambda x: x[1], reverse=True)
        
        return scored_texts[:top_k]