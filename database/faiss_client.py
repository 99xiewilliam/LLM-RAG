from typing import List, Tuple
import numpy as np
import faiss
import pickle
import os

class AsyncFAISSClient:
    def __init__(
        self,
        dim: int = 1024,
        index_type: str = "L2",
        save_path: str = "./faiss_index"
    ):
        self.dim = dim
        self.save_path = save_path
        self.texts = []  # 存储文本
        
        # 创建或加载索引
        if os.path.exists(save_path):
            self.load_index()
        else:
            if index_type == "L2":
                self.index = faiss.IndexFlatL2(dim)
            else:
                self.index = faiss.IndexFlatIP(dim)  # 内积，用于余弦相似度
    
    async def insert(self, texts: List[str], embeddings: np.ndarray) -> bool:
        try:
            self.texts.extend(texts)
            self.index.add(embeddings.astype(np.float32))
            self.save_index()
            return True
        except Exception as e:
            print(f"Error inserting documents: {e}")
            return False
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        try:
            # 确保查询向量是正确的形状
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # 搜索最近邻
            distances, indices = self.index.search(
                query_embedding.astype(np.float32),
                top_k
            )
            
            # 返回文本和距离
            results = [
                (self.texts[idx], dist) 
                for idx, dist in zip(indices[0], distances[0])
            ]
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def save_index(self):
        """保存索引和文本"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.save_path}.index")
        with open(f"{self.save_path}.texts", 'wb') as f:
            pickle.dump(self.texts, f)
    
    def load_index(self):
        """加载索引和文本"""
        self.index = faiss.read_index(f"{self.save_path}.index")
        with open(f"{self.save_path}.texts", 'rb') as f:
            self.texts = pickle.load(f)