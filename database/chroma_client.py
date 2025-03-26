from typing import List, Tuple
import numpy as np
import os
import chromadb
from chromadb.config import Settings

class AsyncChromaClient:
    def __init__(self, host: str, port: int, collection_name: str, embedding_dimension: int, persist_directory: str = None):
        """
        初始化 ChromaDB 客户端
        
        Args:
            host: Chroma服务器主机名
            port: Chroma服务器端口
            collection_name: 集合名称
            embedding_dimension: 嵌入向量维度
            persist_directory: 本地模式的持久化目录 (HTTP模式下不使用)
        """
        import chromadb
        
        # 根据是否提供host和port决定使用HTTP客户端还是本地客户端
        if host and port:
            # 使用HTTP客户端连接到Docker中的Chroma
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # 使用本地客户端
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        self.embedding_dimension = embedding_dimension
    
    async def insert(self, texts: List[str], embeddings: np.ndarray) -> bool:
        try:
            # 确保embeddings是正确的形状
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # 为文档生成唯一ID
            start_id = self.collection.count() if self.collection.count() else 0
            ids = [str(i + start_id) for i in range(len(texts))]
            
            # 插入文档
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),  # 转为列表
                ids=ids,
                metadatas=[{"source": "document"} for _ in texts]
            )
            
            return True
        except Exception as e:
            print(f"Error inserting documents into Chroma: {e}")
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
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "distances"]
            )
            
            # 返回文本和距离
            documents = results["documents"][0]
            distances = results["distances"][0]
            
            return [(doc, dist) for doc, dist in zip(documents, distances)]
        except Exception as e:
            print(f"Error searching documents in Chroma: {e}")
            return []
    
    # 不需要显式保存，Chroma会自动持久化
    def save_index(self):
        """持久化索引 (Chroma自动完成)"""
        # Chroma会自动保存，无需显式操作
        print("Chroma automatically persists data")
        pass