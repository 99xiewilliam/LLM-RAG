from typing import List, Tuple
import numpy as np
import os
import chromadb
from chromadb.config import Settings

class AsyncChromaClient:
    def __init__(
        self,
        persist_directory: str = "./database/chroma_db",
        collection_name: str = "apec_collection",
        embedding_dimension: int = 1024
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # 确保目录存在
        os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
        
        # 初始化Chroma客户端
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))
        
        # 尝试获取或创建集合
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None  # 我们将提供自己的嵌入
            )
            print(f"Created new collection: {collection_name}")
    
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