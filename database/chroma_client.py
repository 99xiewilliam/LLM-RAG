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
           top_k: int = 5,
           max_tokens: int = 2000  # 添加token限制参数
       ) -> List[Tuple[str, float]]:
       try:
           # 执行搜索
           results = self.collection.query(
               query_embeddings=query_embedding.tolist(),
               n_results=top_k * 2,  # 检索更多结果，以便过滤后仍有足够数据
               include=["documents", "distances"]
           )
           
           # 提取文本和距离
           documents = results["documents"][0]
           distances = results["distances"][0]
           
           # 清理并限制token数量
           cleaned_results = []
           total_tokens = 0
           def clean_text(text):
            """清理文本中的乱码和向量数据"""
            # 移除可能的向量数据（包含大量特殊字符的部分）
            import re
            # 移除类似向量或编码的内容
            cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,?!;:()\[\]{}"\'-]+', ' ', text)
            # 移除多余空格
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned
           for doc, dist in zip(documents, distances):
               # 清理文本
               cleaned_doc = clean_text(doc)
               
               # 估算token数
               estimated_tokens = len(cleaned_doc.split()) * 1.5
               
               # 如果添加此文档会超出限制，跳过
               if total_tokens + estimated_tokens > max_tokens:
                   break
               
               # 添加到结果中
               cleaned_results.append((cleaned_doc, dist))
               total_tokens += estimated_tokens
               
               # 如果已达到原始top_k要求，停止
               if len(cleaned_results) >= top_k:
                   break
           
           return cleaned_results
       except Exception as e:
           print(f"Error searching documents in Chroma: {e}")
           return []

        
    
    # 不需要显式保存，Chroma会自动持久化
    def save_index(self):
        """持久化索引 (Chroma自动完成)"""
        # Chroma会自动保存，无需显式操作
        print("Chroma automatically persists data")
        pass

    