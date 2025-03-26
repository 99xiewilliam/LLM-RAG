from typing import List, Tuple, Optional, AsyncGenerator, Optional
import numpy as np
import asyncio
from models.embedding import EmbeddingModel
from models.deepseek_llm import AsyncDeepSeekLLM
from models.openai_llm import AsyncOpenAILLM
from database.chroma_client import AsyncChromaClient
from database.hybrid_search import HybridSearcher
from models.rerank import Reranker
from utils.document_processor import DocumentProcessor

class AsyncRAGPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm: AsyncOpenAILLM,
        vector_store: AsyncChromaClient,
        reranker: Reranker,
        document_processor: DocumentProcessor,
        top_k: int = 5,
        rerank_top_k: int = 3
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.vector_store = vector_store  # 改名
        self.reranker = reranker
        self.document_processor = document_processor
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.hybrid_searcher = HybridSearcher()

    async def hybrid_search(self, query: str) -> List[Tuple[str, float]]:
        try:
            # Vector search
            query_embedding = self.embedding_model.encode(query)
            if isinstance(query_embedding, np.ndarray):
                if len(query_embedding.shape) == 1:
                    query_embedding = query_embedding.reshape(1, -1)
            
            vector_results = await self.vector_store.search(
                query_embedding,
                self.top_k
            )
            
            # 确保返回结果不为空
            if not vector_results:
                print("No results found in vector search")
                return []
            
            # 使用 numpy 的比较操作
            combined_results = self.hybrid_searcher.combine_scores(vector_results, query)
            return combined_results
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []

    async def rerank(self, query: str, texts: List[str]) -> List[Tuple[str, float]]:
        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            None,
            self.reranker.rerank,
            query,
            texts,
            self.rerank_top_k
        )
        return reranked
    
    async def translate_to_english(self, query: str) -> str:
        """将查询转换为英文"""
        prompt = """请将以下文本翻译成英文，只需要返回翻译结果，不需要任何解释：

文本：{query}

英文翻译："""
        
        translated = await self.llm.generate(prompt.format(query=query))
        return translated.strip()

    async def process_query(
    self,
    query: str,
    target_lang: Optional[str] = None
) -> str:
        """处理查询"""
        try:
            # 1. Hybrid search
            search_results = await self.hybrid_search(query)
            
            if not search_results:
                return "抱歉，没有找到相关的信息。"
            
            # 2. Rerank top results
            texts = [r[0] for r in search_results]
            reranked_results = await self.rerank(query, texts)
            
            # 3. Combine context
            context = "\n".join([f"{i+1}. {text}" for i, (text, _) in enumerate(reranked_results)])
            
            # 4. Generate answer
            response = await self.llm.generate_with_context(query, context, target_lang)
            return response if response else "抱歉，生成回答时出现错误。"
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return f"抱歉，处理您的问题时出现错误: {error_msg}"
    
    async def process_query_stream(
        self,
        query: str,
        target_lang: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """处理查询（流式输出）"""
        try:
            # 1. Hybrid search
            search_results = await self.hybrid_search(query)
            
            if not search_results:
                prompt = f"""{query}

                {f'请用{target_lang}回答' if target_lang else '请用与提问相同的语言回答'}。"""
                # 流式生成
                async for token in self.llm.generate_stream(prompt):
                    yield token
                return
            
            # 2. Rerank top results
            texts = [r[0] for r in search_results]
            reranked_results = await self.rerank(query, texts)
            
            # 3. Combine context
            context = "\n".join([f"{i+1}. {text}" for i, (text, _) in enumerate(reranked_results)])
            
            # 4. Generate answer (stream)
            async for token in self.llm.generate_with_context_stream(query, context, target_lang):
                yield token
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            yield f"抱歉，处理您的问题时出现错误: {error_msg}" 

# from typing import List, Tuple, Optional
# import numpy as np
# import asyncio
# from models.embedding import EmbeddingModel
# from models.llm import AsyncDeepSeekLLM
# from database.faiss_client import AsyncFAISSClient
# from database.hybrid_search import HybridSearcher
# from models.rerank import Reranker
# from utils.document_processor import DocumentProcessor

# class AsyncRAGPipeline:
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         llm: AsyncDeepSeekLLM,
#         vector_store: AsyncFAISSClient,  # 改为 FAISS 客户端
#         reranker: Reranker,
#         document_processor: DocumentProcessor,
#         top_k: int = 5,
#         rerank_top_k: int = 3
#     ):
#         self.embedding_model = embedding_model
#         self.llm = llm
#         self.vector_store = vector_store  # 改名
#         self.reranker = reranker
#         self.document_processor = document_processor
#         self.top_k = top_k
#         self.rerank_top_k = rerank_top_k
#         self.hybrid_searcher = HybridSearcher()

#     async def hybrid_search(self, query: str) -> List[Tuple[str, float]]:
#         # Vector search
#         query_embedding = self.embedding_model.encode(query)
#         vector_results = await self.vector_store.search(query_embedding, self.top_k)
        
#         # 其他代码保持不变
#         return self.hybrid_searcher.combine_scores(vector_results, query)

#     async def rerank(self, query: str, texts: List[str]) -> List[Tuple[str, float]]:
#         loop = asyncio.get_event_loop()
#         reranked = await loop.run_in_executor(
#             None,
#             self.reranker.rerank,
#             query,
#             texts,
#             self.rerank_top_k
#         )
#         return reranked
    
#     async def translate_to_english(self, query: str) -> str:
#         """将查询转换为英文"""
#         prompt = """请将以下文本翻译成英文，只需要返回翻译结果，不需要任何解释：

# 文本：{query}

# 英文翻译："""
        
#         translated = await self.llm.generate(prompt.format(query=query))
#         return translated.strip()

#     async def process_query(
#         self, 
#         query: str,
#         target_lang: Optional[str] = None
#     ) -> str:
#         try:
#             # 1. 将查询转换为英文（用于检索）
#             english_query = query
#             if not query.isascii():  # 如果查询包含非ASCII字符（可能是非英文）
#                 english_query = await self.translate_to_english(query)
#                 print(f"Translated query: {english_query}")  # 用于调试

#             # 1. Hybrid search
#             search_results = await self.hybrid_search(english_query)
            
#             if not search_results:
#                 # 如果没有找到相关文档，直接用 LLM 回答
#                 prompt = f"""{query}

# {f'请用{target_lang}回答' if target_lang else '请用与提问相同的语言回答'}。"""
#                 return await self.llm.generate(prompt)
            
#             # 2. Rerank top results
#             texts = [r[0] for r in search_results]
#             reranked_results = await self.rerank(english_query, texts)
            
#             # 3. Combine context
#             context = "\n".join([f"{i+1}. {text}" for i, (text, _) in enumerate(reranked_results)])
            
#             # 4. Generate answer
#             return await self.llm.generate_with_context(query, context, target_lang)
            
#         except Exception as e:
#             error_msg = f"Error processing query: {str(e)}"
#             print(error_msg)
#             return f"抱歉，处理您的问题时出现错误: {error_msg}"

    async def add_documents(
    self,
    directory_path: Optional[str] = None,
    files: Optional[List[str]] = None,
    texts: Optional[List[str]] = None,
    split_method: str = "token"
) -> bool:
        """添加文档到知识库"""
        try:
            # 1. 加载文档
            documents = self.document_processor.load_documents(
                directory_path=directory_path,
                files=files,
                texts=texts
            )
            
            # 2. 处理文档并分块
            chunks = self.document_processor.process_documents(
                documents,
                split_method=split_method
            )
            
            # 3. 生成嵌入
            embeddings = self.embedding_model.encode(chunks)
            
            # 4. 存入 Chroma
            success = await self.vector_store.insert(chunks, embeddings)
            
            if success:
                print(f"Successfully added {len(chunks)} documents to FAISS index")
            else:
                print("Failed to add documents to FAISS index")
                
            return success
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False