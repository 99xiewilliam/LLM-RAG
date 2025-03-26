import asyncio
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from models.embedding import EmbeddingModel
from models.llm import AsyncDeepSeekLLM
from models.rerank import Reranker
from database.chroma_client import AsyncChromaClient  # 修改为Chroma客户端
from utils.document_processor import DocumentProcessor
from core.rag_pipeline import AsyncRAGPipeline

async def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

async def initialize_components(config):
    # Initialize embedding model
    embedding_model = EmbeddingModel(
        model_name=config["model"]["embedding"]["model_name"],
        device=config["model"]["embedding"]["device"],
        model_path=config["model"]["embedding"]["model_path"]
    )
    
    # Initialize LLM
    llm = AsyncDeepSeekLLM(
        model_path=config["model"]["llm"]["model_path"],
        tensor_parallel_size=config["model"]["llm"]["tensor_parallel_size"],
        max_concurrent_requests=config["model"]["llm"]["max_concurrent_requests"]
    )
    
    # 初始化Chroma客户端
    vector_store = AsyncChromaClient(
        persist_directory=config["vector_store"]["persist_directory"],
        collection_name=config["vector_store"]["collection_name"],
        embedding_dimension=config["vector_store"]["dim"]
    )
    
    # Initialize reranker
    reranker = Reranker(
        model_name=config["model"]["rerank"]["model_name"],
        device=config["model"]["rerank"]["device"],
        local_models_dir=config["model"]["rerank"]["model_path"]
    )
    
    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        embedding_model=embedding_model
    )
    
    return AsyncRAGPipeline(
        embedding_model=embedding_model,
        llm=llm,
        vector_store=vector_store,  # 使用Chroma客户端
        reranker=reranker,
        document_processor=document_processor,
        top_k=config["retrieval"]["top_k"],
        rerank_top_k=config["retrieval"]["rerank_top_k"]
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("Starting up...")
    try:
        config = await load_config()
        app.state.rag = await initialize_components(config)
        print("RAG system initialized successfully")
        
        yield  # 应用运行的地方
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    finally:
        # 关闭时执行
        print("Shutting down...")
        if hasattr(app.state, "rag"):
            try:
                # 清理其他资源
                # 例如：清理 embedding model 的 CUDA 缓存
                if hasattr(app.state.rag.embedding_model, "model"):
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        print("CUDA cache cleared")
                    except Exception as e:
                        print(f"Error clearing CUDA cache: {e}")
                
                # 清理 LLM 资源
                if hasattr(app.state.rag.llm, "engine"):
                    # vLLM 引擎的清理
                    engine = app.state.rag.llm.engine
                    if hasattr(engine, "stop"):
                        await engine.stop()
                        print("vLLM engine stopped")
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
        # 可以添加其他清理操作

# 创建 FastAPI 应用实例，使用 lifespan
app = FastAPI(
    title="RAG System API",
    lifespan=lifespan
)

class Query(BaseModel):
    text: str
    target_language: Optional[str] = None

class DocumentInput(BaseModel):
    directory_path: Optional[str] = None
    files: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    split_method: Optional[str] = "token"

@app.post("/query")
async def query(query: Query):
    """处理查询请求"""
    try:
        response = await app.state.rag.process_query(
            query.text,
            target_lang=query.target_language
        )
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/add_documents")
async def add_documents(doc_input: DocumentInput):
    """添加文档到知识库"""
    try:
        success = await app.state.rag.add_documents(
            directory_path=doc_input.directory_path,
            files=doc_input.files,
            texts=doc_input.texts,
            split_method=doc_input.split_method
        )
        return {"success": success}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)