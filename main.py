import asyncio
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncGenerator
from models.embedding import EmbeddingModel
from models.llm_factory import LLMFactory
from models.rerank import Reranker
from database.chroma_client import AsyncChromaClient
from utils.document_processor import DocumentProcessor
from core.rag_pipeline import AsyncRAGPipeline
import json
from fastapi.middleware.cors import CORSMiddleware

async def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

async def initialize_components(config):
    # 初始化 embedding 模型
    embedding_model = EmbeddingModel(
        model_name_or_path=config["model"]["embedding"]["model_path"]
    )
    
    # 使用 LLMFactory 创建 LLM 实例
    llm = LLMFactory.create(config["model"]["llm"])
    
    # 初始化 Chroma 客户端
    vector_store = AsyncChromaClient(
        host=config["vector_store"].get("host"),
        port=config["vector_store"].get("port"),
        collection_name=config["vector_store"]["collection_name"],
        embedding_dimension=config["vector_store"]["dim"],
        persist_directory=config["vector_store"].get("persist_directory")
    )
    
    # 初始化 reranker
    reranker = Reranker(
        model_name=config["model"]["rerank"]["model_name"],
        device=config["model"]["rerank"]["device"],
        local_models_dir=config["model"]["rerank"]["model_path"]
    )
    
    # 初始化文档处理器
    document_processor = DocumentProcessor(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        embedding_model=embedding_model
    )
    
    # 返回 RAG pipeline
    return AsyncRAGPipeline(
        embedding_model=embedding_model,
        llm=llm,
        vector_store=vector_store,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://47.237.107.123"],  # 允许指定的IP
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
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
    
@app.post("/query_stream")
async def query_stream(query: Query):
    """处理查询请求 (流式)"""
    
    async def response_generator() -> AsyncGenerator[str, None]:
        try:
            async for token in app.state.rag.process_query_stream(
                query.text,
                target_lang=query.target_language
            ):
                # 将每个token包装在SSE格式中
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    ) 

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


# Gunicorn 配置
# 可以使用这个来运行: gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:50052
# 注意：由于使用了异步和GPU资源，建议worker数量设为1，避免资源竞争
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50052)