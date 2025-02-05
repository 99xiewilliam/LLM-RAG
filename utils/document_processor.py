from typing import List, Optional, Dict
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.schema import MetadataMode
import re

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        embedding_model = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 通用文本分割器
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=embedding_model.tokenizer if embedding_model else None
        )
        
        # 句子分割器
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_documents(
        self,
        directory_path: Optional[str] = None,
        files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None
    ) -> List[Document]:
        """加载各种格式的文档"""
        try:
            if directory_path:
                reader = SimpleDirectoryReader(
                    input_dir=directory_path,
                    recursive=True,
                    exclude_hidden=True,
                    filename_as_id=True,
                    required_exts=[".pdf", ".md", ".txt", ".docx"],
                )
                documents = reader.load_data()
            elif files:
                reader = SimpleDirectoryReader(
                    input_files=files,
                    exclude_hidden=True,
                    filename_as_id=True,
                )
                documents = reader.load_data()
            elif texts:
                documents = [Document(text=t) for t in texts]
            else:
                raise ValueError("Must provide either directory_path, files, or texts")
            
            print(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []

    def process_documents(
        self,
        documents: List[Document],
        split_method: str = "token"
    ) -> List[str]:
        """处理文档并返回分块"""
        try:
            chunks = []
            
            for doc in documents:
                text = doc.get_content(metadata_mode=MetadataMode.NONE)
                
                # 根据分割方法选择合适的处理方式
                if split_method == "token":
                    # 通用token分割
                    text_chunks = self.text_splitter.split_text(text)
                elif split_method == "sentence":
                    # 基于句子分割
                    nodes = self.sentence_splitter.get_nodes_from_documents([doc])
                    text_chunks = [node.get_content(metadata_mode=MetadataMode.NONE) 
                                for node in nodes]
                else:
                    raise ValueError(f"Unknown split method: {split_method}")
                
                chunks.extend(text_chunks)
            
            # 清理和规范化文本块
            cleaned_chunks = [
                chunk.strip()
                for chunk in chunks
                if chunk.strip()  # 移除空块
            ]
            
            print(f"Successfully processed documents into {len(cleaned_chunks)} chunks")
            return cleaned_chunks
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return []

    def extract_metadata(self, documents: List[Document]) -> List[dict]:
        """提取所有文档的元数据"""
        try:
            metadata_list = []
            for doc in documents:
                metadata = {
                    "file_name": doc.metadata.get("file_name", ""),
                    "file_type": doc.metadata.get("file_type", ""),
                    "file_path": doc.metadata.get("file_path", ""),
                    "creation_date": doc.metadata.get("creation_date", ""),
                }
                metadata_list.append(metadata)
            return metadata_list
            
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return []