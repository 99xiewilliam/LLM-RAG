from typing import List, Tuple
import numpy as np
from pymilvus import Collection, connections, utility
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMilvusClient:
    def __init__(self, host: str, port: int, collection_name: str):
        self.collection_name = collection_name
        self.executor = ThreadPoolExecutor(max_workers=10)
        connections.connect(host=host, port=port)
        
        if utility.exists_collection(collection_name):
            self.collection = Collection(collection_name)
        else:
            self._create_collection()

    def _create_collection(self):
        from pymilvus import CollectionSchema, FieldSchema, DataType
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        schema = CollectionSchema(fields=fields)
        self.collection = Collection(self.collection_name, schema)
        
        # Create IVF_FLAT index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index("embedding", index_params)

    async def insert(self, texts: list, embeddings: np.ndarray):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._insert_sync,
            texts,
            embeddings
        )

    def _insert_sync(self, texts: list, embeddings: np.ndarray):
        entities = [
            {"text": texts},
            {"embedding": embeddings}
        ]
        self.collection.insert(entities)
        self.collection.flush()

    async def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._search_sync,
            query_embedding,
            top_k
        )

    def _search_sync(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Tuple[str, float]]:
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        return [(hit.entity.get('text'), hit.distance) for hit in results[0]]