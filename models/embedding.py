from typing import List, Union, Optional
import torch
import os

class EmbeddingModel:
    def __init__(self, model_name_or_path):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_name_or_path}")
        
        try:
            # 先尝试使用自定义加载方式加载jina模型
            if "jina" in model_name_or_path.lower():
                self._load_jina_model(model_name_or_path)
            else:
                # 对于其他模型，使用常规transformers加载
                from transformers import AutoModel, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name_or_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model.to(self.device)
        except Exception as e:
            print(f"Loading failed: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def _load_jina_model(self, model_path):
        """专门为jina模型设计的加载方法"""
        try:
            # 尝试修复路径问题
            if not os.path.isdir(model_path):
                raise ValueError(f"Model path {model_path} is not a directory")
            
            from transformers import AutoModel, AutoTokenizer
            
            # 明确指定模型类型并加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model.to(self.device)
            print("Successfully loaded Jina model")
        except Exception as e:
            print(f"Failed to load Jina model: {str(e)}")
            raise

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用原生transformers编码
        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get embeddings
        outputs = self.model(**encoded)
        
        # 根据模型输出格式调整处理方式
        if hasattr(outputs, "pooler_output"):
            # 如果模型有pooler_output，直接使用
            embeddings = outputs.pooler_output
        else:
            # 否则使用mean pooling
            embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
        
        embeddings = embeddings.to(torch.float32)
        return embeddings.cpu().numpy()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # 通常是last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)