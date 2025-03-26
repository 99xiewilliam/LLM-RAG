from typing import Dict, Any, Optional
from models.base_llm import BaseLLM
from models.deepseek_llm import AsyncDeepSeekLLM
from models.openai_llm import AsyncOpenAILLM

class LLMFactory:
    """LLM工厂类，负责创建不同类型的LLM实例"""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseLLM:
        """
        根据配置创建LLM实例
        
        Args:
            config: LLM配置字典
            
        Returns:
            BaseLLM实例
        """
        llm_type = config.get("type", "deepseek")
        
        if llm_type == "deepseek":
            return AsyncDeepSeekLLM(
                model_path=config["model_path"],
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                max_concurrent_requests=config.get("max_concurrent_requests", 10)
            )
        
        elif llm_type == "openai":
            return AsyncOpenAILLM(
                api_base_url=config.get("api_base_url", "http://localhost:8000/v1"),
                model=config.get("model", "default"),
                max_tokens=config.get("max_tokens", 1024),
                temperature=config.get("temperature", 0.7),
                timeout=config.get("timeout", 120)
            )
        
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")