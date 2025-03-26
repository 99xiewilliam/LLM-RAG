from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator

class BaseLLM(ABC):
    """LLM抽象基类，定义所有LLM客户端必须实现的接口"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """生成文本完成（非流式）"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """生成文本完成（流式）"""
        pass
    
    @abstractmethod
    async def generate_with_context(
        self, 
        query: str, 
        context: str,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> str:
        """使用上下文生成回答（非流式）"""
        pass
    
    @abstractmethod
    async def generate_with_context_stream(
        self, 
        query: str, 
        context: str,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """使用上下文生成回答（流式）"""
        pass
        
    async def translate_to_english(self, text: str) -> str:
        """将文本翻译成英文（默认实现）"""
        prompt = f"""请将以下文本翻译成英文，只需要返回翻译结果，不需要任何解释：

文本：{text}

英文翻译："""
        return await self.generate(prompt)