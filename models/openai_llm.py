import aiohttp
import json
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from models.base_llm import BaseLLM

class AsyncOpenAILLM(BaseLLM):
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 120,
        **kwargs
    ):
        self.api_base_url = api_base_url
        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.timeout = timeout
        self.logger = logging.getLogger("openai_client")
        
    async def generate(
        self, 
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """生成文本完成"""
        try:
            completion_url = f"{self.api_base_url}/completions"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens or self.default_max_tokens,
                "temperature": temperature or self.default_temperature,
                "stream": False
            }
            
            if stop:
                payload["stop"] = stop
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    completion_url, 
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API错误: {error_text}")
                        return f"生成失败: API错误 {response.status}"
                    
                    response_json = await response.json()
                    return response_json['choices'][0]['text']
                    
        except Exception as e:
            self.logger.error(f"生成文本时发生错误: {str(e)}")
            return f"生成失败: {str(e)}"
    
    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本完成"""
        results = []
        for prompt in prompts:
            result = await self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            results.append(result)
        return results
    
    async def generate_with_context(
        self, 
        query: str, 
        context: str,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> str:
        """使用检索上下文生成回答"""
        
        # 构建提示模板
        language_instruction = f"请用{target_lang}回答" if target_lang else "请用与提问相同的语言回答"
        
        prompt = f"""你是一个知识渊博的助手。使用以下参考信息回答问题。如果参考信息中没有相关内容，请诚实地说你不知道。

参考信息:
{context}

问题: {query}

{language_instruction}。尽可能提供详细和有帮助的回答，直接回答问题而不要复述问题或说"根据参考信息"。
"""
        
        return await self.generate(prompt)
    
    async def generate_stream(
        self, 
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """生成文本完成（流式）"""
        try:
            completion_url = f"{self.api_base_url}/completions"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens or self.default_max_tokens,
                "temperature": temperature or self.default_temperature,
                "stream": True
            }
            
            if stop:
                payload["stop"] = stop
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    completion_url, 
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API错误: {error_text}")
                        yield f"生成失败: API错误 {response.status}"
                        return
                    
                    # 处理流式响应
                    async for line in response.content:
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line.startswith(b"data:"):
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                break
                                
                            try:
                                chunk = json.loads(data)
                                if len(chunk["choices"]) > 0:
                                    text = chunk["choices"][0].get("text", "")
                                    if text:
                                        yield text
                            except Exception as e:
                                self.logger.error(f"解析响应时发生错误: {str(e)}")
                                continue
                
        except Exception as e:
            self.logger.error(f"流式生成文本时发生错误: {str(e)}")
            yield f"生成失败: {str(e)}"
    
    async def generate_with_context_stream(
        self, 
        query: str, 
        context: str,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """使用检索上下文生成回答（流式）"""
        
        # 构建提示模板
        language_instruction = f"请用{target_lang}回答" if target_lang else "请用与提问相同的语言回答"
        
        prompt = f"""你是一个知识渊博的助手。使用以下参考信息回答问题。如果参考信息中没有相关内容，请诚实地说你不知道。

参考信息:
{context}

问题: {query}

{language_instruction}。尽可能提供详细和有帮助的回答，直接回答问题而不要复述问题或说"根据参考信息"。
"""
        
        async for token in self.generate_stream(prompt):
            yield token 