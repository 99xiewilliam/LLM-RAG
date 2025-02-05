from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
import uuid
import time
import asyncio
from typing import List, Optional

class AsyncDeepSeekLLM:
    def __init__(
        self, 
        model_path: str, 
        tensor_parallel_size: int = 1,
        max_concurrent_requests: int = 10
    ):
        # 创建引擎参数
        engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=0.8,         # 增加 GPU 内存使用率
            max_model_len=8192,                 # 减小最大序列长度
            max_num_batched_tokens=8192,        # 减小批处理令牌数
            trust_remote_code=True,
            disable_log_stats=True,
            max_num_seqs=max_concurrent_requests,  # 控制并发序列数
            enforce_eager=True,
            max_parallel_loading_workers=2,
            tensor_parallel_size=tensor_parallel_size,
            dtype='float16',                    # 使用半精度
            swap_space=4,                       # 添加交换空间（GB）
        )
        
        self.model = AsyncLLMEngine.from_engine_args(engine_args)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # 设置停止词
        self.stop_tokens = [
            "Question:", "Question", "USER:", "USER", 
            "ASSISTANT:", "ASSISTANT", "Instruction:", 
            "Instruction", "Response:", "Response"
        ]
        
        # 默认采样参数
        self.default_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1,
            max_tokens=1024,
            stop=self.stop_tokens
        )

    async def generate(
        self, 
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
    ) -> str:
        """异步生成回复"""
        async with self.semaphore:
            try:
                # 生成请求ID
                request_id = str(uuid.uuid4())
                
                # 创建采样参数
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=1,
                    max_tokens=max_tokens,
                    stop=stop or self.stop_tokens
                )
                
                # 添加请求
                request_id = await self.model.add_request(
                    request_id=request_id,
                    prompt=prompt,
                    params=sampling_params
                )
                
                # 收集输出
                full_text = ""
                async for output in self.model.generate(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params
                ):
                    if output.outputs[0].text.strip():
                        full_text = output.outputs[0].text
                
                return full_text.strip()
                
            except Exception as e:
                print(f"Error in generate: {str(e)}")
                return ""

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        """批量异步生成回复"""
        async with self.semaphore:
            try:
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
                
            except Exception as e:
                print(f"Error in generate_batch: {str(e)}")
                return [""] * len(prompts)

    async def generate_with_context(
        self, 
        query: str, 
        context: str,
        target_lang: Optional[str] = None
    ) -> str:
        """根据上下文生成回答"""
        prompt = f"""请基于以下参考信息回答问题。如果参考信息不足以回答问题，请使用你的知识来回答。
        
        {f'请用{target_lang}回答' if target_lang else '请用与提问相同的语言回答'}。

        参考信息：
        {context}

        问题：{query}

        回答："""
        
        return await self.generate(prompt)