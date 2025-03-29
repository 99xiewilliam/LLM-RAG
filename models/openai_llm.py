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
            # 使用 Chat API 而不是 Completions API
            completion_url = f"{self.api_base_url}/chat/completions"
            
            # 构建聊天消息格式
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": self.model,
                "messages": messages,
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
                    # 从聊天完成响应中提取内容
                    return response_json['choices'][0]['message']['content']
                    
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
            # 使用 Chat API 
            completion_url = f"{self.api_base_url}/chat/completions"
            
            # 构建聊天消息格式
            messages = [{"role": "user", "content": prompt}]
            
            # 调整max_tokens以适应模型限制
            # 粗略估计输入tokens
            estimated_input_tokens = len(prompt.split()) * 1.3
            safe_max_tokens = max(256, min(1024, 2048 - int(estimated_input_tokens)))
            
            # 详细记录prompt信息用于调试
            self.logger.info(f"Prompt详细信息:")
            self.logger.info(f"- 字符长度: {len(prompt)}")
            self.logger.info(f"- 预估token数: {estimated_input_tokens}")
            self.logger.info(f"- 前100个字符: {prompt[:100]}...")
            self.logger.info(f"- 后100个字符: {prompt[-100:] if len(prompt) > 100 else prompt}")
            
            # 记录消息数组的大小
            msg_json = json.dumps(messages)
            self.logger.info(f"消息JSON大小: {len(msg_json)} 字节")
            print(f"prompt:{prompt}")
            payload = {
                # "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or safe_max_tokens,  # 使用安全值
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
                    
                    # 改进的流式响应处理
                    buffer = ""
                    async for line in response.content:
                        line_str = line.decode('utf-8')
                        if not line_str.strip():
                            continue
                        
                        # 处理完整的SSE消息
                        if line_str.startswith('data: '):
                            data = line_str[6:].strip()
                            
                            # 调试信息
                            print(f"收到数据: {data}")
                            
                            if data == '[DONE]':
                                break
                                
                            try:
                                json_data = json.loads(data)
                                content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                
                                # 检查content是否存在
                                if content:
                                    print(f"提取到内容: {content}")
                                    yield content
                                else:
                                    print(f"无内容在: {json_data}")
                                    
                            except json.JSONDecodeError as e:
                                print(f"JSON解析错误: {e} - 在数据: {data}")
                            except Exception as e:
                                self.logger.error(f"处理响应时错误: {str(e)}")
            
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
        
        # 大幅降低 token 估算上限，采用更激进的截断策略
        # 模型限制为 4096，但我们将目标设为 3000 以确保安全
        max_total_tokens = 3000
        
        # 为系统提示和用户查询预留空间
        system_template = """你是一个知识渊博的助手。使用以下参考信息回答问题。如果参考信息中没有相关内容，请诚实地说你不知道。

参考信息:
"""
        
        query_template = f"""

问题: {query}

{language_instruction}。尽可能提供详细和有帮助的回答，直接回答问题而不要复述问题或说"根据参考信息"。
"""
        
        # 估算系统提示和查询部分的token (使用更保守的倍数4)
        system_tokens = len(system_template.split()) * 4
        query_tokens = len(query_template.split()) * 4
        
        # 为上下文分配剩余token
        available_context_tokens = max_total_tokens - system_tokens - query_tokens
        
        # 确保有足够的空间
        if available_context_tokens < 500:
            self.logger.warning("查询过长，可用于上下文的token太少")
            available_context_tokens = 500  # 至少保留一些上下文空间
        
        self.logger.info(f"Token分配: 系统={system_tokens}, 查询={query_tokens}, 上下文={available_context_tokens}")
        
        # 估算上下文字符数与token的比例
        # 保守估计：平均每个token对应1.5个字符（考虑中文等非英文语言）
        chars_per_token = 1.5
        
        # 计算可用的上下文字符数
        max_context_chars = int(available_context_tokens * chars_per_token)
        
        # 如果上下文超过限制，进行截断
        if len(context) > max_context_chars:
            self.logger.warning(f"上下文太长({len(context)}字符)，需要截断至{max_context_chars}字符")
            
            # 分段处理上下文
            paragraphs = context.split("\n")
            
            # 提取查询中的关键词（长度>1的词）
            query_keywords = [word.lower() for word in query.split() if len(word) > 1]
            
            # 计算段落得分
            paragraph_scores = []
            for i, para in enumerate(paragraphs):
                if not para.strip():  # 跳过空段落
                    continue
                    
                # 基础分数 - 考虑原始顺序但优先考虑前面的段落
                base_score = 100 - min(i, 99)  # 最多100个段落，前面的段落分数更高
                
                # 关键词匹配分数
                para_lower = para.lower()
                keyword_score = sum(10 for keyword in query_keywords if keyword in para_lower)
                
                # 总分 = 关键词分数 + 基础分数
                total_score = keyword_score + base_score
                
                # 长度惩罚 - 避免过长段落
                length_penalty = min(1.0, 100 / max(len(para), 1))
                adjusted_score = total_score * length_penalty
                
                paragraph_scores.append((i, para, adjusted_score))
            
            # 按分数排序（从高到低）
            sorted_paragraphs = sorted(paragraph_scores, key=lambda x: x[2], reverse=True)
            
            # 重建上下文，优先使用最相关的段落
            truncated_context = ""
            current_length = 0
            
            # 首先添加前3个最相关的段落（如果存在）
            for idx, (_, para, _) in enumerate(sorted_paragraphs[:3]):
                if current_length + len(para) + 1 <= max_context_chars:
                    truncated_context += para + "\n\n"  # 保持段落间的空行
                    current_length += len(para) + 2
                else:
                    # 尝试至少保留第一句
                    first_sentence = para.split(". ")[0] + "."
                    if current_length + len(first_sentence) + 1 <= max_context_chars:
                        truncated_context += first_sentence + "\n\n"
                        current_length += len(first_sentence) + 2
            
            # 如果还有空间，添加更多段落
            for _, para, _ in sorted_paragraphs[3:]:
                if current_length + len(para) + 1 <= max_context_chars:
                    truncated_context += para + "\n\n"
                    current_length += len(para) + 2
                else:
                    break  # 空间已满，停止添加
            
            # 添加截断说明
            if current_length < len(context):
                truncated_context += "[内容已截断以适应模型限制]"
            
            # 更新上下文
            context = truncated_context.strip()
            
            self.logger.info(f"上下文已截断至{len(context)}字符（约{len(context) / chars_per_token:.0f}个token）")
        
        # 构建最终提示
        prompt = f"""你是一个知识渊博的助手。使用以下参考信息回答问题。如果参考信息中没有相关内容，请诚实地说你不知道。

参考信息:
{context}

问题: {query}

{language_instruction}。尽可能提供详细和有帮助的回答，直接回答问题而不要复述问题或说"根据参考信息"。
"""
        
        # 记录最终提示大小
        final_char_count = len(prompt)
        final_token_estimate = final_char_count / chars_per_token
        
        self.logger.info(f"最终提示: {final_char_count}字符，估计{final_token_estimate:.0f}个token")
        
        # 添加备用保护措施 - 如果估计token仍然超过3500，进行额外截断
        if final_token_estimate > 3500:
            self.logger.warning(f"最终token估计仍然过高({final_token_estimate:.0f})，进行紧急截断")
            
            # 构建最小化的提示
            minimal_context = context[:int(2000 * chars_per_token)] + "...[内容已大幅截断]"
            minimal_prompt = f"""根据以下信息回答问题。信息不足时请说不知道。

信息:
{minimal_context}

问题: {query}

{language_instruction}
"""
            self.logger.info(f"紧急截断后大小: {len(minimal_prompt)}字符")
            prompt = minimal_prompt
        
        # 使用流式接口生成回答
        try:
            async for token in self.generate_stream(prompt):
                yield token
        except Exception as e:
            self.logger.error(f"生成回答时出错: {str(e)}")
            error_msg = str(e)
            if "maximum context length" in error_msg:
                yield "很抱歉，您的查询和相关信息超出了模型的处理能力。请尝试简化您的问题，或分成多个小问题来询问。"
            else:
                yield f"生成回答时出错: {error_msg}"