from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Gemma2_LLM(LLM):
    # 基于本地 Gemma2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_name_or_path :str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, device_map="cuda",torch_dtype=torch.bfloat16,)

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        # 调用模型进行对话生成
        chat = [
            { "role": "user", "content": prompt },
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=150)
        outputs = self.tokenizer.decode(outputs[0])
        response = outputs.split('model')[-1].replace('<end_of_turn>\n<eos>', '')

        return response

    @property
    def _llm_type(self) -> str:
        return "Gemma2_LLM"

class LLaMA3_1_LLM(LLM):
    # 基于本地 llama3.1 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
        
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("正在从本地加载llama3.1模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("完成本地模型的加载")

    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        messages = [
            {"role": "system", "content": "你现在是中山大学信息管理学院的迎宾机器人，名字叫小信，要根据提供的信息礼貌热情的回答来宾的问题，当没有提供信息时你要表达：我不知道这层含义。此外以system开头的信息是系统进程信息，表示用户触发了机制的运行，你可以结合用户提问告诉用户系统正在帮他推进。以content开头的信息是你的知识库检索信息"},
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)        
        generated_ids = self.model.generate(model_inputs.input_ids,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        
    @property
    def _llm_type(self) -> str:
        return "LLaMA3_1_LLM"
    
from vllm import LLM as VLLM_LLM, SamplingParams
from langchain.llms.base import LLM
from typing import Any, List, Optional

class VLLM_LLaMA3_1_LLM_Sys(LLM):
    engine: VLLM_LLM = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("正在从本地加载llama3.1模型...")
        # 初始化 vLLM 引擎，使用正确的参数
        #self.engine = VLLM_LLM(model=model_name_or_path, trust_remote_code=True, gpu_memory_utilization=0.95)
        self.engine = VLLM_LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,  # 可以设为 0.95，但根据显存使用情况决定
            #tensor_parallel_size=2,  # 设置为 4，以利用所有四块GPU.无论设置成多少都报错。FileNotFoundError: [Errno 2] No such file or directory: '/proc/2237184/stat'
            max_num_batched_tokens=2048,  # 可调整以优化性能
            max_model_len=2048  # 视具体任务需求而定
        )
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 添加系统消息
        messages = [
            {"role": "system", "content": "你现在是中山大学信息管理学院的迎宾机器人，名字叫小信，要根据提供的信息礼貌热情的回答来宾的问题，当没有提供信息时你要表达：我不知道这层含义。此外以system开头的信息是系统进程信息，表示用户触发了机制的运行，你可以结合用户提问告诉用户系统正在帮他推进。以content开头的信息是你的知识库检索信息,回答时不要有与问题无关的表述"},
            {"role": "user", "content": prompt}
        ]

        # 使用 vLLM 的 generate 方法
        sampling_params = SamplingParams(max_tokens=150)
        # 将系统消息和用户消息合并为字符串
        full_prompt = self._format_messages(messages)
        outputs = self.engine.generate([full_prompt], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        return response

    def _format_messages(self, messages: List[dict]) -> str:
        # 格式化 messages 为模型可以接受的字符串
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    @property
    def _llm_type(self) -> str:
        return "VLLM_LLaMA3_1_LLM"
    
class VLLM_LLaMA3_1_LLM(LLM):
    engine: VLLM_LLM = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("正在从本地加载llama3.1模型...")
        # 初始化 vLLM 引擎，使用正确的参数
        self.engine = VLLM_LLM(model=model_name_or_path, trust_remote_code=True, gpu_memory_utilization=0.9)
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 使用 vLLM 的 generate 方法
        sampling_params = SamplingParams(max_tokens=150)
        outputs = self.engine.generate([prompt], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        return response

    @property
    def _llm_type(self) -> str:
        return "VLLM_LLaMA3_1_LLM"

from langchain.llms.base import LLM
from typing import Any, List, Optional
from openai import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime  # 引入datetime模块

class DeepSeek_LLM(LLM):
    client: OpenAI = None

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print("DeepSeek API 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            messages = [
                {"role": "system", "content": "你现在是中山大学信息管理学院的迎宾机器人，名字叫小信，要根据提供的信息礼貌热情的回答来宾的问题，当没有提供信息时你要表达：我不知道这层含义。此外以system开头的信息是系统进程信息，表示用户触发了机制的运行，你可以结合用户提问告诉用户系统正在帮他推进。以content开头的信息是你的知识库检索信息,回答时不要有与问题无关的表述"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算总响应时间
            total_time = end_time - start_time
            
            # 打印总响应时间
            print(f"Model response time: {total_time}")

            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in DeepSeek API call."

    @property
    def _llm_type(self) -> str:
        return "DeepSeek_LLM"

from langchain.llms.base import LLM
from typing import Any, List, Optional
from zhipuai import ZhipuAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime

from langchain.llms.base import LLM
from typing import Any, List, Optional
from zhipuai import ZhipuAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime

class GLMFlash_LLM(LLM):
    client: ZhipuAI = None

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn"):
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        print("ZhipuAI 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            messages = [
                {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="glm-4-flash",  # 使用GLM-4-flash模型
                messages=messages,
                stream=False,
                max_tokens=1024,  # 你可以根据需要调整
                temperature=0.95,  # 可选参数
                top_p=0.7,  # 可选参数
                stop=stop  # 停止词
            )
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算总响应时间
            total_time = end_time - start_time
            
            # 打印总响应时间
            print(f"Model response time: {total_time}")

            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in GLM-4-flash API call."

    @property
    def _llm_type(self) -> str:
        return "GLMFlash_LLM"




