from abc import ABC
from typing import Any, List, Dict, Optional
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import httpx
import asyncio

class CustomHTTPLLM(BaseLLM, ABC):
    endpoint: str = "https://portal.tools.cmic.online/api/dockplatform/test/groupaceBwaXq/api/v1/jiutian139Dot9B"  # 模型服务地址
    headers: Dict = {"Authorization": "Bearer YOUR_API_KEY"}  # 服务认证头
    timeout: int = 30  # 超时设置（秒）
    max_retries: int = 3  # 失败重试次数

    def _call(self, prompt: str, **kwargs: Any) -> str:
        # 同步请求实现（适用于非异步场景）
        for _ in range(self.max_retries):
            try:
                response = httpx.post(
                    self.endpoint,
                    json={"messages": [{"role": "user", "content": prompt}]},
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError) as e:
                if _ == self.max_retries - 1:
                    raise RuntimeError(f"Model call failed: {str(e)}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> ChatResult:
        # 异步请求实现（适配LangGraph的异步调用）
        content = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
        async with httpx.AsyncClient() as client:
            for _ in range(self.max_retries):
                try:
                    response = await client.post(
                        self.endpoint,
                        json={"input": content},  # 根据实际API字段调整
                        headers=self.headers,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return ChatResult(
                        generations=[ChatGeneration(
                            message=HumanMessage(
                                content=response.json()["output"]  # 根据实际响应字段调整
                            )
                        )]
                    )
                except (httpx.HTTPError, KeyError) as e:
                    if _ == self.max_retries - 1:
                        raise RuntimeError(f"Async model call failed: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom_http_llm"


async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["/home/jean/python/mcp/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8300
                "url": "http://localhost:8300/sse",
                "transport": "sse",
            }
        }
    ) as client:
        system_prompt = "..."  # 保持原系统提示不变

        llm = CustomHTTPLLM(
            endpoint="http://your-model-server:8000/chat",
            headers={"Authorization": "Bearer YOUR_KEY"},
            timeout=30
        )

        memory = MemorySaver()
        config = {"configurable": {"thread_id": "226"}}

        agent = create_react_agent(
            model=llm,
            tools=client.get_tools(),
            prompt=system_prompt,
            checkpointer=memory
        )

        # 测试调用
        response = await agent.ainvoke({"messages": "Hello!"}, config)
        print(response)


asyncio.run(main())