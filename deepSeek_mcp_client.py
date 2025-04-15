import re
from turtle import mode
import os
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

llm=ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="xxxxxxx",
    openai_api_base="https://api.deepseek.com",
    max_tokens=1024,
)


def print_conversation(response):
    for msg in response['messages']:
        # 打印用户提问
        if msg.type == 'human':
            print(f"[User] {msg.content}")

        # 打印AI的思考过程（包括工具调用）
        elif msg.type == 'ai' and hasattr(msg, 'tool_calls'):
            print(f"[AI-Thinking] Decided to call tools:")
            for tool_call in msg.tool_calls:
                print(f"  ✨ {tool_call['name']}({tool_call['args']})")

        # 打印工具执行结果
        elif msg.type == 'tool':
            print(f"[Tool-Result] {msg.name} => {msg.content}")

        # 打印AI最终回复
        elif msg.type == 'ai' and msg.content:
            print(f"[AI-Answer] {msg.content}\n")


async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["D:\\cmic\\code\\testCode\\testMcp\\math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8300/sse",
                "transport": "sse",
            }
        }
    ) as client:
        system_prompt = """
                你是一个有用的助手，能够根据需要自动决定调用哪个工具回答问题。 
                """
        # 在内存中管理对话历史
        memory = MemorySaver()
        config = {"configurable": {"thread_id": "226"}}


        agent = create_react_agent(model=llm, tools=client.get_tools(), prompt=system_prompt, checkpointer=memory,
                                   debug=False)
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"}, config=config)
        print_conversation(math_response)
        math_response = await agent.ainvoke({"messages": "what's the result devided by 6?"}, config=config)
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"}, config=config)
        return math_response, weather_response

math_response, weather_response = asyncio.run(main())
print_conversation(math_response)
print_conversation(weather_response)