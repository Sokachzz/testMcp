# testMcp

mcp-demo

### 虚拟环境创建
依次运行下面的指令

- uv venv .venv
- .\.venv\Scripts\activate
- uv pip install -r requirements.txt


### weather.py
因为weather.py接口是通过sse的形式被agent调用，所以需要启动服务

- python weather.py

### math_server.py
math_server.py是通过stdio的方式被agent调用，知道路径即可不需要启动

### deepSeek_mcp_client.py

把这里的apikey，apibase换成自己的

```python
llm=ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="xxxxxxx",
    openai_api_base="https://api.deepseek.com",
    max_tokens=1024,
)
```

把这里math_server.py的路径换成自己的绝对路径
```python
async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["D:\\cmic\\code\\testCode\\testMcp\\math_server.py"],
                "transport": "stdio",
            },
```

运行程序即可


