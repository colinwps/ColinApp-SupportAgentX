# LangGraph 客服智能体

基于 LangGraph 实现的客服 + 自动化工作流智能体，支持本地知识库和多厂商模型。

## 项目结构

```
langgraph_agent/
├── config/
│   └── settings.py          # 统一配置（模型切换、API Key）
├── knowledge_base/
│   ├── docs/                # 本地知识库文档（.txt / .md）
│   ├── loader.py            # 文档加载 & 向量化
│   └── retriever.py         # RAG 检索
├── tools/
│   ├── order_tool.py        # 订单查询工具
│   ├── refund_tool.py       # 退款申请工具
│   ├── ticket_tool.py       # 工单创建工具
│   └── registry.py          # 工具注册表
├── agent/
│   ├── state.py             # Agent 状态定义
│   ├── nodes.py             # LangGraph 节点
│   ├── graph.py             # 图结构编排
│   └── prompts.py           # Prompt 模板
├── main.py                  # 入口文件
├── requirements.txt
└── .env.example
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key 和选择模型厂商

# 3. 添加知识库文档
# 将 .txt / .md 文件放入 knowledge_base/docs/

# 4. 运行
python main.py
```

## 模型切换

在 `.env` 中设置 `LLM_PROVIDER`：

| 值 | 说明 |
|---|---|
| `anthropic` | Claude（默认）|
| `openai` | GPT-4o 等 |
| `deepseek` | DeepSeek |
| `qwen` | 阿里通义千问 |
| `ollama` | 本地 Ollama 模型 |
