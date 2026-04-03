"""
main.py
客服智能体入口文件
支持：单次对话 / 交互式多轮对话 / 批量测试
"""

import uuid
import argparse
from typing import Generator

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from config.settings import settings, get_llm
from agent.graph import get_graph
from agent.state import AgentState

console = Console()


# ──────────────────────────────────────────
# 核心对话函数
# ──────────────────────────────────────────
def chat(
    user_input: str,
    thread_id: str = "default",
    stream: bool = False,
) -> str:
    """
    发送一条消息给 Agent，返回最终回复。

    Args:
        user_input: 用户消息
        thread_id:  会话 ID（相同 ID = 同一多轮对话）
        stream:     是否流式输出（当前为节点级流式）

    Returns:
        AI 最终回复文本
    """
    app = get_graph(use_memory=True)
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "messages": [HumanMessage(content=user_input)],
    }

    final_response = ""

    if stream:
        # 流式模式：逐步打印每个节点的输出
        for step in app.stream(input_state, config=config, stream_mode="updates"):
            for node_name, node_output in step.items():
                _print_node_step(node_name, node_output)
                # 提取最终 AI 回复
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        final_response = msg.content
    else:
        # 批量模式：直接获取最终状态
        result = app.invoke(input_state, config=config)
        msgs = result.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_response = msg.content
                break

    return final_response or "（无回复，请检查日志）"


def _print_node_step(node_name: str, node_output: dict):
    """流式模式下打印各节点执行情况"""
    node_labels = {
        "intent_classifier": "🧭 意图识别",
        "retrieval": "🔍 知识库检索",
        "agent": "🤖 AI 推理",
        "tools": "⚙️  工具执行",
        "human_handoff": "👤 转人工",
    }
    label = node_labels.get(node_name, f"📌 {node_name}")

    if node_name == "intent_classifier":
        intent = node_output.get("intent", "")
        console.print(f"  {label}: [cyan]{intent}[/cyan]")

    elif node_name == "retrieval":
        ctx = node_output.get("retrieved_context", "")
        if ctx and ctx != "（无相关知识库内容）":
            console.print(f"  {label}: [green]找到相关知识[/green]")
        else:
            console.print(f"  {label}: [dim]无相关知识[/dim]")

    elif node_name == "tools":
        msgs = node_output.get("messages", [])
        for msg in msgs:
            if isinstance(msg, ToolMessage):
                console.print(f"  {label} [{msg.name}]: [yellow]执行完成[/yellow]")

    elif node_name == "agent":
        msgs = node_output.get("messages", [])
        for msg in msgs:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                names = [tc["name"] for tc in msg.tool_calls]
                console.print(f"  {label}: [magenta]调用工具 {names}[/magenta]")


# ──────────────────────────────────────────
# 交互式 CLI
# ──────────────────────────────────────────
def interactive_mode():
    """多轮交互式客服对话"""
    thread_id = str(uuid.uuid4())[:8]

    console.print(Panel.fit(
        "[bold cyan]🤖 智能客服系统[/bold cyan]\n"
        f"[dim]模型: {settings.LLM_PROVIDER.upper()} | 会话ID: {thread_id}[/dim]\n"
        "[dim]输入 'quit' 退出 | 输入 'new' 开始新会话[/dim]",
        border_style="cyan"
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]您[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input.strip():
            continue

        if user_input.lower() == "quit":
            console.print("[dim]感谢使用，再见！[/dim]")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())[:8]
            console.print(f"[dim]✓ 新会话已创建: {thread_id}[/dim]")
            continue

        # 显示思考过程
        console.print("\n[dim]─── 处理过程 ───[/dim]")
        response = chat(user_input, thread_id=thread_id, stream=True)
        console.print("[dim]─── 回复 ───────[/dim]")
        console.print(Panel(
            Markdown(response),
            border_style="blue",
            title="[bold blue]客服[/bold blue]",
            title_align="left",
        ))


# ──────────────────────────────────────────
# 批量测试模式
# ──────────────────────────────────────────
def run_tests():
    """运行预设测试用例，验证各功能"""
    test_cases = [
        ("FAQ 查询", "你们支持7天无理由退货吗？"),
        ("订单查询", "帮我查一下订单 ORD-001 的物流状态"),
        ("退款申请", "我的订单 ORD-002 还没发货，我想退款，原因是不想要了"),
        ("工单创建", "我买的耳机音质很差，完全不值这个价，要投诉！"),
        ("转人工",   "我要转人工客服"),
        ("多工具",   "查一下订单 ORD-003，然后帮我创建一个工单说商品有质量问题"),
    ]

    console.print(Panel.fit(
        "[bold yellow]🧪 批量测试模式[/bold yellow]",
        border_style="yellow"
    ))

    for i, (name, question) in enumerate(test_cases, 1):
        console.rule(f"[cyan]测试 {i}: {name}[/cyan]")
        console.print(f"[green]问题:[/green] {question}")

        thread_id = f"test_{i}"
        response = chat(question, thread_id=thread_id, stream=True)

        console.print(f"[blue]回复:[/blue]")
        console.print(Markdown(response[:500] + ("..." if len(response) > 500 else "")))
        console.print()


# ──────────────────────────────────────────
# 程序入口
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph 客服智能体")
    parser.add_argument(
        "--mode",
        choices=["chat", "test"],
        default="chat",
        help="运行模式: chat=交互对话（默认）, test=批量测试",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "deepseek", "qwen", "ollama"],
        help="覆盖 .env 中的模型提供商（临时切换）",
    )
    args = parser.parse_args()

    # 临时覆盖模型提供商
    if args.provider:
        settings.LLM_PROVIDER = args.provider
        console.print(f"[yellow]模型提供商已切换为: {args.provider}[/yellow]")

    # 预热向量库
    console.print("[dim]正在初始化知识库...[/dim]")
    try:
        from knowledge_base.retriever import get_vectorstore
        get_vectorstore()
        console.print("[dim]✓ 知识库就绪[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ 知识库加载失败: {e}（将跳过RAG功能）[/yellow]")

    if args.mode == "test":
        run_tests()
    else:
        interactive_mode()
