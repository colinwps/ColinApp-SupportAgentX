"""
agent/state.py
LangGraph Agent 状态定义
"""

from typing import Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Agent 在整个工作流中传递的状态对象。
    每个节点读取并更新此状态。
    """

    # 对话消息历史（add_messages 自动追加，不覆盖）
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # RAG 检索到的知识上下文
    retrieved_context: str = ""

    # 意图分类结果
    intent: Literal[
        "order_query",      # 订单查询
        "refund_request",   # 退款申请
        "complaint",        # 投诉
        "general_faq",      # 常见问题
        "human_handoff",    # 转人工
        "unknown",          # 未知
    ] = "unknown"

    # 是否需要转人工
    needs_human: bool = False

    # 工具调用轮次计数（防止死循环）
    iteration_count: int = 0

    # 最终回复（流程结束时填充）
    final_response: str = ""

    class Config:
        arbitrary_types_allowed = True
