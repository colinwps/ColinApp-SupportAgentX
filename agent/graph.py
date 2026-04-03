"""
agent/graph.py
LangGraph 图结构定义：编排各节点和条件路由
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    intent_classifier_node,
    retrieval_node,
    agent_node,
    tool_node,
    human_handoff_node,
    route_after_intent,
    route_after_agent,
)


def build_graph(use_memory: bool = True):
    """
    构建并编译 LangGraph 工作流图。

    流程：
        用户输入
            ↓
        [意图识别]
            ↓
        ┌──────────────────────────────┐
        │  complaint / human_handoff   │→ [转人工] → END
        └──────────────────────────────┘
            ↓ 其他意图
        [RAG 检索]
            ↓
        [Agent 推理]
            ↓
        ┌──────────────────┐
        │  有工具调用？     │→ [工具执行] → [Agent 推理] (循环)
        └──────────────────┘
            ↓ 无工具调用
           END

    Args:
        use_memory: 是否启用多轮对话记忆（基于 MemorySaver）

    Returns:
        compiled LangGraph app
    """
    builder = StateGraph(AgentState)

    # ── 注册节点 ──
    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("human_handoff", human_handoff_node)

    # ── 设置入口 ──
    builder.set_entry_point("intent_classifier")

    # ── 条件路由：意图识别后 ──
    builder.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "retrieval": "retrieval",
            "human_handoff": "human_handoff",
        },
    )

    # ── 线性边：检索 → Agent ──
    builder.add_edge("retrieval", "agent")

    # ── 条件路由：Agent 推理后 ──
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # ── 工具执行后回到 Agent（ReAct 循环）──
    builder.add_edge("tools", "agent")

    # ── 转人工后结束 ──
    builder.add_edge("human_handoff", END)

    # ── 编译（可选：附加记忆检查点）──
    checkpointer = MemorySaver() if use_memory else None
    app = builder.compile(checkpointer=checkpointer)

    return app


# 全局单例
_graph_instance = None


def get_graph(use_memory: bool = True):
    """获取全局 graph 单例，避免重复构建"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph(use_memory=use_memory)
    return _graph_instance
