"""
agent/nodes.py
LangGraph 各节点实现：意图识别、RAG检索、工具调用、回复生成、人工转接
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode

from config.settings import settings, get_llm
from knowledge_base.retriever import retrieve, format_context
from tools.registry import ALL_TOOLS
from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT, INTENT_CLASSIFIER_PROMPT


# ──────────────────────────────────────────
# 节点 1：意图识别
# ──────────────────────────────────────────
def intent_classifier_node(state: AgentState) -> dict:
    """
    分析用户最新消息，判断意图类别。
    使用轻量模型（Haiku/gpt-4o-mini）快速分类，节省成本。
    """
    last_msg = _get_last_human_message(state)
    if not last_msg:
        return {"intent": "unknown"}

    llm = get_llm()
    prompt = INTENT_CLASSIFIER_PROMPT.format(user_message=last_msg)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip().lower()

        valid_intents = {
            "order_query", "refund_request", "complaint",
            "general_faq", "human_handoff", "unknown"
        }
        intent = raw if raw in valid_intents else "unknown"
    except Exception as e:
        print(f"[意图识别] 异常: {e}")
        intent = "unknown"

    print(f"[意图识别] '{last_msg[:30]}...' → {intent}")
    return {"intent": intent}


# ──────────────────────────────────────────
# 节点 2：RAG 知识库检索
# ──────────────────────────────────────────
def retrieval_node(state: AgentState) -> dict:
    """
    从知识库检索与用户问题相关的内容，作为 LLM 上下文。
    对于纯工具调用类意图（订单/退款），可跳过或轻量检索。
    """
    last_msg = _get_last_human_message(state)
    if not last_msg:
        return {"retrieved_context": ""}

    # 订单查询类意图不需要知识库，直接调工具更高效
    if state.intent == "order_query":
        return {"retrieved_context": ""}

    try:
        docs = retrieve(last_msg, top_k=settings.RETRIEVAL_TOP_K)
        context = format_context(docs)
        print(f"[RAG检索] 找到 {len(docs)} 条相关文档")
    except Exception as e:
        print(f"[RAG检索] 异常: {e}")
        context = "（知识库检索失败，请根据通用知识回答）"

    return {"retrieved_context": context}


# ──────────────────────────────────────────
# 节点 3：AI 推理 + 工具调用决策
# ──────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """
    核心 Agent 节点：基于当前状态和知识库上下文，
    决定是直接回复还是调用工具。
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    system_content = SYSTEM_PROMPT.format(
        retrieved_context=state.retrieved_context or "（无相关知识库内容）"
    )

    messages = [SystemMessage(content=system_content)] + list(state.messages)

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        print(f"[Agent] LLM 调用异常: {e}")
        response = AIMessage(content="抱歉，我遇到了一些技术问题，请稍后再试或联系人工客服。")

    return {
        "messages": [response],
        "iteration_count": state.iteration_count + 1,
    }


# ──────────────────────────────────────────
# 节点 4：工具执行（LangGraph 内置）
# ──────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)


# ──────────────────────────────────────────
# 节点 5：人工转接
# ──────────────────────────────────────────
def human_handoff_node(state: AgentState) -> dict:
    """
    转人工坐席节点：记录转接原因，生成转接消息。
    实际项目中可在此触发 WebSocket 推送、工单创建等。
    """
    intent = state.intent
    last_msg = _get_last_human_message(state)

    if intent == "complaint":
        reason = "用户存在投诉，需要人工处理"
    elif intent == "human_handoff":
        reason = "用户主动要求转人工"
    else:
        reason = "AI 无法解决当前问题"

    handoff_msg = f"""我理解您的需求，正在为您转接人工客服。

📋 **转接信息**
- 转接原因：{reason}
- 用户问题：{last_msg or '（未提供）'}

人工客服将在以下时间段为您服务：
- 工作日：9:00 - 21:00
- 节假日：10:00 - 18:00

请稍等，我们会尽快为您安排专属客服！"""

    print(f"[转人工] 原因: {reason}")

    return {
        "messages": [AIMessage(content=handoff_msg)],
        "needs_human": True,
        "final_response": handoff_msg,
    }


# ──────────────────────────────────────────
# 路由函数（条件边）
# ──────────────────────────────────────────
def route_after_intent(state: AgentState) -> str:
    """意图识别后的路由：决定是否先做RAG检索"""
    if state.intent == "human_handoff":
        return "human_handoff"
    if state.intent == "complaint":
        return "human_handoff"
    return "retrieval"


def route_after_agent(state: AgentState) -> str:
    """
    Agent 推理后的路由：
    - 如有工具调用 → 执行工具
    - 超过最大迭代次数 → 结束
    - 否则 → 结束
    """
    last_msg = state.messages[-1] if state.messages else None

    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        if state.iteration_count >= settings.MAX_ITERATIONS:
            print(f"[路由] 达到最大迭代次数 {settings.MAX_ITERATIONS}，强制结束")
            return "end"
        print(f"[路由] 检测到工具调用: {[tc['name'] for tc in last_msg.tool_calls]}")
        return "tools"

    return "end"


# ──────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────
def _get_last_human_message(state: AgentState) -> str | None:
    """从消息历史中取最近一条用户消息"""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None
