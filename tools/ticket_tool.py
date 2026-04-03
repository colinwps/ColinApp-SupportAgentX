"""
tools/ticket_tool.py
工单创建与查询工具（Mock 实现）
"""

import uuid
from datetime import datetime
from langchain_core.tools import tool

TICKETS: dict = {}

PRIORITY_MAP = {
    "低": "low",
    "中": "medium",
    "高": "high",
    "紧急": "urgent",
}


@tool
def create_ticket(
    title: str,
    description: str,
    category: str = "general",
    priority: str = "中",
) -> str:
    """
    创建客服工单，用于需要人工跟进的问题。

    Args:
        title: 工单标题（简短描述问题）
        description: 详细描述
        category: 分类，如 order/refund/logistics/product/other
        priority: 优先级，低/中/高/紧急

    Returns:
        工单创建结果和工单号
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    TICKETS[ticket_id] = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "category": category,
        "priority": PRIORITY_MAP.get(priority, "medium"),
        "status": "待处理",
        "created_at": now,
        "updated_at": now,
        "assignee": None,
    }

    # 高优先级自动分配
    if priority in ("高", "紧急"):
        TICKETS[ticket_id]["assignee"] = "值班客服组"
        TICKETS[ticket_id]["status"] = "处理中"
        note = "已自动分配给值班客服组，将优先处理。"
    else:
        note = "工单已创建，客服将在1个工作日内响应。"

    return f"""
工单已创建：
- 工单号：{ticket_id}
- 标题：{title}
- 分类：{category}
- 优先级：{priority}
- 状态：{TICKETS[ticket_id]['status']}
- 创建时间：{now}
- 说明：{note}

您可以随时使用工单号查询处理进度。
""".strip()


@tool
def query_ticket(ticket_id: str) -> str:
    """
    查询工单处理状态。

    Args:
        ticket_id: 工单号，格式如 TKT-XXXXXX

    Returns:
        工单详情和当前状态
    """
    ticket = TICKETS.get(ticket_id.strip().upper())

    if not ticket:
        return f"工单 {ticket_id} 不存在，请确认工单号是否正确。"

    assignee = ticket["assignee"] or "待分配"
    return f"""
工单详情：
- 工单号：{ticket['ticket_id']}
- 标题：{ticket['title']}
- 状态：{ticket['status']}
- 优先级：{ticket['priority']}
- 负责人：{assignee}
- 创建时间：{ticket['created_at']}
- 最后更新：{ticket['updated_at']}
""".strip()
