"""
tools/registry.py
工具注册表：统一管理所有可用工具，方便按需组合
"""

from .order_tool import query_order, list_user_orders
from .refund_tool import apply_refund, query_refund_status
from .ticket_tool import create_ticket, query_ticket

# 全量工具列表（客服 Agent 可用工具）
ALL_TOOLS = [
    query_order,
    list_user_orders,
    apply_refund,
    query_refund_status,
    create_ticket,
    query_ticket,
]

# 按类别分组（便于构建多 Agent 系统时按职责分配）
ORDER_TOOLS = [query_order, list_user_orders]
REFUND_TOOLS = [apply_refund, query_refund_status]
TICKET_TOOLS = [create_ticket, query_ticket]

# 工具名称 → 实例映射（用于日志展示）
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "ORDER_TOOLS",
    "REFUND_TOOLS",
    "TICKET_TOOLS",
    "TOOL_MAP",
]
