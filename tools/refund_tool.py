"""
tools/refund_tool.py
退款申请工具（Mock 实现）
"""

import uuid
from datetime import datetime
from langchain_core.tools import tool

# 模拟退款记录存储
REFUND_RECORDS: dict = {}


@tool
def apply_refund(order_id: str, reason: str, amount: float | None = None) -> str:
    """
    为指定订单提交退款申请。

    Args:
        order_id: 订单编号
        reason: 退款原因（如"商品质量问题"、"不想要了"等）
        amount: 退款金额，不填则全额退款

    Returns:
        退款申请结果，包含退款单号
    """
    from tools.order_tool import MOCK_ORDERS

    order_id = order_id.strip().upper()
    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"订单 {order_id} 不存在，无法发起退款。"

    # 简单规则校验
    if order["status"] == "待发货":
        refund_status = "自动通过"
        note = "订单未发货，支持直接退款。"
    elif order["status"] == "已发货":
        refund_status = "审核中"
        note = "商品在途中，客服将在1个工作日内审核。"
    elif order["status"] == "已签收":
        refund_status = "审核中"
        note = "签收后退款需人工审核，请耐心等待。"
    else:
        refund_status = "不可退款"
        note = "该订单状态不支持退款申请。"
        return f"退款申请失败：{note}"

    refund_amount = amount or order["amount"]
    refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"

    REFUND_RECORDS[refund_id] = {
        "refund_id": refund_id,
        "order_id": order_id,
        "reason": reason,
        "amount": refund_amount,
        "status": refund_status,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return f"""
退款申请已提交：
- 退款单号：{refund_id}
- 订单编号：{order_id}
- 退款金额：¥{refund_amount:.2f}
- 退款原因：{reason}
- 申请状态：{refund_status}
- 说明：{note}
- 退款到账时间：审核通过后 3-5 个工作日原路返还
""".strip()


@tool
def query_refund_status(refund_id: str) -> str:
    """
    查询退款申请的处理状态。

    Args:
        refund_id: 退款单号，格式如 REF-XXXXXXXX

    Returns:
        退款状态详情
    """
    refund = REFUND_RECORDS.get(refund_id.strip().upper())

    if not refund:
        return f"退款单 {refund_id} 不存在，请确认单号是否正确。"

    return f"""
退款状态查询：
- 退款单号：{refund['refund_id']}
- 关联订单：{refund['order_id']}
- 退款金额：¥{refund['amount']:.2f}
- 当前状态：{refund['status']}
- 申请时间：{refund['created_at']}
""".strip()
