"""
tools/order_tool.py
订单查询工具（Mock 实现，替换为真实 API 即可）
"""

from langchain_core.tools import tool


# 模拟订单数据库
MOCK_ORDERS = {
    "ORD-001": {
        "order_id": "ORD-001",
        "user_name": "张三",
        "product": "无线蓝牙耳机 Pro",
        "status": "已发货",
        "logistics": "顺丰速运",
        "tracking_no": "SF1234567890",
        "amount": 299.00,
        "created_at": "2025-03-28 14:30:00",
        "estimated_delivery": "2025-03-30",
    },
    "ORD-002": {
        "order_id": "ORD-002",
        "user_name": "李四",
        "product": "智能手表 Ultra",
        "status": "待发货",
        "logistics": None,
        "tracking_no": None,
        "amount": 1299.00,
        "created_at": "2025-03-29 09:15:00",
        "estimated_delivery": "2025-04-01",
    },
    "ORD-003": {
        "order_id": "ORD-003",
        "user_name": "王五",
        "product": "机械键盘 87键",
        "status": "已签收",
        "logistics": "京东物流",
        "tracking_no": "JD9876543210",
        "amount": 599.00,
        "created_at": "2025-03-20 16:00:00",
        "estimated_delivery": "2025-03-22",
    },
}


@tool
def query_order(order_id: str) -> str:
    """
    查询订单状态和物流信息。

    Args:
        order_id: 订单编号，格式如 ORD-001

    Returns:
        订单详情字符串
    """
    order_id = order_id.strip().upper()
    order = MOCK_ORDERS.get(order_id)

    if not order:
        return f"未找到订单 {order_id}，请确认订单号是否正确。"

    result = f"""
订单信息：
- 订单号：{order['order_id']}
- 商品：{order['product']}
- 金额：¥{order['amount']:.2f}
- 状态：{order['status']}
- 下单时间：{order['created_at']}
- 预计送达：{order['estimated_delivery']}
"""

    if order.get("tracking_no"):
        result += f"- 物流：{order['logistics']}（单号：{order['tracking_no']}）"
    else:
        result += "- 物流：尚未发货"

    return result.strip()


@tool
def list_user_orders(user_name: str) -> str:
    """
    查询用户的所有订单列表。

    Args:
        user_name: 用户姓名

    Returns:
        该用户所有订单的摘要列表
    """
    user_orders = [
        o for o in MOCK_ORDERS.values()
        if o["user_name"] == user_name
    ]

    if not user_orders:
        return f"未找到用户 {user_name} 的任何订单记录。"

    lines = [f"用户 {user_name} 共有 {len(user_orders)} 条订单：\n"]
    for o in user_orders:
        lines.append(
            f"  [{o['order_id']}] {o['product']} - {o['status']} - ¥{o['amount']:.2f}"
        )

    return "\n".join(lines)
