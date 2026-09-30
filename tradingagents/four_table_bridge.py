# -*- coding: utf-8 -*-
"""四表工作流 → TradingAgents-CN 决策契约的适配层。

背景
----
TradingAgents-CN 的信号处理契约（见 ``tradingagents/graph/signal_processing.py``）是：

    {"action": "买入/持有/卖出", "target_price": 数字,
     "confidence": 0~1, "risk_score": 0~1, "reasoning": "中文摘要"}

四表工作流（量化 ∩ 连板 ∩ 催化，见 a-share-agent-workflows/fusion-daily.md）
产出的是带「共振层」（T1~T5）与「信号共振度」的候选，字段名不同。本模块把
四表候选翻译成上面的契约，口径与 go-stock 侧适配层、Python 侧
``four-table-bridge`` 严格一致，可单测、可复现。

红线（与 fusion-daily.md 对齐）
-------------------------------
- ``confidence`` 严格等于**信号共振度**，不是收益概率。
- 四表只出「观察 / 低吸」，故本适配层**只产 买入 / 持有**，不臆造「卖出」。
- ``target_price`` 契约要求是数字：优先用四表自带「目标」，否则按历史均溢价
  推算，再否则现价 +3%；三者都缺才退化为 0 并写明缺口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ACTION_BUY = "买入"
ACTION_HOLD = "持有"

# 直接买入的共振层；T3/T4 需再看置信度门槛。
_TIER_BUY = {"T1", "T2"}
CONF_BUY_WEAK = 0.30

# 止损占现价的比例达到该值时，risk_score 记满。
STOP_FULL_RISK = 0.10
GAP_PENALTY_PER = 0.10
GAP_PENALTY_MAX = 0.30


@dataclass
class FourTableCandidate:
    """一条四表候选（字段与 bridge 输出的 candidates.json 对齐）。"""

    code: str
    name: str = ""
    sector: str = ""
    tier: str = "T5"
    confidence: float = 0.0
    in_quant: bool = False
    board: str = ""
    catalyst: str = "gap"
    price: Any = None
    stop: Any = None
    target: Any = None
    hist_premium: Any = None
    hist_return: Any = None
    stock_pos: Any = None
    sector_pos: Any = None
    gaps: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FourTableCandidate":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SIM118
        return cls(**{k: v for k, v in d.items() if k in known})


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def action_of(c: FourTableCandidate) -> str:
    """共振层 + 置信度 -> 买入 / 持有。只出这两种，不臆造卖出。"""
    if c.tier in _TIER_BUY:
        return ACTION_BUY
    if c.tier in ("T3", "T4") and c.confidence >= CONF_BUY_WEAK:
        return ACTION_BUY
    return ACTION_HOLD


def target_price_of(c: FourTableCandidate) -> float:
    explicit = _num(c.target)
    if explicit and explicit > 0:
        return round(explicit, 2)
    price = _num(c.price)
    if price and price > 0:
        prem = _num(c.hist_premium) or _num(c.hist_return) or 0.03
        return round(price * (1 + prem), 2)
    return 0.0


def risk_score_of(c: FourTableCandidate) -> float:
    """止损越大、缺口越多、置信度越低 -> 风险越高，有界 [0,1]。"""
    price, stop = _num(c.price), _num(c.stop)
    stop_risk = 0.5
    if price and stop is not None and price > 0:
        stop_risk = min(abs(price - stop) / price / STOP_FULL_RISK, 1.0)
    gap_penalty = min(len(c.gaps) * GAP_PENALTY_PER, GAP_PENALTY_MAX)
    risk = 0.45 * (1 - c.confidence) + 0.40 * stop_risk + gap_penalty
    return round(min(max(risk, 0.0), 1.0), 4)


def reasoning_of(c: FourTableCandidate) -> str:
    legs = " ".join([
        "量化✓" if c.in_quant else "量化✗",
        f"连板{c.board or '—'}",
        f"催化{c.catalyst or 'gap'}",
    ])
    gap = ("；缺口:" + ",".join(c.gaps)) if c.gaps else ""
    return (
        f"{c.tier} 三腿共振：{legs}。"
        f"置信度{c.confidence}（=信号共振度，非收益概率）；"
        f"参考目标 {c.target or '按历史均溢价推算'}，止损 {c.stop or '未给'}。"
        f"仅供参考，非交易指令{gap}"
    )


def to_decision(c: FourTableCandidate) -> dict[str, Any]:
    """单条候选 -> TradingAgents-CN 决策契约。"""
    return {
        "code": c.code,
        "name": c.name,
        "action": action_of(c),
        "target_price": target_price_of(c),
        "confidence": round(float(c.confidence), 4),
        "risk_score": risk_score_of(c),
        "reasoning": reasoning_of(c),
    }


def to_decisions(candidates: list[dict[str, Any]] | list[FourTableCandidate],
                 *, top_n: int | None = None) -> list[dict[str, Any]]:
    """批量适配，按置信度降序（并列按代码）。"""
    items = [c if isinstance(c, FourTableCandidate) else FourTableCandidate.from_dict(c)
             for c in candidates]
    rows = [to_decision(c) for c in items]
    rows.sort(key=lambda r: (-r["confidence"], r["code"]))
    return rows[:top_n] if top_n else rows
