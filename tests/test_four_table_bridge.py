#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四表工作流 → TradingAgents-CN 决策契约适配层的单测。

不需要任何 LLM 密钥，纯函数，可离线跑：

    python3 -m pytest tests/test_four_table_bridge.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.four_table_bridge import (  # noqa: E402
    ACTION_BUY,
    ACTION_HOLD,
    FourTableCandidate,
    action_of,
    risk_score_of,
    target_price_of,
    to_decisions,
)


def _candidate(**kw) -> FourTableCandidate:
    base = dict(code="603928", name="兴业股份", tier="T4", confidence=0.63,
                in_quant=True, board="可成交", catalyst="gap", price=13.07,
                stop=11.76, gaps=["催化腿缺口"])
    base.update(kw)
    return FourTableCandidate.from_dict(base)


def test_action_mapping():
    assert action_of(_candidate(tier="T1")) == ACTION_BUY
    assert action_of(_candidate(tier="T2")) == ACTION_BUY
    assert action_of(_candidate(tier="T3", confidence=0.4)) == ACTION_BUY
    assert action_of(_candidate(tier="T4", confidence=0.63)) == ACTION_BUY
    assert action_of(_candidate(tier="T4", confidence=0.1)) == ACTION_HOLD
    assert action_of(_candidate(tier="T5", confidence=0.9)) == ACTION_HOLD


def test_action_never_sell():
    """四表只出观察/低吸，适配层绝不臆造「卖出」。"""
    for tier in ("T1", "T2", "T3", "T4", "T5"):
        for conf in (0.0, 0.3, 0.6, 0.99):
            assert action_of(_candidate(tier=tier, confidence=conf)) in (ACTION_BUY, ACTION_HOLD)


def test_target_price_prefers_explicit():
    assert target_price_of(_candidate(target=20.93)) == 20.93
    # 无显式目标：现价 13.07 × (1+0.03) 的推算
    assert target_price_of(_candidate(target=None, price=10.0, hist_premium=None,
                                      hist_return=None)) == 10.3


def test_risk_score_bounded():
    for stop in (None, 0.0, 5.0, 13.0):
        r = risk_score_of(_candidate(stop=stop))
        assert 0.0 <= r <= 1.0


def test_to_decisions_sorted_and_shaped():
    rows = to_decisions([
        _candidate(code="000001", tier="T5", confidence=0.9),
        _candidate(code="603928", tier="T4", confidence=0.63),
    ])
    assert rows[0]["code"] == "000001"  # 置信度高者在前
    for r in rows:
        assert set(r) >= {"code", "name", "action", "target_price",
                          "confidence", "risk_score", "reasoning"}
        assert isinstance(r["target_price"], float)
        assert 0.0 <= r["confidence"] <= 1.0
        assert "信号共振度" in r["reasoning"]


def test_top_n():
    rows = to_decisions([_candidate(code=f"{i:06d}") for i in range(30)], top_n=10)
    assert len(rows) == 10
