# tiny_recursive_model_v2.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import math, time, uuid

@dataclass
class StrategyContribution:
    name: str
    weight: float
    confidence: float
    direction: int
    strength: float
    price_hint: Optional[float] = None

@dataclass
class MarketSnapshot:
    bid: float
    ask: float
    mid: float
    spread: float
    atr: Optional[float] = None
    fee_rate: float = 0.0026
    tick_size: float = 0.01
    lot_step: float = 0.0001
    ordemin_eur: Optional[float] = None

@dataclass
class Constraints:
    cash_eur: float
    max_position_eur: Optional[float] = None
    position_qty: float = 0.0
    cooldown_sec: int = 0
    max_leverage: Optional[float] = None
    slippage_bps: float = 6.0
    brand: str = "blend"
    now_ts: Optional[float] = None

@dataclass
class Goals:
    daily_target_pct: float = 0.0
    weekly_target_pct: float = 0.0
    realized_today_pct: float = 0.0
    realized_week_pct: float = 0.0

@dataclass
class Position:
    side: str
    qty: float
    avg_price: float
    leverage: Optional[float] = None

@dataclass
class OpenOrder:
    order_id: str
    side: str
    qty: float
    price: float
    reduce_only: bool = False
    tif: str = "GTC"
    post_only: bool = True
    type: str = "limit"

@dataclass
class TP_SL:
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing: Optional[float] = None

@dataclass
class Action:
    side: str
    qty: float
    limit_price: float
    time_in_force: str = "GTC"
    post_only: bool = True
    reduce_only: bool = False
    tp_sl: TP_SL = field(default_factory=TP_SL)
    split_into: int = 1
    client_id: str = field(default_factory=lambda: f"trm2-{uuid.uuid4().hex[:10]}")
    note: Optional[str] = None

@dataclass
class ActionPlan:
    pair: str
    actions: List[Action] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)

def _clip(x: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None: x = max(lo, x)
    if hi is not None: x = min(hi, x)
    return x

def _round_down(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x/step)*step

def _round_up(x: float, step: float) -> float:
    if step <= 0: return x
    return math.ceil(x/step)*step

def _align_price(px: float, tick: float) -> float:
    return _round_down(px, tick)

def _expected_value(score: float, fee_rate: float, slippage_bps: float, maker: bool) -> float:
    cost = (fee_rate*0.5 if maker else fee_rate) + (0 if maker else slippage_bps/1e4)
    return score - cost

def _side_sign(side: str) -> int:
    return 1 if side in ("buy","long") else -1

class TinyRecursiveModelV2:
    def __init__(self,
                 take_profit_k_atr: float = 1.4,
                 stop_k_atr: float = 1.0,
                 conflict_open_threshold: float = 0.28,
                 reverse_threshold: float = 0.45,
                 partial_reduce_pct: float = 0.5,
                 min_profit_to_lock_pct: float = 0.005,
                 max_slices: int = 3):
        self.tp_k = take_profit_k_atr
        self.sl_k = stop_k_atr
        self.conflict_open_threshold = conflict_open_threshold
        self.reverse_threshold = reverse_threshold
        self.partial_reduce_pct = partial_reduce_pct
        self.min_profit_to_lock_pct = min_profit_to_lock_pct
        self.max_slices = max_slices

    def propose(self, pair: str, market: MarketSnapshot, constraints: Constraints,
                contributions: List[StrategyContribution], goals: Goals,
                position: Optional[Position], open_orders: Optional[List[OpenOrder]] = None) -> ActionPlan:
        long_score, short_score, fair_hint = self._aggregate(contributions)
        net_score = _clip(long_score + short_score, -1.0, 1.0)
        diag = {"score": round(net_score,6), "score_long": round(long_score,6), "score_short": round(short_score,6), "brand": constraints.brand}
        actions: List[Action] = []
        if position and position.qty > 0:
            a, extra = self._decide_with_position(pair, market, constraints, goals, position, net_score, fair_hint)
            actions += a; diag.update(extra)
        else:
            if abs(net_score) > 1e-3:
                a = self._draft_open(pair, market, constraints, net_score, fair_hint)
                if a: actions.append(a)
            diag["conflict_mode"] = "no_position"
        return ActionPlan(pair=pair, actions=actions, diagnostics=diag)

    def _aggregate(self, cs: List[StrategyContribution]):
        long_score = 0.0; short_score = 0.0
        hint_sum = 0.0; hint_w = 0.0
        for c in cs:
            s = c.weight * c.confidence * (c.strength if c.direction >= 0 else -c.strength)
            if s >= 0: long_score += s
            else:      short_score += s
            if c.price_hint is not None:
                hw = max(0.05, c.confidence) * (abs(c.weight) or 1e-6)
                hint_sum += c.price_hint*hw; hint_w += hw
        fair_hint = (hint_sum/hint_w) if hint_w>0 else None
        return _clip(long_score,-1,1), _clip(short_score,-1,1), fair_hint

    def _draft_open(self, pair, m: MarketSnapshot, k: Constraints, net_score: float, fair_hint: Optional[float]) -> Optional[Action]:
        side = "buy" if net_score > 0 else "sell"
        anchor = fair_hint if fair_hint is not None else m.mid
        maker_bias = {"maker": 1.0, "blend": 0.75, "swing": 0.7, "taker": 0.3, "scalper": 0.2}.get(k.brand, 0.7)
        if side == "buy":
            target = min(anchor, m.mid)
            px = _align_price(target - (m.spread * maker_bias * 0.25), m.tick_size); px = max(px, m.tick_size)
        else:
            target = max(anchor, m.mid)
            px = _align_price(target + (m.spread * maker_bias * 0.25), m.tick_size)

        desire_eur = k.cash_eur * _clip(abs(net_score), 0.05, 1.0)
        if k.max_position_eur is not None: desire_eur = min(desire_eur, k.max_position_eur)
        qty = _round_down(desire_eur / max(px, 1e-9), m.lot_step)
        if m.ordemin_eur is not None and px*qty < m.ordemin_eur:
            qty = _round_up(m.ordemin_eur / max(px,1e-9), m.lot_step)
        if px*qty > k.cash_eur: qty = _round_down(k.cash_eur / max(px,1e-9), m.lot_step)
        if qty <= 0: return None
        a = Action(side=side, qty=qty, limit_price=px, post_only=True, note=f"open {side} from score {round(net_score,3)}")
        self._attach_tp_sl(a, m, side); self._slicing(a, m); return a

    def _decide_with_position(self, pair, m: MarketSnapshot, k: Constraints, g: Goals, pos, net_score, fair_hint):
        actions: List[Action] = []
        pos_sign = _side_sign(pos.side); sig_sign = 1 if net_score>0 else -1
        same_dir = (pos_sign == sig_sign)
        pnl_pct = self._unrealized_pct(pos, m.mid)
        daily_gap = max(0.0, g.daily_target_pct - g.realized_today_pct)
        weekly_gap = max(0.0, g.weekly_target_pct - g.realized_week_pct)
        goal_pressure = daily_gap*1.0 + weekly_gap*0.5
        diag = {"conflict_mode":"aligned" if same_dir else "conflict",
                "pos_side":pos.side, "pos_qty":pos.qty, "pos_avg":pos.avg_price,
                "pnl_pct": round(pnl_pct*100,3),
                "goal_pressure_pct": round(goal_pressure*100,3)}

        if same_dir:
            if goal_pressure <= 0 and pnl_pct > self.min_profit_to_lock_pct:
                actions += self._partial_reduce_only(pos, m, fraction=self.partial_reduce_pct, note="lock profit (targets met)")
            else:
                if abs(net_score) >= 0.25:
                    a = self._draft_open(pair, m, k, net_score*0.6, fair_hint)
                    if a: a.note = (a.note or "") + " (add-on)"; actions.append(a)
                diag["mgmt"] = "hold & trail TP/SL"
        else:
            edge = abs(net_score)
            take_profit_min = max(self.min_profit_to_lock_pct, goal_pressure*0.5)
            can_open_against = edge >= self.conflict_open_threshold
            utilities = {}
            utilities["HOLD"] = 0.1 - edge*0.2 + (pnl_pct*0.1) - goal_pressure*0.05
            utilities["REDUCE_ONLY"] = (pnl_pct - take_profit_min)*1.2 + goal_pressure*0.6 + edge*0.2
            utilities["HEDGE"] = (edge*0.6) - 0.02 + (goal_pressure*0.2)
            utilities["REVERSE"] = (edge*1.0) + (pnl_pct*0.3) + goal_pressure*0.5 - 0.01
            if edge < self.reverse_threshold: utilities["REVERSE"] -= 0.3
            best = max(utilities, key=lambda k: utilities[k])
            diag["utilities"] = {k: round(v,4) for k,v in utilities.items()}; diag["decision"] = best

            if best == "HOLD":
                diag["mgmt"] = "hold & monitor"
            elif best == "REDUCE_ONLY":
                actions += self._partial_reduce_only(pos, m, fraction=self.partial_reduce_pct, note="partial TP reduce-only")
                if can_open_against and self.partial_reduce_pct >= 0.5:
                    a = self._draft_open(pair, m, k, -pos_sign*min(edge,0.5), fair_hint)
                    if a: a.note = (a.note or "") + " (post-reduction probe)"; actions.append(a)
            elif best == "HEDGE" and can_open_against:
                a = self._draft_open(pair, m, k, -pos_sign*min(edge*0.7,0.5), fair_hint)
                if a: a.note = (a.note or "") + " (hedge)"; actions.append(a)
            elif best == "REVERSE" and can_open_against:
                actions += self._partial_reduce_only(pos, m, fraction=1.0, note="close & reverse (reduce-only)")
                a = self._draft_open(pair, m, k, -pos_sign*edge, fair_hint)
                if a: a.note = (a.note or "") + " (reverse open)"; actions.append(a)
            else:
                diag["decision_adjusted"] = "fallback to HOLD"

        return actions, diag

    def _unrealized_pct(self, pos: Position, mark: float) -> float:
        if pos.side == "long": return (mark - pos.avg_price) / max(pos.avg_price,1e-9)
        else: return (pos.avg_price - mark) / max(pos.avg_price,1e-9)

    def _partial_reduce_only(self, pos: Position, m: MarketSnapshot, fraction: float, note: str) -> List[Action]:
        fraction = _clip(fraction, 0.05, 1.0)
        qty = _round_down(pos.qty * fraction, m.lot_step)
        if qty <= 0: return []
        if pos.side == "long":
            px = _align_price(m.bid, m.tick_size); side = "sell"
        else:
            px = _align_price(m.ask, m.tick_size); side = "buy"
        a = Action(side=side, qty=qty, limit_price=px, post_only=True, reduce_only=True, note=note)
        self._attach_tp_sl(a, m, side); self._slicing(a, m); return [a]

    def _attach_tp_sl(self, a: Action, m: MarketSnapshot, side_exec: str):
        atr = m.atr or (m.mid*0.012)
        if side_exec == "buy":
            tp = _align_price(a.limit_price + self.tp_k*atr, m.tick_size)
            sl = _align_price(a.limit_price - self.sl_k*atr, m.tick_size)
        else:
            tp = _align_price(a.limit_price - self.tp_k*atr, m.tick_size)
            sl = _align_price(a.limit_price + self.sl_k*atr, m.tick_size)
        a.tp_sl = TP_SL(take_profit=tp, stop_loss=sl, trailing=None)

    def _slicing(self, a: Action, m: MarketSnapshot):
        ticks = max(1, int(round(m.spread / max(m.tick_size, 1e-9))))
        a.split_into = 1 + min(self.max_slices - 1, max(0, ticks - 3))

def propose_actions_v2(pair: str, market: Dict, constraints: Dict, contributions: List[Dict],
                       goals: Dict, position: Optional[Dict], open_orders: Optional[List[Dict]] = None) -> Dict:
    model = TinyRecursiveModelV2()
    m = MarketSnapshot(**market); k = Constraints(**constraints)
    cs = [StrategyContribution(**c) for c in contributions]
    g = Goals(**goals); pos = Position(**position) if position else None
    oo = [] if not open_orders else [OpenOrder(**o) for o in open_orders]
    plan = model.propose(pair, m, k, cs, g, pos, oo)
    return {"pair": plan.pair, "actions": [asdict(a) for a in plan.actions], "diagnostics": plan.diagnostics}
