# Auto-generated TRM Agent for Investimento
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from .Util import _shadow_daily_path, _net_margin_from_open_orders


# -----------------------------
# Utility helpers
# -----------------------------

Number = Union[int, float]

def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None: return float(default)
        return float(v)
    except Exception:
        return float(default)

def _b(v: Any, default: bool = False) -> bool:
    try:
        if isinstance(v, bool):
            return v
        if v in ("true","True","1",1): return True
        if v in ("false","False","0",0,None): return False
        return bool(v)
    except Exception:
        return bool(default)


# -----------------------------
# Action dataclass
# -----------------------------





# ======== AutoFeaturizer: schema-free feature hashing ========
import hashlib
from collections import deque


# --- time helper per timestamp ISO UTC coerente nel JSONL ---
def _iso_now() -> str:
    # usa lo stesso modulo 'datetime' già importato come namespace
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _is_num(x):
    try:
        if x is None: return False
        float(x)
        return True
    except Exception:
        return False

def _flatten_dict(obj, prefix=""):
    """Breadth-first flatten of dict/list with dotted keys."""
    out = {}
    q = deque([(prefix, obj)])
    while q:
        pfx, val = q.popleft()
        if isinstance(val, dict):
            for k,v in val.items():
                q.append((f"{pfx}{k}." if pfx else f"{k}.", v))
        elif isinstance(val, (list, tuple)):
            for i,v in enumerate(val):
                q.append((f"{pfx}{i}.", v))
        else:
            key = pfx[:-1] if pfx.endswith(".") else pfx
            out[key] = val
    return out

def _hash_idx(key: str, dim: int) -> int:
    # stable hash -> [0, dim)
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % dim

class AutoFeaturizer:
    """
    Trasforma row/blend/goal/weights in vettore fissato da feature hashing.
    - Numerici -> value
    - Bool -> {0,1}
    - Stringhe -> hash presence feature (1.0)
    - Prefix per contesti: info.NOW.*, info.1H.*, portfolio.*, pair_limits.*, blend.*, w_*, goal.*
    - Aggiunge qualche derived common (spread, mid fallback, or_pos/or_w se disponibili)
    - Restituisce anche 'aux' per il decoder (pair, bid/ask/mid, decimali, liquidity, budget)
    """
    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def _add(self, vec, key, val=1.0):
        i = _hash_idx(key, self.dim)
        try:
            vec[i] += float(val)
        except Exception:
            pass

    def _num(self, v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    def featurize(self, row: Dict[str,Any], blend: Dict[str,Number],
                  goal_state: Optional[Dict[str,Any]], strategy_weights: Dict[str,Number]) -> Tuple[List[float], Dict[str,List[Any]]]:

        dim = self.dim
        vec = [0.0]*dim

        # -------- pair + NOW basics (per ausili decoder) --------
        pair = row.get("pair") or f"{row.get('base','?')}/{row.get('quote','?')}"
        pl = row.get("pair_limits") or {}
        info = (row.get("info") or {})
        now = (info.get("NOW") or {})
        bid = self._num(now.get("bid"))
        ask = self._num(now.get("ask"))
        last= self._num(now.get("last"))
        mid = self._num(now.get("mid"), (bid+ask)/2 if (bid>0 and ask>0) else last)

        lot_dec   = int(pl.get("lot_decimals") or 8)
        pair_dec  = int(pl.get("pair_decimals") or 5)
        ordermin  = self._num(pl.get("ordermin"))

        # portfolio for aux
        pf = (row.get("portfolio") or {})
        avail = (pf.get("available") or {})
        free_quote = self._num(avail.get("quote"))
        free_base  = self._num(avail.get("base"))

        pos_margin_base = _net_margin_from_open_orders(row)
       # --- regime derivati (robusti)
       # EMA/ATR possono stare in info.NOW oppure in info.1H/info.4H a seconda del tuo loader: prendiamo le 2 strade
        ema50_1h  = self._num(((info.get("1H") or {}).get("EMA50") if isinstance(info.get("1H"), dict) else (now.get("ema50_1h"))))
        ema200_1h = self._num(((info.get("1H") or {}).get("EMA200") if isinstance(info.get("1H"), dict) else (now.get("ema200_1h"))))
        atr_4h    = self._num(((info.get("4H") or {}).get("atr")    if isinstance(info.get("4H"), dict) else (now.get("atr_4h"))))
        trend_bias = 0.0
        vol_regime = 0.0
        if mid > 0.0 and ema50_1h and ema200_1h:
            try:
                trend_bias = math.tanh((ema50_1h - ema200_1h) / max(mid, 1e-12))
            except Exception:
                trend_bias = 0.0
        if mid > 0.0 and atr_4h:
            try:
                vol_regime = atr_4h / max(mid, 1e-12)
            except Exception:
                vol_regime = 0.0
        # in hashing, registriamo anche queste due
        self._add(vec, "derived.trend_bias:num", trend_bias)
        self._add(vec, "derived.vol_regime:num", vol_regime)

        # --- campi per cost estimator in trainer (se disponibili)
        slip_in  = self._num(now.get("slippage_buy_pct"))
        slip_out = self._num(now.get("slippage_sell_pct"))
        fee_maker = self._num((pl.get("fee_maker") or pl.get("maker_fee") or 0.001))
        fee_taker = self._num((pl.get("fee_taker") or pl.get("taker_fee") or 0.001))
        # aux per decoder
        aux = {
            "pair":[pair], "mid":[mid], "bid":[bid], "ask":[ask],
            "lot_decimals":[lot_dec], "pair_decimals":[pair_dec], "ordermin":[ordermin],
            "free_quote":[free_quote], "free_base":[free_base],
            "pos_margin_base":[pos_margin_base],
            "slippage_in":[slip_in], "slippage_out":[slip_out],
            "fee_maker":[fee_maker], "fee_taker":[fee_taker],
            "max_quote_frac":[float(strategy_weights.get("max_quote_frac",0.12))],
            "min_notional_eur":[float(strategy_weights.get("min_notional_eur",5.0))],
        }

        # -------- 1) row flatten (tutti i campi) --------
        flat = _flatten_dict(row)
        for k, v in flat.items():
            key = f"row.{k}"
            if _is_num(v):
                self._add(vec, f"{key}:num", self._num(v))
            elif isinstance(v, bool):
                self._add(vec, f"{key}:bool", 1.0 if v else 0.0)
            elif v is None:
                self._add(vec, f"{key}:none", 1.0)
            else:
                # string presence
                self._add(vec, f"{key}:str={str(v)}", 1.0)

        # -------- 2) info.* blocks (preserva semantica) --------
        for tf, blk in (info.items() if isinstance(info, dict) else []):
            if not isinstance(blk, dict):
                continue
            fblk = _flatten_dict(blk)
            for k, v in fblk.items():
                key = f"info.{tf}.{k}"
                if _is_num(v):
                    self._add(vec, f"{key}:num", self._num(v))
                elif isinstance(v, bool):
                    self._add(vec, f"{key}:bool", 1.0 if v else 0.0)
                elif v is None:
                    self._add(vec, f"{key}:none", 1.0)
                else:
                    self._add(vec, f"{key}:str={str(v)}", 1.0)

        # -------- 3) blends & weights (dinamici) --------
        for name, val in (blend or {}).items():
            self._add(vec, f"blend.{name}", self._num(val))
        for k, v in (strategy_weights or {}).items():
            self._add(vec, f"w.{k}", self._num(v))

        # -------- 4) goal_state --------
        if isinstance(goal_state, dict):
            for k, v in goal_state.items():
                key = f"goal.{k}"
                if _is_num(v):
                    self._add(vec, f"{key}:num", self._num(v))
                elif isinstance(v, bool):
                    self._add(vec, f"{key}:bool", 1.0 if v else 0.0)
                elif v is None:
                    self._add(vec, f"{key}:none", 1.0)
                else:
                    self._add(vec, f"{key}:str={str(v)}", 1.0)

        # -------- 5) derived common (robusti) --------
        spread = (abs(ask - bid) if (ask>0 and bid>0) else 0.0)
        self._add(vec, "derived.spread:num", spread)
        vwap = self._num(now.get("vwap"))
        self._add(vec, "derived.vwap_now:num", vwap)

        # Opening range pos/weight se disponibili
        or_high = now.get("or_high"); or_low = now.get("or_low")
        if _is_num(or_high) and _is_num(or_low) and (self._num(or_high)-self._num(or_low)) != 0.0:
            or_mid = 0.5*(self._num(or_high)+self._num(or_low))
            or_rng = self._num(or_high)-self._num(or_low)
            or_pos = math.tanh((mid - or_mid) / (abs(or_rng)+1e-12))
            or_w   = or_rng / (abs(mid)+1e-12) if mid else 0.0
            self._add(vec, "derived.or_pos:num", or_pos)
            self._add(vec, "derived.or_w:num", or_w)

        return vec, aux

# Flag per attivare il featurizer schema-free
USE_HASH_FEATS = True
HASH_DIM = 2048  # puoi portarlo a 1024/2048 se vuoi più capacità

@dataclass
class Action:
    pair: str
    side: str                     # "buy" | "sell" | "hold"
    qty: float
    price: Optional[float] = None
    ordertype: str = "limit"      # "limit" | "market" | "none"
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    leverage: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False     # <-- nuovo
    score: float = 0.0
    notes: Optional[str] = "TRM proposal"

    def as_dict(self) -> Dict[str, Any]:
        return {k:v for k,v in asdict(self).items() if v is not None}



# -----------------------------
# Config
# -----------------------------

@dataclass
class TRMConfig:
    feature_dim: int = 128
    hidden_dim: int = 64
    mlp_hidden: int = 128
    K_refine: int = 3
    max_actions_per_pair: int = 2
    norm_eps: float = 1e-6
    log_path: Optional[str] = None
    device: str = "cpu"


# -----------------------------
# Running normalizer
# -----------------------------

class RunningNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("count", torch.tensor(eps))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("M2", torch.zeros(dim))
        self.eps = eps

    def update(self, x: torch.Tensor) -> None:
        if x.ndim == 1: x = x.unsqueeze(0)
        batch_n = torch.tensor(x.shape[0], device=x.device, dtype=self.count.dtype)
        batch_mean = x.mean(0)
        batch_var  = x.var(0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_n
        self.M2 = self.M2 + batch_var*batch_n + (delta**2)*self.count*batch_n/total
        self.mean = self.mean + delta*(batch_n/total)
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.M2/torch.clamp(self.count, min=self.eps) + self.eps)


# -----------------------------
# Tiny Recursive Model (GRU + MLP heads)
# -----------------------------

class TinyRecursiveModel(nn.Module):
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = RunningNorm(cfg.feature_dim, eps=cfg.norm_eps)
        self.encoder = nn.GRU(input_size=cfg.feature_dim, hidden_size=cfg.hidden_dim, num_layers=1, batch_first=True)
        self.y_dim = cfg.hidden_dim
        self.residual = nn.Sequential(
            nn.Linear(self.y_dim + cfg.hidden_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, self.y_dim)
        )
        # Heads
        self.head_side = nn.Linear(self.y_dim, 3)   # buy/sell/hold logits
        self.head_qty  = nn.Linear(self.y_dim, 1)   # 0..1 fraction
        self.head_px   = nn.Linear(self.y_dim, 1)   # price offset tanh -> ±5%
        self.head_tp   = nn.Linear(self.y_dim, 1)   # softplus -> tp mult (0..)
        self.head_sl   = nn.Linear(self.y_dim, 1)   # softplus -> sl mult (0..)
        self.head_tif  = nn.Linear(self.y_dim, 3)   # IOC/FOK/GTC logits
        self.head_lev  = nn.Linear(self.y_dim, 1)   # softplus + 1
        self.head_score= nn.Linear(self.y_dim, 1)   # ranking
        self.head_ordertype = nn.Linear(self.y_dim, 2)  # logits: 0=limit, 1=market
        self.head_reduce    = nn.Linear(self.y_dim, 1)  # sigmoid -> prob reduce_only
        self.to(cfg.device)

    def encode_once(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        out, h1 = self.encoder(x.unsqueeze(1), h)   # x as 1-step sequence
        return out[:, -1, :], h1

    def forward(self, x: torch.Tensor, y: torch.Tensor, h: Optional[torch.Tensor] = None):
        enc, h1 = self.encode_once(x, h)
        y_in = torch.cat([y, enc], dim=-1)
        dy = self.residual(y_in)
        return y + dy, h1

    @torch.no_grad()
    def improve(self, x: torch.Tensor, y0: Optional[torch.Tensor] = None, K: Optional[int] = None) -> torch.Tensor:
        self.eval()
        K = int(K or self.cfg.K_refine)
        x_n = self.norm(x)
        B = x_n.shape[0]
        y = torch.zeros(B, self.y_dim, device=x_n.device) if y0 is None else (y0.to(x_n.device).view(B, self.y_dim))
        h = None
        for _ in range(K):
            y, h = self.forward(x_n, y, h)
        return y

    @torch.no_grad()
    def decode_actions(self, y: torch.Tensor, aux: Dict[str, List[Any]]) -> List[Action]:
        self.eval()
        logits_side = self.head_side(y)                   # [B,3]
        p_side = torch.softmax(logits_side, dim=-1)       # buy/sell/hold probs
        qty_frac = torch.sigmoid(self.head_qty(y)).squeeze(-1)        # 0..1
        px_off   = torch.tanh(self.head_px(y)).squeeze(-1) * 0.05     # ±5%
        tp_mult  = F.softplus(self.head_tp(y)).squeeze(-1) * 0.01 # 0..
        sl_mult  = F.softplus(self.head_sl(y)).squeeze(-1) * 0.01
        tif_idx  = torch.argmax(self.head_tif(y), dim=-1)
        lev      = F.softplus(self.head_lev(y)).squeeze(-1) + 1.0
        score    = torch.tanh(self.head_score(y)).squeeze(-1)
        lev      = F.softplus(self.head_lev(y)).squeeze(-1) + 1.0
        score    = torch.tanh(self.head_score(y)).squeeze(-1)
        ord_logits = self.head_ordertype(y)                 # [B,2]
        ord_idx    = torch.argmax(ord_logits, dim=-1)       # 0=limit, 1=market
        p_reduce   = torch.sigmoid(self.head_reduce(y)).squeeze(-1)  # 0..1

        actions: List[Action] = []
        B = y.shape[0]
        for i in range(B):
            pair = aux["pair"][i]
            mid  = _f(aux["mid"][i], 0.0)
            bid  = _f(aux["bid"][i], mid)
            ask  = _f(aux["ask"][i], mid)
            lot_dec   = int(aux["lot_decimals"][i])
            pair_dec  = int(aux["pair_decimals"][i])
            ordermin  = _f(aux["ordermin"][i], 0.0)
            free_quote= _f(aux["free_quote"][i], 0.0)
            free_base = _f(aux["free_base"][i], 0.0)
            max_qfrac = float(aux.get("max_quote_frac", [0.12])[i] if isinstance(aux.get("max_quote_frac"), list) else aux.get("max_quote_frac", 0.12))
            min_notional = float(aux.get("min_notional_eur", [5.0])[i] if isinstance(aux.get("min_notional_eur"), list) else aux.get("min_notional_eur", 5.0))

            side_idx = int(torch.argmax(p_side[i]).item())   # 0=buy, 1=sell, 2=hold
            tif_map = {0:"IOC", 1:"FOK", 2:"GTC"}

            if side_idx == 2:
                # --- HOLD: nessun ordine, qty 0, nessun TP/SL ---
                actions.append(Action(
                    pair=pair, side="hold", qty=0.0, price=None,
                    ordertype="none", take_profit=None, stop_loss=None,
                    leverage=None, time_in_force="GTC", score=float(score[i].item())
                ))
            else:
                is_buy = (side_idx == 0)
                side = "buy" if is_buy else "sell"

                anchor = bid if is_buy else ask
                base_px = mid if mid > 0 else anchor
                px = base_px * (1.0 + float(px_off[i].item()))
                px = round(px, pair_dec)

                # TP/SL direzionati
                tp = px * (1.0 + (float(tp_mult[i].item()) if is_buy else -float(tp_mult[i].item())))
                sl = px * (1.0 - (float(sl_mult[i].item()) if is_buy else -float(sl_mult[i].item())))
                tp = round(tp, pair_dec); sl = round(sl, pair_dec)

                # size & vincoli
                frac = max(min(float(qty_frac[i].item()), max_qfrac), 0.02)
                notional = free_quote * frac
                if notional < min_notional and free_quote > 0.0:
                    notional = min_notional
                qty = (notional / px) if px > 0 else 0.0
                if not is_buy and free_base > 0:
                    qty = min(qty if qty>0 else free_base, free_base)
                qty = round(qty, lot_dec)

                # ---- ordertype ----
                oi = int(ord_idx[i].item())           # 0=limit, 1=market
                ordertype = "market" if oi == 1 else "limit"
                # per i market non forziamo il prezzo; lascia None o mid:
                price_out = px
                # TIF: per market usa IOC, altrimenti mappa come prima
                tif_map = {0:"IOC", 1:"FOK", 2:"GTC"}
                tif_out = "IOC" if ordertype == "market" else tif_map.get(int(tif_idx[i].item()), "GTC")

                # ---- reduce_only ----
                reduce_pred = bool(p_reduce[i].item() >= 0.5)
                # regola gestionale: se stiamo vendendo con leva >= 2, forza reduce_only per evitare aperture short non desiderate
                # ---- reduce_only con gate sulla posizione a leva ----
                reduce_pred = bool(p_reduce[i].item() >= 0.5)
                pos_m = _f(aux["pos_margin_base"][i], 0.0)

                is_buy  = (side_idx == 0)
                is_sell = (side_idx == 1)

                # stai riducendo una posizione a leva solo se:
                # - hai long>0 e vendi, oppure short<0 e compri
                is_reducing_lev = (pos_m > 0 and is_sell) or (pos_m < 0 and is_buy)

                reduce_only = bool(reduce_pred and is_reducing_lev)
                lev_raw = float(lev[i].item())
                sc = float(score[i].item())
                lev_out = (lev_raw if (sc >= 0.6 and lev_raw > 1.0) else None)

                actions.append(Action(
                    pair=pair, side=side, qty=qty, price=price_out,
                    ordertype=ordertype,
                    take_profit=tp,  # opzionale: su market spesso gestisci TP/SL a parte
                    leverage=lev_out,
                    stop_loss=sl,
                    # leverage=float(lev[i].item()),
                    time_in_force=tif_out,
                    reduce_only=reduce_only,
                    score=float(score[i].item())
                ))


        return actions


# -----------------------------
# Featurizer
# -----------------------------

STRATEGY_LIST = [
    "momentum","value","pairs","ml","meanrevsr","neural","inventor","trendpb","squeezebo","micro","lgbm"
]

@dataclass
class FeatureSpec:
    fields: List[str]

DEFAULT_FIELDS = [
    "mid","bid","ask","spread","vwap_now",
    "ema50_1h","ema200_1h","ema50_4h","ema200_4h",
    "atr_1h","atr_4h",
    "chg_5m","chg_30m","chg_1h","chg_4h","chg_24h","chg_48h",
    "or_ok","or_range","liq_bid_sum","liq_ask_sum","slip_buy","slip_sell",
    "pos_base","pos_val_eur","avg_buy_eur","pnl_pct","free_base","free_quote",
    "pend_base",
    "urgency","near_daily","near_weekly",
]
DEFAULT_FIELDS += [f"blend_{s}" for s in STRATEGY_LIST]
DEFAULT_FIELDS += [f"w_{s}" for s in STRATEGY_LIST]
FEATURE_SPEC = FeatureSpec(fields=DEFAULT_FIELDS)

def _blk(row: Dict[str,Any], tf: str) -> Dict[str,Any]:
    return ((row.get("info") or {}).get(tf) or {})

def _now_feats(row: Dict[str,Any]) -> Dict[str,float]:
    now = _blk(row,"NOW")
    bid = _f(now.get("bid"))
    ask = _f(now.get("ask"))
    last= _f(now.get("last"))
    mid = _f(now.get("mid")) if _f(now.get("mid"))>0 else ((bid+ask)/2 if bid>0 and ask>0 else last)
    return {"mid": mid, "bid": bid, "ask": ask, "spread": (abs(ask-bid) if (ask>0 and bid>0) else 0.0), "vwap_now": _f(now.get("vwap"))}

def _bias_ma(row: Dict[str,Any]) -> Dict[str,float]:
    now = _blk(row,"NOW")
    return {
        "ema50_1h": _f(now.get("ema50_1h")),
        "ema200_1h": _f(now.get("ema200_1h")),
        "ema50_4h": _f(now.get("ema50_4h")),
        "ema200_4h": _f(now.get("ema200_4h")),
        "atr_1h": _f(now.get("atr")),
        "atr_4h": _f((_blk(row,"4H")).get("atr"))
    }

def _changes(row: Dict[str,Any]) -> Dict[str,float]:
    def ch(tf:str)->float: return _f((_blk(row,tf)).get("change_pct"))
    return {"chg_5m": ch("5M"), "chg_30m": ch("30M"), "chg_1h": ch("1H"), "chg_4h": ch("4H"), "chg_24h": ch("24H"), "chg_48h": ch("48H")}

def _or_slip(row: Dict[str,Any]) -> Dict[str,float]:
    now = _blk(row,"NOW")
    return {"or_ok": 1.0 if _b(now.get("or_ok")) else 0.0, "or_range": _f(now.get("or_range")),
            "liq_bid_sum": _f(now.get("liquidity_bid_sum")), "liq_ask_sum": _f(now.get("liquidity_ask_sum")),
            "slip_buy": _f(now.get("slippage_buy_pct")), "slip_sell": _f(now.get("slippage_sell_pct"))}

def _portfolio(row: Dict[str,Any]) -> Dict[str,float]:
    pf = (row.get("portfolio") or {}); r = (pf.get("row") or {}); avail = pf.get("available") or {}
    pos_base = _f(r.get("qty")); avg_buy = _f(r.get("avg_buy_EUR") or r.get("avg_buy"))
    px_eur   = _f(r.get("px_EUR") or _now_feats(row)["mid"])
    pos_val  = pos_base * px_eur; pnl_pct = _f(r.get("pnl_pct"))
    free_base= _f(avail.get("base")); free_quote = _f(avail.get("quote"))
    return {"pos_base":pos_base,"pos_val_eur":pos_val,"avg_buy_eur":avg_buy,"pnl_pct":pnl_pct,"free_base":free_base,"free_quote":free_quote}

def _pend_delta(row: Dict[str,Any]) -> float:
    oo = row.get("open_orders") or row.get("opens_orders") or []
    delta = 0.0
    for o in oo:
        side = (o.get("type") or o.get("side") or "").lower()
        vol  = _f(o.get("vol_rem") or o.get("volume") or o.get("vol") or o.get("qty"))
        if side == "buy": delta += vol
        elif side == "sell": delta -= vol
    return float(delta)

def _goal(goal_state: Optional[Dict[str,Any]]) -> Tuple[float,float,float]:
    if not isinstance(goal_state, dict): return 0.0,0.0,0.0
    urg = _f(goal_state.get("urgency") or goal_state.get("urg") or 0.0)
    nd  = 1.0 if _b(goal_state.get("near_daily_target") or goal_state.get("near_d")) else 0.0
    nw  = 1.0 if _b(goal_state.get("near_weekly_target") or goal_state.get("near_w")) else 0.0
    return urg, nd, nw

def _blend_vec(blend: Dict[str,Number], weights: Dict[str,Number]) -> Tuple[List[float],List[float]]:
    v, w = [], []
    for s in STRATEGY_LIST:
        v.append(_f(blend.get(s))); w.append(_f(weights.get(s,1.0)))
    return v, w

def featurize_one(row: Dict[str,Any], blend: Dict[str,Number], goal_state: Optional[Dict[str,Any]], strategy_weights: Dict[str,Number]) -> Tuple[List[float], Dict[str,List[Any]]]:
    nowf = _now_feats(row); bias = _bias_ma(row); chgs = _changes(row); ors = _or_slip(row); pf = _portfolio(row); pend = _pend_delta(row)
    urg, nd, nw = _goal(goal_state); bvec, wvec = _blend_vec(blend, strategy_weights)
    flat = [nowf["mid"], nowf["bid"], nowf["ask"], nowf["spread"], nowf["vwap_now"],
            bias["ema50_1h"], bias["ema200_1h"], bias["ema50_4h"], bias["ema200_4h"],
            bias["atr_1h"], bias["atr_4h"],
            chgs["chg_5m"], chgs["chg_30m"], chgs["chg_1h"], chgs["chg_4h"], chgs["chg_24h"], chgs["chg_48h"],
            ors["or_ok"], ors["or_range"], ors["liq_bid_sum"], ors["liq_ask_sum"], ors["slip_buy"], ors["slip_sell"],
            pf["pos_base"], pf["pos_val_eur"], pf["avg_buy_eur"], pf["pnl_pct"], pf["free_base"], pf["free_quote"],
            pend, urg, nd, nw, *bvec, *wvec]
    pair = row.get("pair") or f"{row.get('base','?')}/{row.get('quote','?')}"
    pl = row.get("pair_limits") or {}
    aux = {"pair":[pair], "mid":[nowf["mid"]], "bid":[nowf["bid"]], "ask":[nowf["ask"]],
           "lot_decimals":[int(pl.get("lot_decimals") or 8)], "pair_decimals":[int(pl.get("pair_decimals") or 5)],
           "ordermin":[pl.get("ordermin") or 0.0], "free_quote":[pf["free_quote"]], "free_base":[pf["free_base"]],
           "max_quote_frac":[strategy_weights.get("max_quote_frac",0.12)], "min_notional_eur":[strategy_weights.get("min_notional_eur",5.0)]}
    return flat, aux

# --- Router featurizer: hashing schema-free oppure set fisso (fallback) ---
_auto_feat = AutoFeaturizer(dim=HASH_DIM)

def featurize_batch(rows: List[Dict[str,Any]], blends: List[Dict[str,Number]],
                    goal_state: Optional[Dict[str,Any]],
                    strategy_weights: Dict[str,Number]) -> Tuple[torch.Tensor, Dict[str,List[Any]]]:
    if USE_HASH_FEATS:
        X = []
        aux = {k:[] for k in ["pair","mid","bid","ask","lot_decimals","pair_decimals",
                               "ordermin","free_quote","free_base","pos_margin_base",
                               "slippage_in","slippage_out","fee_maker","fee_taker","max_quote_frac","min_notional_eur"]}
        for row, blend in zip(rows, blends):
            v, a = _auto_feat.featurize(row, blend, goal_state, strategy_weights)
            X.append(v)
            for k in aux.keys(): aux[k].extend(a[k])
        X_t = torch.tensor(X, dtype=torch.float32)
        return X_t, aux
    else:
        # fallback alle features hardcoded esistenti
        X = []; aux = {k:[] for k in ["pair","mid","bid","ask","lot_decimals","pair_decimals","ordermin","free_quote","free_base","pos_margin_base","slippage_in","slippage_out","fee_maker","fee_taker","max_quote_frac","min_notional_eur"]}
        for row, blend in zip(rows, blends):
            flat, a = featurize_one(row, blend, goal_state, strategy_weights)
            X.append(flat)
            for k in aux.keys(): aux[k].extend(a[k])
        X_t = torch.tensor(X, dtype=torch.float32)
        return X_t, aux



# -----------------------------
# Logger
# -----------------------------

class JsonlLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        if path: os.makedirs(os.path.dirname(path), exist_ok=True)
    def log(self, payload: Dict[str,Any]) -> None:
        if not self.path: return
        real = _shadow_daily_path(self.path)
        if not real: return
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------
# High-level Agent
# -----------------------------

class TRMAgent:
    def __init__(self, cfg: Optional[TRMConfig] = None):

        if cfg is None:
            cfg = TRMConfig()
        if USE_HASH_FEATS:
            cfg.feature_dim = HASH_DIM
        else:
            cfg.feature_dim = len(FEATURE_SPEC.fields)
        self.cfg = cfg
        self.model = TinyRecursiveModel(self.cfg)
        self.logger = JsonlLogger(self.cfg.log_path)

    def bootstrap_norm(self, rows: List[Dict[str,Any]], blends: List[Dict[str,Number]], goal_state: Dict[str,Any], strategy_weights: Dict[str,Number]) -> None:
        X, _ = featurize_batch(rows, blends, goal_state, strategy_weights)
        with torch.no_grad():
            self.model.norm.update(X.to(self.cfg.device))

    def _seed_y0(self, Xn: torch.Tensor) -> torch.Tensor:
        B, D = Xn.shape
        W = torch.randn(D, self.cfg.hidden_dim, device=Xn.device) * 0.01
        return Xn @ W

    @torch.no_grad()
    def propose_actions_for_batch(self, rows: List[Dict[str,Any]], blends: List[Dict[str,Number]], goal_state: Dict[str,Any], strategy_weights: Dict[str,Number], K: Optional[int] = None, shadow_log: bool = True) -> List[Action]:
        assert len(rows) == len(blends), "rows/blends length mismatch"
        X, aux = featurize_batch(rows, blends, goal_state, strategy_weights)
        X = X.to(self.cfg.device)
        self.model.norm.update(X)
        Xn = self.model.norm(X)
        y0 = self._seed_y0(Xn)
        yK = self.model.improve(X, y0=y0, K=K)
        actions = self.model.decode_actions(yK, aux)
        by_pair: Dict[str, List[Action]] = {}
        for a in actions: by_pair.setdefault(a.pair, []).append(a)
        pair2input = {}
        for row, blend in zip(rows, blends):
            now = ((row.get("info") or {}).get("NOW") or {})
            pf  = (row.get("portfolio") or {})
            rpf = (pf.get("row") or {})
            liq_bid = float(now.get("liquidity_bid_sum") or 0.0)
            liq_ask = float(now.get("liquidity_ask_sum") or 0.0)
            inputs = {
                "now": {
                    "bid": float(now.get("bid") or 0.0),
                    "ask": float(now.get("ask") or 0.0),
                    "mid": float(now.get("mid") or ((float(now.get("bid") or 0.0)+float(now.get("ask") or 0.0))/2.0)),
                    "spread": abs(float(now.get("ask") or 0.0) - float(now.get("bid") or 0.0)) if (now.get("bid") and now.get("ask")) else 0.0,
                    "vwap": float(now.get("vwap") or 0.0),
                },
                "liquidity": {
                    "liq_bid_sum": liq_bid,
                    "liq_ask_sum": liq_ask,
                    "slippage_buy_pct": float(now.get("slippage_buy_pct") or 0.0),
                    "slippage_sell_pct": float(now.get("slippage_sell_pct") or 0.0),
                },
                "derived": {
                    "or_ok": bool(now.get("or_ok") or False),
                    "or_range": float(now.get("or_range") or 0.0),
                },
                "portfolio": {
                    "pos_base": float(rpf.get("qty") or 0.0),
                    "pos_val_eur": float(rpf.get("px_EUR") or 0.0) * float(rpf.get("qty") or 0.0),
                    "pnl_pct": float(rpf.get("pnl_pct") or 0.0),
                    "free_base": float((pf.get("available") or {}).get("base") or 0.0),
                    "free_quote": float((pf.get("available") or {}).get("quote") or 0.0),
                },
            }
            pair = row.get("pair") or f"{row.get('base','?')}/{row.get('quote','?')}"
            pair2input[pair] = {
                "inputs": inputs,
                "blend": dict(blend or {}),
            }

        # selezione finale + logging esteso
        by_pair = {}
        for a in actions:
            by_pair.setdefault(a.pair, []).append(a)
        final = []
        for pair, acts in by_pair.items():
            acts.sort(key=lambda a: a.score, reverse=True)
            final.extend(acts[: self.cfg.max_actions_per_pair])

        if shadow_log:
            for a in final:
                decision_id = str(uuid.uuid4())
                meta_in = pair2input.get(a.pair, {"inputs": {}, "blend": {}})
                payload = {
                    "event_type": "decision",
                    "ts": _iso_now(),
                    "decision_id": decision_id,
                    "pair": a.pair,
                    "action": a.as_dict(),
                    "score": a.score,
                    "inputs": meta_in["inputs"],
                    "blend": meta_in["blend"],
                    "weights": {k: float(v) for k, v in (strategy_weights or {}).items()},
                    "goal_state": goal_state or {},
                }
                # scrivi nel JSONL
                self.logger.log(payload)
                # opzionale: attacca l'id all'azione così chi esegue può ributtarlo nel log ‘execution/close’
                a.notes = (a.notes or "") + f" | decision_id={decision_id}"
        return final

    @torch.no_grad()
    def propose_actions_legacy(
        self,
        rows: List[Dict[str,Any]],
        blends: List[Dict[str,Number]],
        goal_state: Dict[str,Any],
        strategy_weights: Dict[str,Number],
        K: Optional[int] = None,
        shadow_log: bool = True
    ) -> List[Dict[str, Any]]:
        # 1) generiamo azioni TRM “native”
        actions = self.propose_actions_for_batch(
            rows=rows, blends=blends,
            goal_state=goal_state, strategy_weights=strategy_weights,
            K=K, shadow_log=shadow_log
        )

        # 2) ri-costruiamo aux per avere decimali e mid per ogni pair
        _, aux = featurize_batch(rows, blends, goal_state, strategy_weights)

        # 3) mappa pair -> (row, blend_val, pair_decimals)
        by_pair = {}
        for r, b, pd in zip(rows, blends, aux["pair_decimals"]):
            pair = r.get("pair") or f"{r.get('base','?')}/{r.get('quote','?')}"
            # blend in molti tuoi esempi è un singolo float ‘blend’ del meta / ensemble
            # se non c’è, proviamo a stimarlo: media pesata dei segnali non la conosciamo qui -> fallback 0.5
            blend_val = (r.get("meta") or {}).get("blend")
            if blend_val is None:
                blend_val = 0.5
            by_pair[pair] = (r, float(blend_val), int(pd))

        # 4) serializza in schema legacy
        out_list: List[Dict[str,Any]] = []
        for a in actions:
            row, blend_val, pair_dec = by_pair.get(a.pair, ({}, 0.5, 5))
            out_list.append(_legacy_one(row, a, blend_val, pair_dec))
        return out_list


# -----------------------------
# Legacy serializer (schema ITA)
# -----------------------------
def _legacy_reason_for_side(side: str) -> str:
    s = (side or "").lower()
    if s == "buy":  return "Entrata con prudenza"
    if s == "sell": return "Uscita prudente"
    return "Meglio Attendere Momenti Migliori"  # hold

def _legacy_motivo_for(side: str, ordertype: str) -> str:
    s = (side or "").lower()
    if s == "hold": return "looking"
    if ordertype == "market":
        return f"{s} market"
    return f"{s} limite"  # default

def _get_tf_from_row(row: dict) -> str:
    # fallback robusto: molti tuoi file usano sempre "24H"
    meta = (row.get("meta") or {})
    return meta.get("tf") or row.get("timeframe") or "24H"

def _legacy_meta_from_row(row: dict, blend_val: float, score: float) -> dict:
    meta_in = dict(row.get("meta") or {})
    # prendi liste se già presenti (signals/weights/weights_eff/contribs/w_names)
    out = {
        "reason": meta_in.get("reason") or "",  # lo rimpiazzeremo se vuoto
        "signals": meta_in.get("signals") or [],
        "weights": meta_in.get("weights") or [],
        "weights_eff": meta_in.get("weights_eff") or [],
        "contribs": meta_in.get("contribs") or [],
        "w_names": meta_in.get("w_names") or meta_in.get("weights_names") or [],
        "blend": float(blend_val),
        "conf": float(blend_val),   # nei tuoi esempi conf == blend
        # opzionale: info TRM attuale
        "trm": {"score": float(score)},
    }
    return out

def _euros(qty: float, price: float) -> float:
    try:
        return float(qty) * float(price)
    except Exception:
        return 0.0

def _legacy_one(row: dict, a: "Action", blend_val: float, pair_decimals: int) -> dict:
    # prezzo: se market provo a usare NOW.mid come best-effort, altrimenti quello dell'azione
    now_mid = ((row.get("info") or {}).get("NOW") or {}).get("mid")
    prezzo = a.price if a.ordertype != "market" else (now_mid if now_mid is not None else a.price)

    quantita = a.qty if a.qty is not None else 0.0
    quantita_eur = (quantita * prezzo) if (prezzo is not None) else None

    meta = _legacy_meta_from_row(row, blend_val, a.score)
    if not meta.get("reason"):
        meta["reason"] = _legacy_reason_for_side(a.side)

    tf = _get_tf_from_row(row)
    motivo = _legacy_motivo_for(a.side, a.ordertype)
    limite = prezzo if a.ordertype == "limit" else None

    # --- NUOVO: estrai decision_id dalle note dell'Action (set da propose_actions_for_batch)
    decision_id = None
    try:
        if a.notes:
            m = re.search(r"decision_id=([0-9a-fA-F-]{36})", str(a.notes))
            if m:
                decision_id = m.group(1)
    except Exception:
        decision_id = None

    out = {
        "pair": a.pair,
        "tipo": a.side,                           # "buy" | "sell" | "hold"
        "prezzo": prezzo,                         # nessun round
        "quando": ("hold" if a.side=="hold" else a.ordertype),
        "quantita": quantita,                     # nessun round
        "quantita_eur": quantita_eur,             # qty * prezzo, nessun round
        "stop_loss": (None if a.side=="hold" else a.stop_loss),
        "take_profit": (None if a.side=="hold" else a.take_profit),
        "timeframe": tf,
        "lato": a.side,
        "break_price": None,
        "limite": limite,
        "leverage":(None if a.side=="hold" else (int(a.leverage) if a.leverage is not None else 1)),
        "motivo": motivo,
        "meta": meta,
        "reduce_only": bool(getattr(a, "reduce_only", False)),
        "blend": float(blend_val),

        # extra utili per downstream (solo informativi, NON usati per arrotondare qui):
        "pair_decimals": pair_decimals,
        "lot_decimals": (row.get("pair_limits") or {}).get("lot_decimals"),

        "cancel_order_id": "",

        # nuovi per training (placeholders, li valorizzerai dopo):
        "pnl": None,
        "orderId": None,
        "error": None,
        "decision_id": decision_id
    }
    return out





if USE_HASH_FEATS:
    FEATURE_NAMES = [f"hashed_{i}" for i in range(HASH_DIM)]
else:
    FEATURE_NAMES = FEATURE_SPEC.fields
