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

from .Util import _shadow_daily_path, _net_margin_from_open_orders, _finite, _rel, _rel_to


# -----------------------------
# Utility helpers
# -----------------------------

Number = Union[int, float]

_EPS = 1e-9

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
# Trade phase helpers
# -----------------------------

def infer_inventory_state(pos_base: float, pos_margin_base: float) -> str:
    """Infer current inventory regime before taking a new action."""
    try:
        pos_base = float(pos_base or 0.0)
    except Exception:
        pos_base = 0.0
    try:
        pos_margin_base = float(pos_margin_base or 0.0)
    except Exception:
        pos_margin_base = 0.0

    if pos_base > _EPS and pos_margin_base <= _EPS:
        return "long_spot"
    if pos_margin_base > _EPS:
        return "long_margin"
    if pos_margin_base < -_EPS:
        return "short_open"
    return "flat"


def infer_trade_phase(side: str,
                      reduce_only: bool,
                      leverage: Optional[Number],
                      pos_base: float,
                      pos_margin_base: float) -> str:
    """Infer the trade phase (entry/exit) for supervision and decoding."""
    side = (side or "hold").lower()
    lev = 0.0
    try:
        lev = float(leverage or 0.0)
    except Exception:
        lev = 0.0
    inv = infer_inventory_state(pos_base, pos_margin_base)

    if side == "hold":
        return "hold"

    if side == "buy":
        # closing an open short either because reduce_only is flagged or inventory is short
        if reduce_only or inv == "short_open":
            return "close_short"
        return "open_long"

    if side == "sell":
        # selling spot inventory / margin long counts as a close
        if inv in ("long_spot", "long_margin") and not (lev > 1.0 and inv != "long_spot"):
            return "close_long"
        if lev > 1.0:
            return "open_short"
        # if we still have spot size assume partial close
        if inv in ("long_spot", "long_margin"):
            return "close_long"
        return "open_short"

    return "hold"

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



class DailyMemory:
    """
    Carica dal file shadow_actions della data odierna info semplici:
    - per pair: last_side, last_price, last_score, last_decision_ts
    - conteggi buy/sell/hold del giorno
    """
    def __init__(self):
        self.by_pair = {}
        self.counts = {}
        self.no_trade = set()   # es: "Invalid permissions:* for IT"
        self.no_margin = set()  # es: "Margin trading in asset is restricted"

    @staticmethod
    def _today_path(base_dir: str) -> str:
        # se hai già un helper _shadow_daily_path usa quello
        if base_dir:
            base_dir = os.path.join(base_dir, "shadow_actions.jsonl")
        try:
            return _shadow_daily_path(base_dir)  # già presente nel tuo Util
        except Exception:
            # fallback: unico file
            return os.path.join(base_dir, "shadow_actions.jsonl")

    def load_today(self, base_dir: str) -> None:
        path = self._today_path(base_dir)
        self.by_pair, self.counts = {}, {}
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                act = (rec.get("action") or {})
                pair = act.get("pair") or ((rec.get("pair")) if isinstance(rec.get("pair"), str) else None)
                side = act.get("side")
                px   = act.get("price")
                sc   = act.get("score")
                ts   = rec.get("ts") or rec.get("timestamp")

                if not pair:
                    continue
                # aggiorna ultimo
                self.by_pair[pair] = {"last_side": side, "last_price": px, "last_score": sc, "last_ts": ts, "last_entry_price": act.get("prezzo") or act.get("price"), "last_take_profit": act.get("take_profit") or act.get("tp")}
                # self.by_pair[pair]["tp_hit_now"] = rec.get("tp_hit_now") or d.get("tp_hit_now")
                # conteggi
                c = self.counts.get(pair, {"buy":0, "sell":0, "hold":0})
                if side in c:
                    c[side] += 1
                self.counts[pair] = c
                try:
                    err = ((act.get("after") or {}).get("kraken_result") or {}).get('error')
                except Exception:
                    err = ''
                em = err if err else ''
                if "EAccount:Invalid permissions" in em:
                    self.no_trade.add(pair)
                if "EOrder:Margin trading in asset is restricted" in em and "restricted" in em:
                    self.no_margin.add(pair)

    def memory_feats_for(self, pair: str) -> dict:
        info = self.by_pair.get(pair, {})
        cnt  = self.counts.get(pair, {"buy":0, "sell":0, "hold":0})
        # ritorna feature “piatte” pronte per hashing
        ls = info.get("last_side")
        side_onehot = {"buy":0.0,"sell":0.0,"hold":0.0}
        if ls in side_onehot: side_onehot[ls] = 1.0
        return {
            "mem.last_side_buy": side_onehot["buy"],
            "mem.last_side_sell": side_onehot["sell"],
            "mem.last_side_hold": side_onehot["hold"],
            "mem.last_price": float(info.get("last_price") or 0.0),
            "mem.last_score": float(info.get("last_score") or 0.0),
            "mem.cnt_buy": float(cnt["buy"]),
            "mem.cnt_sell": float(cnt["sell"]),
            "mem.cnt_hold": float(cnt["hold"]),
            "mem.last_entry_price": float(info.get("last_entry_price") or 0.0),
            "mem.last_take_profit": float(info.get("last_take_profit") or 0.0),
        }



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
            fv = float(val)
            if math.isfinite(fv):
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
        pivot_ctx = _pivot_context(row, mid)

        lot_dec   = int(pl.get("lot_decimals") or 8)
        pair_dec  = int(pl.get("pair_decimals") or 5)
        ordermin  = self._num(pl.get("ordermin"))

        # portfolio for aux
        pf = (row.get("portfolio") or {})
        avail = (pf.get("available") or {})
        pos_row = (pf.get("row") or {})
        pos_base_qty = self._num(pos_row.get("qty"))
        free_quote = self._num(avail.get("quote"))
        free_base  = self._num(avail.get("base"))

        pos_margin_base = _net_margin_from_open_orders(row)
        # --- regime derivati (robusti)
        # EMA/ATR possono stare in info.NOW oppure in info.1H/info.4H a seconda del tuo loader: prendiamo le 2 strade
        ema50_1h  = self._num(((info.get("1H") or {}).get("EMA50") if isinstance(info.get("1H"), dict) else (now.get("ema50_1h"))))
        ema200_1h = self._num(((info.get("1H") or {}).get("EMA200") if isinstance(info.get("1H"), dict) else (now.get("ema200_1h"))))
        atr_4h    = self._num(((info.get("4H") or {}).get("atr")    if isinstance(info.get("4H"), dict) else (now.get("atr_4h"))))

        fc_meta = (info.get("FORECAST_META") or {})
        p50 = fc_meta.get("p50") if isinstance(fc_meta.get("p50"), (list, tuple)) else None
        t_m1 = self._num(((info.get("24H") or {}).get("start_price")))
        t_m2 = self._num(((info.get("48H") or {}).get("start_price")))
        t0 = mid if (mid and mid > 0.0) else 0
        t_p1 = float(p50[0]) if (isinstance(p50, (list, tuple)) and len(p50) >= 1 and _is_num(p50[0])) else 0
        t_p2 = float(p50[1]) if (isinstance(p50, (list, tuple)) and len(p50) >= 2 and _is_num(p50[1])) else 0


        tpv_vals = [
            float(t_m2 or 0.0),  # t-2  (48H start_price)
            float(t_m1 or 0.0),  # t-1  (24H start_price)
            float(t0   or 0.0),  # t0   (NOW.mid)
            float(t_p1 or 0.0),  # t+1  (forecast p50[0])
            float(t_p2 or 0.0),  # t+2  (forecast p50[1])
        ]
        tpv_has = 1.0 if any([t_m2, t_m1, t0, t_p1, t_p2]) else 0.0

        if t0 and t0 > 0.0:
            self._add(vec, "tpv.t-2.rel_t0:num", _rel(t_m2, t0))
            self._add(vec, "tpv.t-1.rel_t0:num", _rel(t_m1, t0))
            self._add(vec, "tpv.t+1.rel_t0:num", _rel(t_p1, t0))
            self._add(vec, "tpv.t+2.rel_t0:num", _rel(t_p2, t0))
            # pendenza forward (t+2 - t+1) / t0
            try:
                if t_p2 and t_p1:
                    self._add(vec, "tpv.fwd_slope:num", (float(t_p2) - float(t_p1)) / t0)
            except Exception:
                pass
            # direzione forward semplice (segno di t+1 - t0)
            try:
                self._add(vec, "tpv.dir_fwd:num", math.copysign(1.0, (float(t_p1) - t0)) if (t_p1 and t_p1 != t0) else 0.0)
            except Exception:
                pass
        # 2) associazione ai pivot principali: distanze relative per t0 e t+1
        levels = pivot_ctx.get("levels", {}) if isinstance(pivot_ctx, dict) else {}

        for name in ("pp","r1","r2","s1","s2"):
            lvl = levels.get(name)
            if lvl is None:
                continue
            rel_t0 = _rel_to(lvl, t0)
            rel_t1 = _rel_to(lvl, t_p1)
            if rel_t0 is not None:
                self._add(vec, f"tpv.t0.rel_{name}:num", rel_t0)
            if rel_t1 is not None:
                self._add(vec, f"tpv.t+1.rel_{name}:num", rel_t1)

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

        if pivot_ctx.get("has"):
            zone = pivot_ctx.get("zone")
            if zone is not None:
                zone_f = float(zone)
                self._add(vec, "derived.pivot.zone:num", zone_f)
                self._add(vec, "derived.pivot.zone_abs:num", abs(zone_f))
            bias = pivot_ctx.get("bias")
            if bias is not None:
                self._add(vec, "derived.pivot.bias:num", float(bias))
            rng = pivot_ctx.get("range_norm")
            if rng is not None:
                self._add(vec, "derived.pivot.range_norm:num", float(rng))
            dist_up = pivot_ctx.get("dist_up")
            if dist_up is not None:
                self._add(vec, "derived.pivot.dist_up:num", float(dist_up))
            dist_down = pivot_ctx.get("dist_down")
            if dist_down is not None:
                self._add(vec, "derived.pivot.dist_down:num", float(dist_down))
            closeness = pivot_ctx.get("closeness")
            if closeness is not None:
                closeness_f = float(closeness)
                self._add(vec, "derived.pivot.closeness:num", closeness_f)
                self._add(vec, "derived.pivot.closeness_inv:num", math.exp(-abs(closeness_f) * 120.0))

        # --- campi per cost estimator in trainer (se disponibili)
        slip_in  = self._num(now.get("slippage_buy_pct"))
        slip_out = self._num(now.get("slippage_sell_pct"))
        fee_maker = self._num((pl.get("fee_maker") or pl.get("maker_fee") or 0.001))
        fee_taker = self._num((pl.get("fee_taker") or pl.get("taker_fee") or 0.001))
        # aux per decoder
        lev_buy_max  = int(pl.get("leverage_buy_max") or 0)
        lev_sell_max = int(pl.get("leverage_sell_max") or 0)
        # feature hashed (aiuta il modello a capire dove può usare leva)
        self._add(vec, "pair_limits.leverage_buy_max:num", lev_buy_max)
        self._add(vec, "pair_limits.leverage_sell_max:num", lev_sell_max)

        aux = {
            "pair":[pair], "mid":[mid], "bid":[bid], "ask":[ask],
            "lot_decimals":[lot_dec], "pair_decimals":[pair_dec], "ordermin":[ordermin],
            "free_quote":[free_quote], "free_base":[free_base],
            "pos_margin_base":[pos_margin_base],
            "pos_base":[pos_base_qty],
            "slippage_in":[slip_in], "slippage_out":[slip_out],
            "fee_maker":[fee_maker], "fee_taker":[fee_taker],
            "max_quote_frac":[float(strategy_weights.get("max_quote_frac",0.12))],
            "min_notional_eur":[float(strategy_weights.get("min_notional_eur",5.0))],
            "lev_buy_max":  [lev_buy_max],
            "lev_sell_max": [lev_sell_max],
        }

        levels = pivot_ctx.get("levels", {})

        def _pval(name: str) -> Optional[float]:
            val = levels.get(name)
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        aux.update({
            "pivot_has_meta": [1.0 if pivot_ctx.get("has") else 0.0],
            "pivot_pp": [_pval("pp")],
            "pivot_r1": [_pval("r1")],
            "pivot_r2": [_pval("r2")],
            "pivot_s1": [_pval("s1")],
            "pivot_s2": [_pval("s2")],
            "pivot_near_up": [pivot_ctx.get("near_up")],
            "pivot_near_down": [pivot_ctx.get("near_down")],
            "pivot_near_up_dist": [pivot_ctx.get("dist_up")],
            "pivot_near_down_dist": [pivot_ctx.get("dist_down")],
            "pivot_zone": [pivot_ctx.get("zone")],
            "pivot_bias": [pivot_ctx.get("bias")],
            "pivot_range": [pivot_ctx.get("range_norm")],
            "pivot_closeness": [pivot_ctx.get("closeness")],
        })

        unit = max(mid, 1e-9)  # per normalizzare EUR -> "pezzi" circa

        _ratio = (free_quote / unit) if unit > 0 else 0.0
        _ratio = max(_ratio, -0.999999)  # FIX: evita log1p(x<=-1)
        self._add(vec, "aux.free_quote_scaled:num", math.log1p(_ratio))
        self._add(vec, "aux.free_quote_sign:num", 1.0 if free_quote >= 0 else -1.0)
        # liquidità disponibile
        # self._add(vec, "aux.free_quote_scaled:num", math.log1p(free_quote / unit))
        self._add(vec, "aux.free_base_scaled:num",  math.log1p(max(0.0, free_base)))

        # minimo scambiabile (ordermin è in base units): normalizza e log1p
        self._add(vec, "aux.ordermin_scaled:num",   math.log1p(max(0.0, ordermin)))

        # pressione da posizione a margine (positiva long, negativa short)
        self._add(vec, "aux.pos_margin_base_tanh:num", math.tanh(pos_margin_base))

        inv_state = infer_inventory_state(pos_base_qty, pos_margin_base)
        self._add(vec, f"inventory.state:{inv_state}", 1.0)
        if inv_state == "long_spot":
            self._add(vec, "inventory.long_spot:num", math.log1p(max(pos_base_qty, 0.0)))
        elif inv_state == "long_margin":
            self._add(vec, "inventory.long_margin:num", math.log1p(max(pos_margin_base, 0.0)))
        elif inv_state == "short_open":
            self._add(vec, "inventory.short_open:num", math.log1p(max(-pos_margin_base, 0.0)))

        # policy di run/config
        self._add(vec, "aux.max_quote_frac:num", float(strategy_weights.get("max_quote_frac", 0.12)))
        self._add(vec, "aux.min_notional_eur_scaled:num", math.log1p(float(strategy_weights.get("min_notional_eur", 5.0)) / unit))

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

        fc = ((row.get("info") or {}).get("FORECAST") or {})
        try:
            e = float(fc.get("timesfm_edge_1d") or 0.0)
            u = float(fc.get("timesfm_uncert_1d") or 0.0)
            conf = abs(e) / max(u, 1e-6)
            self._add(vec, "derived.timesfm_conf:num", float(max(0.0, min(1.0, conf))))
            self._add(vec, "derived.timesfm_dir:num",  1.0 if e>0 else (-1.0 if e<0 else 0.0))
        except Exception:
            pass

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

        # --- NEW: 1) Liquidity imbalance & total (safe-scaled) ---
        liq_bid = self._num(now.get("liquidity_bid_sum"))
        liq_ask = self._num(now.get("liquidity_ask_sum"))
        liq_tot = liq_bid + liq_ask
        liq_imb = ((liq_bid - liq_ask) / (abs(liq_tot) + 1e-12)) if liq_tot != 0.0 else 0.0
        self._add(vec, "derived.liq_imbalance:num", liq_imb)
        self._add(vec, "derived.liq_tot:num", (liq_tot / (abs(mid) + 1e-12)) if mid > 0 else 0.0)

        # --- NEW: 2) Expected execution cost (bps) for buy/sell ---
        spread_abs = (abs(ask - bid) if (ask > 0 and bid > 0) else 0.0)
        half_spread_bps = (10000.0 * (0.5 * spread_abs / max(mid, 1e-12))) if mid > 0 else 0.0
        slip_buy_bps  = 100.0 * self._num(now.get("slippage_buy_pct"))
        slip_sell_bps = 100.0 * self._num(now.get("slippage_sell_pct"))
        fee_maker = self._num((pl.get("fee_maker") or pl.get("maker_fee") or 0.001))
        fee_taker = self._num((pl.get("fee_taker") or pl.get("taker_fee") or 0.001))
        fee_taker_bps = 10000.0 * fee_taker
        exp_cost_buy_bps  = half_spread_bps + slip_buy_bps  + fee_taker_bps
        exp_cost_sell_bps = half_spread_bps + slip_sell_bps + fee_taker_bps
        self._add(vec, "derived.exp_cost_buy_bps:num",  exp_cost_buy_bps)
        self._add(vec, "derived.exp_cost_sell_bps:num", exp_cost_sell_bps)

        # --- NEW: 3) Inventory & pending pressure ---
        # pos_base stimata dal portfolio.row.qty (come nel dataset) e mid

        pos_val_eur = pos_base_qty * mid if mid > 0 else 0.0
        inv_pressure = (pos_val_eur / (abs(free_quote) + 1e-9)) if free_quote > 0 else 0.0
        # pendency: usa la tua funzione già calcolata (netto ordini aperti)
        pend_pressure = (abs(pos_margin_base) / (abs(pos_base_qty) + abs(free_base) + 1e-9))
        self._add(vec, "derived.inv_pressure:num", inv_pressure)
        self._add(vec, "derived.pend_pressure:num", pend_pressure)

        # --- NEW: 4) Momentum agreement multi-timeframe ---
        def _chg(tf: str) -> float:
            blk = (info.get(tf) or {})
            return self._num(blk.get("change_pct"))
        pos_cnt = neg_cnt = tot_cnt = 0
        for tf in ["5M", "30M", "1H", "4H", "24H", "48H"]:
            c = _chg(tf)
            if c > 0:  pos_cnt += 1; tot_cnt += 1
            elif c < 0: neg_cnt += 1; tot_cnt += 1
            elif c == 0: tot_cnt += 1  # conta il flat per stabilizzare il denominatore
        mom_agree = (pos_cnt - neg_cnt) / float(tot_cnt or 1)
        self._add(vec, "derived.mtf_mom_agreement:num", mom_agree)

        # --- NEW: cross utile tra i due regimi che già calcoli sopra ---
        self._add(vec, "derived.trendXvol:num", float(trend_bias) * float(vol_regime))

        return vec, aux

# Flag per attivare il featurizer schema-free
USE_HASH_FEATS = True
HASH_DIM = 8192 # puoi portarlo a 1024/2048 se vuoi più capacità

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
    _side_prob: Optional[float] = None
    _entropy: Optional[float] = None
    trade_phase: Optional[str] = None

    def as_dict(self, keep_none: bool = False) -> dict:
        d = asdict(self)            # non rimuove None
        if not keep_none:
            # comportamento precedente: togli i None
            d = {k: v for k, v in d.items() if v is not None}
        return d



# -----------------------------
# Config
# -----------------------------

@dataclass
class TRMConfig:
    feature_dim: int = 128
    hidden_dim: int = 128
    mlp_hidden: int = 256
    K_refine: int = 6
    max_actions_per_pair: int = 2
    norm_eps: float = 1e-6
    log_path: Optional[str] = None
    device: str = "cpu"
    total_budget_quote: Optional[float] = None
    reserve_frac: float = 0.20                   # 20% riserva
    super_conf_score: float = 0.85               # soglia "super convinta" su score (tanh ∈ [-1,1])
    super_conf_sideprob: float = 0.65
    run_currencies_left: Optional[int] = None
    memory_path: Optional[str] = None
    act_enabled: bool = True           # abilita halting adattivo (se False usa K fisso)
    act_threshold: float = 0.60
    act_min_steps: int = 1
    # passi minimi prima di poter fermare
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
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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
        self.y0_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim, bias=False)
        nn.init.xavier_normal_(self.y0_proj.weight)
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
        self.head_halt      = nn.Linear(self.y_dim, 1)  # sigmoid -> p_halt (ACT)
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
    def improve_adaptive(
        self,
        x: torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        threshold: float = 0.60,
        return_stats: bool = True,
    ):
        """
        ACT: ad ogni passo predice p_halt e decide se fermarsi per elemento di batch.
        Usa la stessa normalizzazione di improve() ed è retro-compatibile.
        """
        self.eval()
        T = int(max_steps or self.cfg.K_refine)
        m = int(min_steps or getattr(self.cfg, "act_min_steps", 1) or 1)
        x_n = self.norm(x)
        B = x_n.shape[0]
        y = torch.zeros(B, self.y_dim, device=x_n.device) if y0 is None else y0.to(x_n.device).view(B, self.y_dim)
        h = None

        stopped = torch.zeros(B, dtype=torch.bool, device=x_n.device)
        steps   = torch.zeros(B, dtype=torch.int64, device=x_n.device)
        last_p  = torch.zeros(B, dtype=torch.float32, device=x_n.device)
        last_t  = 0

        for t in range(T):
            last_t = t + 1
            y_new, h_new = self.forward(x_n, y, h)
            p_halt = torch.sigmoid(self.head_halt(y_new)).squeeze(-1)
            last_p = p_halt

            can_halt = (last_t >= m)
            stop_now = (~stopped) & can_halt & (p_halt >= threshold)

            if stop_now.any():
                steps[stop_now] = last_t
            stopped = stopped | stop_now

            if (~stopped).any():
                y[~stopped] = y_new[~stopped]
                if h_new is not None:
                    if h is None:
                        h = h_new
                    else:
                        h[:, ~stopped, :] = h_new[:, ~stopped, :]

            if stopped.all():
                break

        steps = torch.where(steps == 0, torch.full_like(steps, last_t), steps)
        if return_stats:
            return y, {
                "steps": steps.detach().cpu().tolist(),
                "p_halt": last_p.detach().cpu().tolist(),
                "threshold": float(threshold),
            }
        return y



    @torch.no_grad()
    def decode_actions(self, y: torch.Tensor, aux: Dict[str, List[Any]]) -> List[Action]:
        self.eval()
        logits_side = self.head_side(y)
        p_side = torch.softmax(logits_side, dim=-1)
        qty_frac = torch.sigmoid(self.head_qty(y)).squeeze(-1)

        px_raw = self.head_px(y)
        px_raw = torch.nan_to_num(px_raw, nan=0.0, posinf=0.0, neginf=0.0)            # FIX
        px_off = torch.tanh(px_raw).squeeze(-1) * 0.05

        tp_mult = F.softplus(self.head_tp(y)).squeeze(-1) * 0.01
        sl_mult = F.softplus(self.head_sl(y)).squeeze(-1) * 0.01
        tif_idx = torch.argmax(self.head_tif(y), dim=-1)
        lev = F.softplus(self.head_lev(y)).squeeze(-1) + 1.0
        score = torch.tanh(self.head_score(y)).squeeze(-1)
        ord_logits = self.head_ordertype(y)
        ord_idx = torch.argmax(ord_logits, dim=-1)
        p_reduce = torch.sigmoid(self.head_reduce(y)).squeeze(-1)

        def _aux_val(key: str, idx: int, default: Any = None):
            arr = aux.get(key)
            if arr is None:
                return default
            try:
                if isinstance(arr, list):
                    if idx < len(arr):
                        val = arr[idx]
                    else:
                        return default
                else:
                    val = arr
            except Exception:
                return default
            return default if val is None else val

        actions: List[Action] = []
        B = y.shape[0]
        for i in range(B):
            pair = aux["pair"][i]
            mid = _f(aux["mid"][i], 0.0)
            bid = _f(aux["bid"][i], mid)
            ask = _f(aux["ask"][i], mid)
            lot_dec = int(aux["lot_decimals"][i])
            pair_dec = int(aux["pair_decimals"][i])
            ordermin = _f(aux["ordermin"][i], 0.0)
            free_quote = _f(aux["free_quote"][i], 0.0)
            free_base = _f(aux["free_base"][i], 0.0)
            max_qfrac = float(aux.get("max_quote_frac", [0.12])[i] if isinstance(aux.get("max_quote_frac"), list) else aux.get("max_quote_frac", 0.12))
            min_notional = float(aux.get("min_notional_eur", [5.0])[i] if isinstance(aux.get("min_notional_eur"), list) else aux.get("min_notional_eur", 5.0))
            minOrder =  float(aux.get("minOrder", [0.0])[i])
            tp_hint = float((aux.get("tp_mult_hint") or [0.0])[i])
            sl_hint = float((aux.get("sl_mult_hint") or [0.0])[i])
            tfs = float((aux.get("timesfm_sig") or [0.0])[i])
            hint_weight = min(1.0, max(0.0, abs(tfs)))
            tp_eff = float(tp_mult[i].item()) * 0.5 + tp_hint * 0.5 * hint_weight
            sl_eff = float(sl_mult[i].item()) * 0.5 + sl_hint * 0.5 * hint_weight

            side_idx = int(torch.argmax(p_side[i]).item())
            p_hat = float(p_side[i, side_idx].item())
            H = float((-(p_side[i] * torch.log(p_side[i] + 1e-12)).sum()).item())

            is_buy  = (side_idx == 0)
            is_hold = (side_idx == 1)
            is_sell = (side_idx == 2)

            side = "buy" if is_buy else ("hold" if is_hold else "sell")

            try:
                anchor = bid if is_buy else ask
                base_px = mid if mid > 0 else anchor
                px = base_px * (1.0 + float(px_off[i].item()))
                px = round(px, pair_dec)
            except Exception:
                px = _finite(bid or ask, 0.0)

            tp = px * (1.0 + (tp_eff if is_buy else -tp_eff))
            sl = px * (1.0 - (sl_eff if is_buy else -sl_eff))
            tp = round(tp, pair_dec)
            sl = round(sl, pair_dec)

            def _to_float(val):
                try:
                    return float(val)
                except Exception:
                    return None

            pivot_has = bool(_aux_val("pivot_has_meta", i, 0.0))
            pivot_up = _to_float(_aux_val("pivot_near_up", i)) if pivot_has else None
            pivot_down = _to_float(_aux_val("pivot_near_down", i)) if pivot_has else None
            pivot_zone = _to_float(_aux_val("pivot_zone", i)) if pivot_has else None
            pivot_bias = _to_float(_aux_val("pivot_bias", i)) if pivot_has else None
            pivot_closeness = _to_float(_aux_val("pivot_closeness", i)) if pivot_has else None
            pivot_pp = _to_float(_aux_val("pivot_pp", i)) if pivot_has else None

            pivot_notes: List[str] = []
            pivot_force_limit = False

            if pivot_has and mid > 0.0 and px > 0.0:
                def _mix_from_dist(delta: float) -> float:
                    if delta <= 0.0015:
                        return 0.6
                    if delta <= 0.004:
                        return 0.4
                    if delta <= 0.01:
                        return 0.22
                    if delta <= 0.02:
                        return 0.12
                    return 0.0

                closeness_factor = 1.0
                if pivot_closeness is not None:
                    closeness_factor = max(0.0, min(1.0, math.exp(-abs(pivot_closeness) * 80.0)))

                if is_buy:
                    if pivot_up is not None and pivot_up > px:
                        delta = abs((pivot_up - px) / max(px, 1e-9))
                        mix = _mix_from_dist(delta) * closeness_factor
                        if mix > 0.0:
                            tp = round(tp * (1.0 - mix) + pivot_up * mix, pair_dec)
                            pivot_notes.append(f"tp→{pivot_up:.4f}")
                    if pivot_down is not None and pivot_down < px:
                        delta = abs((px - pivot_down) / max(px, 1e-9))
                        mix = _mix_from_dist(delta) * closeness_factor
                        if mix > 0.0:
                            px = round(px * (1.0 - mix) + pivot_down * mix, pair_dec)
                            pivot_force_limit = True
                            pivot_notes.append(f"px→sup@{pivot_down:.4f}")
                        guard_buffer = 0.0015 if delta <= 0.004 else 0.003
                        guard = pivot_down * (1.0 - guard_buffer)
                        if guard > 0.0:
                            sl = round(min(sl, guard), pair_dec)
                            pivot_notes.append(f"sl≤{guard:.4f}")
                else:
                    if pivot_down is not None and pivot_down < px:
                        delta = abs((px - pivot_down) / max(px, 1e-9))
                        mix = _mix_from_dist(delta) * closeness_factor
                        if mix > 0.0:
                            tp = round(tp * (1.0 - mix) + pivot_down * mix, pair_dec)
                            pivot_notes.append(f"tp→{pivot_down:.4f}")
                    if pivot_up is not None and pivot_up > px:
                        delta = abs((pivot_up - px) / max(px, 1e-9))
                        mix = _mix_from_dist(delta) * closeness_factor
                        if mix > 0.0:
                            px = round(px * (1.0 - mix) + pivot_up * mix, pair_dec)
                            pivot_force_limit = True
                            pivot_notes.append(f"px→res@{pivot_up:.4f}")
                        guard_buffer = 0.0015 if delta <= 0.004 else 0.003
                        guard = pivot_up * (1.0 + guard_buffer)
                        if guard > 0.0:
                            sl = round(max(sl, guard), pair_dec)
                            pivot_notes.append(f"sl≥{guard:.4f}")

                if is_buy and tp <= px:
                    tp = round(px * 1.001, pair_dec)
                elif (not is_buy) and tp >= px:
                    tp = round(px * 0.999, pair_dec)

                if is_buy and sl >= px:
                    sl = round(px * 0.995, pair_dec)
                elif (not is_buy) and sl <= px:
                    sl = round(px * 1.005, pair_dec)

                if pivot_zone is not None:
                    pivot_notes.append(f"zone={pivot_zone:+.4f}")
                if pivot_bias is not None:
                    pivot_notes.append(f"bias={pivot_bias:+.4f}")
                if pivot_pp is not None:
                    pivot_notes.append(f"pp={pivot_pp:.4f}")

            frac = max(min(float(qty_frac[i].item()), max_qfrac), 0.02)

            rb = None
            rl = 0
            try:
                rb = float((aux.get("run_budget_quote") or [None])[i]) if ("run_budget_quote" in aux) else None
                rl = int((aux.get("run_currencies_left") or [0])[i]) if ("run_currencies_left" in aux) else 0
            except Exception:
                rb, rl = None, 0

            reserve_frac = float((aux.get("reserve_frac", [self.cfg.reserve_frac])[i] if isinstance(aux.get("reserve_frac"), list) else aux.get("reserve_frac", self.cfg.reserve_frac)))
            super_sc = float((aux.get("super_conf_score", [self.cfg.super_conf_score])[i] if isinstance(aux.get("super_conf_score"), list) else aux.get("super_conf_score", self.cfg.super_conf_score)))
            super_p = float((aux.get("super_conf_sideprob", [self.cfg.super_conf_sideprob])[i] if isinstance(aux.get("super_conf_sideprob"), list) else aux.get("super_conf_sideprob", self.cfg.super_conf_sideprob)))

            pos_base = _f((aux.get("pos_base") or [0.0])[i], 0.0)
            pos_m = _f(aux["pos_margin_base"][i], 0.0)
            long_exposure = max(pos_base, 0.0) + max(pos_m, 0.0)
            short_exposure = max(-pos_m, 0.0)

            lev_raw = max(float(lev[i].item()), 0.0)
            sc = float(score[i].item())
            lev_max_allowed = int(aux["lev_buy_max"][i]) if is_buy else int(aux["lev_sell_max"][i])
            lev_out = None
            if lev_max_allowed > 1 and lev_raw > 1.0 and sc >= 0.35:
                lev_out = float(min(lev_raw, float(lev_max_allowed)))

            cap_hint = None
            if "cap_eur_hint" in aux and i < len(aux["cap_eur_hint"]):
                try:
                    cap_hint = float(aux["cap_eur_hint"][i])
                except Exception:
                    cap_hint = None

            if cap_hint is None and rb is not None:
                budget_after_res = max(0.0, float(rb)) * max(0.0, 1.0 - float(reserve_frac))
                denom = float(max(1, int(rl) or 1))
                cap_hint = budget_after_res / denom

            if cap_hint is not None:
                cap_hint = max(cap_hint, min_notional)

            cap_eur = cap_hint if (cap_hint is not None) else float("inf")
            p_hat_exec = float(torch.max(p_side[i, 0], p_side[i, 2]).item())
            is_super = (abs(float(score[i].item())) >= super_sc) and (p_hat_exec >= super_p)
            if is_super and (rb is not None):
                cap_eur = min(float(cap_eur) * 3.0, float(rb))

            reduce_only = bool(p_reduce[i].item() >= 0.5)
            trade_phase = infer_trade_phase(side, reduce_only, lev_out or lev_raw, pos_base, pos_m)

            if trade_phase == "open_long" and free_quote <= 0.0:
                trade_phase = "hold"
            if trade_phase == "open_short" and free_quote <= 0.0:
                trade_phase = "hold"

            if trade_phase == "close_short":
                reduce_only = True
                lev_out = aux["levOpenOrders"][i]
            elif trade_phase == "close_long":
                if pos_m > _EPS:
                    reduce_only = True
                if reduce_only:
                    lev_out = None
            elif trade_phase == "open_short":
                if lev_out is None and lev_max_allowed > 1 and lev_raw > 1.0:
                    lev_out = float(min(lev_raw, float(lev_max_allowed)))
                if lev_max_allowed <= 1 or lev_out is None or lev_out <= 1.0:
                    trade_phase = "hold"
            else:
                if is_buy:
                    lev_out = None

            if trade_phase == "hold":
                actions.append(Action(
                    pair=pair, side="hold", qty=0.0, price=px,
                    ordertype="none", take_profit=tp, stop_loss=None,
                    leverage=None, time_in_force="GTC", score=float(score[i].item()),
                    _side_prob=p_hat, _entropy=H, notes="trade_phase=hold", trade_phase="hold"
                ))
                continue

            if trade_phase == "open_long":
                notional_cap = float(free_quote)
                if cap_eur != float("inf"):
                    notional_cap = min(float(free_quote), float(cap_eur))
                notional = notional_cap * frac
                qty = (notional / px) if px > 0 else 0.0
            elif trade_phase == "open_short":
                margin_cap = float(free_quote)
                if cap_eur != float("inf"):
                    margin_cap = min(float(margin_cap), float(cap_eur))
                notional = margin_cap * max(lev_out or 1.0, 1.0) * frac
                qty = (notional / px) if px > 0 else 0.0
            elif trade_phase == "close_long":
                base_available = max(long_exposure, 0.0)
                qty = base_available * frac
                notional = qty * px
            else:
                short_available = max(short_exposure, 0.0)
                qty = short_available * frac
                notional = qty * px

            if qty < minOrder:
                qty = minOrder

            qty = round(qty, lot_dec)
            notional = float(qty * px)

            if trade_phase in ("close_long", "close_short") and qty <= 0.0:
                qty = round((long_exposure if trade_phase == "close_long" else short_exposure), lot_dec)
                notional = qty * px

            oi = int(ord_idx[i].item())
            ordertype = "market" if oi == 1 else "limit"
            if pivot_has and pivot_force_limit and trade_phase != "hold":
                ordertype = "limit"
            price_out = px
            tif_map = {0: "IOC", 1: "FOK", 2: "GTC"}
            tif_out = "IOC" if ordertype == "market" else tif_map.get(int(tif_idx[i].item()), "GTC")

            note_parts = [f"trade_phase={trade_phase}"]
            if pivot_has and pivot_notes:
                note_parts.append("pivot=" + ";".join(pivot_notes))
            note = " | ".join(note_parts)

            actions.append(Action(
                pair=pair, side=side, qty=qty, price=price_out,
                ordertype=ordertype, take_profit=tp,
                leverage=lev_out, stop_loss=sl,
                time_in_force=tif_out, reduce_only=reduce_only,
                score=float(score[i].item()), notes=note, trade_phase=trade_phase,
                _side_prob=p_hat, _entropy=H
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


def _pivot_context(row: Dict[str, Any], mid: float) -> Dict[str, Any]:
    """Estrarre informazioni di contesto sui pivot point dal row TimesFM."""

    info = (row.get("info") or {})
    meta_sources: List[Dict[str, Any]] = []
    fc_meta = info.get("FORECAST_META")
    if isinstance(fc_meta, dict):
        meta_sources.append(fc_meta)
    row_meta = row.get("meta")
    if isinstance(row_meta, dict):
        meta_sources.append(row_meta)

    levels: Dict[str, float] = {}
    for src in meta_sources:
        for key in ("pp", "r1", "r2", "s1", "s2"):
            raw = src.get(f"pivot_{key}")
            if _is_num(raw):
                levels[key] = float(raw)

    ctx = {
        "has": bool(levels),
        "levels": levels,
        "near_up": None,
        "near_down": None,
        "dist_up": None,
        "dist_down": None,
        "zone": None,
        "bias": None,
        "range_norm": None,
        "closeness": None,
    }

    if not levels or not _is_num(mid) or mid <= 0.0:
        return ctx

    mid_f = float(mid)
    denom = max(mid_f, 1e-9)

    above = sorted([val for val in levels.values() if val >= mid_f])
    below = sorted([val for val in levels.values() if val <= mid_f])

    near_up = above[0] if above else None
    near_down = below[-1] if below else None

    if near_up == near_down:
        above_strict = sorted([val for val in levels.values() if val > mid_f])
        below_strict = sorted([val for val in levels.values() if val < mid_f])
        near_up = above_strict[0] if above_strict else near_up
        near_down = below_strict[-1] if below_strict else near_down

    dist_up = ((near_up - mid_f) / denom) if (near_up is not None) else None
    dist_down = ((near_down - mid_f) / denom) if (near_down is not None) else None

    hi = max(levels.values())
    lo = min(levels.values())
    rng = (hi - lo) / denom if hi != lo else 0.0

    pp = levels.get("pp")
    zone = ((mid_f - pp) / denom) if (pp is not None) else None

    bias = None
    if dist_up is not None or dist_down is not None:
        up_abs = abs(dist_up) if dist_up is not None else float("inf")
        down_abs = abs(dist_down) if dist_down is not None else float("inf")
        if math.isfinite(up_abs) and math.isfinite(down_abs):
            bias = (down_abs - up_abs)

    closeness = None
    finite_dists = [abs(d) for d in [dist_up, dist_down] if d is not None]
    if finite_dists:
        closeness = min(finite_dists)

    ctx.update({
        "near_up": near_up,
        "near_down": near_down,
        "dist_up": dist_up,
        "dist_down": dist_down,
        "zone": zone,
        "bias": bias,
        "range_norm": rng,
        "closeness": closeness,
    })
    return ctx

def featurize_one(row: Dict[str,Any], blend: Dict[str,Number], goal_state: Optional[Dict[str,Any]], strategy_weights: Dict[str,Number]) -> Tuple[List[float], Dict[str,List[Any]]]:
    nowf = _now_feats(row); bias = _bias_ma(row); chgs = _changes(row); ors = _or_slip(row); pf = _portfolio(row); pend = _pend_delta(row)
    pivot_ctx = _pivot_context(row, nowf["mid"])
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
    pos_margin_base = _net_margin_from_open_orders(row)
    aux = {"pair":[pair], "mid":[nowf["mid"]], "bid":[nowf["bid"]], "ask":[nowf["ask"]],
           "lot_decimals":[int(pl.get("lot_decimals") or 8)], "pair_decimals":[int(pl.get("pair_decimals") or 5)],
           "ordermin":[pl.get("ordermin") or 0.0], "free_quote":[pf["free_quote"]], "free_base":[pf["free_base"]],
           "pos_margin_base":[pos_margin_base], "pos_base":[pf["pos_base"]],
           "max_quote_frac":[strategy_weights.get("max_quote_frac",0.12)], "min_notional_eur":[strategy_weights.get("min_notional_eur",5.0)]}

    levels = pivot_ctx.get("levels", {})

    def _pval(name: str) -> Optional[float]:
        val = levels.get(name)
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    aux.update({
        "pivot_has_meta": [1.0 if pivot_ctx.get("has") else 0.0],
        "pivot_pp": [_pval("pp")],
        "pivot_r1": [_pval("r1")],
        "pivot_r2": [_pval("r2")],
        "pivot_s1": [_pval("s1")],
        "pivot_s2": [_pval("s2")],
        "pivot_near_up": [pivot_ctx.get("near_up")],
        "pivot_near_down": [pivot_ctx.get("near_down")],
        "pivot_near_up_dist": [pivot_ctx.get("dist_up")],
        "pivot_near_down_dist": [pivot_ctx.get("dist_down")],
        "pivot_zone": [pivot_ctx.get("zone")],
        "pivot_bias": [pivot_ctx.get("bias")],
        "pivot_range": [pivot_ctx.get("range_norm")],
        "pivot_closeness": [pivot_ctx.get("closeness")],
    })
    return flat, aux

# --- Router featurizer: hashing schema-free oppure set fisso (fallback) ---
_auto_feat = AutoFeaturizer(dim=HASH_DIM)

def featurize_batch(rows: List[Dict[str,Any]], blends: List[Dict[str,Number]],
                    goal_state: Optional[Dict[str,Any]],
                    strategy_weights: Dict[str,Number], run_aux: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str,List[Any]]]:
    if USE_HASH_FEATS:
        X = []
        aux = {k:[] for k in [
            "pair","mid","bid","ask","lot_decimals","pair_decimals",
            "ordermin","free_quote","free_base","pos_margin_base","pos_base",
            "slippage_in","slippage_out","fee_maker","fee_taker",
            "max_quote_frac","min_notional_eur","lev_buy_max","lev_sell_max",
            "pivot_has_meta","pivot_pp","pivot_r1","pivot_r2","pivot_s1","pivot_s2",
            "pivot_near_up","pivot_near_down","pivot_near_up_dist","pivot_near_down_dist",
            "pivot_zone","pivot_bias","pivot_range","pivot_closeness"
        ]}
        for row, blend in zip(rows, blends):
            v, a = _auto_feat.featurize(row, blend, goal_state, strategy_weights)
            # --- INIETTA segnali di contesto per il GRU (per ogni riga del batch) ---
            # --- NEW: inietta feature di memoria (se presenti in run_aux) ---
            if run_aux is not None and isinstance(run_aux.get("mem"), list):
                idx = len(X)  # X è la lista vettori che stai costruendo: len(X) è l'indice corrente
                if idx < len(run_aux["mem"]):
                    m = run_aux["mem"][idx] or {}
                    for k, val in m.items():
                        try:
                            _auto_feat._add(v, f"{k}:num", float(val))
                        except Exception:
                            pass
            if run_aux is not None:
                rb = float(run_aux.get("run_budget_quote") or 0.0)
                rl = float(run_aux.get("run_currencies_left") or 0.0)
                _auto_feat._add(v, "derived.run_budget_quote:num", rb)
                _auto_feat._add(v, "derived.run_currencies_left:num", rl)
                try:
                    mems = (run_aux or {}).get("mem")
                except Exception:
                    mems = None
                try:
                    mid_now = float(a.get("mid")[0] if isinstance(a.get("mid"), list) else (a.get("mid") or 0.0))
                except Exception:
                    mid_now = 0.0

                if isinstance(mems, list):
                    idx = len(X)  # X è la lista dei vettori riga che stai costruendo
                    if isinstance(mems, list) and 0 <= idx < len(mems) and isinstance(mems[idx], dict):
                        mcur = mems[idx]
                        # dump mem.* come numeric
                        for kk, vv in mcur.items():
                            key = kk if str(kk).startswith("mem.") else f"mem.{kk}"
                            try:
                                _auto_feat._add(v, f"{key}:num", float(vv))
                            except Exception:
                                print('errore 1')
                                pass
                        # derived: TP colpito ora?
                        try:
                            ls = mcur.get("last_side")
                            tp = float(mcur.get("last_take_profit") or 0.0)
                            if mid_now and tp:
                                hit = 1.0 if ((ls == "buy" and mid_now >= tp) or (ls == "sell" and mid_now <= tp)) else 0.0
                                _auto_feat._add(v, "mem.tp_hit_now:bool", hit)
                        except Exception:
                            print('errore')
                            pass
                        try:
                            entry = float(mcur.get("last_entry_price") or 0.0)
                            if entry != 0.0:
                                print('dentro')
                            if mid_now > 0 and entry > 0:
                                unrealized = (mid_now - entry) / entry
                                _auto_feat._add(v, "mem.unrealized_pct:num", unrealized)
                                direction = str(mcur.get("last_side") or "").lower()
                                _auto_feat._add(v, f"mem.last_cycle:{direction}", 1.0)
                        except Exception:
                            pass
            X.append(v)
            for k in aux.keys(): aux[k].extend(a[k])
        X_t = torch.tensor(X, dtype=torch.float32)
        return X_t, aux
    else:
        # fallback alle features hardcoded esistenti
        X = []; aux = {k:[] for k in [
            "pair","mid","bid","ask","lot_decimals","pair_decimals","ordermin",
            "free_quote","free_base","pos_margin_base","pos_base","slippage_in",
            "slippage_out","fee_maker","fee_taker","max_quote_frac","min_notional_eur",
            "pivot_has_meta","pivot_pp","pivot_r1","pivot_r2","pivot_s1","pivot_s2",
            "pivot_near_up","pivot_near_down","pivot_near_up_dist","pivot_near_down_dist",
            "pivot_zone","pivot_bias","pivot_range","pivot_closeness"
        ]}
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

        if self.cfg.memory_path is None and self.cfg.log_path:
            base_dir = os.path.dirname(self.cfg.log_path)
            self.cfg.memory_path = os.path.join(base_dir, "trm_memory.jsonl")

        try:
            # default_ckpt = os.path.join(os.getcwd(), "aiConfig", "trm_from_paired.ckpt")
            # self.load_brain(default_ckpt)  # se non esiste, stampa un warning e continua
            ckpt_path = self.cfg.__dict__.get("ckpt_path") \
                    or os.path.join(os.getcwd(), "aiConfig", "trm_from_paired_full.ckpt")
            self._safe_load_ckpt(ckpt_path)
        except Exception:
            pass

        self.memory = DailyMemory()
        # base dir del log (cartella che contiene i file giornalieri)
        mem_dir = os.path.dirname(self.cfg.log_path) if self.cfg.log_path else os.path.join(os.getcwd(), "storico_output","trm_log")
        self.memory.load_today(mem_dir)
        print(f"[TRM] memoria giornaliera caricata da: {mem_dir}")

    def bootstrap_norm(self, rows: List[Dict[str,Any]], blends: List[Dict[str,Number]], goal_state: Dict[str,Any], strategy_weights: Dict[str,Number]) -> None:
        run_aux = {
            "run_budget_quote": self.cfg.total_budget_quote,
            "run_currencies_left": self.cfg.run_currencies_left,
        }
        X, _ = featurize_batch(rows, blends, goal_state, strategy_weights, run_aux=run_aux)
        with torch.no_grad():
            self.model.norm.update(X.to(self.cfg.device))

    def _log_uncertain_decision_id(self, decision_id: str, confidence: float):
        """
        Logga solo la decision_id se la confidenza è bassa.
        """
        try:
            base_dir = os.path.join(self.cfg.log_path, "trm_log")
            os.makedirs(base_dir, exist_ok=True)
            log_path = os.path.join(
                base_dir, f"uncertain_decisions_{datetime.date.today():%Y_%m_%d}.jsonl"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "decision_id": decision_id,
                    "confidence": confidence
                }) + "\n")
        except Exception as e:
            print(f"[WARN] could not log uncertain decision_id: {e}")

    def _seed_y0(self, Xn: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.y0_proj(Xn)

    @torch.no_grad()
    def propose_actions_for_batch(self, rows: List[Dict[str,Any]], blends: List[Dict[str,Number]], goal_state: Dict[str,Any], strategy_weights: Dict[str,Number], K: Optional[int] = None, shadow_log: bool = True) -> List[Action]:
        assert len(rows) == len(blends), "rows/blends length mismatch"
        run_aux = {
            "run_budget_quote": self.cfg.total_budget_quote,
            "run_currencies_left": self.cfg.run_currencies_left,
        }
        # --- NEW: prepara le feature di memoria per ogni riga (pair) ---
        mem_list = []
        for row in rows:
            pair = row.get("pair") or f"{row.get('base','?')}/{row.get('quote','?')}"
            mem_list.append(self.memory.memory_feats_for(pair))
        run_aux["mem"] = mem_list
        X, aux = featurize_batch(rows, blends, goal_state, strategy_weights, run_aux=run_aux)

        print("FEAT", X.shape, "nonzero_total:", int((X!=0).sum()))
        nnz_per_row = (X!=0).sum(dim=1)
        print("nnz[min,max,mean] =", int(nnz_per_row.min()), int(nnz_per_row.max()), float(nnz_per_row.float().mean()))
        assert (X!=0).any(), "X è tutto zero: featurize_batch sta perdendo le feature"
        assert torch.isfinite(X).all(), "Non-finite in X"
        print("nonfinite in X:", (~torch.isfinite(X)).sum().item())
        # Trm_agent.py → dentro propose_actions_for_batch(...), dopo featurize_batch
        # --- NEW: cap per-pair guidato da TimesFM (se presente in row.info.FORECAST)
        cap_list = []
        levOpenOrders = []
        minOrder = []
        # Trm_agent.py (dentro propose_actions_for_batch, dopo cap_list)
        tp_hints, sl_hints, side_hints, sigs = [], [], [], []
        for r in rows:
            fc  = ((r.get("info") or {}).get("FORECAST") or {})
            fch = ((r.get("info") or {}).get("FORECAST_HINTS") or {})
            # hints (fallback a None/0.0)
            tp_hints.append(float(fch.get("tp_mult_hint") or 0.0))
            sl_hints.append(float(fch.get("sl_mult_hint") or 0.0))
            side_hints.append(float(fch.get("side_hint") or 0.0))
            sigs.append(float(fc.get("timesfm_signal") or 0.0))
            levOrd = (r.get('open_orders')[0]).get('Lev') if len(r.get('open_orders') or []) > 0 else 0
            limit =  (r.get('pair_limits')).get('ordermin') or 0
            levOpenOrders.append(levOrd)
            minOrder.append(limit)

        # rendili disponibili al decoder (dimensione B)
        aux = dict(aux or {})
        aux["tp_mult_hint"] = tp_hints
        aux["sl_mult_hint"] = sl_hints
        aux["side_hint"]    = side_hints
        aux["timesfm_sig"]  = sigs
        aux["levOpenOrders"]  = levOpenOrders
        aux["minOrder"]  = minOrder

        B = len(rows)
        base_cap = None
        try:
            if self.cfg.total_budget_quote is not None and self.cfg.run_currencies_left:
                base_cap = float(self.cfg.total_budget_quote) * max(0.0, 1.0 - float(self.cfg.reserve_frac))
                base_cap = base_cap / float(max(1, int(self.cfg.run_currencies_left)))
        except Exception:
            base_cap = None

        for r in rows:
            fc = ((r.get("info") or {}).get("FORECAST") or {})
            e = float(fc.get("timesfm_edge_1d") or 0.0)
            u = float(fc.get("timesfm_uncert_1d") or 0.0)
            # conf semplice: edge / uncert (clippato 0..1)
            conf = max(0.0, min(1.0, abs(e) / max(u, 1e-6)))
            # 50%..100% del cap equal-share, proporzionale alla confidenza
            cap = None if base_cap is None else float(base_cap) * (0.5 + 0.5 * conf)
            cap_list.append(cap if cap is not None else 0.0)

        if cap_list:
            aux["cap_eur_hint"] = cap_list  # il decoder lo consumerà per calcolare la qty
        # --- valori dinamici per run (budget/currencies) ---
        if self.cfg.total_budget_quote is not None and B > 0:
            aux["run_budget_quote"]    = [float(self.cfg.total_budget_quote)] * B
            aux["reserve_frac"]        = [float(self.cfg.reserve_frac)] * B
            aux["super_conf_score"]    = [float(self.cfg.super_conf_score)] * B
            aux["super_conf_sideprob"] = [float(self.cfg.super_conf_sideprob)] * B

        if self.cfg.run_currencies_left is not None and B > 0:
            aux["run_currencies_left"] = [int(self.cfg.run_currencies_left)] * B

        X = X.to(self.cfg.device)
        self.model.norm.update(X)
        Xn = self.model.norm(X)
        y0 = self._seed_y0(Xn)
        act_stats = None
        if getattr(self.cfg, "act_enabled", False):
            yK, act_stats = self.model.improve_adaptive(
                X, y0=y0,
                max_steps=int(K or self.cfg.K_refine),
                min_steps=int(getattr(self.cfg, "act_min_steps", 1)),
                threshold=float(getattr(self.cfg, "act_threshold", 0.60)),
                return_stats=True,
            )
        else:
            yK = self.model.improve(X, y0=y0, K=K)
        if pair == "FIL/EUR":
            print('fil')
        actions = self.model.decode_actions(yK, aux)

        if act_stats is not None:
            for i, a in enumerate(actions):
                a.notes = (a.notes or "") + f" | act_steps={int(act_stats['steps'][i])} | p_halt={float(act_stats['p_halt'][i]):.2f}"

        by_pair: Dict[str, List[Action]] = {}
        for a in actions: by_pair.setdefault(a.pair, []).append(a)
        pair2input = {}
        for row, blend in zip(rows, blends):
            now = ((row.get("info") or {}).get("NOW") or {})
            pf  = (row.get("portfolio") or {})
            rpf = (pf.get("row") or {})
            liq_bid = float(now.get("liquidity_bid_sum") or 0.0)
            liq_ask = float(now.get("liquidity_ask_sum") or 0.0)
            # dentro loop che riempie pair2input
            meta = ((row.get("info") or {}).get("FORECAST") or {})
            pm   = ((row.get("info") or {}).get("FORECAST_META") or ({}))
            p10_T1 = ((pm.get("p10") or [None])[0] if isinstance(pm.get("p10"), list) else None)
            p50_T1 = ((pm.get("p50") or [None])[0] if isinstance(pm.get("p50"), list) else None)
            p90_T1 = ((pm.get("p90") or [None])[0] if isinstance(pm.get("p90"), list) else None)
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
            inputs["forecast"] = {
                "edge_1d": meta.get("timesfm_edge_1d"),
                "uncert_1d": meta.get("timesfm_uncert_1d"),
                "edge_h": meta.get("timesfm_edge_h"),
                "uncert_h": meta.get("timesfm_uncert_h"),
                "p10_T1": p10_T1, "p50_T1": p50_T1, "p90_T1": p90_T1,
            }
            pivots_meta = {k: pm.get(f"pivot_{k}") for k in ("pp","r1","r2","s1","s2") if pm.get(f"pivot_{k}") is not None}
            if pivots_meta:
                inputs["forecast"]["pivots"] = pivots_meta
            piv_ctx = _pivot_context(row, inputs["now"]["mid"])
            if piv_ctx.get("has"):
                if piv_ctx.get("zone") is not None:
                    inputs["forecast"]["pivot_zone"] = float(piv_ctx.get("zone"))
                if piv_ctx.get("bias") is not None:
                    inputs["forecast"]["pivot_bias"] = float(piv_ctx.get("bias"))
                if piv_ctx.get("range_norm") is not None:
                    inputs["forecast"]["pivot_range_norm"] = float(piv_ctx.get("range_norm"))
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

        rows_index = {}
        for idx, r in enumerate(rows):
            rows_index[r.get("pair") or f"{r.get('base','?')}/{r.get('quote','?')}"] = idx

        if shadow_log:
            for a in final:
                decision_id = str(uuid.uuid4())

                # Calcolo confidenza aggregata
                p_hat = float(getattr(a, "_side_prob", 0.0))        # max softmax
                H = float(getattr(a, "_entropy", 0.0))              # entropia
                H_norm = H / math.log(3.0)
                score_abs = abs(float(getattr(a, "score", 0.0)))    # già in [-1,1]
                conf = p_hat * (1.0 - max(0.0, min(1.0, H_norm))) * (0.5 + 0.5*score_abs)
                if conf < 0.60:
                    a.side = "hold"
                    self._log_uncertain_decision_id(decision_id, conf)
                # End Calcolo confidenza aggregata

                meta_in = pair2input.get(a.pair, {"inputs": {}, "blend": {}})
                payload = {
                    "event_type": "decision",
                    "ts": _iso_now(),
                    "decision_id": decision_id,
                    "pair": a.pair,
                    "action": a.as_dict(keep_none=True),
                    "score": a.score,
                    "inputs": meta_in["inputs"],
                    "blend": meta_in["blend"],
                    "weights": {k: float(v) for k, v in (strategy_weights or {}).items()},
                    "goal_state": goal_state or {},
                }

                # poco prima di self.logger.log(payload)
                fc  = ((pair2input.get(a.pair, {}).get("inputs", {}) or {}).get("forecast") or {})
                payload["timesfm"] = {
                    "edge_1d": fc.get("edge_1d"), "uncert_1d": fc.get("uncert_1d"),
                    "edge_h": fc.get("edge_h"), "uncert_h": fc.get("uncert_h"),
                    "p10_T1": fc.get("p10_T1"), "p50_T1": fc.get("p50_T1"), "p90_T1": fc.get("p90_T1"),
                }
                if act_stats is not None:
                    _idx = rows_index.get(a.pair, None)
                    if _idx is not None:
                        payload["act"] = {
                            "steps": int(act_stats["steps"][_idx]),
                            "p_halt": float(act_stats["p_halt"][_idx]),
                            "threshold": float(getattr(self.cfg, "act_threshold", 0.60)),
                        }
                # scrivi nel JSONL
                self.logger.log(payload)
                # opzionale: attacca l'id all'azione così chi esegue può ributtarlo nel log ‘execution/close’
                a.notes = (a.notes or "") + f" | decision_id={decision_id}"
                # --- update stato dinamico nel cfg per le prossime run ---
            try:
                # spendo budget solo per i BUY (qty*prezzo)
                spent = 0.0
                for a in final:
                    if a.price is None:
                        continue
                    if getattr(a, "trade_phase", None) == "open_long" and a.side == "buy" and not a.reduce_only:
                        spent += float(a.qty) * float(a.price)
                    elif getattr(a, "trade_phase", None) == "open_short" and a.leverage is not None:
                        spent += float(a.qty) * float(a.price)

                if self.cfg.total_budget_quote is not None:
                    self.cfg.total_budget_quote = max(float(self.cfg.total_budget_quote) - spent, 0.0)

                # currency rimanenti = rimanenti - batch lavorato
                if self.cfg.run_currencies_left is not None:
                    self.cfg.run_currencies_left = max(int(self.cfg.run_currencies_left) - len(rows), 0)
            except Exception:
                pass
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

    # --- in Trm_agent.py, dentro la classe TRMAgent ---
    def load_brain(self, ckpt_path: str) -> bool:
        """
        Carica i pesi addestrati (state_dict) da un checkpoint .ckpt.
        Ritorna True se il caricamento è andato a buon fine.
        """
        try:
            if not ckpt_path or not os.path.exists(ckpt_path):
                print(f"[TRM] checkpoint non trovato: {ckpt_path}")
                return False
            chk = torch.load(ckpt_path, map_location=self.cfg.device)
            state = chk.get("state_dict") or chk  # fallback, se salvato come puro state_dict
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.cfg.device)
            self.model.eval()  # in produzione normalmente eval()
            # print facoltativo: un assaggio dei pesi
            with torch.no_grad():
                wsample = self.model.head_side.weight.view(-1)[:5].detach().cpu().numpy()
            print("[TRM] brain loaded. head_side weight sample:", wsample)
            return True
        except Exception as e:
            print(f"[TRM] errore load_brain: {e}")
            return False

    # === LOAD TRAINED BRAIN (.ckpt) ===
    def _safe_load_ckpt(self, path: str):
        if not path or not os.path.exists(path):
            print(f"[TRM] ckpt non trovato: {path}")
            return
        ckpt = torch.load(path, map_location=self.cfg.device)
        sd = ckpt.get("state_dict")
        if not sd:
            print("[TRM] ckpt senza 'state_dict'")
            return
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        self.model.to(self.cfg.device)
        print("[TRM] mente caricata:", path)
        if missing or unexpected:
            print("[TRM] load_state warn -> missing:", missing, " | unexpected:", unexpected)
        # opzionale: sneak peek di un head per verifica
        try:
            with torch.no_grad():
                samp = self.model.head_side.weight.view(-1)[:6].detach().cpu().numpy().tolist()
            print("[TRM] head_side weight sample:", samp)
        except Exception:
            pass
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
