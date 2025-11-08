# trm_training.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, math, glob, datetime as _dt, tempfile, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
# Importa dal tuo agente TRM esistente
from ..Trm_agent import (
    TRMAgent,
    featurize_batch,
    TinyRecursiveModel,
    TRMConfig,
    infer_trade_phase,
)
from ..Trm_agent import Action  # solo per typing
from ..Util import _shadow_daily_path  # helper giornaliero
import torch.nn.functional as F

Number = float | int

# -----------------------------
# Utils robusti per jsonl
# -----------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not (path and os.path.exists(path)):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _write_jsonl_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="train_", suffix=".jsonl.tmp",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _parse_ts(val: Any) -> _dt.datetime:
    if isinstance(val, _dt.datetime):
        return val
    if not val:
        return _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)
    txt = str(val)
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            if fmt:
                dt = _dt.datetime.strptime(txt, fmt)
            else:
                dt = _dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return dt
        except Exception:
            continue
    return _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    try:
        if str(val).lower() in {"true", "1", "yes"}:
            return True
        if str(val).lower() in {"false", "0", "no", "none"}:
            return False
    except Exception:
        pass
    return bool(default)

# -----------------------------
# Dataset dai shadow_actions
# -----------------------------
@dataclass
class Sample:
    x_row: Dict[str, Any]            # row ricostruito per featurizer
    blend: Dict[str, Number]
    goal_state: Dict[str, Any]
    weights: Dict[str, Number]
    target: Dict[str, Any]           # bersagli per training (side/qty/price/ordertype/…)
    pnl: Optional[float]             # per reinforcement-like
    decision_id: Optional[str]       # per assisted override
    pair: str
    trade_phase: str

class ShadowActionsDataset(torch.utils.data.Dataset):
    """
    Costruisce esempi dai file giornalieri shadow_actions_{dd_mm_yyyy}.jsonl
    Usa record 'decision' come input + target (azione fatta), e se presenti
    record successivi con 'pnl' (allo stesso livello di event_type) come reward.
    """
    def __init__(self,
                 log_base_path: str,
                 dates: Optional[List[_dt.date]] = None,
                 assisted_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 max_samples: Optional[int] = None):
        self.samples: List[Sample] = []
        assisted_overrides = assisted_overrides or {}
        paths: List[str] = []
        if dates:
            for d in dates:
                p = _shadow_daily_path(log_base_path, _dt.datetime(d.year, d.month, d.day))
                if p and os.path.exists(p):
                    paths.append(p)
        else:
            # fallback: prendi tutti gli shadow_actions_*.jsonl nella cartella
            base_dir = os.path.dirname(log_base_path)
            paths = sorted(glob.glob(os.path.join(base_dir, "shadow_actions_*.jsonl")))

        # indicizza per decision_id per agganciare pnl post-operazione
        decisions: Dict[str, Dict[str, Any]] = {}
        ordered_decisions: List[Tuple[_dt.datetime, Dict[str, Any]]] = []
        for p in paths:
            for r in _read_jsonl(p):
                if (r.get("event_type") or r.get("eventType")) == "decision":
                    did = str(r.get("decision_id") or r.get("_decision_id") or "")
                    if not did:
                        continue
                    dt = _parse_ts(r.get("ts") or r.get("timestamp") or r.get("time"))
                    row = dict(r)
                    row["_parsed_ts"] = dt
                    decisions[did] = row
                    ordered_decisions.append((dt, row))

        # seconda passata: attacca pnl dove presente
        did2pnl: Dict[str, float] = {}
        for p in paths:
            for r in _read_jsonl(p):
                if "pnl" in r:
                    did = str(r.get("decision_id") or r.get("_decision_id") or "")
                    if did:
                        try:
                            did2pnl[did] = float(r["pnl"])
                        except Exception:
                            continue

        ordered_decisions.sort(key=lambda t: t[0])
        open_cycles: Dict[Tuple[str, str], int] = {}

        # costruisci samples
        for _, dec in ordered_decisions:
            did = str(dec.get("decision_id") or dec.get("_decision_id") or "")
            pair = dec.get("pair") or ((dec.get("action") or {}).get("pair"))
            if not pair:
                continue

            inputs = dec.get("inputs") or {}
            now = inputs.get("now") or {}
            liq = inputs.get("liquidity") or {}
            derived = inputs.get("derived") or {}
            pf = inputs.get("portfolio") or {}

            row_like = {
                "pair": pair,
                "info": {
                    "NOW": {
                        "bid": now.get("bid"), "ask": now.get("ask"), "mid": now.get("mid"), "vwap": now.get("vwap"),
                        "liquidity_bid_sum": liq.get("liq_bid_sum"),
                        "liquidity_ask_sum": liq.get("liq_ask_sum"),
                        "slippage_buy_pct": liq.get("slippage_buy_pct"),
                        "slippage_sell_pct": liq.get("slippage_sell_pct"),
                        "or_ok": derived.get("or_ok"),
                        "or_range": derived.get("or_range"),
                    }
                },
                "portfolio": {
                    "row": {
                        "qty": pf.get("pos_base"),
                        "px_EUR": None,
                        "pnl_pct": pf.get("pnl_pct"),
                        "pos_margin_base": pf.get("pos_margin_base") or derived.get("pos_margin_base"),
                    },
                    "available": {
                        "base": pf.get("free_base"),
                        "quote": pf.get("free_quote"),
                    }
                },
                "pair_limits": {}
            }

            blend = dec.get("blend") or {}
            weights = dec.get("weights") or {}
            goal_state = dec.get("goal_state") or {}

            target = (assisted_overrides.get(did) or (dec.get("action") or {})).copy()
            target["side"] = (str(target.get("side") or target.get("tipo") or "hold")).lower()
            target["ordertype"] = (str(target.get("ordertype") or target.get("quando") or "limit")).lower()
            if target["ordertype"] == "hold":
                target["ordertype"] = "none"

            target["reduce_only"] = _to_bool(target.get("reduce_only") or target.get("reduce") or target.get("reduceOnly"))
            target["leverage"] = _to_float(target.get("leverage") or 0.0)

            pos_base_val = _to_float(pf.get("pos_base") or pf.get("qty"))
            pos_margin_val = _to_float(pf.get("pos_margin_base") or derived.get("pos_margin_base") or 0.0)

            trade_phase = infer_trade_phase(
                target["side"],
                target["reduce_only"],
                target["leverage"],
                pos_base_val,
                pos_margin_val,
            )

            target["trade_phase"] = trade_phase
            pnl = did2pnl.get(did)

            sample = Sample(
                x_row=row_like,
                blend=blend,
                goal_state=goal_state,
                weights=weights,
                target=target,
                pnl=pnl,
                decision_id=did,
                pair=pair,
                trade_phase=trade_phase
            )
            self.samples.append(sample)

            idx = len(self.samples) - 1
            cycle_key = (pair, "long" if trade_phase in ("open_long", "close_long") else "short")
            if trade_phase == "open_long":
                open_cycles[cycle_key] = idx
            elif trade_phase == "open_short":
                open_cycles[cycle_key] = idx
            elif trade_phase == "close_long":
                entry_idx = open_cycles.pop(cycle_key, None)
                if entry_idx is not None and pnl is not None:
                    self.samples[entry_idx].pnl = pnl
            elif trade_phase == "close_short":
                entry_idx = open_cycles.pop(cycle_key, None)
                if entry_idx is not None and pnl is not None:
                    self.samples[entry_idx].pnl = pnl

            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

class PairedJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_samples: int | None = None, require_executed: bool = False):
        rows = _read_jsonl(path)
        self.samples: List[Sample] = []
        for r in rows:
            pair = r.get("pair") or ((r.get("action") or {}).get("pair"))
            if not pair:
                continue
            x_row = r.get("x_row") or _row_from_inputs_block(pair, r.get("inputs") or {})
            target = (r.get("target") or r.get("action") or {}).copy()
            target["side"] = str(target.get("side") or target.get("tipo") or "hold").lower()
            target["ordertype"] = str(target.get("ordertype") or target.get("quando") or "limit").lower()
            if target["ordertype"] == "hold":
                target["ordertype"] = "none"

            target["reduce_only"] = _to_bool(target.get("reduce_only") or target.get("reduce") or target.get("reduceOnly"))
            target["leverage"] = _to_float(target.get("leverage") or 0.0)

            row_port = (x_row.get("portfolio") or {}).get("row") or {}
            pos_base_val = _to_float(row_port.get("qty"))
            pos_margin_val = _to_float(row_port.get("pos_margin_base") or 0.0)

            trade_phase = infer_trade_phase(
                target["side"],
                target["reduce_only"],
                target["leverage"],
                pos_base_val,
                pos_margin_val,
            )

            target["trade_phase"] = trade_phase

            pnl = r.get("pnl")
            decision_id = r.get("decision_id")
            order_id = r.get("order_id") or r.get("orderId")
            # opzionale: tieni solo esempi con ordine eseguito
            if require_executed and target["side"] in ("buy","sell") and not order_id:
                continue

            self.samples.append(Sample(
                x_row=x_row,
                blend=r.get("blend") or {},
                goal_state=r.get("goal_state") or {},
                weights=r.get("weights") or {},
                target=target,
                pnl=(float(pnl) if pnl is not None else None),
                decision_id=decision_id,
                pair=pair,
                trade_phase=trade_phase
            ))
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def _row_from_inputs_block(pair: str, inputs: dict) -> dict:
    """Converte il blocco 'inputs' nel 'row' atteso da featurize_batch."""
    inputs = inputs or {}
    now = inputs.get("now") or {}
    liq = inputs.get("liquidity") or {}
    derived = inputs.get("derived") or {}
    pf = inputs.get("portfolio") or {}
    row_like = {
        "pair": pair,
        "info": {
            "NOW": {
                "bid": now.get("bid"), "ask": now.get("ask"), "mid": now.get("mid"), "vwap": now.get("vwap"),
                "liquidity_bid_sum": liq.get("liq_bid_sum"),
                "liquidity_ask_sum": liq.get("liq_ask_sum"),
                "slippage_buy_pct": liq.get("slippage_buy_pct"),
                "slippage_sell_pct": liq.get("slippage_sell_pct"),
                "or_ok": derived.get("or_ok"),
                "or_range": derived.get("or_range"),
            }
        },
        "portfolio": {
            "row": {
                "qty": pf.get("pos_base"),
                "px_EUR": None,
                "pnl_pct": pf.get("pnl_pct"),
            },
            "available": {
                "base": pf.get("free_base"),
                "quote": pf.get("free_quote"),
            }
        },
        "pair_limits": {}
    }
    return row_like
# -----------------------------
# Memoria leggera per correzioni/“hard cases”
# -----------------------------
class TRMMemory:
    """
    JSONL append-only per correzioni (human-in-the-loop) o hard cases.
    Ogni riga: {decision_id, input(row/blend/goal/weights), target}
    """
    def __init__(self, memory_path: Optional[str]):
        self.path = memory_path

    def append(self, decision_id: str, x_row: dict, blend: dict, goal: dict, weights: dict, target: dict) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        rec = {
            "decision_id": decision_id,
            "x_row": x_row, "blend": blend, "goal_state": goal, "weights": weights,
            "target": target
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_overrides(self) -> Dict[str, Dict[str, Any]]:
        """
        Carica il memory file come mappa decision_id -> target.
        Utile per alimentarli in 'assisted_overrides'.
        """
        m = {}
        if not (self.path and os.path.exists(self.path)):
            return m
        for r in _read_jsonl(self.path):
            did = r.get("decision_id")
            tgt = r.get("target")
            if did and isinstance(tgt, dict):
                m[did] = tgt
        return m


# -----------------------------
# Trainer
# -----------------------------
class TRMTrainer:
    """
    Allena TinyRecursiveModel su dati shadow_actions con tre modalità:
      - 'supervised'      : behavior cloning dall'azione loggata/override
      - 'reinforce_like'  : come sopra ma pesato da reward tanh(k*pnl)
      - 'assisted'        : come supervised ma con targets passati esplicitamente
    """
    def __init__(self, agent: TRMAgent, lr: float = 1e-3, pnl_scale: float = 0.05,
                 device: Optional[str] = None,
                 memory_path: Optional[str] = None):
        self.agent = agent
        self.model: TinyRecursiveModel = agent.model
        self.device = device or agent.cfg.device
        self.cfg = agent.cfg
        self.pnl_scale = float(pnl_scale)
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.side_class_weights = None

        # losses
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # memoria
        self.memory = TRMMemory(memory_path)

    def _budget_penalty(self, heads: Dict[str, torch.Tensor], x_aux: Dict[str, List[Any]]) -> torch.Tensor:
        """
        Penalizza la parte di qty_frac che eccede il tetto per pair.
        Usa aux['max_quote_frac'] (es. 0.12). Output: tensore [B] (qui B=1 nel tuo loop).
        """
        qf = heads["qty"].view(-1)  # [B], già sigmoid 0..1
        try:
            # aux["max_quote_frac"] può essere lista o scalare
            mxf = x_aux.get("max_quote_frac", 0.12)
            max_qf = float(mxf[0] if isinstance(mxf, list) else mxf)
        except Exception:
            max_qf = 0.12
        return torch.relu(qf - max_qf)

    def _market_regularizer(self, heads, x_aux):
        ord_logits = heads["ord_logits"]                 # [B,2]
        p_market = torch.softmax(ord_logits, dim=-1)[:, 1]  # [B]

        # costo atteso (EUR)
        cost_eur = self._trade_cost_est(x_aux, {"ordertype": "market", "side": "buy"}).to(p_market.device).view(-1)

        # recupero mid da aux con fallback e guardie
        mid_val = x_aux.get("mid") or x_aux.get("vwap") or x_aux.get("price_now")
        if isinstance(mid_val, list):
            mid_val = (mid_val[0] if len(mid_val) > 0 else 0.0)
        try:
            mid = float(mid_val or 0.0)
        except Exception:
            mid = 0.0

        # se non ho mid>0, NON penalizzo
        if not (mid > 0):
            return torch.zeros_like(p_market)

        # costo relativo con cap (2%)
        rel = torch.clamp(cost_eur / max(mid, 1e-6), min=0.0, max=0.02)
        return p_market * rel

    def _cycle_hinge(self, pnl_value: Optional[float], x_aux: Dict[str, List[Any]],
                    heads: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                    margin: float = 0.0) -> torch.Tensor:
        """
        Hinge: max(0, margin - (PNL_netto)) applicata SOLO alle chiusure (reduce==1),
        ponderata dalla probabilità di esecuzione (1 - P(hold)).
        Se non c’è pnl_value, torna zero.
        """
        reduces = tgt["reduce"].view(-1)  # [B]  (1.0 = chiusura)
        if pnl_value is None:
            return torch.zeros_like(reduces)

        # costo stimato (eur) dal punto di vista generico
        cost_eur = self._trade_cost_est(x_aux, {"ordertype": "limit", "side": "buy"}).to(reduces.device).view(-1)  # [B]
        pnl = torch.tensor([float(pnl_value)], dtype=torch.float32, device=reduces.device)  # [1]
        edge = pnl - cost_eur  # pnl netto

        # p_exec ≈ 1 - P(hold)
        p_side = torch.softmax(heads["logits_side"], dim=-1)  # [B,3]
        p_hold = p_side[:, 2]                                 # [B]
        p_exec = (1.0 - p_hold).view(-1)

        notional = max(abs(float(x_aux.get("mid",[0])[0] or 0.0) *
                        float((x_aux.get("qty") or [0])[0] or 0.0)), 1.0)
        edge = ((pnl - cost_eur) / notional).clamp(-0.05, 0.05)
        return reduces * torch.relu(float(margin) - edge) * p_exec

    def _estimate_side_class_weights(self, dataset) -> torch.Tensor:
        # 0=buy, 1=sell, 2=hold
        from collections import Counter
        c = Counter(int(self._targets_from_action({}, s.target)["side"].item()) for s in dataset)
        total = sum(c.values()) or 1
        # inversa della frequenza (clippata)
        w = [total / max(c.get(i,1),1) for i in range(3)]
        # normalizza
        s = sum(w); w = [x/s for x in w]
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    # ---------- supervised targets ----------
    def _targets_from_action(self, x_aux: Dict[str, List[Any]], target: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        side_map = {"buy":0, "sell":1, "hold":2}
        side_idx = side_map.get(str(target.get("side","hold")).lower(), 2)

        free_quote = float(x_aux.get("free_quote",[0.0])[0] or 0.0)
        pos_base  = float((x_aux.get("pos_base") or [0.0])[0])
        price = float(target.get("price") or x_aux.get("mid",[0.0])[0] or 0.0)
        qty = float(target.get("qty") or target.get("quantita") or 0.0)
        pos_m = float(x_aux.get("pos_margin_base", [0.0])[0] or 0.0)
        leverage = float(target.get("leverage") or 0.0)
        trade_phase = str(target.get("trade_phase") or "hold")

        long_exposure = max(pos_base + max(pos_m, 0.0), 0.0)
        short_exposure = max(-pos_m, 0.0)

        if side_idx == 0:  # BUY
            if trade_phase == "close_short" and short_exposure > 0:
                qty_frac = min(max(qty / max(short_exposure, 1e-9), 0.0), 1.0)
            else:
                notional = qty * price if (qty and price) else 0.0
                qty_frac = 0.0 if free_quote <= 0 else min(max(notional / max(free_quote, 1e-9), 0.0), 1.0)
        elif side_idx == 1:  # SELL
            if trade_phase == "close_long" and long_exposure > 0:
                qty_frac = min(max(qty / max(long_exposure, 1e-9), 0.0), 1.0)
            elif trade_phase == "open_short":
                notional = qty * price if (qty and price) else 0.0
                margin_cap = free_quote * max(leverage, 1.0)
                qty_frac = 0.0 if margin_cap <= 0 else min(max(notional / max(margin_cap, 1e-9), 0.0), 1.0)
            else:
                qty_frac = 0.0 if pos_base <= 0 else min(max(qty / max(pos_base, 1e-9), 0.0), 1.0)
        else:
            qty_frac = 0.0

        mid = float(x_aux.get("mid",[0.0])[0] or 0.0)
        px = float(target.get("price") or mid)
        px_off = 0.0 if mid <= 0 else max(min((px - mid) / mid, 0.05), -0.05)

        ord_map = {"limit":0, "market":1, "none":0}
        ord_idx = ord_map.get(str(target.get("ordertype","limit")).lower(), 0)

        tif_map = {"ioc":0, "fok":1, "gtc":2}
        tif = tif_map.get(str(target.get("time_in_force") or target.get("tif") or "gtc").lower(), 2)

        reduces = 1.0 if trade_phase in ("close_long", "close_short") else 0.0

        return {
            "side": torch.tensor([side_idx], dtype=torch.long, device=self.device),
            "qty_frac": torch.tensor([qty_frac], dtype=torch.float32, device=self.device),
            "px_off": torch.tensor([px_off], dtype=torch.float32, device=self.device),
            "ordertype": torch.tensor([ord_idx], dtype=torch.long, device=self.device),
            "tif": torch.tensor([tif], dtype=torch.long, device=self.device),
            "reduce": torch.tensor([reduces], dtype=torch.float32, device=self.device),
        }

    def _forward_heads(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        m = self.model
        out = {
            "logits_side": m.head_side(y),
            "qty": torch.sigmoid(m.head_qty(y)).squeeze(-1),
            "px": torch.tanh(m.head_px(y)).squeeze(-1),
            "ord_logits": m.head_ordertype(y),
            "tif_logits": m.head_tif(y),
            "reduce_logit": m.head_reduce(y).squeeze(-1),
            "halt_logit": m.head_halt(y).squeeze(-1)
        }
        return out

    def _loss_supervised(self, heads, tgt, reward_sig: Optional[torch.Tensor] = None,
                        class_w: Optional[torch.Tensor] = None,
                        gate_nonexec: bool = True) -> torch.Tensor:
        # CE con pesi di classe opzionali
        ce_fn = nn.CrossEntropyLoss(reduction="none", weight=class_w.to(heads["logits_side"].device) if class_w is not None else None)

        ce_side = ce_fn(heads["logits_side"], tgt["side"])   # imitazione
        ce_ord  = self.ce(heads["ord_logits"], tgt["ordertype"])
        ce_tif  = self.ce(heads["tif_logits"], tgt["tif"])
        l_qty   = self.l1(heads["qty"], tgt["qty_frac"])
        l_px    = self.l1(heads["px"], tgt["px_off"])
        b_reduce= self.bce(heads["reduce_logit"], tgt["reduce"])
     # ---- self-distilled halt target: conf = (1 - H_norm) * (1 - P(hold)) ----
        with torch.no_grad():
            p_side = torch.softmax(heads["logits_side"], dim=-1)
            H = (-(p_side * torch.log(p_side + 1e-12)).sum(dim=-1))               # entropia
            Hn = H / math.log(3.0)
            conf = (1.0 - Hn) * (1.0 - p_side[:, 2])                               # 2 = hold
            conf = conf.clamp(0.0, 1.0)
        b_halt = self.bce(heads["halt_logit"], conf)
        # gating per HOLD: se hold, non alleni qty/px/reduce
        if gate_nonexec:
            is_hold = (tgt["side"] == 2).float() if tgt["side"].dtype == torch.long else (tgt["side"] == torch.tensor([2], device=tgt["side"].device)).float()
            # keep side/tif/ord sempre, ma disattiva qty/px/reduce
            l_qty = l_qty * (1.0 - is_hold)
            l_px  = l_px  * (1.0 - is_hold)
            b_reduce = b_reduce * (1.0 - is_hold)

        # rinforzo: se reward negativo, aggiungi anti-imitazione (opposto) per la side
        if reward_sig is not None:
            r = reward_sig.clamp(-1.0, 1.0)
            pos = (r.clamp(min=0.0) + 1.0)          # [1,2]
            neg = (-r).clamp(min=0.0)               # [0,1]
            # opp label
            side_opp = tgt["side"].clone()
            side_opp[:] = self._opp_side_idx(int(tgt["side"].item()))
            ce_side_opp = ce_fn(heads["logits_side"], side_opp)
            # loss finale: premia imitazione se r>0; se r<0 aggiungi spinta opposta
            loss_side = pos * ce_side + 0.5 * neg * ce_side_opp
        else:
            loss_side = ce_side

            extra = getattr(self, "_extra_terms", None)
            if extra is not None:
                loss_budget, loss_market, loss_cycle = extra  # ciascuno [B]
                # pesi iniziali (tunable)
                lam_budget = 0.50
                lam_market = 0.25
                lam_cycle  = 0.01
                return (loss_side + 0.5*ce_ord + 0.25*ce_tif + 0.5*l_qty + 0.5*l_px + 0.25*b_reduce
                        + lam_budget*loss_budget.mean()
                        + lam_market*loss_market.mean()
                        + lam_cycle*loss_cycle.mean() + 0.05*b_halt).mean()

        return (loss_side + 0.5*ce_ord + 0.25*ce_tif + 0.5*l_qty + 0.5*l_px + 0.25*b_reduce + 0.05*b_halt).mean()

    def _trade_cost_est(self, x_aux: dict, target: dict) -> torch.Tensor:
        device = self.device
        try:
            bid = float((x_aux.get("bid") or [0.0])[0] or 0.0)
            ask = float((x_aux.get("ask") or [0.0])[0] or 0.0)
            mid = 0.5*(bid+ask) if (bid>0 and ask>0) else max(bid, ask, 0.0)
            spread = (ask - bid) if (ask>0 and bid>0) else 0.0
            side = str(target.get("side","hold")).lower()
            if side == "buy":
                slip_pct = float((x_aux.get("slippage_buy_pct")  or [0.0])[0] or 0.0)
            elif side == "sell":
                slip_pct = float((x_aux.get("slippage_sell_pct") or [0.0])[0] or 0.0)
            else:
                slip_pct = 0.0  # hold: niente costo

            fee_maker = float((x_aux.get("fee_maker") or [0.001])[0] or 0.001)
            fee_taker = float((x_aux.get("fee_taker") or [0.001])[0] or 0.001)
            ordtype = str(target.get("ordertype","limit")).lower()
            fee_in  = fee_taker if ordtype == "market" else fee_maker

            half_spread = 0.5*spread
            slip_abs = abs(slip_pct) * mid
            cost_eur = half_spread + slip_abs + fee_in*mid
            return torch.tensor([cost_eur], dtype=torch.float32, device=device)
        except Exception:
            return torch.tensor([0.0], dtype=torch.float32, device=device)

    def _make_reward_weight(self, pnl_value: Optional[float], cost_est: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pnl_value is None:
            return torch.tensor([1.0], dtype=torch.float32, device=self.device)
        k = float(self.pnl_scale)
        pnl = torch.tensor([float(pnl_value)], dtype=torch.float32, device=self.device).clamp(-10.0, 10.0)
        base = torch.tanh(k * pnl).abs() + 1e-6
        if cost_est is not None:
            edge = (pnl - cost_est.to(self.device).float()).clamp(-10.0, 10.0)
            eco = torch.tanh(k * edge).abs() + 1e-6
            return base * eco
        return base

    # --- helper: indice opposto per side (0=buy, 1=sell, 2=hold) ---
    def _opp_side_idx(self, idx: int) -> int:
        if idx == 0: return 1
        if idx == 1: return 0
        return 2  # hold resta hold

    def _make_reward_signal(self, pnl_value: Optional[float], cost_est: Optional[torch.Tensor] = None) -> torch.Tensor:
        # r in [-1, +1] con segno
        if pnl_value is None:
            return torch.tensor([0.0], dtype=torch.float32, device=self.device)
        k = float(self.pnl_scale)
        pnl = torch.tensor([float(pnl_value)], dtype=torch.float32, device=self.device).clamp(-10.0, 10.0)
        edge = pnl - (cost_est.to(self.device).float() if cost_est is not None else 0.0)
        return torch.tanh(k * edge)  # <-- mantiene il segno

    # ---------- training loop ----------
    def train_epoch(self,
                    dataset: ShadowActionsDataset,
                    mode: str = "supervised",
                    batch_size: int = 32) -> float:
        """
        Un'epoca di training. 'mode' in {'supervised','reinforce_like','assisted'}.
        """
        if self.side_class_weights is None:
            self.side_class_weights = self._estimate_side_class_weights(dataset)

        self.model.train()
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True,
            collate_fn=lambda b: b[0]
        )  # 1->K refine
        running = 0.0; n = 0

        for s in dl:
            # s: Sample = sample[0] if isinstance(sample, list) else sample
            # if isinstance(s, (list, tuple)):
            #     s = s[0]

            run_aux = {
                "run_budget_quote": self.cfg.total_budget_quote or 150.0,
                "run_currencies_left": self.cfg.run_currencies_left or 200,
                "mem": [ self.agent.memory.memory_feats_for(s.pair) ]  # un solo sample nel batch
            }
            X, aux = featurize_batch([s.x_row], [s.blend], s.goal_state, s.weights, run_aux=run_aux)  # :contentReference[oaicite:1]{index=1}
            X = X.to(self.device)
            self.model.norm.update(X)
            Xn = self.model.norm(X)
            y0 = self.model.y0_proj(Xn).detach()
            if getattr(self.agent.cfg, "act_enabled", False):
                # allineato a propose_actions_for_batch (runtime)
                # print('dentro')
                yK, _ = self.model.improve_adaptive(
                    X, y0=y0,
                    max_steps=int(self.agent.cfg.K_refine),
                    min_steps=int(getattr(self.agent.cfg, "act_min_steps", 1)),
                    threshold=float(getattr(self.agent.cfg, "act_threshold", 0.60)),
                    return_stats=True,
                )
                yK = yK.detach().clone()
            else:
                yK = self.model.improve(X, y0=y0, K=self.agent.cfg.K_refine)                 # y0 learnable, NO random
            # yK = self.model.improve(X, y0=y0, K=self.agent.cfg.K_refine)  # passa X (raw), non Xn   u should use improve_adaptive
            h = self._forward_heads(yK)
            tgt = self._targets_from_action(aux, s.target)

            # --- costruisci le tre componenti ---
            pen_budget = self._budget_penalty(h, aux)
            pen_market = self._market_regularizer(h, aux)
            cycle_h    = self._cycle_hinge(s.pnl, aux, h, tgt, margin=0.0)  # margin 0 = “almeno non perdere”
            # # esponi al calcolo della loss
            self._extra_terms = (pen_budget, pen_market, cycle_h)
            # self._extra_terms = None

            reward_w = None
            m = mode.lower()
            # if m == "reinforce_like":
            #     cost_est = self._trade_cost_est(aux, s.target)
            #     reward_w = self._make_reward_signal(s.pnl, cost_est)

            reward_sig = None
            if mode.lower() == "reinforce_like":
                cost_est = self._trade_cost_est(aux, s.target)
                reward_sig = self._make_reward_signal(s.pnl, cost_est)
            loss = self._loss_supervised(h, tgt, reward_sig=reward_sig, class_w=self.side_class_weights)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self._extra_terms = None
            running += float(loss.item()); n += 1

        return (running / max(n,1))

    @torch.no_grad()
    def evaluate(self, dataset: ShadowActionsDataset, batch_size: int = 64) -> Dict[str, float]:
        self.model.eval()
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=lambda b: b[0]
        )
        tot, n = 0.0, 0
        for s in dl:
            s = s[0] if isinstance(s, list) else s
            run_aux = {
                "run_budget_quote": self.cfg.total_budget_quote or 150.0,
                "run_currencies_left": self.cfg.run_currencies_left or 200,
                "mem": [ self.agent.memory.memory_feats_for(s.pair) ]  # un solo sample nel batch
            }
            # print(run_aux)
            # print(s.pair)
            X, aux = featurize_batch([s.x_row], [s.blend], s.goal_state, s.weights, run_aux=run_aux)
            X = X.to(self.device)
            Xn = self.model.norm(X)
            y0 = self.model.y0_proj(Xn).detach()                  # y0 learnable, NO random

            if getattr(self.agent.cfg, "act_enabled", False):
                yK, _ = self.model.improve_adaptive(
                    X, y0=y0,
                    max_steps=int(self.agent.cfg.K_refine),
                    min_steps=int(getattr(self.agent.cfg, "act_min_steps", 1)),
                    threshold=float(getattr(self.agent.cfg, "act_threshold", 0.60)),
                    return_stats=True,
                )
                yK = yK.detach().clone()
            else:
                yK = self.model.improve(X, y0=y0, K=self.agent.cfg.K_refine)
            h = self._forward_heads(yK)
            tgt = self._targets_from_action(aux, s.target)
            pen_budget = self._budget_penalty(h, aux)
            pen_market = self._market_regularizer(h, aux)
            cycle_h    = self._cycle_hinge(s.pnl, aux, h, tgt, margin=0.0)  # margin 0 = “almeno non perdere”
            # esponi al calcolo della loss
            self._extra_terms = (pen_budget, pen_market, cycle_h)
            # self._extra_terms = None
            loss = self._loss_supervised(h, tgt, reward_sig=None, class_w=self.side_class_weights)
            self._extra_terms = None
            tot += float(loss.item()); n += 1
        return {"loss": tot / max(n,1)}

    def save(self, path: str, best_val: float = 0.0, epoch: int | None = None, batch_size: int | None = None) -> None:
        """
        Salva il checkpoint del modello con pesi, config e meta-info.
        - best_val: loss di validazione migliore (opzionale, per log)
        - epoch: indice epoca (opzionale)
        - batch_size: per tracciare i parametri del training
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rn = self.agent.model.norm  # scorciatoia
        ckpt = {
            "version": 2,
            "state_dict": self.agent.model.state_dict(),     # include anche i buffer della RunningNorm
            "model_cfg": {
                "feature_dim": self.agent.cfg.feature_dim,
                "hidden_dim": self.agent.cfg.hidden_dim,
                "mlp_hidden": self.agent.cfg.mlp_hidden,
                "K_refine": self.agent.cfg.K_refine,
            },
            "hash": {
                "use": "AutoFeaturizer" in globals(),
                "dim": getattr(self.agent, "hash_dim", None) or 2048,
            },
            "train_meta": {
                "timestamp": _dt.datetime.now().isoformat(),
                "best_val_loss": float(best_val or 0.0),
                "epochs_done": int(epoch or 0),
                "batch_size": int(batch_size or 0),
                "device": str(self.agent.cfg.device),
            },
            # opzionale: numeretti per debug veloce (non necessari al load)
            "norm_state": {
                "count": float(rn.count.detach().cpu().item()),
                "mean_sample": rn.mean.detach().cpu().view(-1)[:8].tolist(),
                "var_sample": (
                    rn.M2.detach().cpu().view(-1)[:8] /
                    max(rn.count.detach().cpu().item(), 1e-6)
                ).tolist(),
            }
        }

        torch.save(ckpt, path)
        print(f"[ckpt] salvato modello -> {path}")
        # torch.save({"cfg": self.agent.cfg, "state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk["state_dict"])
        self.model.to(self.device)

    # ---------- Active learning: confidence + raccolta correzioni ----------
    @torch.no_grad()
    def predict_with_confidence(self, x_row: dict, blend: dict, goal_state: dict, weights: dict, K: Optional[int] = None) -> Dict[str, Any]:
        X, aux = featurize_batch([x_row], [blend], goal_state, weights, run_aux=None)
        X = X.to(self.device)
        self.model.norm.update(X)
        Xn = self.model.norm(X)
        y0 = self.model.y0_proj(Xn)                  # y0 learnable, NO random
        yK = self.model.improve(X, y0=y0, K=self.agent.cfg.K_refine)
        logits_side = self.model.head_side(yK)
        p_side = torch.softmax(logits_side, dim=-1)
        conf, idx = p_side.max(dim=-1)
        # decode azione per feedback
        acts = self.agent.model.decode_actions(yK, aux)  # usa lo stesso decoder dell'agente
        pred = acts[0].as_dict()
        pred["pair"] = acts[0].pair
        return {"action": pred, "confidence": float(conf.item()), "aux": aux}

    def active_annotate_once(self,
                             decision_id: str,
                             x_row: dict, blend: dict, goal_state: dict, weights: dict,
                             labeler: Callable[[dict], dict],
                             threshold: float = 0.6) -> Optional[dict]:
        """
        Se confidence < threshold, richiama labeler(input)->target e salva
        la correzione in memoria (e.g. manual_training / trm_memory.jsonl).
        Ritorna il target se annotato, altrimenti None.
        """
        pred = self.predict_with_confidence(x_row, blend, goal_state, weights)
        if pred["confidence"] >= threshold:
            return None  # sufficientemente sicuro, non chiediamo
        # chiedi al “trainer umano”
        target = labeler(pred) or {}
        # normalizza campi minimi
        target["side"] = str(target.get("side","hold")).lower()
        target["ordertype"] = str(target.get("ordertype","limit")).lower()
        # salva in memoria
        if self.memory and getattr(self.memory, "append", None):
            self.memory.append(decision_id, x_row, blend, goal_state, weights, target)
        return target


# -----------------------------
# Helper per caricamento override manuali
# -----------------------------
def load_manual_overrides(path: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not (path and os.path.exists(path)):
        return out
    for r in _read_jsonl(path):
        did = r.get("decision_id")
        tgt = r.get("target")
        if did and isinstance(tgt, dict):
            out[did] = tgt
    return out

def append_manual_override(path: str, decision_id: str, target: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"decision_id": decision_id, "target": target}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    import argparse, random

    parser = argparse.ArgumentParser(description="TRM Supervised Training")
    # === NUOVO: allenamento da file paired (input+output nello stesso record)
    parser.add_argument("--paired_file", type=str, default=None,
                        help="JSONL con (input + target) nello stesso record. Se presente, useremo questo dataset.")
    # === esistenti per training da shadow logs
    parser.add_argument("--log_path", type=str, default=None,
                        help="Percorso al file base dei log shadow_actions.jsonl (usato solo se --paired_file è assente).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="Checkpoint .ckpt (best su validation). Se non passato, salva in ./models/")
    # opzionali: overrides/memoria per supervised da shadow (non servono per paired)
    parser.add_argument("--manual_overrides", type=str, default=None)
    parser.add_argument("--memory_overrides", type=str, default=None)
    parser.add_argument("--hold_max_frac", type=float, default=0.6)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--mode", type=str, default="reinforce_like",
                    choices=["supervised","reinforce_like","assisted"])
    args = parser.parse_args()

    # --- SEED ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Agent / cfg (serve comunque per modello e featurizer) ---
    if args.paired_file:
        # se usi paired_file, log_path non serve per costruire il dataset
        agent = TRMAgent(TRMConfig(log_path=args.log_path or "./storico_output/trm_log/shadow_actions.jsonl",
                                   device=args.device))
    else:
        if not args.log_path:
            raise RuntimeError("Devi passare --log_path oppure --paired_file.")
        agent = TRMAgent(TRMConfig(log_path=args.log_path, device=args.device))

    # --- Costruzione dataset ---
    if args.paired_file:
        full_ds = PairedJsonlDataset(args.paired_file)
        print(f"[dataset] paired samples: {len(full_ds)}")
    else:
        overrides = {}
        if args.manual_overrides:
            overrides.update(load_manual_overrides(args.manual_overrides))
        if args.memory_overrides:
            overrides.update(load_manual_overrides(args.memory_overrides))
        print(f'ciao {agent.cfg.log_path}')
        full_ds = ShadowActionsDataset(log_base_path=agent.cfg.log_path, assisted_overrides=(overrides or None))
        print(f"[dataset] shadow samples: {len(full_ds)}")

    if len(full_ds) == 0:
        raise RuntimeError("Nessun sample trovato (controlla --paired_file o --log_path).")

    # --- Split train/val (vale per entrambi i dataset) ---
    idxs = list(range(len(full_ds)))
    random.shuffle(idxs)
    n_val = max(1, int(len(idxs) * args.val_frac))
    val_idxs = set(idxs[:n_val])
    tr_idxs  = idxs[n_val:]

    # Bilanciamento HOLD solo se NON usi paired_file (nei paired decidi tu i target)
    if not args.paired_file and args.hold_max_frac < 1.0:
        def _is_hold(sample: Sample) -> bool:
            return (sample.target.get("side","hold").lower() == "hold")
        hold_bucket, nonhold_bucket = [], []
        for i in tr_idxs:
            (hold_bucket if _is_hold(full_ds[i]) else nonhold_bucket).append(i)
        import random as _r
        _r.shuffle(hold_bucket)
        max_hold = int(args.hold_max_frac * max(1, len(tr_idxs)))
        tr_idxs = nonhold_bucket + hold_bucket[:max_hold]
        _r.shuffle(tr_idxs)

    train_ds = torch.utils.data.Subset(full_ds, tr_idxs)
    val_ds   = torch.utils.data.Subset(full_ds, list(val_idxs))

    # --- Trainer ---
    trainer = TRMTrainer(agent, lr=args.lr, device=args.device, memory_path=agent.cfg.memory_path)

    # --- Loop epoche con early stopping ---
    best_val = float("inf")
    best_path = args.save
    patience = args.early_stop_patience
    patience_left = patience
    min_delta = 1e-3
    min_epochs = 3

    for epoch in range(1, args.epochs + 1):
        loss_tr = trainer.train_epoch(dataset=train_ds, mode=args.mode, batch_size=args.batch_size)
        metrics_val = trainer.evaluate(dataset=val_ds)
        loss_val = metrics_val["loss"]
        print(f"[epoch {epoch:02d}] train_loss={loss_tr:.5f} | val_loss={loss_val:.5f}")

        improved = (best_val - loss_val) > min_delta
        if improved:
            best_val = loss_val
            patience_left = patience
            if best_path:
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                trainer.save(best_path, best_val=best_val, epoch=epoch, batch_size=args.batch_size)
                print(f"[ckpt] salvato best -> {best_path}")
        else:
            if patience > 0:
                patience_left -= 1
                print(f"[early-stop] pazienza {patience_left}/{patience}")
                if patience_left <= 0:
                    print("[early-stop] stop anticipato: nessun miglioramento su validation")
                    break

    if not args.save:
        out = os.path.join(os.path.dirname(agent.cfg.log_path) or ".", "models", "trm_supervised.ckpt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        trainer.save(out, best_val=best_val, epoch=epoch, batch_size=args.batch_size)
        print(f"[ckpt] salvato modello finale -> {out}")



# run on root

# python -m Class.Training.trm_training --paired_file ".\storico_output\trm_log\training\test.jsonl" --device cpu --epochs 8 --batch_size 32 --save ".\aiConfig\trm_from_paired.ckpt"
# python -m Class.Training.trm_training --paired_file ".\storico_output\trm_log\training\trm_paired_dataset_20251030_192606.jsonl" --device cpu --epochs 8 --batch_size 32 --lr 1e-3 --val_frac 0.2 --save ".\aiConfig\trm_from_paired.ckpt"

# python -m Class.Training.trm_training --log_path ".\storico_output\trm_log\shadow_actions.jsonl" --device cpu --epochs 8 --batch_size 32 --lr 1e-3 --val_frac 0.2 --save ".\aiConfig\trm_from_shadow.ckpt" --mode supervised --hold_max_frac 1.0
# python -m Class.Training.trm_training --paired_file "storico_output/trm_log/training/trm_paired_dataset_synthetic_levels_20251107_175144.jsonl" --device cpu --epochs 8 --batch_size 48 --lr 5e-4 --mode supervised --save "aiConfig/trm_from_paired_full.ckpt"
