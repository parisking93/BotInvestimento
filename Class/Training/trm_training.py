# trm_training.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, math, glob, datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Importa dal tuo agente TRM esistente
from ..Trm_agent import TRMAgent, featurize_batch, TinyRecursiveModel, TRMConfig
from ..Trm_agent import Action  # solo per typing
from ..Util import _shadow_daily_path  # helper giornaliero

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

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

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
        # prima passata: raccogli decisions
        decisions: Dict[str, Dict[str, Any]] = {}
        for p in paths:
            for r in _read_jsonl(p):
                if (r.get("event_type") or r.get("eventType")) == "decision":
                    did = str(r.get("decision_id") or r.get("_decision_id") or "")
                    if not did:
                        continue
                    decisions[did] = r

        # seconda passata: attacca pnl dove presente
        did2pnl: Dict[str, float] = {}
        for p in paths:
            for r in _read_jsonl(p):
                # schema robusto: il tuo messaggio dice "pnl allo stesso livello di eventype"
                if "pnl" in r:
                    did = str(r.get("decision_id") or r.get("_decision_id") or "")
                    if did:
                        try:
                            did2pnl[did] = float(r["pnl"])
                        except Exception:
                            continue

        # costruisci samples
        for did, dec in decisions.items():
            pair = dec.get("pair") or ((dec.get("action") or {}).get("pair"))
            if not pair:
                continue

            inputs = dec.get("inputs") or {}
            # ricostruisci row-like minimale: funzionerà sia col featurizer hashing (consigliato)
            # sia col fallback a campi fissi se presenti nel blocco NOW/portfolio
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
                        "px_EUR": None,  # opzionale: non indispensabile con hashing
                        "pnl_pct": pf.get("pnl_pct"),
                    },
                    "available": {
                        "base": pf.get("free_base"),
                        "quote": pf.get("free_quote"),
                    }
                },
                # opzionali: pair_limits ecc. Se non ci sono, il featurizer hashing non ne ha bisogno.
                "pair_limits": {}
            }

            blend = dec.get("blend") or {}
            weights = dec.get("weights") or {}
            goal_state = dec.get("goal_state") or {}

            # target supervision: dall'azione loggata o da assisted override
            target = (assisted_overrides.get(did) or (dec.get("action") or {})).copy()
            # normalizza campi minimi
            target["side"] = (str(target.get("side") or target.get("tipo") or "hold")).lower()
            target["ordertype"] = (str(target.get("ordertype") or target.get("quando") or "limit")).lower()
            if target["ordertype"] == "hold":  # in schema legacy "quando": "hold"
                target["ordertype"] = "none"

            pnl = did2pnl.get(did)

            self.samples.append(Sample(
                x_row=row_like,
                blend=blend,
                goal_state=goal_state,
                weights=weights,
                target=target,
                pnl=pnl,
                decision_id=did,
                pair=pair
            ))
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


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
    def __init__(self, agent: TRMAgent, lr: float = 1e-3, pnl_scale: float = 0.05, device: Optional[str] = None):
        self.agent = agent
        self.model: TinyRecursiveModel = agent.model
        self.device = device or agent.cfg.device
        self.cfg = agent.cfg
        self.pnl_scale = float(pnl_scale)
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        # losses
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _targets_from_action(self, x_aux: Dict[str, List[Any]], target: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Converte l'azione target in tensori bersaglio per le head.
        """
        side_map = {"buy":0, "sell":1, "hold":2}
        side_idx = side_map.get(str(target.get("side","hold")).lower(), 2)

        # qty come frazione eur/quote massima disponibile (proxy stabile):
        free_quote = float(x_aux.get("free_quote",[0.0])[0] or 0.0)
        price = float(target.get("price") or x_aux.get("mid",[0.0])[0] or 0.0)
        qty = float(target.get("qty") or target.get("quantita") or 0.0)
        notional = qty * price if (qty and price) else 0.0
        # target frazione in [0,1]
        qty_frac = 0.0 if free_quote <= 0 else min(max(notional / max(free_quote, 1e-9), 0.0), 1.0)

        # prezzo offset ±5% relativo al mid
        mid = float(x_aux.get("mid",[0.0])[0] or 0.0)
        px = float(target.get("price") or mid)
        px_off = 0.0 if mid <= 0 else max(min((px - mid) / mid, 0.05), -0.05)

        ord_map = {"limit":0, "market":1, "none":0}
        ord_idx = ord_map.get(str(target.get("ordertype","limit")).lower(), 0)

        tif_map = {"ioc":0, "fok":1, "gtc":2}
        tif = tif_map.get(str(target.get("time_in_force") or target.get("tif") or "gtc").lower(), 2)

        # reduce_only = 1.0 if bool(target.get("reduce_only", False)) else 0.0k

        # === Label 'reduce' derivato dallo STATO, non dal log ===
        # usa la stessa logica dell'inference:
        #   - pos_margin_base > 0  -> SELL riduce (chiude long)
        #   - pos_margin_base < 0  -> BUY  riduce (chiude short)
        pos_m = float(x_aux.get("pos_margin_base", [0.0])[0] or 0.0)
        reduces = 0.0
        if side_idx == 1 and pos_m > 0:   # sell su long
            reduces = 1.0
        if side_idx == 0 and pos_m < 0:   # buy su short
            reduces = 1.0

        return {
            "side": torch.tensor([side_idx], dtype=torch.long, device=self.device),
            "qty_frac": torch.tensor([qty_frac], dtype=torch.float32, device=self.device),
            "px_off": torch.tensor([px_off], dtype=torch.float32, device=self.device),
            "ordertype": torch.tensor([ord_idx], dtype=torch.long, device=self.device),
            "tif": torch.tensor([tif], dtype=torch.long, device=self.device),
            "reduce": torch.tensor([reduces], dtype=torch.float32, device=self.device),
        }

    def _forward_heads(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Applica le head del modello; ritorna logits/valori necessari per i loss.
        """
        m = self.model
        out = {
            "logits_side": m.head_side(y),
            "qty": torch.sigmoid(m.head_qty(y)).squeeze(-1),
            "px": torch.tanh(m.head_px(y)).squeeze(-1),
            "ord_logits": m.head_ordertype(y),
            "tif_logits": m.head_tif(y),
            "reduce_logit": m.head_reduce(y).squeeze(-1),
        }
        return out

    def _loss_supervised(self, heads: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                         reward_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Behavior cloning multi-task; se reward_w è fornito, pesa i termini.
        """
        ce_side = self.ce(heads["logits_side"], tgt["side"])
        ce_ord  = self.ce(heads["ord_logits"], tgt["ordertype"])
        ce_tif  = self.ce(heads["tif_logits"], tgt["tif"])
        l_qty   = self.l1(heads["qty"], tgt["qty_frac"])
        l_px    = self.l1(heads["px"], tgt["px_off"])
        b_reduce= self.bce(heads["reduce_logit"], tgt["reduce"])

        # loss = (ce_side + 0.5*ce_ord + 0.25*ce_tif + 0.5*l_qty + 0.5*l_px + 0.25*b_reduce)
        # if reward_w is not None:
        #     loss = loss * reward_w
        # return loss.mean()
        if reward_w is None:
            loss = (ce_side + 0.5*ce_ord + 0.25*ce_tif + 0.5*l_qty + 0.5*l_px + 0.25*b_reduce).mean()
        else:
            loss = ((reward_w * ce_side) + (0.5 * reward_w * ce_ord) + 0.25*ce_tif + 0.5*l_qty + 0.5*l_px + 0.25*b_reduce).mean()
        return loss

    # --- STIMA COSTO DI INGRESSO (EUR o bps equivalenti) ---
    def _trade_cost_est(self, x_aux: dict, target: dict) -> torch.Tensor:
        """
        Usa NOW.bid/ask (spread), slippage_* e fees_* in pair_limits per stimare il costo.
        Ritorna un tensore scalare on-device. Se mancano i campi => 0.0
        """
        device = self.device
        try:
            bid = float((x_aux.get("bid") or [0.0])[0] or 0.0)
            ask = float((x_aux.get("ask") or [0.0])[0] or 0.0)
            mid = 0.5*(bid+ask) if (bid>0 and ask>0) else max(bid, ask, 0.0)
            spread = (ask - bid) if (ask>0 and bid>0) else 0.0
            # slippage/lq (se presenti in aux – opzionale)
            sl_in  = float((x_aux.get("slippage_in")  or [0.0])[0] or 0.0)
            sl_out = float((x_aux.get("slippage_out") or [0.0])[0] or 0.0)
            # fee: maker/taker (se presenti – opzionale). Default: 0.001 (0.1%)
            fee_maker = float((x_aux.get("fee_maker") or [0.001])[0] or 0.001)
            fee_taker = float((x_aux.get("fee_taker") or [0.001])[0] or 0.001)
            # ordertype del target per stimare fee di ingresso
            ordtype = str(target.get("ordertype","limit")).lower()
            fee_in  = fee_taker if ordtype == "market" else fee_maker
            # costo totale: spread/2 + slippage + fee*mid (approssimazione)
            half_spread = 0.5*spread
            slip = max(sl_in, 0.0)  # ingresso; se vuoi includere anche uscita, aggiungi sl_out
            cost_eur = half_spread + slip + fee_in*mid
            return torch.tensor([cost_eur], dtype=torch.float32, device=device)
        except Exception:
            return torch.tensor([0.0], dtype=torch.float32, device=device)


    def _make_reward_weight(self, pnl_value: Optional[float], cost_est: Optional[torch.Tensor] = None) -> torch.Tensor:
        "Peso stile RL: tanh(pnl_scale * pnl) con opzionale boost cost-aware."
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

    def train_epoch(self,
                    dataset: ShadowActionsDataset,
                    mode: str = "supervised",
                    batch_size: int = 32) -> float:
        """
        Un'epoca di training. 'mode' in {'supervised','reinforce_like','assisted'}.
        """
        self.model.train()
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)  # 1->K refine dentro al modello
        running = 0.0; n = 0

        for sample in dl:
            # scarta il batch dimension del DataLoader (usiamo 1 per semplicità)
            s: Sample = sample[0] if isinstance(sample, list) else sample
            if isinstance(s, (list, tuple)):  # difesa
                s = s[0]

            # featurize (riutilizziamo il batch builder originale – 1 record)
            X, aux = featurize_batch([s.x_row], [s.blend], s.goal_state, s.weights)  # :contentReference[oaicite:1]{index=1}
            X = X.to(self.device)
            self.model.norm.update(X)              # stessa normalizzazione online dell'agente
            Xn = self.model.norm(X)
            y0 = torch.zeros(1, self.model.y_dim, device=self.device)

            # ricorsivo K passi (come in inference)
            yK = self.model.improve(Xn, y0=y0, K=self.agent.cfg.K_refine)

            # head forward
            h = self._forward_heads(yK)

            # targets (assisted override già applicato a livello di dataset)
            tgt = self._targets_from_action(aux, s.target)

            # reward weight
            reward_w = None
            m = mode.lower()
            if m == "reinforce_like":
                # reward_w = self._make_reward_weight(s.pnl)
                cost_est = self._trade_cost_est(aux, s.target)
                reward_w = self._make_reward_weight(s.pnl, cost_est)
            # 'assisted' usa gli stessi loss del supervised (targets sono già corretti)

            loss = self._loss_supervised(h, tgt, reward_w=reward_w)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            running += float(loss.item()); n += 1

        return (running / max(n,1))

    @torch.no_grad()
    def evaluate(self, dataset: ShadowActionsDataset, batch_size: int = 64) -> Dict[str, float]:
        """
        Valutazione semplice: loss supervisionato medio.
        """
        self.model.eval()
        dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        tot, n = 0.0, 0
        for s in dl:
            s = s[0] if isinstance(s, list) else s
            X, aux = featurize_batch([s.x_row], [s.blend], s.goal_state, s.weights)
            X = X.to(self.device)
            Xn = self.model.norm(X)
            y0 = torch.zeros(1, self.model.y_dim, device=self.device)
            yK = self.model.improve(Xn, y0=y0, K=self.agent.cfg.K_refine)
            h = self._forward_heads(yK)
            tgt = self._targets_from_action(aux, s.target)
            loss = self._loss_supervised(h, tgt, reward_w=None)
            tot += float(loss.item()); n += 1
        return {"loss": tot / max(n,1)}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"cfg": self.agent.cfg, "state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk["state_dict"])
        self.model.to(self.device)
