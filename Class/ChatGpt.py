from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import os, time, json

from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass
class TradeAction:
    tipo: str
    pair: str
    prezzo: Optional[float]
    quando: Optional[str]
    quantita: Optional[float]
    quantita_eur: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timeframe: str
    lato: Optional[str] = None
    break_price: Optional[float] = None
    limite: Optional[float] = None
    leverage: Optional[float] = None
    motivo: Optional[str] = None


class CurrencyAnalyzerForYourPayloadV3:
    """
    - Stesse API e stesso I/O.
    - Batching dinamico per evitare input troppo lunghi.
    - L’input è una LISTA di oggetti currency completi (come da file JSON).
    - Post-processing: forza pair=kr_pair quando disponibile e rispetta tick size
      (decimali prezzo e quantità) usando i dati presenti nel payload.
    - **PATCH 2025-10-02**: sizing ora considera la cassa EUR reale dal payload
      (portfolio.available.quote su coppie quotate in EUR o asset ZEUR/EUR),
      evitando i falsi "available_total=0".
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_batch_size: int = 40,
        request_timeout_s: int = 300,
        max_retries: int = 4,
        # default di sessione (override in analyze)
        budget_eur: float = 0.0,
        reserve_eur: float = 0.0,
        risk_level: int = 5,            # 1..10
        prefer_limit: bool = True,
        per_trade_cap_pct_map: Optional[Dict[int, float]] = None,
        # soglia di sicurezza per input molto lunghi (in caratteri)
        max_input_chars: int = 120_000,
        # --- Parametri strategici aggiuntivi (safe defaults) ---
        or_min_atr_mult: float = 0.5,       # OR valido se OR_range ≥ 0.5*ATR
        spread_max_pct: float = 0.5,        # % massima di spread consentita
        min_volume_label: str = "Medium",   # Low/Medium/High
        enable_shorts: bool = False,        # facoltativo
        sl_atr_mult: float = 1.5,           # SL = entry - 1.5*ATR (long)
        tp_atr_mult: float = 2.5,           # TP = entry + 2.5*ATR (long)
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_batch_size = max_batch_size
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.max_input_chars = max_input_chars

        self.per_trade_cap_pct_map = per_trade_cap_pct_map or {
            1: 0.01, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.08,
            6: 0.10, 7: 0.12, 8: 0.15, 9: 0.20, 10: 0.25
        }
        self._defaults = dict(
            budget_eur=float(budget_eur),
            reserve_eur=float(reserve_eur),
            risk_level=max(1, min(10, int(risk_level))),
            prefer_limit=bool(prefer_limit),
        )

        # Parametri strategici runtime
        self.or_min_atr_mult = float(or_min_atr_mult)
        self.spread_max_pct = float(spread_max_pct)
        self.min_volume_label = str(min_volume_label).capitalize()  # Low/Medium/High
        self.enable_shorts = bool(enable_shorts)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)

        # Messaggio di sistema: stessa logica, ma descrizione input più chiara
        self.system_instructions = (
            "Sei un analista disciplinato. Ricevi una lista di strumenti (oggetti completi) "
            "che includono prezzi/indicatori e il mio portfolio. "
            "Produci azioni nel formato richiesto; se i dati non bastano ⇒ hold con motivo. "
            "SL/TP solo se sensati; usa ATR come guida quando opportuno; motivazione ≤80 parole."
        )

    # ---------- PUBLIC ----------
    def analyze(
        self,
        currencies_payload: List[Dict[str, Any]],
        budget_eur: Optional[float] = None,
        rischio: Optional[int] = None,
        reserve_eur: Optional[float] = None,
        prefer_limit: Optional[bool] = None,
    ) -> List[TradeAction]:

        budget = float(budget_eur if budget_eur is not None else self._defaults["budget_eur"])
        reserve = float(reserve_eur if reserve_eur is not None else self._defaults["reserve_eur"])
        risk = int(rischio if rischio is not None else self._defaults["risk_level"])
        risk = max(1, min(10, risk))
        prefer_lim = bool(prefer_limit if prefer_limit is not None else self._defaults["prefer_limit"])

        invested_now = self._sum_invested_eur(currencies_payload)

        # PATCH: stima robusta della cassa EUR dal payload per evitare available_total=0
        eur_cash = self._estimate_available_eur(currencies_payload)
        model_cash_estimate = budget - invested_now  # cassa stimata dal chiamante (prima di reserve)

        # Applica la reserve a entrambe le stime di cassa, poi prendi il massimo (>=0)
        eur_cash_net = eur_cash - reserve
        model_cash_net = model_cash_estimate - reserve
        available_total = max(0.0, eur_cash_net, model_cash_net)

        per_trade_cap = available_total * self.per_trade_cap_pct_map[risk]

        # compatta PRIMA per calcolare batch per lunghezza (includo anche il raw completo)
        compact_all = [self._compact_item(item) for item in currencies_payload]
        sizing = {
            "budget_eur": budget,
            "reserve_eur": reserve,
            "invested_now_eur": invested_now,
            "eur_cash_detected": float(eur_cash),
            "available_total_eur": available_total,
            "risk_level": risk,
            "per_trade_cap_eur": per_trade_cap,
            "prefer_limit": prefer_lim,
            # Passo al modello anche i parametri strategici (solo lettura)
            "rules": {
                "or_min_atr_mult": self.or_min_atr_mult,
                "spread_max_pct": self.spread_max_pct,
                "min_volume_label": self.min_volume_label,
                "enable_shorts": self.enable_shorts,
                "sl_atr_mult": self.sl_atr_mult,
                "tp_atr_mult": self.tp_atr_mult,
            }
        }

        results: List[TradeAction] = []
        # batching dinamico per size + cap massimo per batch
        for batch in self._split_batches_by_size(compact_all, sizing):
            json_obj = self._call(batch, sizing)
            for a in json_obj.get("actions", []):
                act = self._map_action(a)
                act = self._postprocess_action(act, batch, per_trade_cap, prefer_lim)
                results.append(act)

        # dedup per pair: tieni la prima azione proposta per quella coppia
        deduped: Dict[str, TradeAction] = {}
        for a in results:
            if a.pair and a.pair not in deduped:
                deduped[a.pair] = a
        return list(deduped.values())

    # ---------- INTERNALS ----------
    def _sum_invested_eur(self, currencies: List[Dict[str, Any]]) -> float:
        total = 0.0
        for it in currencies:
            row = ((it.get("portfolio") or {}).get("row")) or {}
            val = row.get("val_EUR")
            try:
                if val is not None:
                    total += float(val)
            except Exception:
                pass
        return total

    def _estimate_available_eur(self, currencies: List[Dict[str, Any]]) -> float:
        """ stima NON duplicata della cassa EUR dal payload.
        - Se le coppie quotate in EUR ripetono la stessa cassa su molti item
          in portfolio.available.quote, prendiamo il MAX (non la somma).
        - Se esiste un asset fiat dedicato (ZEUR/EUR) con qty, consideriamo anche quello.
        """
        candidates: List[float] = []
        for it in currencies:
            try:
                quote = (it.get("quote") or "").upper()
                avail = ((it.get("portfolio") or {}).get("available") or {})
                if quote == "EUR":
                    q = avail.get("quote")
                    if q is not None:
                        qf = float(q)
                        if qf >= 0:
                            candidates.append(qf)
                row = ((it.get("portfolio") or {}).get("row")) or {}
                code = (row.get("code") or "").upper()
                asset = (row.get("asset") or "").upper()
                if code in {"ZEUR", "EUR"} or asset in {"ZEUR", "EUR"}:
                    qty = row.get("qty")
                    if qty is not None:
                        qf = float(qty)
                        if qf >= 0:
                            candidates.append(qf)
            except Exception:
                continue
        if not candidates:
            return 0.0
        # Evita il doppio conteggio cross-item: prendi la migliore stima singola
        return max(candidates)

    def _compact_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mantengo i campi chiave come prima (per compatibilità con la logica di sizing/postprocess)
        e in più passo TUTTO l'oggetto originale in 'raw' per dare al modello accesso completo.
        """
        pair = item.get("pair")
        kr_pair = item.get("kr_pair")
        try:
            info_24h = (item.get("info", {}) or {}).get("24H", {}) or {}
        except Exception:
            info_24h = {}
        portfolio = item.get("portfolio", {}) or {}
        row = portfolio.get("row") or {}

        return {
            "pair": pair,
            "kr_pair": kr_pair,
            "base": item.get("base"),
            "quote": item.get("quote"),
            # estratto essenziale 24H
            "info_24H": {k: info_24h.get(k) for k in [
                "range","interval_min","since","open","close","start_price","current_price",
                "change_pct","direction","high","low","volume","volume_label",
                "bid","ask","last","mid","spread","ema_fast","ema_slow","atr"
            ] if k in info_24h},
            # snapshot portfolio sintetico
            "portfolio": {
                "qty": row.get("qty"),
                "px_EUR": row.get("px_EUR"),
                "val_EUR": row.get("val_EUR"),
                "avg_buy_EUR": row.get("avg_buy_EUR"),
                "pnl_pct": row.get("pnl_pct"),
                "asset": row.get("asset"),
                "code": row.get("code"),
            },
            # OGGETTO COMPLETO
            "raw": item,
        }

    def _build_messages(self, compact_batch: List[Dict[str, Any]], sizing: Dict[str, Any]):
        # Guida aggiornata: stessa logica operativa, input descritto meglio
        guide = (
            "Ti passo una LISTA di oggetti 'item'. Ogni item include:\n"
            "- 'pair','kr_pair','base','quote'\n"
            "- 'info_24H' (estratto 24H) e 'raw' (oggetto completo con tutte le finestre in 'info': NOW/1H/4H/24H/30D/90D/1Y, "
            "open_orders, pair_limits, portfolio completo, ecc.).\n\n"
            "Obiettivo: opportunità a breve/medio termine (long/short). Non inventare dati: se servono dettagli usa 'raw'. "
            "Se i dati non giustificano un'azione ⇒ 'hold' con breve motivo.\n\n"
            "Regole pratiche:\n"
            "- Nelle azioni usa la chiave 'pair' impostata a **kr_pair** dell'item.\n"
            "- Se proponi stop e take profit, includili nei campi 'stop_loss' e 'take_profit' (NON mischiarli in un unico prezzo).\n"
            "- Rispetta il sizing: 'per_trade_cap_eur' è la spesa max per BUY; per SELL/close usa la qty in portfolio. Se preferisci nuovi posizioni procedi cosi, rimuovi gli ordini aperti stop e loss e poi se non basta fail il sell di qualche cripto.\n"
            "- Evita 'market' se non necessario; preferisci 'limit' salvo breakout chiaro.\n"
            "- Considera che il sistema gira periodicamente: evita over-trading, ma non perdere occasioni chiare.\n"
            "\n"
            # --- Estensioni: stesso significato, più dettagli operativi ---
            "Linee guida aggiuntive (mantieni il senso di quelle sopra):\n"
            "- Breakout valido solo se 'opening range' del giorno è significativo: usa 'or_ok' se presente, altrimenti "
            "valuta se 'range' della 24H è almeno >= {or_mult}*ATR.\n"
            "- Preferisci long quando bias 1H e 4H sono allineati UP (oppure ema_fast > ema_slow su più finestre). "
            "Se bias discordanti, riduci o evita il trade.\n"
            "- Evita operazioni su coppie con 'volume_label' basso o 'spread' percentuale superiore a {spread}%.\n"
            "- Rispetta 'ordermin/lot' dai 'pair_limits': se il sizing minimo non è raggiungibile, imposta 'hold'.\n"
            "- Se proponi un BUY e ATR è disponibile ma non dai SL/TP, calcola SL = entry - {sl}*ATR e TP = entry + {tp}*ATR.\n"
            "- Evita duplicazioni con 'open_orders': se c'è già un ordine coerente, preferisci hold; se non coerente, proponi cancel/replace.\n"
            "- Usa 'motivo' sintetico (≤80 parole) e, quando utile, aggiungi un reason_code tra: CAP_ZERO, OR_ZERO, LOW_LIQ, "
            "HIGH_SPREAD, MIXED_BIAS, ORDMIN_BLOCK, DUP_ORDER.\n"
        ).format(
            or_mult=self.or_min_atr_mult,
            spread=self.spread_max_pct,
            sl=self.sl_atr_mult,
            tp=self.tp_atr_mult,
        )
        sizing_line = f"\n\nSIZING: {repr(sizing)}\n"
        return [
            {"role": "system", "content": self.system_instructions},
            {"role": "user", "content": guide + sizing_line + "DATI:\n" + repr(compact_batch)}
        ]

    # --------- batching per size ----------
    def _split_batches_by_size(self, compact_all: List[Dict[str, Any]], sizing: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """
        Crea batch che rispettano sia max_batch_size sia una soglia di caratteri
        dell'input totale (system+user).
        """
        batches: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []

        def fits(candidate: List[Dict[str, Any]]) -> bool:
            messages = self._build_messages(candidate, sizing)
            total_chars = sum(len(m.get("content", "")) for m in messages)
            # margine di sicurezza ~10%
            return total_chars <= int(self.max_input_chars * 0.9)

        for item in compact_all:
            if len(cur) >= self.max_batch_size:
                batches.append(cur)
                cur = []
            tentative = cur + [item]
            if fits(tentative):
                cur = tentative
            else:
                if cur:
                    batches.append(cur)
                    cur = [item] if fits([item]) else []
                else:
                    batches.append([item])
                    cur = []
        if cur:
            batches.append(cur)
        return batches

    def _call(self, compact_batch, sizing) -> Dict[str, Any]:
        messages = self._build_messages(compact_batch, sizing)
        messages[-1]["content"] += (
            "\n\nIMPORTANTISSIMO: restituisci **SOLO** un singolo JSON valido con questa struttura:\n"
            "{ \"actions\": [...], \"disclaimer\": \"...\" }\n"
            "Nessun testo prima o dopo. Nessun commento. Nessun markdown."
        )

        backoff = [1, 2, 4, 8]
        last_err = None

        for attempt, sleep_s in enumerate([0] + backoff):
            if sleep_s:
                time.sleep(sleep_s)
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    timeout=self.request_timeout_s,
                )
                try:
                    parsed = resp.output[0].parsed
                    if parsed:
                        return parsed
                except Exception:
                    pass

                text = getattr(resp, "output_text", None) or ""
                if not text:
                    parts = []
                    try:
                        for block in resp.output or []:
                            for c in getattr(block, "content", []) or []:
                                if getattr(c, "type", "") == "output_text":
                                    parts.append(getattr(c, "text", "") or "")
                    except Exception:
                        pass
                    text = "".join(parts)

                try:
                    return json.loads(text)
                except Exception:
                    s, e = text.find("{"), text.rfind("}")
                    if s != -1 and e != -1 and e > s:
                        return json.loads(text[s:e+1])
                    raise RuntimeError(f"JSON parsing failed\n{text[:2000]}")
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                continue

        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

    # ---------- Helpers per rounding e lookup ----------
    @staticmethod
    def _decimals_from_number(x, cap=8) -> int:
        try:
            s = f"{float(x):.12f}".rstrip("0").rstrip(".")
            return min(len(s.split(".")[1]) if "." in s else 0, cap)
        except Exception:
            return 2

    @staticmethod
    def _round_to(x: Optional[float], dec: int) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(f"{float(x):.{int(dec)}f}")
        except Exception:
            return x

    def _infer_price_decimals(self, item: Dict[str, Any]) -> int:
        """Prova a inferire i decimali prezzo dal payload (NOW/24H bid/ask/last)."""
        raw = item.get("raw") or {}
        info = (raw.get("info") or {})
        for k in ("NOW","1M","5M","30M","1H","24H", "1H", "30D"):
            v = info.get(k) or {}
            for key in ("bid", "ask", "last", "mid", "close", "current_price"):
                val = v.get(key)
                if val is not None:
                    d = self._decimals_from_number(val)
                    if d is not None:
                        return d
        # fallback conservativo
        base = (raw.get("base") or "").upper()
        return 1 if base in {"BTC","XBT"} else 4

    def _find_item_for_pair(self, compact_batch: List[Dict[str, Any]], any_pair: str) -> Optional[Dict[str, Any]]:
        # match su kr_pair, pair, oppure "BASE/QUOTE"
        for it in compact_batch:
            raw = it.get("raw") or {}
            if it.get("kr_pair") == any_pair or it.get("pair") == any_pair:
                return it
            if raw:
                human = f"{raw.get('base')}/{raw.get('quote')}"
                if human == any_pair:
                    return it
        return None

    def _find_price_for_pair(self, compact_batch: List[Dict[str, Any]], pair: str) -> Optional[float]:
        for it in compact_batch:
            if it.get("pair") == pair or it.get("kr_pair") == pair:
                p = (it.get("info_24H") or {}).get("current_price")
                try:
                    return float(p) if p is not None else None
                except Exception:
                    return None
        return None

    # ---------- Estrattori di indicatori dal payload ----------
    def _get_atr(self, item: Dict[str, Any]) -> Optional[float]:
        v = (item.get("info_24H") or {}).get("atr")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _get_or_range(self, item: Dict[str, Any]) -> Optional[float]:
        # se nel raw esiste un flag or_ok o un campo or_range, usali; altrimenti usa 'range' 24H
        raw = item.get("raw") or {}
        info24 = (item.get("info_24H") or {})
        # supporto a or_range/or_ok se presenti
        or_range = raw.get("or_range") or info24.get("or_range") or info24.get("range")
        try:
            return float(or_range) if or_range is not None else None
        except Exception:
            return None

    def _get_bias_flags(self, item: Dict[str, Any]) -> Dict[str, Optional[str]]:
        raw = item.get("raw") or {}
        info = (raw.get("info") or {})
        # provo a leggere 'direction' su NOW/1H/24H; fallback su ema_fast vs ema_slow
        def bias_from(v: Dict[str, Any]) -> Optional[str]:
            d = (v or {}).get("direction")
            if isinstance(d, str) and d:
                return d.upper()
            ef, es = (v or {}).get("ema_fast"), (v or {}).get("ema_slow")
            try:
                if ef is not None and es is not None:
                    return "UP" if float(ef) > float(es) else "DOWN"
            except Exception:
                pass
            return None

        b1h = bias_from(info.get("1H") or {})
        b4h = bias_from(info.get("4H") or {}) or bias_from(info.get("24H") or {})
        return {"b1h": b1h, "b4h": b4h}

    def _volume_ok(self, item: Dict[str, Any]) -> bool:
        label = (item.get("info_24H") or {}).get("volume_label")
        order = {"Low": 0, "Medium": 1, "High": 2}
        return order.get(str(label).capitalize(), 0) >= order.get(self.min_volume_label, 1)

    def _spread_pct(self, item: Dict[str, Any]) -> Optional[float]:
        v = item.get("info_24H") or {}
        bid, ask, mid = v.get("bid"), v.get("ask"), v.get("mid")
        try:
            bid, ask = float(bid), float(ask)
            mid = float(mid) if mid is not None else (bid + ask) / 2 if (bid and ask) else None
            if mid and mid > 0 and bid is not None and ask is not None:
                return abs(ask - bid) / mid * 100.0
        except Exception:
            return None
        return None

    def _ordermin_qty(self, item: Dict[str, Any]) -> Optional[float]:
        limits = ((item.get("raw") or {}).get("pair_limits") or {})
        v = limits.get("ordermin") or limits.get("min_order") or limits.get("order_min")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    # ---------- Postprocess azione ----------
    def _postprocess_action(
        self,
        act: TradeAction,
        compact_batch: List[Dict[str, Any]],
        per_trade_cap_eur: float,
        prefer_limit: bool
    ) -> TradeAction:
        # 1) se market ma prefer_limit=True, prova a convertirlo in limit vicino al prezzo corrente
        if prefer_limit and (act.quando == "market"):
            if not act.break_price and not act.limite:
                px = self._find_price_for_pair(compact_batch, act.pair)
                if px:
                    act.quando = "limit"
                    act.prezzo = px

        # 2) forza pair = kr_pair (se lo troviamo nel batch)
        item = self._find_item_for_pair(compact_batch, act.pair) or \
               self._find_item_for_pair(compact_batch, (act.pair or "").replace("XBT", "BTC"))
        if item:
            kr = (item.get("kr_pair") or (item.get("raw") or {}).get("kr_pair") or act.pair)
            if kr:
                act.pair = kr

        # 3) rounding a tick: prezzi ai decimali della coppia, quantità a lot_decimals
        price_dec = self._infer_price_decimals(item) if item else 2
        lot_dec = int((((item or {}).get("raw") or {}).get("pair_limits") or {}).get("lot_decimals") or 8)

        def rp(x): return self._round_to(x, price_dec)
        def rq(x): return self._round_to(x, lot_dec)

        act.prezzo = rp(act.prezzo)
        act.stop_loss = rp(act.stop_loss)
        act.take_profit = rp(act.take_profit)
        act.quantita = rq(act.quantita)

        # 3.1 — Regole HARD prima del sizing
        def add_reason(r):
            act.motivo = (act.motivo + f" | {r}") if act.motivo else r

        # CAP zero → nessun buy
        if act.tipo == "buy" and per_trade_cap_eur <= 0:
            act.tipo = "hold"
            add_reason("CAP_ZERO")
            return act

        # Liquidity / spread
        sp = self._spread_pct(item) if item else None
        if act.tipo == "buy":
            if not self._volume_ok(item):
                act.tipo = "hold"
                add_reason("LOW_LIQ")
                return act
            if sp is not None and sp > self.spread_max_pct:
                act.tipo = "hold"
                add_reason("HIGH_SPREAD")
                return act

        # Opening range / ATR
        atr = self._get_atr(item) if item else None
        or_range = self._get_or_range(item) if item else None
        if act.tipo == "buy":
            # se non c'è un or_ok esplicito, usa range vs ATR
            if atr is not None and or_range is not None and (or_range < self.or_min_atr_mult * atr):
                act.tipo = "hold"
                add_reason("OR_ZERO")
                return act

            # bias gating (long se 1H/4H UP)
            bias = self._get_bias_flags(item) if item else {}
            if bias.get("b1h") and bias.get("b4h"):
                if not (bias["b1h"] == "UP" and bias["b4h"] == "UP"):
                    # non blocco sempre: preferisco prudenza e spiego
                    add_reason("MIXED_BIAS")

        # 4) sizing BUY
        price = act.prezzo or self._find_price_for_pair(compact_batch, act.pair)
        if price and act.tipo == "buy":
            max_eur = max(10.0, per_trade_cap_eur)
            if act.quantita_eur is None and act.quantita is not None:
                act.quantita_eur = act.quantita * price
            if act.quantita_eur is not None:
                if act.quantita_eur > max_eur:
                    act.quantita_eur = max_eur
                if act.quantita is None:
                    act.quantita = act.quantita_eur / price if price > 0 else None
            elif act.quantita is not None:
                eur = act.quantita * price
                if eur > max_eur:
                    act.quantita = max_eur / price
                    act.quantita_eur = max_eur
            else:
                act.quantita_eur = max_eur
                act.quantita = max_eur / price if price > 0 else None

            if act.quantita is not None:
                act.quantita = float(f"{act.quantita:.6f}")
            if act.quantita_eur is not None:
                act.quantita_eur = float(f"{act.quantita_eur:.2f}")

            # Order-min check
            qmin = self._ordermin_qty(item) if item else None
            if qmin is not None and act.quantita is not None and act.quantita < qmin:
                # prova ad aumentare fino al min se rientra nel cap, altrimenti hold
                needed_eur = qmin * price
                if needed_eur <= max_eur:
                    act.quantita = qmin
                    act.quantita_eur = float(f"{needed_eur:.2f}")
                else:
                    act.tipo = "hold"
                    add_reason("ORDMIN_BLOCK")
                    return act

        # 5) SL/TP auto se mancano e ATR disponibile (long)
        if act.tipo == "buy" and price:
            if atr is not None:
                if act.stop_loss is None:
                    act.stop_loss = rp(price - self.sl_atr_mult * atr)
                if act.take_profit is None:
                    act.take_profit = rp(price + self.tp_atr_mult * atr)

        # 6) leverage: per spot meglio None se non specificato in input
        if act.leverage is not None:
            try:
                if float(act.leverage) == 1.0:
                    act.leverage = None
            except Exception:
                pass

        return act

    @staticmethod
    def as_dicts(actions: List[TradeAction]) -> List[Dict[str, Any]]:
        return [asdict(a) for a in actions]

    def _map_action(self, a: Dict[str, Any]) -> TradeAction:
        def num(key):
            v = a.get(key)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        default_quando = "limit"
        return TradeAction(
            tipo=str(a.get("tipo") or "hold"),
            pair=str(a.get("pair") or ""),
            prezzo=num("prezzo"),
            quando=(a.get("quando") or default_quando),
            quantita=num("quantita"),
            quantita_eur=num("quantita_eur"),
            stop_loss=num("stop_loss"),
            take_profit=num("take_profit"),
            timeframe=str(a.get("timeframe") or "24H"),
            lato=a.get("lato"),
            break_price=num("break_price"),
            limite=num("limite"),
            leverage=num("leverage"),
            motivo=a.get("motivo"),
        )


# --- USAGE (esempio) ---
# analyzer = CurrencyAnalyzerForYourPayloadV3(
#     model="gpt-5",
#     request_timeout_s=300,
#     reserve_eur=50,
#     risk_level=5,
#     prefer_limit=True,
#     or_min_atr_mult=0.5,
#     spread_max_pct=0.5,
#     min_volume_label="Medium",
#     sl_atr_mult=1.5,
#     tp_atr_mult=2.5,
#     enable_shorts=True,
# )
# actions = analyzer.analyze(merged, rischio=6)
