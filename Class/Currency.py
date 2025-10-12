# Class/Currency.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import time

try:
    from .Order import Order
except ImportError:
    from Order import Order


@dataclass
class Currency:
    # Identità
    base: str
    quote: str
    pair_human: str
    kr_pair: Optional[str] = None

    # Quote / Realtime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mid: Optional[float] = None
    spread: Optional[float] = None

    # Riepilogo range (search)
    range: Optional[str] = None
    since: Optional[int] = None
    interval_min: Optional[int] = None
    open: Optional[float] = None
    close: Optional[float] = None
    change_pct: Optional[float] = None
    change_dir: Optional[str] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    volume_label: Optional[str] = None

    # Indicatori
    vwap: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    atr: Optional[float] = None

    # Opening Range (OR)
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_range: Optional[float] = None
    day_start: Optional[int] = None

    # Segnale corrente
    strategy: Optional[str] = None
    signal: Optional[str] = None
    signal_reason: Optional[str] = None

    # --- NUOVO: Ordini collegati alla currency ---
    max_ordini_per_lista: int = 10
    ordini_attivi: List[Order] = field(default_factory=list)       # open + pending (ultimi N)
    ordini_cancellati: List[Order] = field(default_factory=list)   # annullati/scaduti/rifiutati (ultimi N)
    ordini_chiusi: List[Order] = field(default_factory=list)       # chiusi (ultimi N)

    # conteggi totali (prima del taglio a N)
    conteggio_attivi_totale: int = 0
    conteggio_cancellati_totale: int = 0
    conteggio_chiusi_totale: int = 0


    # Liquidity / Slippage
    liquidity_depth_used: Optional[int] = None
    liquidity_bid_sum: Optional[float] = None
    liquidity_ask_sum: Optional[float] = None
    liquidity_total_sum: Optional[float] = None
    slippage_buy_pct: Optional[float] = None
    slippage_sell_pct: Optional[float] = None

    # Multi-timeframe EMA
    ema50_1h: Optional[float] = None
    ema200_1h: Optional[float] = None
    bias_1h: Optional[str] = None
    ema50_4h: Optional[float] = None
    ema200_4h: Optional[float] = None
    bias_4h: Optional[str] = None

    # Qualità OR
    or_ok: Optional[bool] = None
    or_reason: Optional[str] = None

    # Meta
    updated_at: float = field(default_factory=lambda: time.time())

    # --- Update helpers ---
    def touch(self) -> "Currency":
        self.updated_at = time.time()
        return self

    def update_from_ticker(self, d: Dict[str, Any]) -> "Currency":
        self.kr_pair = d.get("pair", self.kr_pair)
        self.bid = d.get("bid", self.bid)
        self.ask = d.get("ask", self.ask)
        self.last = d.get("last", self.last)
        self.mid = d.get("mid", self.mid)
        self.spread = d.get("spread", self.spread)
        return self.touch()

    def update_from_realtime(self, d: Dict[str, Any]) -> "Currency":
        return self.update_from_ticker(d)

    def update_from_search(self, d: Dict[str, Any]) -> "Currency":
        self.kr_pair = d.get("kr_pair", self.kr_pair)
        self.range = d.get("range", self.range)
        self.since = d.get("since", self.since)
        self.interval_min = d.get("interval_min", self.interval_min)
        self.open = d.get("open", self.open)
        self.close = d.get("close", self.close)
        self.change_pct = d.get("change_pct", self.change_pct)
        self.change_dir = d.get("change_dir", self.change_dir)
        self.high = d.get("high", self.high)
        self.low = d.get("low", self.low)
        self.volume = d.get("volume", self.volume)
        self.volume_label = d.get("volume_label", self.volume_label)
        return self.touch()

    def update_indicators(self, *, vwap=None, ema_fast=None, ema_slow=None,
                          atr=None, volume_label=None) -> "Currency":
        if vwap is not None: self.vwap = vwap
        if ema_fast is not None: self.ema_fast = ema_fast
        if ema_slow is not None: self.ema_slow = ema_slow
        if atr is not None: self.atr = atr
        if volume_label is not None: self.volume_label = volume_label
        return self.touch()

    def update_or(self, d: Dict[str, Any]) -> "Currency":
        if not d: return self
        self.or_high = d.get("or_high", self.or_high)
        self.or_low = d.get("or_low", self.or_low)
        self.or_range = d.get("or_range", self.or_range)
        self.day_start = d.get("day_start", self.day_start)
        return self.touch()

    def update_signal(self, d: Dict[str, Any]) -> "Currency":
        self.strategy = d.get("strategy", self.strategy)
        self.signal = d.get("signal", self.signal)
        self.signal_reason = d.get("reason", self.signal_reason)
        return self.touch()

    # --- NUOVO: gestione ordini collegati ---
    def set_limite_ordini(self, n: int) -> "Currency":
        try:
            self.max_ordini_per_lista = max(1, int(n))
        except Exception:
            self.max_ordini_per_lista = 10
        return self

    def attach_orders(
        self,
        *,
        attivi: Optional[List[Order]] = None,
        cancellati: Optional[List[Order]] = None,
        chiusi: Optional[List[Order]] = None,
        conteggi: Optional[Dict[str, int]] = None
    ) -> "Currency":
        if attivi is not None:
            self.conteggio_attivi_totale = len(attivi)
            self.ordini_attivi = list(attivi[: self.max_ordini_per_lista])
        if cancellati is not None:
            self.conteggio_cancellati_totale = len(cancellati)
            self.ordini_cancellati = list(cancellati[: self.max_ordini_per_lista])
        if chiusi is not None:
            self.conteggio_chiusi_totale = len(chiusi)
            self.ordini_chiusi = list(chiusi[: self.max_ordini_per_lista])
        if conteggi:
            self.conteggio_attivi_totale = conteggi.get("attivi", self.conteggio_attivi_totale)
            self.conteggio_cancellati_totale = conteggi.get("cancellati", self.conteggio_cancellati_totale)
            self.conteggio_chiusi_totale = conteggi.get("chiusi", self.conteggio_chiusi_totale)
        return self.touch()

    # --- Stampa ordini in maniera compatta ---
    def print_ordini_breve(self):
        def _p(lista: List[Order], titolo: str):
            print(f"[currency][{self.pair_human}] {titolo} (mostrati {len(lista)}/{getattr(self, f'conteggio_{titolo}_totale', len(lista))})")
            for o in lista:
                o.print_breve()

        _p(self.ordini_attivi, "attivi")
        _p(self.ordini_cancellati, "cancellati")
        _p(self.ordini_chiusi, "chiusi")

    # --- Serializzazione sicura con liste di Order ---
    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        # sovrascrivo le liste di ordini con versioni serializzate
        base["ordini_attivi"] = [o.to_dict() for o in self.ordini_attivi]
        base["ordini_cancellati"] = [o.to_dict() for o in self.ordini_cancellati]
        base["ordini_chiusi"] = [o.to_dict() for o in self.ordini_chiusi]
        return base

    def __repr__(self) -> str:
        return (f"<Currency {self.pair_human} last={self.last} "
                f"chg={self.change_pct}% vol={self.volume_label} sig={self.signal}>")


def update_liquidity(self, d: dict) -> "Currency":
    self.liquidity_depth_used = d.get("depth_used", self.liquidity_depth_used)
    self.liquidity_bid_sum    = d.get("bid_sum", self.liquidity_bid_sum)
    self.liquidity_ask_sum    = d.get("ask_sum", self.liquidity_ask_sum)
    self.liquidity_total_sum  = d.get("total_sum", self.liquidity_total_sum)
    self.slippage_buy_pct     = d.get("slippage_buy_pct", self.slippage_buy_pct)
    self.slippage_sell_pct    = d.get("slippage_sell_pct", self.slippage_sell_pct)
    return self.touch()

def update_mtf(self, d: dict) -> "Currency":
    self.ema50_1h = d.get("ema50_1h", self.ema50_1h)
    self.ema200_1h = d.get("ema200_1h", self.ema200_1h)
    self.bias_1h   = d.get("bias_1h", self.bias_1h)
    self.ema50_4h = d.get("ema50_4h", self.ema50_4h)
    self.ema200_4h = d.get("ema200_4h", self.ema200_4h)
    self.bias_4h   = d.get("bias_4h", self.bias_4h)
    return self.touch()

def update_or_quality(self, ok: bool | None, reason: str | None) -> "Currency":
    self.or_ok = ok
    self.or_reason = reason
    return self.touch()
