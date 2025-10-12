# Class/Order.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import time


@dataclass
class Order:
    # Identità / Riferimenti
    id_ordine: Optional[str] = None           # es. txid Kraken
    id_riferimento: Optional[str] = None      # refid / userref / clOrdID se disponibili
    strategia: Optional[str] = None           # nome strategia (es. or_breakout, ema_pullback, ecc.)
    motivo: Optional[str] = None              # nota/descrizione libera
    origine: Optional[str] = None             # 'manuale' | 'bot' | 'strategia'

    # Strumento
    coppia: Optional[str] = None              # es. "BTC/EUR"
    kr_pair: Optional[str] = None             # es. "XXBTZEUR"
    lato: Optional[str] = None                # 'long' | 'short' (per spot lo short richiede margin/futures)
    tipo_operazione: Optional[str] = None     # 'acquisto' | 'vendita' (buy/sell)

    # Tipo ordine e parametri
    tipo_ordine: Optional[str] = None         # 'market' | 'limit' | 'stop-loss' | 'take-profit' | 'stop-limit' ...
    durata: Optional[str] = None              # GTC | IOC | FOK
    post_only: Optional[bool] = None
    reduce_only: Optional[bool] = None
    leva: Optional[float] = None              # se margin/futures
    margine_usato: Optional[float] = None     # in QUOTE (se fornito da exchange)

    # Prezzi / Quantità
    prezzo_limite: Optional[float] = None     # price
    prezzo_stop: Optional[float] = None       # price2 / stop
    prezzo_take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None     # distanza trailing in valore (facoltativo)
    quantita: Optional[float] = None          # vol richiesto
    quantita_eseguita: Optional[float] = None # vol_exec
    prezzo_medio: Optional[float] = None      # avg_price
    prezzo_apertura: Optional[float] = None   # primo fill
    prezzo_chiusura: Optional[float] = None   # media finale a chiusura (se chiuso)

    # Costi / Valori
    valore_totale: Optional[float] = None     # cost (in QUOTE)
    commissioni: Optional[float] = None       # fee (in QUOTE)
    guadagno_perdita: Optional[float] = None  # P&L realizzato in QUOTE (se chiuso)
    pnl_percentuale: Optional[float] = None   # P&L% realizzato

    # Stato ordine
    stato: Optional[str] = None               # 'in_attesa'|'attivo'|'parziale'|'eseguito'|'chiuso'|'annullato'|'scaduto'|'rifiutato'
    sottostato: Optional[str] = None          # 'in_attesa_trigger'|'in_coda'|...
    motivo_annullamento: Optional[str] = None

    # Tempi
    ts_creazione: Optional[int] = None        # opentm
    ts_avvio: Optional[int] = None            # starttm
    ts_scadenza: Optional[int] = None         # expiretm
    ts_chiusura: Optional[int] = None         # closetm

    # Trigger logici (per bot)
    attende_trigger: Optional[bool] = None    # True se deve scattare su condizione
    condizione_trigger: Optional[str] = None  # es. "breakout OR", "prezzo < VWAP - 1*ATR", ecc.

    # Meta
    aggiornato_il: float = field(default_factory=lambda: time.time())

    # -------- Helper di calcolo / aggiornamento --------

    def touch(self) -> "Order":
        self.aggiornato_il = time.time()
        return self

    def tempo_dall_apertura(self) -> Optional[float]:
        """Secondi dalla creazione (se definita)."""
        if self.ts_creazione is None:
            return None
        return time.time() - float(self.ts_creazione)

    def durata_ordine(self) -> Optional[float]:
        """Secondi tra creazione e chiusura, se chiuso."""
        if self.ts_creazione is None or self.ts_chiusura is None:
            return None
        return float(self.ts_chiusura) - float(self.ts_creazione)

    def e_attivo(self) -> bool:
        """True se l’ordine è aperto/attivo (anche parzialmente)."""
        return self.stato in {"attivo", "parziale", "in_attesa"}

    def calcola_pnl_realtime(self, prezzo_corrente: Optional[float]) -> Optional[float]:
        """
        P&L teorico (non realizzato) in QUOTE:
        - long: (prezzo_corrente - prezzo_medio) * qty
        - short: (prezzo_medio - prezzo_corrente) * qty
        Commissioni non sottratte (o sottrai se vuoi).
        """
        if prezzo_corrente is None or self.prezzo_medio is None or not self.quantita_eseguita:
            return None
        qty = float(self.quantita_eseguita)
        if (self.lato or "").lower() == "short":
            pnl = (self.prezzo_medio - prezzo_corrente) * qty
        else:
            pnl = (prezzo_corrente - self.prezzo_medio) * qty
        return pnl

    def aggiorna_da_kraken(self, oid: str, payload: Dict[str, Any]) -> "Order":
        """
        Mappa i campi più comuni del modello Kraken sull’ordine locale.
        Accetta record da: GetOpenOrders / GetClosedOrders / QueryOrders.
        """
        self.id_ordine = oid or self.id_ordine
        self.touch()

        # campi di primo livello
        self.stato = payload.get("status", self.stato)
        self.ts_creazione = payload.get("opentm", self.ts_creazione)
        self.ts_avvio = payload.get("starttm", self.ts_avvio)
        self.ts_scadenza = payload.get("expiretm", self.ts_scadenza)
        self.ts_chiusura = payload.get("closetm", self.ts_chiusura)

        self.quantita = _f(payload.get("vol", self.quantita))
        self.quantita_eseguita = _f(payload.get("vol_exec", self.quantita_eseguita))
        self.valore_totale = _f(payload.get("cost", self.valore_totale))
        self.commissioni = _f(payload.get("fee", self.commissioni))
        self.prezzo_medio = _f(payload.get("price", self.prezzo_medio)) or _f(payload.get("avg_price", self.prezzo_medio))
        # 'price' in Kraken può essere "avg price" a chiusura; in open orders è spesso 0

        # campi in 'descr'
        d = payload.get("descr", {}) or {}
        self.coppia = d.get("pair", self.coppia)
        self.tipo_operazione = _map_buy_sell(d.get("type")) or self.tipo_operazione
        self.tipo_ordine = d.get("ordertype", self.tipo_ordine)
        self.prezzo_limite = _f(d.get("price", self.prezzo_limite))
        self.prezzo_stop = _f(d.get("price2", self.prezzo_stop))
        self.leva = _f(d.get("leverage", self.leva))
        oflags = (d.get("oflags") or "").lower()
        self.post_only = ("post" in oflags) or self.post_only
        self.reduce_only = ("reduce-only" in oflags) or self.reduce_only
        self.durata = d.get("timeinforce", self.durata)

        # lato (long/short) inferito da leva/tipo + tipo_operazione (semplificazione)
        # Nota: su spot puro 'short' non esiste senza margin/futures.
        self.lato = self.lato or _infer_lato(self.tipo_operazione, self.leva)

        # stato leggibile
        self._normalizza_stato()
        return self

    def _normalizza_stato(self):
        """
        Converte gli stati Kraken in etichette semplici:
        open -> 'attivo'
        pending -> 'in_attesa'
        closed -> 'chiuso'
        canceled -> 'annullato'
        expired -> 'scaduto'
        rejected -> 'rifiutato'
        partial (derivato se vol_exec < vol) -> 'parziale'
        """
        m = {
            "open": "attivo",
            "pending": "in_attesa",
            "closed": "chiuso",
            "canceled": "annullato",
            "expired": "scaduto",
            "rejected": "rifiutato",
        }
        if self.stato in m:
            self.stato = m[self.stato]
        # parziale
        try:
            if (self.stato == "attivo") and self.quantita and self.quantita_eseguita and (0 < float(self.quantita_eseguita) < float(self.quantita)):
                self.stato = "parziale"
        except Exception:
            pass

    def chiudi(self, prezzo_chiusura: Optional[float] = None, pnl: Optional[float] = None, pnl_pct: Optional[float] = None, motivo: Optional[str] = None) -> "Order":
        """Segna l’ordine come chiuso (gestione lato client)."""
        self.prezzo_chiusura = prezzo_chiusura or self.prezzo_chiusura
        self.guadagno_perdita = pnl if pnl is not None else self.guadagno_perdita
        self.pnl_percentuale = pnl_pct if pnl_pct is not None else self.pnl_percentuale
        self.ts_chiusura = int(time.time())
        self.stato = "chiuso"
        if motivo:
            self.motivo_annullamento = motivo
        return self.touch()

    def annulla(self, motivo: Optional[str] = None) -> "Order":
        self.stato = "annullato"
        if motivo:
            self.motivo_annullamento = motivo
        return self.touch()

    def in_attesa_di_trigger(self, condizione: str) -> "Order":
        self.attende_trigger = True
        self.condizione_trigger = condizione
        self.sottostato = "in_attesa_trigger"
        return self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # --- Stampa leggibile in italiano ---
    def print_breve(self):
        print(f"[ordine] {self.id_ordine} | {self.coppia} | {self.tipo_operazione}/{self.tipo_ordine} "
              f"| stato:{self.stato} qty:{self.quantita_eseguita or self.quantita} px_med:{self.prezzo_medio}")

    def print_completo(self):
        print("[ordine][completo]")
        for k, v in self.to_dict().items():
            print(f"  - {k}: {v}")


# ---------- utility locali ----------

def _f(x) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def _map_buy_sell(t: Optional[str]) -> Optional[str]:
    if not t: return None
    t = t.lower()
    if t == "buy": return "acquisto"
    if t == "sell": return "vendita"
    return t

def _infer_lato(tipo_operazione: Optional[str], leva: Optional[float]) -> Optional[str]:
    """
    Heuristica: se c’è leva > 1 e tipo_operazione='vendita' -> 'short', altrimenti 'long'.
    (Con spot senza margin, lato di solito è 'long')
    """
    try:
        if (tipo_operazione or "").lower() == "vendita" and (leva or 1.0) > 1.0:
            return "short"
        return "long"
    except Exception:
        return None
