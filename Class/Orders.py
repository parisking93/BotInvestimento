# Class/Orders.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

try:
    from .Order import Order
except ImportError:
    from Order import Order


@dataclass
class Orders:
    """
    Collezione ordini con metodi di utilità:
      - aggiungi/aggiorna da payload Kraken
      - liste per stato
      - P&L totale calcolato a partire da prezzi correnti
      - stampa riepiloghi in italiano
    """
    elenco: List[Order] = field(default_factory=list)
    _idx: Dict[str, int] = field(default_factory=dict)   # id_ordine -> index in elenco

    # ---- CRUD ----
    def aggiungi(self, ordine: Order) -> Order:
        if ordine.id_ordine and ordine.id_ordine in self._idx:
            # già presente: sovrascrive
            i = self._idx[ordine.id_ordine]
            self.elenco[i] = ordine
        else:
            self.elenco.append(ordine)
            if ordine.id_ordine:
                self._idx[ordine.id_ordine] = len(self.elenco) - 1
        return ordine

    def get(self, id_ordine: str) -> Optional[Order]:
        i = self._idx.get(id_ordine)
        if i is None:
            return None
        return self.elenco[i]

    def rimuovi(self, id_ordine: str) -> bool:
        i = self._idx.get(id_ordine)
        if i is None:
            return False
        self.elenco.pop(i)
        # ricostruisci indice
        self._idx.clear()
        for j, o in enumerate(self.elenco):
            if o.id_ordine:
                self._idx[o.id_ordine] = j
        return True

    # ---- Integrazione Kraken ----
    def add_or_update_from_kraken(self, oid: str, payload: Dict[str, Any]) -> Order:
        """
        Aggiunge o aggiorna un ordine a partire da un record Kraken:
        - oid è il txid (chiave del dict ritornato dalle API)
        - payload è il dizionario dell'ordine
        """
        o = self.get(oid)
        if o is None:
            o = Order(id_ordine=oid)
        o.aggiorna_da_kraken(oid, payload)
        return self.aggiungi(o)

    def bulk_update_from_kraken_dict(self, payload: Dict[str, Any]) -> int:
        """
        Accetta una mappa {txid: order_dict, ...} come quelle di:
        - GetOpenOrders["result"]["open"]
        - GetClosedOrders["result"]["closed"]
        - QueryOrders["result"]
        Ritorna quanti ordini sono stati aggiornati/aggiunti.
        """
        if not payload:
            return 0
        n = 0
        for txid, od in payload.items():
            self.add_or_update_from_kraken(txid, od)
            n += 1
        return n

    # ---- Filtri / Query ----
    def per_stato(self, stato: str) -> List[Order]:
        return [o for o in self.elenco if (o.stato or "").lower() == stato.lower()]

    def attivi(self) -> List[Order]:
        return [o for o in self.elenco if o.e_attivo()]

    def chiusi(self) -> List[Order]:
        return self.per_stato("chiuso")

    def in_attesa(self) -> List[Order]:
        return self.per_stato("in_attesa")

    def parziali(self) -> List[Order]:
        return self.per_stato("parziale")

    # ---- P&L / Riepilogo ----
    def pnl_teorico_totale(self, prezzi_correnti: Dict[str, float]) -> float:
        """
        Somma il P&L teorico di tutti gli ordini ATTIVI, usando i prezzi correnti per coppia.
        prezzi_correnti: dict { "BTC/EUR": 93500.0, ... }
        """
        tot = 0.0
        for o in self.attivi():
            px = prezzi_correnti.get(o.coppia) if o.coppia else None
            pnl = o.calcola_pnl_realtime(px)
            if pnl is not None:
                tot += pnl
        return tot

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [o.to_dict() for o in self.elenco]

    # ---- Stampa (italiano semplice) ----
    def stampa_breve(self):
        print("[ordini][riepilogo_breve]")
        for o in self.elenco:
            o.print_breve()

    def stampa_completa(self):
        print("[ordini][riepilogo_completo]")
        for o in self.elenco:
            o.print_completo()
            print("-" * 40)
