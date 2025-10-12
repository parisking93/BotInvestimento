# Class/SaveOrder.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
import os
import json
import time
import math
import csv

try:
    from .Orders import Orders
    from .Order import Order
except ImportError:
    from Orders import Orders
    from Order import Order


class SaveOrder:
    """
    Salva su file gli ordini (da Orders) filtrati per stato.

    Uso rapido:
        saver = SaveOrder(ordini, cartella="exports", stato="chiuso")
        saver.salva()       # JSON -> exports/saveOrderedata_chiuso_YYYYMMDD-HHMMSS.json
        saver.salva_csv()   # CSV  -> exports/saveOrderedata_chiuso_YYYYMMDD-HHMMSS.csv
    """

    def __init__(self, ordini: Orders, cartella: str = "ordini_export", stato: Optional[str] = "chiuso"):
        self.ordini = ordini
        self.cartella = cartella
        self.stato = stato  # 'attivo' | 'in_attesa' | 'parziale' | 'chiuso' | 'annullato' | 'scaduto' | 'rifiutato' | None/'tutti'

    # ---------- helpers interni ----------
    def _filtra(self) -> List[Order]:
        if not self.stato or str(self.stato).lower() in {"tutti", "all", "*"}:
            return list(self.ordini.elenco)
        return self.ordini.per_stato(self.stato)

    def _ensure_dir(self) -> None:
        os.makedirs(self.cartella, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def _clean_numbers(obj):
        """
        Ripulisce ricorsivamente NaN/Inf non serializzabili in JSON, sostituendoli con None.
        """
        if isinstance(obj, dict):
            return {k: SaveOrder._clean_numbers(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SaveOrder._clean_numbers(v) for v in obj]
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        return obj

    @staticmethod
    def _none_to_empty(x):
        return "" if x is None else x

    def crea_filename(self, base: Optional[str] = None, includi_timestamp: bool = True, ext: str = ".json") -> str:
        """
        Genera un nome file. Per richiesta: base predefinita 'saveOrderedata'.
        """
        stato_norm = (self.stato or "tutti").replace(" ", "_").lower()
        base = base or f"saveOrderedata_{stato_norm}"
        if includi_timestamp:
            base += f"_{self._timestamp()}"
        if not ext.startswith("."):
            ext = "." + ext
        return os.path.join(self.cartella, base + ext)

    # ---------- API: salva JSON ----------
    def salva(self, filename: Optional[str] = None, includi_timestamp: bool = True, indent: int = 2) -> Dict[str, Any]:
        """
        Crea il file JSON con la lista di ordini nello stato richiesto.
        - filename: se non passato, genera automaticamente "saveOrderedata_<stato>_<timestamp>.json"
        - includi_timestamp: False per sovrascrivere sempre lo stesso nome
        - indent: indentazione JSON (default 2)
        """
        self._ensure_dir()
        ordini_filtrati = self._filtra()
        lista = [o.to_dict() for o in ordini_filtrati]

        payload = {
            "versione": 1,
            "stato_filtrato": self.stato or "tutti",
            "conteggio": len(lista),
            "generato_il": int(time.time()),
            "ordini": self._clean_numbers(lista),
        }

        path = filename or self.crea_filename(includi_timestamp=includi_timestamp, ext=".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent, allow_nan=False)

        print(f"[SaveOrder] salvati {len(lista)} ordini (stato='{self.stato or 'tutti'}') in: {path}")
        return {"ok": True, "path": path, "conteggio": len(lista), "stato": self.stato or "tutti"}

    # ---------- API: salva CSV ----------
    def salva_csv(
        self,
        filename: Optional[str] = None,
        includi_timestamp: bool = True,
        delimiter: str = ",",
        include_header: bool = True
    ) -> Dict[str, Any]:
        """
        Crea il file CSV con gli ordini nello stato richiesto.
        - filename: se non passato, genera "saveOrderedata_<stato>_<timestamp>.csv"
        - delimiter: separatore CSV (default ',')
        - include_header: se True scrive l'intestazione con i nomi colonna
        """
        self._ensure_dir()
        ordini_filtrati = self._filtra()
        righe = [o.to_dict() for o in ordini_filtrati]

        # costruisci set di tutte le colonne presenti (unione delle chiavi)
        colonne: List[str] = []
        seen = set()
        for d in righe:
            for k in d.keys():
                if k not in seen:
                    seen.add(k)
                    colonne.append(k)
        # se vuoto, salva comunque file con 0 righe
        path = filename or self.crea_filename(includi_timestamp=includi_timestamp, ext=".csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=colonne, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            if include_header:
                w.writeheader()
            for d in righe:
                # None -> "", NaN/Inf -> "" (già ripulito nel JSON, qui gestiamo al volo)
                row = {}
                for k in colonne:
                    v = d.get(k)
                    if isinstance(v, float) and not math.isfinite(v):
                        v = None
                    row[k] = self._none_to_empty(v)
                w.writerow(row)

        print(f"[SaveOrder] CSV salvato ({len(righe)} ordini) in: {path}")
        return {"ok": True, "path": path, "conteggio": len(righe), "stato": self.stato or "tutti"}
