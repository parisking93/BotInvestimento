# Class/Currencies.py
from __future__ import annotations
from typing import List, Dict, Optional, Iterable

# import robusto di Currency
try:
    from .Currency import Currency
except ImportError:
    from Currency import Currency


class Currencies:
    """
    Collezione di Currency:
    - mantiene una LISTA per l'ordine (come richiesto)
    - e un INDICE per accesso rapido base/quote -> Currency
    """

    def __init__(self) -> None:
        self._list: List[Currency] = []
        self._index: Dict[str, int] = {}

    # --- internals ---
    @staticmethod
    def _norm(base: str, quote: str):
        b = base.upper()
        q = quote.upper()
        if b == "XBT":  # normalizza XBT -> BTC
            b = "BTC"
        return b, q, f"{b}/{q}"

    # --- CRUD ---
    def add(self, cur: Currency) -> Currency:
        _, _, key = self._norm(cur.base, cur.quote)
        if key in self._index:
            # già presente: aggiorna riferimento
            self._list[self._index[key]] = cur
            return cur
        self._index[key] = len(self._list)
        self._list.append(cur)
        return cur

    def get(self, base: str, quote: str) -> Optional[Currency]:
        _, _, key = self._norm(base, quote)
        i = self._index.get(key)
        return self._list[i] if i is not None else None

    def get_by_pair(self, pair_human: str) -> Optional[Currency]:
        base, quote = pair_human.upper().split("/", 1)
        return self.get(base, quote)

    def get_or_create(self, base: str, quote: str, *, kr_pair: Optional[str] = None) -> Currency:
        cur = self.get(base, quote)
        if cur is not None:
            if kr_pair and not cur.kr_pair:
                cur.kr_pair = kr_pair
            return cur
        b, q, key = self._norm(base, quote)
        cur = Currency(base=b, quote=q, pair_human=key, kr_pair=kr_pair)
        return self.add(cur)

    # --- utils ---
    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterable[Currency]:
        return iter(self._list)

    def to_list(self) -> List[Currency]:
        return list(self._list)

    def to_dicts(self) -> List[dict]:
        return [c.to_dict() for c in self._list]

    def filter_by_quote(self, quote: str) -> List[Currency]:
        q = quote.upper()
        return [c for c in self._list if c.quote == q]

    def filter_by_base(self, base: str) -> List[Currency]:
        b = base.upper() if base.upper() != "XBT" else "BTC"
        return [c for c in self._list if c.base == b]

    def top_by_change(self, n: int = 10, *, descending: bool = True) -> List[Currency]:
        return sorted(
            self._list,
            key=lambda c: (c.change_pct if c.change_pct is not None else float("-inf")),
            reverse=descending,
        )[:n]
