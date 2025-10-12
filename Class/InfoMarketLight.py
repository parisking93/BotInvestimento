# info_market_lite.py
# -*- coding: utf-8 -*-
import csv, json, argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Alias MINIMI e sicuri (niente replace a tappeto)
ALIAS_MAP = {
    "XBT": "BTC",
    "XDG": "DOGE",  # opzionale (storico Kraken)
}
def asset_to_human(asset: Optional[str]) -> str:
    a = (asset or "").upper()
    # togli SOLO un prefisso X/Z (come fa Kraken sugli asset code)
    if a.startswith(("X", "Z")) and len(a) >= 4:
        a = a[1:]
    return ALIAS_MAP.get(a, a)

@dataclass
class PairRow:
    kr_code: Optional[str]     # es. "XXBTZEUR"
    wsname: Optional[str]      # es. "XBT/EUR"
    altname: Optional[str]     # es. "XBTEUR"
    base: Optional[str]        # es. "XXBT"
    quote: Optional[str]       # es. "ZEUR"
    pair_decimals: Optional[int]
    lot_decimals: Optional[int]
    ordermin: Optional[float]

def _to_int(x) -> Optional[int]:
    try:
        return int(x) if x not in (None, "", "None") else None
    except Exception:
        return None

def _to_float(x) -> Optional[float]:
    try:
        return float(x) if x not in (None, "", "None") else None
    except Exception:
        return None

def _read_csv(path: str) -> List[PairRow]:
    rows: List[PairRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                PairRow(
                    kr_code      = d.get("kr_code") or d.get("code") or d.get("pair") or d.get("krpair"),
                    wsname       = d.get("wsname"),
                    altname      = d.get("altname"),
                    base         = d.get("base"),
                    quote        = d.get("quote"),
                    pair_decimals= _to_int(d.get("pair_decimals")),
                    lot_decimals = _to_int(d.get("lot_decimals")),
                    ordermin     = _to_float(d.get("ordermin")),
                )
            )
    return rows

DEFAULT_RANGES = ["NOW","1M","5M","15M","30M","1H","24H","30D","90D","1Y"]

def _empty_info_entry(pair_human: str, kr_pair: Optional[str], rng: str) -> Dict[str, Any]:
    return {
        "pair": pair_human, "kr_pair": kr_pair, "range": rng,
        "interval_min": None, "since": None,
        "open": None, "close": None, "start_price": None, "current_price": None,
        "change_pct": None, "direction": None,
        "high": None, "low": None, "volume": None, "volume_label": None,
        "bid": None, "ask": None, "last": None, "mid": None, "spread": None,
        "ema_fast": None, "ema_slow": None, "atr": None, "vwap": None,
        "or_high": None, "or_low": None, "or_range": None, "day_start": None,
        "or_ok": None, "or_reason": None,
        "liquidity_depth_used": None, "liquidity_bid_sum": None,
        "liquidity_ask_sum": None, "liquidity_total_sum": None,
        "slippage_buy_pct": None, "slippage_sell_pct": None,
        "ema50_1h": None, "ema200_1h": None, "ema50_4h": None, "ema200_4h": None,
        "bias_1h": None, "bias_4h": None,
    }

class InfoMarketLite:
    """
    Costruisce JSON 'currency' dal CSV (export AssetPairs) prendendo TUTTE le pair.
    - base/quote/pair in formato umano (BTC/EUR, ETH/EUR, ...).
    - kr_pair = codice REST grezzo (XXBTZEUR, XETHZEUR, ...).
    - nessuna chiamata di rete.
    """
    def __init__(self, csv_path: str, ranges: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.rows = _read_csv(csv_path)
        self.ranges = (ranges or DEFAULT_RANGES)

    def build(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()  # evita duplicati su pair umano
        for row in self.rows:
            base_h = asset_to_human(row.base)
            quote_h = asset_to_human(row.quote)
            if not base_h or not quote_h:
                continue
            pair_human = f"{base_h}/{quote_h}"
            if pair_human in seen:
                continue
            seen.add(pair_human)

            kr_pair = row.kr_code or row.altname  # preferisci il code grezzo

            info: Dict[str, Any] = {rng: _empty_info_entry(pair_human, kr_pair, rng) for rng in self.ranges}

            limits = {}
            if row.lot_decimals is not None:
                limits["lot_decimals"] = row.lot_decimals
            if row.ordermin is not None:
                limits["ordermin"] = row.ordermin
            limits = limits or None

            out.append({
                "base": base_h,
                "quote": quote_h,
                "pair": pair_human,   # es. "BTC/EUR"
                "kr_pair": kr_pair,   # es. "XXBTZEUR"
                "info": info,
                "pair_limits": limits,
                "open_orders": [],
                "portfolio": {
                    "row": None,
                    "trades": [],
                    "ledgers": [],
                    "available": {"base": None, "quote": None}
                }
            })
        return out

    def save(self, path: str) -> None:
        data = self.build()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[InfoMarketLite] salvato: {path}  (records={len(data)})")

# CLI
def main():
    # import argparse
    # ap = argparse.ArgumentParser(description="Build currencies JSON from Kraken AssetPairs CSV (offline).")
    # ap.add_argument("--csv", required=True)
    # ap.add_argument("--pairs", nargs="*", default=None, help="Filtra: accetta BTC/EUR, XBT/EUR, XBTEUR, XXBTZEUR...")
    # ap.add_argument("--ranges", nargs="*", default=None)
    # ap.add_argument("--out", default="currencies.json")
    # args = ap.parse_args()

    im = InfoMarketLite(
        csv_path="kraken_pairs_EUR.csv",
        # pairs=["BTC/EUR", "ETH/EUR"],    # opzionale: ometti per tutte
        ranges=["NOW", "1M", "1H", "24H"]  # opzionale
    )

    data = im.build()          # ottieni la lista di oggetti currency
    im.save("currencies.json") # salva su file

if __name__ == "__main__":
    main()
