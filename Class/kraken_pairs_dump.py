# kraken_pairs_dump.py
import requests
import csv

KRAKEN_API = "https://api.kraken.com/0/public/AssetPairs"

def get_asset_pairs():
    r = requests.get(KRAKEN_API, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    return data["result"]  # dict: { pair_code: {meta...}, ... }

def iter_pairs(pairs_dict, quote_filter=None):
    """
    quote_filter: es. 'EUR' per tenere solo */EUR.
    Nessun replace: uso i campi forniti da Kraken.
    """
    for kr_code, meta in pairs_dict.items():
        base  = meta.get("base")     # es. 'XXBT'
        quote = meta.get("quote")    # es. 'ZEUR'
        ws    = meta.get("wsname")   # es. 'XBT/EUR' (comodo per UI)
        alt   = meta.get("altname")  # es. 'XBTEUR' o simile

        # filtra per quote se richiesto (accetta sia 'EUR' che 'ZEUR')
        if quote_filter:
            q = (quote or "").upper()
            if q != quote_filter.upper() and q != ("Z" + quote_filter.upper()):
                continue

        yield {
            "kr_code": kr_code,
            "wsname": ws,
            "altname": alt,
            "base": base,
            "quote": quote,
            "pair_decimals": meta.get("pair_decimals"),
            "lot_decimals": meta.get("lot_decimals"),
            "ordermin": meta.get("ordermin"),  # può essere None se non fornito
        }

if __name__ == "__main__":
    pairs = get_asset_pairs()
    rows = list(iter_pairs(pairs, quote_filter="EUR"))

    # stampa a video
    print(f"Totale pair con quote=EUR: {len(rows)}")
    for r in rows[:20]:
        print(f"- {r['wsname'] or r['altname']}  (base={r['base']} quote={r['quote']})")

    # salva CSV completo
    # out = "kraken_pairs_EUR.csv"
    # with open(out, "w", newline="", encoding="utf-8") as f:
    #     w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    #     w.writeheader()
    #     w.writerows(rows)
    # print(f"Salvato: {out}")
