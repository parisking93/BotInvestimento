import os, json, tempfile, time
import datetime as _dt
from typing import Any, Dict

def _shadow_jsonl_upsert_after(log_path: str, decision_id: str, after_payload: dict) -> bool:
    """
    Cerca il PRIMO record nel JSONL che abbia decision_id == decision_id e
    aggiunge/aggiorna il campo 'after' con `after_payload`. Salvataggio ATOMICO.
    Ritorna True se ha aggiornato, False se non ha trovato nulla.
    """
    if not (log_path and os.path.exists(log_path)):
        return False

    updated = False
    dirn = os.path.dirname(log_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="shadow_", suffix=".jsonl", dir=dirn)
    os.close(fd)

    try:
        with open(log_path, "r", encoding="utf-8") as src, \
             open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                try:
                    obj = json.loads(line)
                except Exception:
                    # riga corrotta → passa attraverso
                    dst.write(line)
                    continue

                did = obj.get("decision_id") or obj.get("_decision_id")
                if (not updated) and str(did) == str(decision_id):
                    obj["after"] = after_payload
                    updated = True

                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # se ho aggiornato, rimpiazzo il file
        if updated:
            os.replace(tmp_path, log_path)
        else:
            # niente match: butta via il tmp
            try: os.remove(tmp_path)
            except Exception: pass

        return updated

    except Exception:
        # best-effort: pulizia tmp
        try: os.remove(tmp_path)
        except Exception: pass
        return False



# ---- JSONL helpers (leggono/scrivono tutto il file) ----
def _jsonl_read_all(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return rows

def _jsonl_write_all(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _jsonl_write_all_atomic(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="shadow_", suffix=".jsonl.tmp",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

# ---------- helpers ----------
def _extract_order_id_from_record(rec: dict) -> str | None:
    oid = rec.get("order_id")
    if oid:
        return oid
    try:
        return ((rec.get("after") or {}).get("kraken_result") or {})\
               .get("_echo", {}).get("order_id") \
               or (rec.get("after") or {}).get("order_id")
    except Exception:
        return None

def _query_orders(kraken, order_id: str) -> dict:
    # ritorna il payload dell'ordine (dict per quell'order_id)
    resp = kraken.query_private("QueryOrders", {"txid": order_id, "trades": True})
    r = (resp or {}).get("result") or {}
    return r.get(order_id) or r  # alcune lib ritornano già l'oggetto piatto

def _query_trades_by_ids(kraken, trade_ids: list[str]) -> list[dict]:
    if not trade_ids:
        return []
    out = []
    for tid in trade_ids:
        # QueryTrades accetta un trade-id alla volta e ritorna un dict {tid: {...}}
        resp = kraken.query_private("QueryTrades", {"txid": tid})
        trades = ((resp or {}).get("result") or {})
        t = trades.get(tid)
        if t:
            t = dict(t)
            t["_id"] = tid
            out.append(t)
        # micro-sleep per non martellare
        time.sleep(0.05)
    return out

def _pnl_from_trades(trades: list[dict]) -> float | None:
    if not trades:
        return None
    buys = sells = fees = 0.0
    seen = False
    for t in trades:
        try:
            typ  = str(t.get("type") or "").lower()  # 'buy' | 'sell'
            cost = float(t.get("cost"))
            fee  = float(t.get("fee") or 0.0)
            seen = True
            if typ == "buy":
                buys += cost
            elif typ == "sell":
                sells += cost
            fees += fee
        except Exception:
            pass
    return ((sells - buys) - fees) if seen else None

# ---------- main ----------
def reconcile_shadow_with_query_orders(log_path: str, kraken, sleep_s: float = 0.0) -> int:
    """
    Per ogni record con order_id:
      - QueryOrders(txid, trades=true)
      - trade_ids = closetxid (se presenti) altrimenti trades (riempimenti dell'ordine)
      - QueryTrades per quei trade_ids
      - pnl = Σ(cost sell) − Σ(cost buy) − Σ(fee)
      - aggiorna record con: pnl (float), closedIds (lista di trade-id)
    """
    now = _dt.datetime.now()
    daily_log_path = _shadow_daily_path(log_path, now)
    rows = _jsonl_read_all(daily_log_path)
    if not rows:
        return 0

    updated = 0
    for i, rec in enumerate(rows):
        oid = _extract_order_id_from_record(rec)
        if not oid:
            continue
        # salta se già calcolato
        if "pnl" in rec and "closedIds" in rec:
            continue

        try:
            o = _query_orders(kraken, oid)  # dict dell'ordine
            # priorità: closetxid (trade-id che chiudono la posizione aperta da quest'ordine)
            cx = o.get("closetxid")
            if isinstance(cx, str) and cx:
                trade_ids = [cx]
            elif isinstance(cx, list):
                trade_ids = [str(t) for t in cx]
            else:
                # fallback: i trade generati da QUESTO ordine
                tids = o.get("trades")
                trade_ids = [str(t) for t in (tids or [])]

            if not trade_ids:
                continue

            trades = _query_trades_by_ids(kraken, trade_ids)
            pnl = _pnl_from_trades(trades)

            rec["pnl"] = pnl
            rec["closedIds"] = trade_ids
            rows[i] = rec
            updated += 1

            if sleep_s:
                time.sleep(sleep_s)

        except Exception as e:
            rec.setdefault("reconcile_error", str(e))
            rows[i] = rec

    if updated:
        _jsonl_write_all_atomic(log_path, rows)
    return updated


# --- daily shadow_actions helper ---
def _shadow_daily_path(base_path, when=None):
    if not base_path:
        return None
    d = (when or _dt.datetime.now()).strftime("%d_%m_%Y")
    base_dir = os.path.dirname(base_path)
    return os.path.join(base_dir, f"shadow_actions_{d}.jsonl")



def attach_currencies_to_decision(log_path: str,
                                  decision_id: str,
                                  currencies: list[dict] | None) -> bool:
    """
    Trova il record 'decision' con decision_id==... dentro lo shadow JSONL (giornaliero)
    e aggiunge/aggiorna il campo 'currencies' con l'oggetto della pair corrispondente.
    Cerca oggi, poi ieri. Salvataggio ATOMICO. Ritorna True se aggiornato.
    """
    if not (log_path and decision_id):
        return False
    currencies = list(currencies or [])

    # Costruisci lista di candidati: oggi -> ieri
    now = _dt.datetime.now()
    paths = [
        _shadow_daily_path(log_path, now),
        _shadow_daily_path(log_path, now - _dt.timedelta(days=1)),
    ]
    # fallback: se esiste ancora il file piatto
    if os.path.exists(log_path):
        paths.append(log_path)

    for p in [pp for pp in paths if pp and os.path.exists(pp)]:
        try:
            rows = _jsonl_read_all(p)
            if not rows:
                continue

            # trova indice del record per decision_id
            idx = -1
            for i, r in enumerate(rows):
                did = r.get("decision_id") or r.get("_decision_id")
                if str(did) == str(decision_id):
                    idx = i
                    break
            if idx < 0:
                continue

            rec = rows[idx]
            # pair dal record (prima action.pair poi pair)
            pair = None
            try:
                pair = ((rec.get("action") or {}).get("pair")) or rec.get("pair")
            except Exception:
                pair = rec.get("pair")
            if not pair:
                continue

            # match della currency su quella pair
            match = None
            for c in currencies:
                cp = c.get("pair") or c.get("pairname") or c.get("symbol")
                if str(cp) == str(pair):
                    match = c
                    break
            if match is None:
                continue

            rec["currencies"] = match
            rows[idx] = rec
            _jsonl_write_all_atomic(p, rows)
            return True
        except Exception:
            continue
    return False



def _net_margin_from_open_orders(row: Dict[str, Any]) -> float:
    """
    Somma le posizioni a leva presenti in row['open_orders'] con ordertype == 'position'.
    Convenzione: buy (long) = +vol, sell (short) = -vol. Usa 'vol_rem' se presente, altrimenti 'vol'.
    """
    net = 0.0
    for o in (row.get("open_orders") or []):
        try:
            if str(o.get("ordertype","")).lower() != "position":
                continue
            side = (o.get("type") or "").lower()   # 'buy' (long) | 'sell' (short)
            vol  = o.get("vol_rem", o.get("vol", 0.0))
            v    = float(vol or 0.0)
            if side == "buy":
                net += v
            elif side == "sell":
                net -= v
        except Exception:
            continue
    return float(net)
