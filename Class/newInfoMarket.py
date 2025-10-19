# -*- coding: utf-8 -*-
"""
KrakenNowArchiver
-----------------
Obiettivo: chiamare **sempre e solo il NOW** da Kraken per ogni pair, salvare in un file JSON per-currency
con **la stessa struttura** dei tuoi input (info.{NOW e tanti bucket}, pair_limits, open_orders, portfolio...)

Logica di "shift": ad ogni nuovo NOW:
- inseriamo la nuova fotografia in `info.NOW` (con `date_iso`)
- la **vecchia** `info.NOW` viene spostata in un bucket calcolato come **differenza temporale** dall'ultima
  snapshot, es. 2 minuti -> chiave `"2M"`. Se la differenza è 1h 15m -> chiave `"75M"`.
- Dopo molte chiamate avrai uno storico completo (3gg, 6gg, ...), tutto derivato solo da NOW.

In più:
- Popoliamo i campi che già usavi (bid/ask/last/mid/spread, volume, ema_fast/slow se li mandi, ecc.).
- Calcoliamo in **locale** EMA 1H/4H e semplici ATR/VWAP incrementali dalla history (senza round).
- Opzionale: prendiamo anche il **Depth** (order book) per riempire slippage/liquidity (è comunque NOW).
- **Timing reale**: misuriamo durata fetch/process e la scriviamo in `_meta.timings`.

Dipendenze: `aiohttp` (per REST), nessun websocket necessario.
Niente arrotondamenti: i numeri restano raw (qualsiasi quantizzazione solo nel runner).
"""
from __future__ import annotations
import os, json, time, math, asyncio, datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import aiohttp

# ---------------------- Util ----------------------

def utc_ts() -> float:
    return time.time()

def iso_from_ts(ts: float) -> str:
    return dt.datetime.utcfromtimestamp(ts).replace(tzinfo=dt.timezone.utc).isoformat()

def safe_pair_file(pair: str) -> str:
    return pair.replace("/", "") + ".json"

def _utc_midnight(ts: float) -> int:
    dt_ = dt.datetime.utcfromtimestamp(ts).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)
    return int(dt_.timestamp())

# Delta -> bucket name (in minuti interi). Esempi: 2M, 5M, 75M, 1440M, 2880M
# Se vuoi tassonomie fisse (1M/5M/15M/1H/4H/12H/24H/48H) basta mappare qui come vuoi.

def bucket_from_delta_secs(delta_s: float) -> str:
    if delta_s <= 0:
        return "0M"
    minutes = int(round(delta_s / 60.0))
    return f"{minutes}M"

# ---------------------- Kraken client (REST) ----------------------

class KrakenPublic:
    BASE = "https://api.kraken.com/0/public"

    def __init__(self, session: aiohttp.ClientSession, qps: float = 1.6):
        self.sess = session
        self.qps = qps
        self._last = 0.0

    async def _throttle(self):
        # semplice rate-limit client side
        delta = utc_ts() - self._last
        min_interval = 1.0 / max(self.qps, 0.1)
        if delta < min_interval:
            await asyncio.sleep(min_interval - delta)
        self._last = utc_ts()

    async def ticker(self, pair: str) -> Dict[str, Any]:
        await self._throttle()
        url = f"{self.BASE}/Ticker?pair={pair.replace('/', '')}"
        async with self.sess.get(url, timeout=20) as r:
            r.raise_for_status()
            data = await r.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken Ticker error: {data['error']}")
        return data["result"]

    async def depth(self, pair: str, count: int = 25) -> Dict[str, Any]:
        await self._throttle()
        url = f"{self.BASE}/Depth?pair={pair.replace('/', '')}&count={count}"
        async with self.sess.get(url, timeout=20) as r:
            r.raise_for_status()
            data = await r.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken Depth error: {data['error']}")
        return data["result"]

    async def asset_pairs(self, pair: str) -> Dict[str, Any]:
        await self._throttle()
        url = f"{self.BASE}/AssetPairs?pair={pair.replace('/', '')}"
        async with self.sess.get(url, timeout=20) as r:
            r.raise_for_status()
            data = await r.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken AssetPairs error: {data['error']}")
        return data["result"]

    async def ohlc(self, pair: str, interval: int = 1, since: Optional[int] = None) -> Dict[str, Any]:
        await self._throttle()
        url = f"{self.BASE}/OHLC?pair={pair.replace('/','')}&interval={interval}"
        if since is not None:
            url += f"&since={since}"
        async with self.sess.get(url, timeout=20) as r:
            r.raise_for_status()
            data = await r.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken OHLC error: {data['error']}")
        return data["result"]

# ---------------------- Archiver ----------------------

class KrakenNowArchiver:
    def __init__(self, store_dir: str = "./currency", depth_levels: int = 25, get_depth: bool = True, qps: float = 1.6):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.depth_levels = int(depth_levels)
        self.get_depth = bool(get_depth)
        self.qps = qps
        self._session: Optional[aiohttp.ClientSession] = None
        self._kraken: Optional[KrakenPublic] = None
        # ---- indicator params ----
        self.ema_fast_len = 12
        self.ema_slow_len = 26
        self.atr_len = 14
        self.vwap_len = 60      # 60 minuti
        self.rsi_len = 14
        self.bb_len = 20
        self.bb_k = 2.0


            # ---- indicator helpers (nuovi) ----
    def _sma(self, xs: list[float]) -> Optional[float]:
        xs = [x for x in xs if x is not None]
        if not xs: return None
        return sum(xs) / float(len(xs))

    def _std(self, xs: list[float]) -> Optional[float]:
        xs = [x for x in xs if x is not None]
        n = len(xs)
        if n <= 1: return None
        m = self._sma(xs)
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        return math.sqrt(var)

    def _ema_last(self, closes: list[Optional[float]], length: int) -> Optional[float]:
        xs = [c for c in closes if c is not None]
        if not xs: return None
        alpha = 2.0 / (length + 1.0)
        ema = None
        for c in xs:
            ema = c if ema is None else (alpha * c + (1 - alpha) * ema)
        return ema

    def _rsi_last(self, closes: list[Optional[float]], length: int) -> Optional[float]:
        xs = [c for c in closes if c is not None]
        if len(xs) <= length: return None
        gains = []
        losses = []
        for i in range(1, len(xs)):
            d = xs[i] - xs[i-1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        g = self._sma(gains[-length:])
        l = self._sma(losses[-length:])
        if g is None or l is None: return None
        if l == 0: return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))


    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        self._kraken = KrakenPublic(self._session, qps=self.qps)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
        self._session = None
        self._kraken = None

    # ---------- File I/O ----------
    def _path(self, pair: str) -> str:
        return os.path.join(self.store_dir, safe_pair_file(pair))

    def _load(self, pair: str) -> Dict[str, Any]:
        p = self._path(pair)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        return {}

    def _save(self, pair: str, obj: Dict[str, Any]):
        p = self._path(pair)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, p)

    # ---------- Helpers di calcolo locali (nessun round) ----------
    def _calc_mid(self, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _append_history_minute(self, obj: Dict[str, Any], now_ts: float, close: Optional[float], high: Optional[float], low: Optional[float], volume: Optional[float]):
        hist = obj.setdefault("_history", {})
        arr = hist.setdefault("minutes", [])
        arr.append({"ts": now_ts, "close": close, "high": high, "low": low, "volume": volume})
        if len(arr) > 10000:
            arr[:] = arr[-10000:]
        hist["last_now"] = arr[-1]

    def _ema_update(self, prev: Optional[float], x: Optional[float], alpha: float) -> Optional[float]:
        if x is None:
            return prev
        if prev is None:
            return x
        return alpha * x + (1.0 - alpha) * prev

    def _atr_from_hist(self, window: List[Dict[str, Any]]) -> Optional[float]:
        # semplificato: usando high/low/close della finestra. Se non disponibili, usa close range.
        if not window:
            return None
        trs = []
        prev_close = None
        for e in window:
            h = e.get("high")
            l = e.get("low")
            c = e.get("close")
            if h is None or l is None:
                if c is None:
                    continue
                h = c if h is None else h
                l = c if l is None else l
            if prev_close is None or c is None:
                tr = (h - l) if (h is not None and l is not None) else None
            else:
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            if tr is not None:
                trs.append(tr)
            prev_close = c if c is not None else prev_close
        if not trs:
            return None
        return sum(trs) / float(len(trs))

    def _vwap_from_hist(self, window: List[Dict[str, Any]]) -> Optional[float]:
        num = 0.0
        den = 0.0
        for e in window:
            c = e.get("close")
            v = e.get("volume") or 0.0
            if c is None:
                continue
            num += c * v
            den += v
        if den <= 0.0:
            return None
        return num / den

    def _slippage_from_depth(self, depth: Dict[str, Any], kr_pair: str, budget_eur: float = 1000.0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        # ritorna: bid, ask, liq_bid_sum, liq_ask_sum, (avg slippage % su budget)
        # depth[kr_pair] -> {"bids": [[price, vol, ts],...], "asks": [...]}
        try:
            node = next(iter(depth.values())) if kr_pair not in depth else depth[kr_pair]
            bids = node.get("bids", [])
            asks = node.get("asks", [])
        except Exception:
            return None, None, None, None, None
        bid = float(bids[0][0]) if bids else None
        ask = float(asks[0][0]) if asks else None

        def consume(side: List[List[Any]], want_quote: float, is_buy: bool) -> Tuple[float, float]:
            # restituisce (avg_price, total_quote_filled)
            remain = want_quote
            spent = 0.0
            qty_acc = 0.0
            for px, qty, *_ in (side if is_buy else side):
                px = float(px); qty = float(qty)
                # assumiamo quote = qty*px (base*px)
                have = qty * px
                take_q = min(remain, have)
                if take_q <= 0.0:
                    break
                take_base = take_q / px
                spent += take_base * px
                qty_acc += take_base
                remain -= take_q
                if remain <= 0.0:
                    break
            avg = (spent / qty_acc) if qty_acc > 0 else None
            return avg, (want_quote - remain)

        # slippage su 1000 EUR lato buy e sell
        avg_buy, filled_buy = consume(asks, budget_eur, True)
        avg_sell, filled_sell = consume(bids, budget_eur, False)
        liq_bid_sum = sum(float(q)*float(p) for p, q, *_ in bids[:self.depth_levels])
        liq_ask_sum = sum(float(q)*float(p) for p, q, *_ in asks[:self.depth_levels])
        # slippage % vs best
        buy_slip = ((avg_buy - ask) / ask * 100.0) if (avg_buy and ask) else None
        sell_slip = ((bid - avg_sell) / bid * 100.0) if (avg_sell and bid) else None
        return bid, ask, liq_bid_sum, liq_ask_sum, (buy_slip, sell_slip)

    # ---------- Public: fetch & ingest ----------
    async def fetch_now(self, pair: str) -> Dict[str, Any]:
        assert self._kraken is not None, "Use 'async with KrakenNowArchiver(...) as arch:'"
        t0 = utc_ts()
        # 1) AssetPairs per limiti e kr_pair canonical
        ap = await self._kraken.asset_pairs(pair)
        kr_pair = next(iter(ap.keys()))
        ap_node = ap[kr_pair]
        pair_decimals = int(ap_node.get("pair_decimals", ap_node.get("pair_decimals")))
        lot_decimals = int(ap_node.get("lot_decimals", ap_node.get("lot_decimals")))
        ordermin = ap_node.get("ordermin")
        pair_limits = {
            "lot_decimals": lot_decimals,
            "ordermin": float(ordermin) if ordermin is not None else None,
            "pair_decimals": pair_decimals,
            # fees e fee_volume_currency se vuoi, disponibili in ap_node
        }
        # 2) Ticker
        t1 = utc_ts()
        tk = await self._kraken.ticker(pair)
        t2 = utc_ts()
        # estrai valori
        node = tk[kr_pair]
        bid = float(node["b"][0]) if node.get("b") else None
        ask = float(node["a"][0]) if node.get("a") else None
        last = float(node["c"][0]) if node.get("c") else None
        mid = self._calc_mid(bid, ask)
        vol = float(node["v"][1]) if node.get("v") else None  # volume 24h

        # 3) Depth (opzionale ma consigliato per liquidity/slippage) — ancora NOW
        depth_data = {}
        if self.get_depth:
            t3 = utc_ts()
            depth_data = await self._kraken.depth(pair, self.depth_levels)
            t4 = utc_ts()
        else:
            t3 = t4 = t2

        fetch_ms = int((t4 - t0) * 1000) if self.get_depth else int((t2 - t0) * 1000)

        ohlc = await self._kraken.ohlc(pair, interval=1, since=int(t2) - 3600)  # ultima ora basta
        node_ohlc = next(iter(ohlc.values()))
        # default None, nel caso non ci sia alcuna candela
        c_open = c_high = c_low = c_close = c_vwap = c_volume = None
        last_candle = node_ohlc[-1] if node_ohlc else None
        # formati Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
        if last_candle:
            c_ts, c_open, c_high, c_low, c_close, c_vwap, c_volume, _ = last_candle
            c_open = float(c_open); c_high = float(c_high); c_low = float(c_low)
            c_close = float(c_close); c_volume = float(c_volume)

        now_block = {
            "pair": pair,
            "kr_pair": kr_pair,
            "range": "NOW",
            "interval_min": 1,
            "since": int(t2),
            "open": c_open,
            "close": c_close,
            "start_price": None,
            "current_price": None,
            "change_pct": None,
            "direction": None,
            "high": c_high,
            "low": c_low,
            "volume": vol,
            "volume_1m": c_volume,
            "volume_label": None,
            "bid": bid,
            "ask": ask,
            "last": last,
            "mid": mid,
            "spread": (ask - bid) if (ask is not None and bid is not None) else None,
            "ema_fast": None,
            "ema_slow": None,
            "atr": None,
            "vwap": c_vwap if isinstance(c_vwap, (int,float)) else None,
            "or_high": None,
            "or_low": None,
            "or_range": None,
            "day_start": None,
            "or_ok": None,
            "or_reason": None,
            # liquidity/slippage se depth disponibile
            "liquidity_depth_used": self.depth_levels if self.get_depth else None,
        }

        if depth_data:
            bid0, ask0, liq_b, liq_a, slip = self._slippage_from_depth(depth_data, kr_pair, 1000.0)
            if bid0 is not None: now_block["bid"] = bid0
            if ask0 is not None: now_block["ask"] = ask0
            if slip is not None:
                now_block["slippage_buy_pct"], now_block["slippage_sell_pct"] = slip
            if liq_b is not None: now_block["liquidity_bid_sum"] = liq_b
            if liq_a is not None: now_block["liquidity_ask_sum"] = liq_a
            if (liq_b is not None) or (liq_a is not None):
                now_block["liquidity_total_sum"] = (liq_b or 0.0) + (liq_a or 0.0)

        return now_block, pair_limits, fetch_ms

    def _ensure_base_structure(self, obj: Dict[str, Any], pair: str, kr_pair: str):
        base = pair.split("/")[0] if "/" in pair else None
        quote = pair.split("/")[1] if "/" in pair else None
        info = obj.setdefault("info", {})
        info.setdefault("NOW", {"pair": pair, "kr_pair": kr_pair, "range": "NOW"})
        obj.setdefault("base", base)
        obj.setdefault("quote", quote)
        obj.setdefault("pair", pair)
        obj.setdefault("kr_pair", kr_pair)
        obj.setdefault("open_orders", obj.get("open_orders", []))
        obj.setdefault("portfolio", obj.get("portfolio", {"row":{}, "trades":[], "ledgers":[], "available":{"base": None, "quote": None}}))
        obj.setdefault("_history", {})
        obj.setdefault("_meta", {"timings": []})

    def __key_to_minutes(self, k: str) -> Optional[int]:
        """Parsa una chiave temporale ('5M','1H','2D') in minuti interi.
        Ritorna None se la chiave non è riconosciuta."""
        try:
            if k.endswith("M"):
                return int(k[:-1])
            if k.endswith("H"):
                return int(k[:-1]) * 60
            if k.endswith("D"):
                return int(k[:-1]) * 1440
        except Exception:
            return None
        return None

    def __minutes_to_key(self, minutes: int) -> str:
        """Mappa i minuti al bucket *a scatti*:
        - >= 1440 -> floor in 'xD'
        - elif >= 60 -> floor in 'xH'
        - else -> 'xM'"""
        if minutes >= 1440:
            return f"{minutes // 1440}D"
        if minutes >= 60:
            return f"{minutes // 60}H"
        return f"{minutes}M"

    def _shift_previous_now(self, obj: Dict[str, Any], new_since_ts: float):
        """Riscalo tutte le chiavi temporali in obj['info'] in base alla differenza
        tra new_since_ts e previous NOW. Mantiene liste se già presenti o se
        più snapshot cadono sullo stesso bucket dopo lo shift.
        Esempio: se delta = 1 minuto, '5M' -> '6M', 'NOW' -> '1M', '1H' resta '1H'
        fino a raggiungere 60 minuti (diventa '2H' solo se multiplo di 60)."""
        info = obj.setdefault("info", {})
        prev_now = info.get("NOW")
        if not prev_now:
            return

        prev_since = prev_now.get("since") or (obj.get("_history", {}).get("last_now", {}).get("ts"))
        if not prev_since:
            return

        delta_s = float(new_since_ts) - float(prev_since)
        delta_min = int(round(delta_s / 60.0))
        if delta_min <= 0:
            # nulla da shiftare (stesso ts o indietro)
            return

        # Costruiamo la nuova mappa temporanea per le chiavi in 'info'
        new_buckets: Dict[str, Any] = {}
        # Prima: processa il prev_now (vecchio NOW) come se fosse 0 minuti
        prev_minutes = 0
        new_key = self.__minutes_to_key(prev_minutes + delta_min)
        # inseriamo prev_now nel nuovo bucket
        if new_key not in new_buckets:
            new_buckets[new_key] = prev_now
        else:
            # gestisci collisione: promuovi a lista
            if isinstance(new_buckets[new_key], dict):
                new_buckets[new_key] = [new_buckets[new_key]]
            new_buckets[new_key].append(prev_now)

        # Poi processiamo tutte le altre chiavi già presenti in info (escluse chiavi non-temporali)
        for k, v in list(info.items()):
            if k == "NOW":
                continue
            # Parso la chiave in minuti; se non è una chiave temporale la saltiamo
            km = self.__key_to_minutes(k)
            if km is None:
                # chiave NON temporale (es. metadata custom) -> mantienila così com'è
                # (ma non dentro new_buckets; le lasciamo in info più tardi)
                continue
            new_min = km + delta_min
            nk = self.__minutes_to_key(new_min)
            if nk not in new_buckets:
                # se v è lista o dict, lo riutilizziamo così com'è
                new_buckets[nk] = v
            else:
                # se esiste già una voce, promuovi/append
                if isinstance(new_buckets[nk], dict):
                    new_buckets[nk] = [new_buckets[nk]]
                # se v è lista, estendi
                if isinstance(v, list):
                    new_buckets[nk].extend(v)
                else:
                    new_buckets[nk].append(v)

        # Ora ricomponi info: mantieni tutte le chiavi NON temporali invariate,
        # e sovrascrivi le chiavi temporali con new_buckets.
        # Raccogliamo le non-temporali
        non_temp = {k: v for k, v in info.items() if self.__key_to_minutes(k) is None and k != "NOW"}
        # Ricreiamo info: prima le non-temp, poi NOW sarà creato dal chiamante (noi puliamo le vecchie temporal)
        info.clear()
        info.update(non_temp)
        # Inseriamo tutti i nuovi bucket (temporali)
        # Se vuoi un ordine particolare (es. crescente minutes) potremmo ordinarli:
        def key_minutes_for_sort(kname: str) -> int:
            m = self.__key_to_minutes(kname)
            return m if m is not None else 10**12
        for nk in sorted(new_buckets.keys(), key=key_minutes_for_sort):
            info[nk] = new_buckets[nk]
        # Nota: non reinseriamo 'NOW' qui — il chiamante scriverà la nuova NOW subito dopo.


    def _update_indicators_local(self, obj: Dict[str, Any]):
        # 1) prendi la history 1m
        hist = obj.setdefault("_history", {})
        arr = hist.get("minutes", [])
        if not arr:
            return

        closes = [e.get("close") for e in arr if e.get("close") is not None]
        if not closes:
            return

        # 2) calcola una volta gli indicatori “globali” sulla serie 1m
        ema_fast = self._ema_last(closes, self.ema_fast_len)   # es. 12
        ema_slow = self._ema_last(closes, self.ema_slow_len)   # es. 26

        window = arr[-max(60, self.atr_len):]  # finestra comoda per atr/vwap
        atr  = self._atr_from_hist(window)
        vwap = self._vwap_from_hist(window)

        # ema50/200 "1h" e "4h" come già facevi (ema cumulativa sulla serie 1m)
        def ema_series(alpha: float) -> Optional[float]:
            val = None
            for e in arr[-1000:]:
                c = e.get("close")
                if c is None:
                    continue
                val = self._ema_update(val, c, alpha)
            return val

        ema50_1h  = ema_series(2.0/(50+1))
        ema200_1h = ema_series(2.0/(200+1))
        ema50_4h  = ema_series(2.0/(50+1))
        ema200_4h = ema_series(2.0/(200+1))

        # 3) funzione che aggiorna SOLO nodi esistenti (nessuna creazione)
        def _touch(node: Dict[str, Any]):
            if ema_fast  is not None: node["ema_fast"]  = ema_fast
            if ema_slow  is not None: node["ema_slow"]  = ema_slow
            if atr       is not None: node["atr"]       = atr
            if vwap      is not None: node["vwap"]      = vwap
            if ema50_1h  is not None: node["ema50_1h"]  = ema50_1h
            if ema200_1h is not None: node["ema200_1h"] = ema200_1h
            if ema50_4h  is not None: node["ema50_4h"]  = ema50_4h
            if ema200_4h is not None: node["ema200_4h"] = ema200_4h
            if node.get("ema50_1h") is not None and node.get("ema200_1h") is not None:
                node["bias_1h"] = "UP" if node["ema50_1h"] >= node["ema200_1h"] else "DOWN"
            if node.get("ema50_4h") is not None and node.get("ema200_4h") is not None:
                node["bias_4h"] = "UP" if node["ema50_4h"] >= node["ema200_4h"] else "DOWN"

        # 4) applica agli elementi ESISTENTI in info (dict e liste)
        info = obj.get("info", {})
        for key, node in info.items():
            if isinstance(node, dict):
                _touch(node)
            elif isinstance(node, list):
                for elem in node:
                    if isinstance(elem, dict):
                        _touch(elem)
            # altri tipi: ignora



    def _append_minute_history(self, obj: Dict[str, Any], now_ts: float, now_node: Dict[str, Any]):
        close = now_node.get("close")
        if close is None:
            close = now_node.get("last") or now_node.get("mid")
        self._append_history_minute(obj, now_ts, close, now_node.get("high"), now_node.get("low"), now_node.get("volume"))

            # --------- API principale ---------
    async def update_pair(self, pair: str) -> Dict[str, Any]:
            t0 = utc_ts()
            now_block, pair_limits, fetch_ms = await self.fetch_now(pair)
            t1 = utc_ts()

            # 1) carica una sola volta e prepara struttura
            obj = self._load(pair) or {}
            self._ensure_base_structure(obj, pair, now_block.get("kr_pair", pair.replace("/", "")))
            obj["pair_limits"] = pair_limits or obj.get("pair_limits", {})

            # 2) calcolo day-start e change su history ESISTENTE (prima di shift/append)
            c_close_eff = now_block.get("close") or now_block.get("last") or now_block.get("mid")
            day0 = _utc_midnight(now_block["since"])
            hist_arr = obj.setdefault("_history", {}).get("minutes", [])
            start_price = None
            for e in hist_arr:
                if int(e.get("ts", 0)) >= day0 and e.get("close") is not None:
                    start_price = e["close"]
                    break
            if start_price is None:
                start_price = now_block.get("open") or c_close_eff

            if start_price and c_close_eff:
                change_pct = (c_close_eff - start_price) / start_price * 100.0
                direction = "UP" if change_pct >= 0 else "DOWN"
            else:
                change_pct = None
                direction = None

            now_block["start_price"] = start_price
            now_block["current_price"] = c_close_eff
            now_block["change_pct"] = change_pct
            now_block["direction"] = direction

            # 3) shift del vecchio NOW -> bucket
            self._shift_previous_now(obj, new_since_ts=now_block["since"])

            # 4) scrivi nuovo NOW + date_iso
            info = obj.setdefault("info", {})
            now_node = info.setdefault("NOW", {})
            now_ts = utc_ts()
            now_node.update(now_block)
            now_node["date_iso"] = iso_from_ts(now_ts)
            # 5) append history minuto e indicatori
            self._append_minute_history(obj, now_ts, now_node)
            self._update_indicators_local(obj)

            # 6) timings e salvataggio
            t2 = utc_ts()
            obj.setdefault("_meta", {}).setdefault("timings", []).append({
                "ts": now_node["since"],
                "fetch_ms": fetch_ms,
                "process_ms": int((t2 - t1) * 1000),
                "total_ms": int((t2 - t0) * 1000)
            })

            self._save(pair, obj)
            return obj


# ---------------------- Esempio d'uso ----------------------
# Esegui:
#   python -m asyncio KrakenNowArchiver.py
# oppure integra in un tuo orchestratore async.

async def _demo():
    pairs = ["BTC/EUR", "ETH/EUR"]
    async with KrakenNowArchiver(store_dir="./currency", get_depth=True) as arch:
        for p in pairs:
            obj = await arch.update_pair(p)
            print(p, "NOW mid=", obj.get("info",{}).get("NOW",{}).get("mid"), "timings(ms)=", obj.get("_meta",{}).get("timings", [])[-1])

if __name__ == "__main__":
    asyncio.run(_demo())
