# -*- coding: utf-8 -*-
from __future__ import annotations
import time, os, json, math, threading, random
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING, List, Tuple, Set, Iterable
import requests
import asyncio, datetime as _dt
try:
    from Class.KrakenPortfolio import KrakenPortfolio
except ImportError:
    from KrakenPortfolio import KrakenPortfolio


# --- AGGIUNTA IN FONDO A InfoMarket.py ---------------------------------------
# Richiede: websockets (pip install websockets)
import asyncio, json, time
try:
    import websockets
except Exception:
    websockets = None

import asyncio, json, time
from typing import Set, Iterator, List, AsyncIterator


try:
    from .Currencies import Currencies
except ImportError:
    from Currencies import Currencies

try:
    from .auth import authorize
except ImportError:
    from auth import authorize

if TYPE_CHECKING:
    from .SaveOrder import SaveOrder as _SaveOrder
    from .Orders import Orders as _Orders

API_BASE = "https://api.kraken.com"

_ASSET_ALIAS = {"BTC": "XBT", "XBT": "XBT", "ETH": "ETH", "EUR": "EUR", "USD": "USD"}

def _date_str(ts: Optional[float] = None) -> str:
    d = _dt.datetime.fromtimestamp(ts or _dt.datetime.now().timestamp())
    return d.strftime("%Y%m%d")



# --- merge helpers (portati da bot3.py) ---
def _nonnull(a, b):
    return b if (b is not None and b != {}) else a

def _count_fields(d):
    return sum(1 for v in (d or {}).values() if v not in (None, {}, []))

def _dedupe(items, keyfunc):
    seen = {}
    for it in items or []:
        k = keyfunc(it)
        if k is None:
            import json as _json
            k = _json.dumps(it, sort_keys=True, default=str)
        seen[k] = it
    return list(seen.values())

def _merge_info(a, b):
    out = dict(a or {})
    for rng, val in (b or {}).items():
        out[rng] = val
    return out

def _merge_portfolio(a, b):
    a, b = a or {}, b or {}
    row_a, row_b = a.get("row") or {}, b.get("row") or {}
    row = row_b if _count_fields(row_b) >= _count_fields(row_a) else row_a
    def k_trade(t): return t.get("ordertxid") or t.get("trade_id")
    trades = _dedupe((a.get("trades") or []) + (b.get("trades") or []), k_trade)
    def k_ledger(l): return l.get("refid") or (l.get("time"), l.get("amount"), l.get("asset"))
    ledgers = _dedupe((a.get("ledgers") or []) + (b.get("ledgers") or []), k_ledger)
    av_a, av_b = a.get("available") or {}, b.get("available") or {}
    available = {"base": _nonnull(av_a.get("base"), av_b.get("base")),
                 "quote": _nonnull(av_a.get("quote"), av_b.get("quote"))}
    return {"row": row, "trades": trades, "ledgers": ledgers, "available": available}

def _merge_open_orders(a, b):
    def k(o):
        return ((o.get("kr_pair") or o.get("pair") or "").upper(),
                o.get("type"), o.get("ordertype"),
                o.get("price"), o.get("price2"), o.get("vol_rem"))
    return _dedupe((a or []) + (b or []), k)

def _merge_one_currency(a, b):
    out = {}
    for k in ("base","quote","pair","kr_pair"):
        out[k] = _nonnull(a.get(k), b.get(k))
    out["info"] = _merge_info(a.get("info"), b.get("info"))
    out["pair_limits"] = _nonnull(a.get("pair_limits"), b.get("pair_limits"))
    out["open_orders"] = _merge_open_orders(a.get("open_orders"), b.get("open_orders"))
    out["portfolio"] = _merge_portfolio(a.get("portfolio"), b.get("portfolio"))
    known = {"base","quote","pair","kr_pair","info","pair_limits","open_orders","portfolio"}
    extra = {k:v for k,v in a.items() if k not in known}
    extra.update({k:v for k,v in b.items() if k not in known})
    out.update(extra)
    return out



async def _gather_full_snapshot(self,
                                ranges_hint: Optional[Tuple[str, ...]],
                                max_total: Optional[int]) -> List[Dict[str, Any]]:
    """
    Raccoglie TUTTE le entries usando la stessa sorgente che alimenta gli input_*.json.
    Non cambia la logica: usa stream_async() e unisce i batch.
    """
    # mantieni i parametri correnti dell'istanza
    per_run = getattr(self, "per_run", 20)
    total   = max_total if (max_total is not None) else getattr(self, "total", 400)

    # Se la tua pipeline legge un "ranges_hint", lo passiamo via attributo (compatibility: se non esiste, viene ignorato)
    old_hint = getattr(self, "_ranges_hint", None)
    if ranges_hint is not None:
        setattr(self, "_ranges_hint", tuple(ranges_hint))

    try:
        agg: Dict[str, Dict[str, Any]] = {}
        async for batch in self.stream_async():   # <-- RIUSA il tuo stream
            for obj in batch:
                pair = obj.get("pair")
                if not pair:
                    continue
                # merge "soft" per mantenere il formato identico ai tuoi input_*.json
                if pair not in agg:
                    agg[pair] = obj
                else:
                    # unisci info/range senza perdere campi
                    info_new = obj.get("info", {})
                    info_old = agg[pair].setdefault("info", {})
                    for rng, payload in info_new.items():
                        info_old[rng] = payload
                    # mantieni eventuali altre chiavi aggiornate (open_orders, portfolio, limits, ecc.)
                    for k, v in obj.items():
                        if k in ("info",):
                            continue
                        agg[pair][k] = v
        # ritorna lista nello stesso formato dei tuoi input_*.json
        return list(agg.values())
    finally:
        if ranges_hint is not None:
            # ripristina eventuale hint precedente
            setattr(self, "_ranges_hint", old_hint)

def _normalize_pair(pair: str) -> str:
    base, quote = pair.upper().split("/", 1)
    base = _ASSET_ALIAS.get(base, base)
    quote = _ASSET_ALIAS.get(quote, quote)
    kr_base = "XXBT" if base == "XBT" else ("XETH" if base == "ETH" else base)
    kr_quote = "ZEUR" if quote == "EUR" else ("ZUSD" if quote == "USD" else quote)
    return f"{kr_base}{kr_quote}"

@dataclass
class AuthState:
    ok: bool
    ws_token: Optional[str]
    ws_expires_in: Optional[int]
    server_unixtime: Optional[int]
    balances: Dict[str, str]
    key_last4: str
    error: Optional[str]

class InfoMarket:
    """
    - Rate limiter con backoff anti-429 (configurabile)
    - Cache soft
    - Ticker e AssetPairs in batch
    - OHLC prefetched per INTERVAL e riutilizzato per tutti i range (corretto)
    - MTF da candele già scaricate (fallback REST)
    - Export multi-range per *ogni* coppia + salvataggio JSON in ./currency/
    """

    _PUBLIC_QPS   = 1.6           # prudente: evita 429; puoi alzare con public_qps
    _MAX_RETRIES  = 4
    _CACHE_TTL    = 15.0
    _HTTP_TIMEOUT = 12

    def __init__(self, pair: str = "BTC/EUR", verbose: bool = False, public_qps: Optional[float] = None):
        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=4, backoff_factor=0.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"])
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://",  HTTPAdapter(max_retries=retry))

        self.verbose = bool(verbose)
        self._rate = float(public_qps) if public_qps else self._PUBLIC_QPS

        self.pair_human = pair
        self.cset = Currencies()
        self._pair_cache: Dict[str, str] = {}
        self.pair = self._norm_pair(pair)

        # token bucket
        self._lock = threading.Lock()
        self._tokens = max(1.0, self._rate)
        self._last_refill = time.time()

        self._soft_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}

        a = authorize()
        self.auth = AuthState(
            ok=a.ok, ws_token=a.ws_token, ws_expires_in=a.ws_expires_in,
            server_unixtime=a.server_unixtime, balances=a.balances,
            key_last4=a.key_last4, error=a.error
        )
        if self.verbose:
            print("[InfoMarket] Authorized:", self.auth.ok, "key_last4:", self.auth.key_last4)

    # ---------------- Rate limiter + cache ----------------
    def _acquire_token(self):
        with self._lock:
            now = time.time()
            rate = self._rate
            capacity = max(1.0, rate * 2.0)
            self._tokens = min(capacity, self._tokens + (now - self._last_refill) * rate)
            self._last_refill = now
            if self._tokens < 1.0:
                sleep_s = (1.0 - self._tokens) / rate
                time.sleep(max(0.0, sleep_s))
                self._tokens = 0.0
                self._last_refill = time.time()
            else:
                self._tokens -= 1.0

    def _cache_get(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        item = self._soft_cache.get(key)
        if not item: return None
        ts, data = item
        if (time.time() - ts) <= self._CACHE_TTL:
            return data
        return None

    def _cache_set(self, key: Tuple[str, str], data: Dict[str, Any]):
        self._soft_cache[key] = (time.time(), data)

    def _public(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        key = (endpoint, json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        url = f"{API_BASE}/0/public/{endpoint}"
        last_err = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                self._acquire_token()
                r = self.session.get(url, params=params, timeout=self._HTTP_TIMEOUT)
                r.raise_for_status()
                j = r.json()
                if j.get("error"):
                    msg = "; ".join(j["error"])
                    if "Too many requests" in msg or "EGeneral:Too many requests" in msg:
                        raise requests.HTTPError("429 too many requests (Kraken payload)")
                self._cache_set(key, j)
                return j
            except Exception as e:
                last_err = e
                base = 1.0 if ("429" in str(e) or "too many requests" in str(e).lower()) else 0.4
                jitter = random.uniform(0.05, 0.25)
                if attempt >= self._MAX_RETRIES: break
                time.sleep(base * (2 ** attempt) + jitter)

        raise RuntimeError(f"Public API error {endpoint} params={params}: {last_err}")

    # ---------------- Pair & params helpers ----------------
    def _norm_pair(self, pair: str) -> str:
        key = pair.upper()
        if key in self._pair_cache:
            return self._pair_cache[key]
        if key in (v.upper() for v in self._pair_cache.values()):
            return pair
        try:
            return _normalize_pair(pair)
        except Exception:
            return pair

    def _cur(self, base: Optional[str] = None, quote: Optional[str] = None):
        p = getattr(self, "_params", {"base": "BTC", "quote": "EUR"})
        b = (base or p.get("base", "BTC")).upper()
        q = (quote or p.get("quote", "EUR")).upper()
        return self.cset.get_or_create(b, q, kr_pair=self.pair)

    def getCurrencies(self) -> Currencies:
        return self.cset

    # ---------------- Public market methods ----------------
    def ticker(self, pair: Optional[str] = None) -> Dict[str, Any]:
        p = self._norm_pair(pair or self.pair_human)
        data = self._public("Ticker", {"pair": p})
        if data.get("error"): raise RuntimeError("Ticker error: " + "; ".join(data["error"]))
        res = list(data["result"].values())[0]
        bid, ask, last = float(res["b"][0]), float(res["a"][0]), float(res["c"][0])
        mid = (bid + ask) / 2.0
        out = {"pair": p, "bid": bid, "ask": ask, "last": last, "mid": mid}
        cur = self._cur(); cur.update_from_ticker(out)
        out["currency"] = cur.to_dict(); out["currencies"] = self.cset.to_dicts()
        return out

    def order_book(self, depth: int = 10, pair: Optional[str] = None) -> Dict[str, Any]:
        p = self._norm_pair(pair or self.pair_human)
        data = self._public("Depth", {"pair": p, "count": depth})
        if data.get("error"): raise RuntimeError("Depth error: " + "; ".join(data["error"]))
        return list(data["result"].values())[0]

    def ohlc(self, interval: int = 1, since: Optional[int] = None, pair: Optional[str] = None) -> Dict[str, Any]:
        p = self._norm_pair(pair or self.pair_human)
        params: Dict[str, Any] = {"pair": p, "interval": int(interval)}
        if since: params["since"] = int(since)
        data = self._public("OHLC", params)
        if data.get("error"): raise RuntimeError("OHLC error: " + "; ".join(data["error"]))
        res = data["result"][p]; last = data["result"].get("last")
        return {"candles": res, "last": last}

    def asset_info(self, pair: Optional[str] = None) -> Dict[str, Any]:
        p = self._norm_pair(pair or self.pair_human)
        data = self._public("AssetPairs", {"pair": p})
        if data.get("error"): raise RuntimeError("AssetPairs error: " + "; ".join(data["error"]))
        return list(data["result"].values())[0]

    # ---------------- Auth ----------------
    def refresh_auth(self, verbose: bool = True) -> AuthState:
        a = authorize()
        self.auth = AuthState(
            ok=a.ok, ws_token=a.ws_token, ws_expires_in=a.ws_expires_in,
            server_unixtime=a.server_unixtime, balances=a.balances,
            key_last4=a.key_last4, error=a.error,
        )
        if verbose: print("[InfoMarket] Refreshed auth. OK:", self.auth.ok)
        return self.auth

    @staticmethod
    def _range_to_since_and_interval(range_str: str) -> Dict[str, Any]:
        now = int(time.time())
        rs = range_str.upper().strip()

        # --- realtime / now ---
        if rs in {"NOW", "LIVE", "REALTIME"}:
            return {"since": now - 60, "interval": 1}

        # --- minuti (nuovi) ---
        if rs in {"1M", "1MIN", "1MINUTE", "M1", "1'", "1m"}:
            return {"since": now - 60, "interval": 1}
        if rs in {"5M", "5MIN", "5MINUTE", "M5", "5'", "5m"}:
            return {"since": now - 5 * 60, "interval": 5}
        if rs in {"15M", "15MIN", "15MINUTE", "M15", "15'", "15m"}:
            return {"since": now - 15 * 60, "interval": 15}
        if rs in {"30M", "30MIN", "30MINUTE", "M30", "30'", "30m"}:
            return {"since": now - 30 * 60, "interval": 30}

        # --- ore/giorni già presenti ---
        if rs in {"1H"}:
            return {"since": now - 3600, "interval": 1}
        if rs in {"4H"}:
            return {"since": now - 4*3600, "interval": 5}   # 5m bars per 4H  ✅ NUOVO
        if rs in {"24H", "1D"}:
            return {"since": now - 24 * 3600, "interval": 5}
        if rs in {"48H"}:
            return {"since": now - 2 * 24 * 3600, "interval": 5}
        if rs in {"7D", "1W"}:
            return {"since": now - 7 * 24 * 3600, "interval": 15}
        if rs in {"30D", "1M"}:
            return {"since": now - 30 * 24 * 3600, "interval": 60}
        if rs in {"90D", "3M"}:
            return {"since": now - 90 * 24 * 3600, "interval": 240}
        if rs in {"1Y", "12M"}:
            return {"since": now - 365 * 24 * 3600, "interval": 1440}
        if rs in {"ALL", "MAX", "*"}:
            return {"since": 0, "interval": 1440}

        # --- date YYYY-MM-DD (resto invariato) ---
        try:
            from datetime import datetime, timezone
            if "-" in rs and len(rs) >= 8:
                dt = datetime.fromisoformat(rs)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                since = int(dt.timestamp())
                return {"since": since, "interval": 60}
        except Exception:
            pass

        # fallback conservativo
        return {"since": now - 24 * 3600, "interval": 5}


    # ---------------- setParams / realtime ----------------
    def setParams(self, base: Optional[str] = None, range_str: Optional[str] = None,
                  quote: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not hasattr(self, "_params"):
            self._params = {"base": "BTC", "quote": "EUR", "range": "24H",
                            "since": None, "until": None, "interval": 1}
        if base:  self._params["base"] = base.upper()
        if quote: self._params["quote"] = quote.upper()
        if range_str:
            rr = self._range_to_since_and_interval(range_str)
            self._params.update({"range": range_str, **rr, "until": int(time.time())})
        for k in ("since", "until", "interval"):
            if k in kwargs and kwargs[k] is not None:
                self._params[k] = int(kwargs[k])
        self.pair_human = f"{self._params['base']}/{self._params['quote']}"
        self.pair = self._norm_pair(self.pair_human)
        if self.verbose: print("[setParams]", self._params)
        return dict(self._params)

    def getRealtime(self, pair: Optional[str] = None, depth: int = 0) -> Dict[str, Any]:
        p = self._norm_pair(pair or self.pair_human)
        t = self._public("Ticker", {"pair": p})
        if t.get("error"): raise RuntimeError("Ticker error: " + "; ".join(t["error"]))
        tr = list(t["result"].values())[0]
        bid, ask, last = float(tr["b"][0]), float(tr["a"][0]), float(tr["c"][0])
        mid = (bid + ask) / 2.0; spread = ask - bid
        out = {"pair": p, "bid": bid, "ask": ask, "last": last, "mid": mid, "spread": spread}
        cur = self._cur(); cur.update_from_realtime(out)
        out["currency"] = cur.to_dict(); out["currencies"] = self.cset.to_dicts()
        return out

    # ------- indicatori veloci -------
    @staticmethod
    def _ema_last(values: list, period: int) -> float:
        if not values: return 0.0
        k = 2.0/(period+1.0); ema = values[0]
        for v in values[1:]: ema = v*k + ema*(1.0-k)
        return ema

    @staticmethod
    def _atr(candles, period: int = 14) -> float:
        if len(candles) < 2: return 0.0
        trs = []
        for i in range(1, len(candles)):
            h = float(candles[i][2]); l = float(candles[i][3]); pc = float(candles[i-1][4])
            trs.append(max(h-l, abs(h-pc), abs(l-pc)))
        n = min(period, len(trs)); avg = sum(trs[:n]) / n
        for tr in trs[n:]: avg = (avg*(period-1) + tr) / period
        return avg

    @staticmethod
    def _vwap_global(candles) -> float:
        num = den = 0.0
        for c in candles:
            tp = (float(c[2]) + float(c[3]) + float(c[4]))/3.0
            v  = float(c[6]); num += tp*v; den += v
        return (num/den) if den else 0.0

    @staticmethod
    def _classify_volume(candles) -> str:
        from statistics import median
        vols = [float(c[6]) for c in candles if len(c) > 6]
        if not vols: return "Volume Low"
        m = median(vols); avg = sum(vols)/len(vols)
        if m <= 0: return "Volume Low" if avg <= 0 else "Volume High"
        ratio = avg/m
        return "Volume High" if ratio >= 1.5 else ("Volume Low" if ratio <= 0.75 else "Volume Medium")

    # ---------------- util ----------------
    def _utc_day_start(self, ts: Optional[int] = None) -> int:
        from datetime import datetime, timezone
        if ts is None: ts = int(time.time())
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        zero = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        return int(zero.timestamp())

    # ---------------- OHLC prefetch per INTERVAL ----------------
    def _fetch_ohlc_for_intervals(self, kr_code: str, interval_since: Dict[int, int],
                                  sleep_per_call: float = 0.0) -> Dict[int, List[list]]:
        out: Dict[int, List[list]] = {}
        # ordino per since (più antico prima → cache coerente)
        for interval, since in sorted(interval_since.items(), key=lambda kv: kv[1]):
            data = self._public("OHLC", {"pair": kr_code, "interval": int(interval), "since": int(since)})
            out[interval] = data["result"].get(kr_code, []) or []
            if sleep_per_call:
                time.sleep(float(sleep_per_call))
        return out

    # ---------------- MTF da candele già in cache ----------------
    @staticmethod
    def _mtf_from_candles(c_by_iv: Dict[int, List[list]]) -> dict:
        def ema(vals: List[float], n: int) -> Optional[float]:
            if not vals: return None
            k = 2.0/(n+1.0); e = vals[0]
            for v in vals[1:]: e = v*k + e*(1.0-k)
            return e
        out = {}
        for iv, tag in ((60,"1h"), (240,"4h")):
            closes = [float(c[4]) for c in c_by_iv.get(iv, [])]
            if len(closes) >= 200:
                e50 = ema(closes, 50); e200 = ema(closes, 200)
                out[f"ema50_{tag}"] = e50; out[f"ema200_{tag}"] = e200
                if e50 is not None and e200 is not None:
                    up = 1.001*e200; down = 0.999*e200
                    out[f"bias_{tag}"] = "UP" if e50>up else ("DOWN" if e50<down else "FLAT")
        return out

    # ---------------- Hydrater da subset corretto ----------------
    def _hydrate_from_subset(self, human: str, kr_code: str, range_str: str,
                             *, candles: List[list],
                             ws_bidask: Optional[Dict[str, float]] = None,
                             pre_liq: Optional[dict] = None,
                             pre_mtf: Optional[dict] = None,
                             pre_or: Optional[dict] = None,
                             since: int, interval: int) -> Dict[str, Any]:

        base, quote = human.split("/", 1)

        if candles:
            o = float(candles[0][1]); c = float(candles[-1][4])
            h = max(float(x[2]) for x in candles); l = min(float(x[3]) for x in candles)
            vol_sum = sum(float(x[6]) for x in candles)
            closes = [float(x[4]) for x in candles]
            ema_f = self._ema_last(closes, 9); ema_s = self._ema_last(closes, 21)
            atr_v = self._atr(candles, 14); vwap_g = self._vwap_global(candles)
            change_pct = ((c - o) / o) * 100.0 if o else 0.0
            change_dir = "UP" if change_pct > 0.01 else ("DOWN" if change_pct < -0.01 else "FLAT")
            volume_label = self._classify_volume(candles)
        else:
            o=c=h=l=vol_sum=ema_f=ema_s=atr_v=vwap_g=None
            change_pct= None; change_dir=None; volume_label=None

        bid=ask=last=mid=spread=None
        if ws_bidask:
            bid = ws_bidask.get("bid"); ask = ws_bidask.get("ask"); last = ws_bidask.get("last")
            if (bid is not None) and (ask is not None):
                mid = (bid+ask)/2.0; spread = ask-bid

        or_high=or_low=or_range=day_start=None
        if pre_or:
            or_high = pre_or.get("or_high"); or_low = pre_or.get("or_low")
            or_range = pre_or.get("or_range"); day_start = pre_or.get("day_start")

        rec = {
            "pair": human, "kr_pair": kr_code,
            "range": range_str, "interval_min": int(interval), "since": int(since),
            "open": round(o,6) if o is not None else None,
            "close": round(c,6) if c is not None else None,
            "start_price": round(o,6) if o is not None else None,
            "current_price": round(c,6) if c is not None else None,
            "change_pct": round(change_pct,4) if change_pct is not None else None,
            "direction": change_dir,
            "high": round(h,6) if h is not None else None,
            "low":  round(l,6) if l is not None else None,
            "volume": round(vol_sum,6) if vol_sum is not None else None,
            "volume_label": volume_label,
            "bid": bid, "ask": ask, "last": last, "mid": mid, "spread": spread,
            "ema_fast": ema_f, "ema_slow": ema_s, "atr": atr_v, "vwap": vwap_g,
            "or_high": or_high, "or_low": or_low, "or_range": or_range, "day_start": day_start,
            "or_ok": None, "or_reason": None,
            "liquidity_depth_used": pre_liq.get("depth_used") if pre_liq else None,
            "liquidity_bid_sum": pre_liq.get("bid_sum") if pre_liq else None,
            "liquidity_ask_sum": pre_liq.get("ask_sum") if pre_liq else None,
            "liquidity_total_sum": pre_liq.get("total_sum") if pre_liq else None,
            "slippage_buy_pct": pre_liq.get("slippage_buy_pct") if pre_liq else None,
            "slippage_sell_pct": pre_liq.get("slippage_sell_pct") if pre_liq else None,
            "ema50_1h": pre_mtf.get("ema50_1h") if pre_mtf else None,
            "ema200_1h": pre_mtf.get("ema200_1h") if pre_mtf else None,
            "ema50_4h": pre_mtf.get("ema50_4h") if pre_mtf else None,
            "ema200_4h": pre_mtf.get("ema200_4h") if pre_mtf else None,
            "bias_1h": pre_mtf.get("bias_1h") if pre_mtf else None,
            "bias_4h": pre_mtf.get("bias_4h") if pre_mtf else None,
        }

        if rec["or_range"] is not None:
            if rec["or_range"] == 0:
                rec["or_ok"], rec["or_reason"] = False, "OR range = 0"
            elif (rec["atr"] is not None) and rec["or_range"] < 0.5 * float(rec["atr"]):
                rec["or_ok"], rec["or_reason"] = False, "OR range < 0.5*ATR"
            else:
                rec["or_ok"], rec["or_reason"] = True, "OR ok"

        return rec

    # ---------------- EXPORT MULTI-RANGE (come la vecchia, più veloce) ----------------
    def export_currencies_ws(
        self,
        pairs: Optional[List[str]] = None,
        quote: str = "EUR",
        ranges: Optional[List[str]] = None,
        *,
        max_pairs: int = 200,
        depth_top_n: int = 5,
        liquidity_budget_quote: float = 1000.0,
        with_liquidity: bool = False,
        with_mtf: bool = True,
        with_or: bool = True,
        save_folder: str = "currency",
        sleep_per_call: float = 0.03,
        sleep_per_pair: float = 0.01,
        AssetPair = None
    ) -> List[Dict[str, Any]]:
        """
        Export completo per più coppie. **PATCH**: ora accetta sia kr-codes
        (es. 'XXBTZEUR') sia human pairs ('BTC/EUR') e li normalizza in 'BASE/QUOTE'.
        """
        if ranges is None:
            ranges = ["NOW","1M","5M","15M","30M","1H","24H","30D","90D","1Y"]

        # --- AssetPairs una volta (serve per tradurre kr-code -> wsname) ---
        ap_all = self._public("AssetPairs") if AssetPair == None else AssetPair

        if ap_all.get("error"):
            print(f"[InfoMarket] skip invalid pairs in AssetPairs: {ap_all['error']}")
        ap_map = (ap_all.get("result", {}) or {}) if AssetPair == None else AssetPair

        def _to_human(p: str) -> str:
            """Se p è un kr-code, usa wsname per restituire 'BASE/QUOTE';
            se è già 'BASE/QUOTE' limitati a normalizzare BTC/XBT."""
            p = (p or "").strip()
            if "/" in p:
                b,q = p.split("/",1)
                b = "BTC" if b.upper()=="XBT" else b.upper()
                return f"{b}/{q.upper()}"
            row = ap_map.get(p)
            if row and row.get("wsname"):
                b,q = row["wsname"].split("/",1)
                b = "BTC" if b.upper()=="XBT" else b.upper()
                return f"{b}/{q.upper()}"
            # fallback: non rischio lo split -> lascio il codice così
            return p

        # Normalizza SEMPRE la lista in ingresso
        if pairs:
            pairs = [_to_human(p) for p in pairs]
        else:
            # se non passata, costruisci l’elenco human dalla mappa
            tmp: List[str] = []
            for row in ap_map.values():
                ws = row.get("wsname")
                if not ws or "/" not in ws:
                    continue
                b,q = ws.split("/",1)
                if q.upper() != quote.upper():
                    continue
                b = "BTC" if b.upper()=="XBT" else b.upper()
                tmp.append(f"{b}/{q.upper()}")
            priority = {"BTC","ETH"}
            tmp.sort(key=lambda p: (0 if p.split('/')[0] in priority else 1, p))
            pairs = tmp[:max_pairs]

        # mappa human->kr per chiamate batch successive
        human_to_kr: Dict[str, str] = {}
        for code, row in ap_map.items():
            ws = row.get("wsname")
            if not ws or "/" not in ws:
                continue
            b,q = ws.split("/",1)
            b = "BTC" if b.upper()=="XBT" else b.upper()
            human_to_kr[f"{b}/{q.upper()}"] = code

        # Ticker batch (usa kr-codes derivati dall’human)
        kr_codes = [human_to_kr.get(h) or self._norm_pair(h) for h in pairs]
        try:
            t = self._public("Ticker", {"pair": ",".join(kr_codes)})
            ticker_map = t.get("result", {}) if not t.get("error") else {}
        except Exception:
            ticker_map = {}

        # Limits batch
        ai_batch = self._public("AssetPairs", {"pair": ",".join(kr_codes)})
        limits_map: Dict[str, Optional[dict]] = {}
        if not ai_batch.get("error"):
            for code, row in ai_batch["result"].items():
                # -- estrai array di leve (se la coppia supporta il margine) --
                lb = row.get("leverage_buy") or []
                ls = row.get("leverage_sell") or []
                limits_map[code] = {
                    "lot_decimals": row.get("lot_decimals") or row.get("pair_decimals"),
                    "ordermin": float(row.get("ordermin")) if row.get("ordermin") is not None else None,
                    "pair_decimals": row.get("pair_decimals") or row.get("pair_decimals"),
                    "fees": row.get("fees") or [], # [[threshold_usd, pct], ...] (taker)
                    "fees_maker": row.get("fees_maker") or row.get("fees_maker_tier") or [],
                    "fee_volume_currency": row.get("fee_volume_currency") or "ZUSD",
                    "leverage_buy": lb,                      # array es. [2,3,5] oppure []
                    "leverage_sell": ls,                     # array es. [2,3,5] oppure []
                    "leverage_buy_max": (max(lb) if lb else 0),
                    "leverage_sell_max": (max(ls) if ls else 0),
                    "can_leverage_buy": bool(lb),
                    "can_leverage_sell": bool(ls),
                }

        out_array: List[Dict[str, Any]] = []

        for human in pairs:
            # **FIX**: qui prima andava in errore se human non aveva '/'
            if "/" not in human:
                # non rischio lo split: salto questa coppia (non valida in formato human)
                if self.verbose:
                    print(f"[export] skip '{human}' (non è 'BASE/QUOTE')")
                continue

            kr_code = human_to_kr.get(human) or self._norm_pair(human)
            base, quote_ = human.split("/", 1)

            # earliest since per INTERVAL richiesti dai range
            interval_since: Dict[int, int] = {}
            now = int(time.time())
            for r in ranges:
                rr = self._range_to_since_and_interval(r)
                iv = int(rr["interval"]); sc = int(rr["since"])
                if iv not in interval_since or sc < interval_since[iv]:
                    interval_since[iv] = sc

            # OR 1m se richiesto
            day_start = self._utc_day_start()
            if with_or:
                interval_since[1] = min(interval_since.get(1, now - 60), day_start)

            # prendo le candele una volta per INTERVAL
            candles_by_interval = self._fetch_ohlc_for_intervals(
                kr_code, interval_since, sleep_per_call=sleep_per_call
            )

            # MTF (da candele)
            pre_mtf = self._mtf_from_candles(candles_by_interval) if with_mtf else {}

            # OR una volta
            pre_or = None
            if with_or:
                try:
                    or_c = [c for c in candles_by_interval.get(1, []) if int(c[0]) >= day_start]
                    if len(or_c) >= 2:
                        first15 = or_c[:15]
                        or_high = max(float(x[2]) for x in first15)
                        or_low  = min(float(x[3]) for x in first15)
                        pre_or = {"or_high": or_high, "or_low": or_low, "or_range": or_high-or_low, "day_start": day_start}
                except Exception:
                    pre_or = None

            # Liquidity opzionale (lento)
            pre_liq = {}
            if with_liquidity and depth_top_n > 0:
                try:
                    pre_liq = self._liquidity_block(kr_code, top_n=depth_top_n, budget_quote=float(liquidity_budget_quote))
                except Exception:
                    pre_liq = {}

            # costruisci info per ciascun range usando SOTTOINSEDI corretti
            info_block: Dict[str, Any] = {}
            for r in ranges:
                rr = self._range_to_since_and_interval(r)
                iv = int(rr["interval"]); sc = int(rr["since"])
                all_c = candles_by_interval.get(iv, [])
                subset = [c for c in all_c if int(c[0]) >= sc]  # <<< QUI la correzione per i dati giusti
                # aggancio bid/ask/last dal batch ticker (se disponibile)
                trow = ticker_map.get(kr_code)
                ws_bidask = None
                if trow:
                    try:
                        b = float(trow["b"][0]); a = float(trow["a"][0]); lc = float(trow["c"][0])
                        ws_bidask = {"bid": b, "ask": a, "last": lc, "mid": (b+a)/2.0}
                    except Exception:
                        ws_bidask = None

                info_block[r] = self._hydrate_from_subset(
                    human, kr_code, r,
                    candles=subset, ws_bidask=ws_bidask,
                    pre_liq=pre_liq, pre_mtf=pre_mtf, pre_or=pre_or,
                    since=sc, interval=iv
                )

            # balances disponibili
            avail = self._balances_available(base, quote_)
            portfolio = {"row": None, "trades": [], "ledgers": [], "available": avail}

            out_array.append({
                "base": base, "quote": quote_, "pair": human, "kr_pair": kr_code,
                "info": info_block,
                "pair_limits": limits_map.get(kr_code),
                "open_orders": [],
                "portfolio": portfolio,
            })

            if sleep_per_pair:
                time.sleep(float(sleep_per_pair))

        # # === salva JSON in ./currency/
        # try:
        #     base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        #     out_dir = os.path.join(base_dir, save_folder)
        #     os.makedirs(out_dir, exist_ok=True)
        #     stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        #     path = os.path.join(out_dir, f"currencies_{stamp}.json")
        #     with open(path, "w", encoding="utf-8") as f:
        #         json.dump(out_array, f, ensure_ascii=False, separators=(",", ":"), indent=2)
        #     if self.verbose:
        #         print(f"[export_currencies_ws] Saved {len(out_array)} items to {path}")
        # except Exception as e:
        #     if self.verbose:
        #         print(f"[export_currencies_ws] WARN cannot save JSON: {e}")

        return out_array

    # ---------------- balances helper ----------------
    def _balances_available(self, base: str, quote: str) -> Dict[str, Optional[float]]:
        try:
            bals = self.auth.balances or {}
        except Exception:
            bals = {}
        def _find(asset_code: str) -> Optional[float]:
            if asset_code in bals:
                try: return float(bals[asset_code])
                except Exception: return None
            return None
        base_alias = _ASSET_ALIAS.get(base, base)
        base_kr = "XXBT" if base_alias == "XBT" else ("XETH" if base_alias == "ETH" else base_alias)
        quote_alias = _ASSET_ALIAS.get(quote, quote)
        quote_kr = "ZEUR" if quote_alias == "EUR" else ("ZUSD" if quote_alias == "USD" else quote_alias)
        return {"base": _find(base_kr), "quote": _find(quote_kr)}


    # ---------------- Liquidity helper ----------------
    def _liquidity_block(self, kr_pair: str, top_n: int = 5, budget_quote: float = 1000.0) -> dict:
        """
        Calcola un riassunto di liquidità usando il book (Depth) fino a top_n livelli
        e stima la slippage % per comprare/vendere 'budget_quote' di QUOTE.
        """
        data = self._public("Depth", {"pair": kr_pair, "count": int(top_n)})
        if data.get("error"):
            raise RuntimeError("Depth error: " + "; ".join(data["error"]))
        book = list(data["result"].values())[0]

        bids = [(float(p), float(q)) for (p, q, *_) in book.get("bids", [])]
        asks = [(float(p), float(q)) for (p, q, *_) in book.get("asks", [])]

        if not bids and not asks:
            return {
                "depth_used": int(top_n),
                "bid_sum": 0.0, "ask_sum": 0.0, "total_sum": 0.0,
                "slippage_buy_pct": None, "slippage_sell_pct": None,
            }

        def _avg_price(levels: List[tuple], budget_q: float) -> Optional[float]:
            remain = float(budget_q); cost = 0.0; qty = 0.0
            for px, q in levels:
                take = min(q, remain / px)
                cost += take * px
                qty  += take
                remain -= take * px
                if remain <= 1e-9:
                    break
            return None if qty <= 0.0 else (cost / qty)

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        ref = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else (best_ask or best_bid)

        avg_buy  = _avg_price(asks, budget_quote)  # comprare budget_quote a mercato → prezzo medio
        avg_sell = _avg_price(bids, budget_quote)  # vendere budget_quote a mercato → prezzo medio

        def _slip_buy(ref_px: Optional[float], avg_px: Optional[float]) -> Optional[float]:
            return None if (ref_px is None or avg_px is None) else (avg_px / ref_px - 1.0) * 100.0

        def _slip_sell(ref_px: Optional[float], avg_px: Optional[float]) -> Optional[float]:
            return None if (ref_px is None or avg_px is None) else (1.0 - avg_px / ref_px) * 100.0

        bid_sum = sum(q for _, q in bids[:top_n])
        ask_sum = sum(q for _, q in asks[:top_n])

        return {
            "depth_used": int(top_n),
            "bid_sum": bid_sum,
            "ask_sum": ask_sum,
            "total_sum": bid_sum + ask_sum,
            "slippage_buy_pct": _slip_buy(ref, avg_buy),
            "slippage_sell_pct": _slip_sell(ref, avg_sell),
        }









# --- InfoMarket2 (stream async + fallback REST, con log) ----------------------


# --- InfoMarket2 (stream async, dati completi via InfoMarket.export_currencies_ws) --

# --- InfoMarket2 (FAST streaming: WS init only + export in thread) ------------


# --- InfoMarket2 (portfolio-aware, liquidity, EMA/ATR/VWAP, positions filter) --


# --- InfoMarket2 (portfolio-aware + MTF EMA/bias + liquidity + fast streaming) -


_WS_PUBLIC = "wss://ws.kraken.com"

class InfoMarket2:
    """
    Top-movers via WS (prima passata) + export completo via InfoMarket (REST),
    con:
      - EMA50/EMA200 (1h e 4h) + bias_1h/bias_4h calcolati ad ogni batch
      - Liquidity/slippage (opzionale)
      - Portfolio preload e filtro only_positions
      - Streaming asincrono: export in thread, senza re-subscribe WS tra i batch
    Output compatibile con i tuoi file 'inputFirst' (info NOW/4H/24H/48H, pair_limits,
    open_orders, portfolio, liquidity, ecc.).
    """

    def __init__(
        self,
        per_run: int,
        total: int,
        quote: str = "EUR",
        max_pairs: Optional[int] = None,
        verbose: bool = False,
        public_qps: Optional[float] = None,
        allow_rest_fallback: bool = True,
        *,
        with_liquidity: bool = True,   # abilita depth/slippage nell’exporter
        only_positions: bool = False   # se True: lavora solo su asset in portfolio / ordini aperti
    ):
        self._ws_ok = websockets is not None
        self.allow_rest_fallback = bool(allow_rest_fallback)
        if (not self._ws_ok) and (not self.allow_rest_fallback):
            raise RuntimeError("Serve il pacchetto 'websockets' (pip install websockets)")

        self.per_run = int(per_run)
        self.total = int(total)
        self.quote = quote.upper()
        self.max_pairs = max_pairs
        self.verbose = bool(verbose)
        self.with_liquidity = bool(with_liquidity)
        self.only_positions = bool(only_positions)

        # Re-uso della tua InfoMarket (autenticazione, rate, exporter, helpers)
        self._rest = InfoMarket(pair=f"BTC/{self.quote}", verbose=verbose, public_qps=public_qps)

        # mappe & cache
        self._pairs_all: list[str] = []         # "BTC/EUR", "ETH/EUR", ...
        self._kr_to_human: dict[str, str] = {}  # "XXBTZEUR" -> "BTC/EUR"
        self._human_to_kr: dict[str, str] = {}  # "BTC/EUR"  -> "XXBTZEUR"

        # ticker per ranking
        self._ticker: dict[str, dict] = {}
        self._seen: Set[str] = set()

        # snapshot portfolio & ordini
        self._portfolio_snapshot: dict = {}
        self._open_orders_pairs: Set[str] = set()
        self._positions_pairs: Set[str] = set()

        self._preload_portfolio_and_orders()
        time.sleep(2)
        self.run_kraken_portfolio()

    # ------------------- alias helpers -------------------
    @staticmethod
    def _alias_asset_to_human(asset: str) -> str:
        a = asset.upper().replace("X", "").replace("Z", "")
        return "BTC" if a == "XBT" else a

    @staticmethod
    def _fiat_to_kr_code(f):
        return {'EUR':'ZEUR','USD':'ZUSD','GBP':'ZGBP','USDT':'USDT','USDC':'USDC'}.get(f, f)


    # ------------------- portfolio preload ----------------

        # --- ex run_kraken_portfolio ---
    def run_kraken_portfolio(self):
        """
        Esegue il report portafoglio da KrakenPortfolio, stampa e
        salva in attributi: rows/total/trades/ledgers.
        """
        print("\n========== KrakenPortfolio: portafoglio ==========")
        self.kp = KrakenPortfolio()
        rows, total, trades, ledgers, AssetPair = self.kp.portfolio_view()

        # salva negli attributi
        self.portfolio_rows = rows
        self.portfolio_total = total
        self.portfolio_trades = trades
        self.portfolio_ledgers = ledgers
        self.AssetPair = AssetPair

        # self.portfolioIn = kp.investable_eur()



        # --- NEW: leggi anche OpenOrders / OpenPositions dal client krakenex di KrakenPortfolio
        try:
            resp_oo = self.kp.k.query_private('OpenOrders') or {}
            open_orders = (resp_oo.get('result') or {}).get('open', {}) or {}
        except Exception:
            open_orders = {}

        self.portfolioIn = self.kp.investable_eur(resp_oo)
        try:
            resp_pos = self.kp.k.query_private('OpenPositions') or {}
            open_positions = (resp_pos.get('result') or {}) or {}
        except Exception:
            open_positions = {}

        self._portfolio_snapshot['rows'] = rows
        self._portfolio_snapshot['trades'] = trades
        self._portfolio_snapshot['ledgers'] = ledgers
        self._portfolio_snapshot['open_orders'] = open_orders           # <--- chiave già letta in _export_block
        self._portfolio_snapshot['open_positions'] = open_positions     # <--- la useremo per fondere nelle open_orders
        self._portfolio_snapshot['account_totals'] = {
            'total_eur': total,
            'investable_eur': self.portfolioIn,
        }
        return rows, total, trades, ledgers

    def _preload_portfolio_and_orders(self):
        try:
            self._rest.refresh_auth(verbose=self.verbose)
        except Exception:
            pass

        balances = {}
        try:
            balances = dict(self._rest.auth.balances or {})
        except Exception:
            balances = {}

        base_assets: Set[str] = set()
        for k, v in balances.items():
            try:
                qty = float(v)
            except Exception:
                qty = 0.0
            if qty <= 0:
                continue
            human = self._alias_asset_to_human(k)
            if human == self.quote:
                continue
            base_assets.add(human)

        self._positions_pairs = set(f"{b}/{self.quote}" for b in base_assets)

        pairs_from_orders: Set[str] = set()
        oo_fn = getattr(self._rest, "open_orders", None)  # opzionale
        if callable(oo_fn):
            try:
                oo = oo_fn()
            except Exception:
                oo = []
            for row in (oo or []):
                p = (row.get("pair") or row.get("kr_pair") or "").strip()
                if p:
                    pairs_from_orders.add(p)

        self._open_orders_pairs = pairs_from_orders
        self._portfolio_snapshot = {
            "balances": balances,
            "positions_pairs": list(self._positions_pairs),
            "open_order_pairs": list(self._open_orders_pairs),
        }
        if self.verbose:
            print(f"[InfoMarket2] Portfolio preload: {len(base_assets)} assets, "
                  f"{len(self._open_orders_pairs)} pairs with open orders")

    # ------------------- discovery coppie ------------------
    def _discover_pairs(self):
        ap = self._rest._public("AssetPairs")
        result = ap.get("result", {})
        for name, meta in result.items():
            q = meta.get("quote", "")
            if not q:
                continue
            if q.endswith(self.quote) or q == f"Z{self.quote}" or q == self.quote:
                kr = name
                base = meta.get("base", "").replace("X", "").replace("Z", "")
                quote = meta.get("quote", "").replace("X", "").replace("Z", "")
                base = base or kr[:-4]
                quote = quote or kr[-4:]
                b_alias = self._alias_asset_to_human(base)
                q_alias = self._alias_asset_to_human(quote)
                human = f"{b_alias}/{q_alias}"
                self._kr_to_human[kr] = human
                self._human_to_kr[human] = kr
                self._pairs_all.append(human)

        # filtro opzionale (portfolio + ordini)
        if self.only_positions:
            keep: Set[str] = set()
            keep.update(self._positions_pairs)
            for p in list(self._open_orders_pairs):
                if "/" in p:
                    keep.add(p)
                else:
                    human = self._kr_to_human.get(p)
                    if human:
                        keep.add(human)
            self._pairs_all = [p for p in self._pairs_all if p in keep]

        if self.max_pairs:
            self._pairs_all = self._pairs_all[: self.max_pairs]

        if self.verbose:
            print(f"[InfoMarket2] scoperte {len(self._pairs_all)} coppie con quote {self.quote}"
                  + (" (only_positions)" if self.only_positions else ""))

    # ------------------- WS ranking (una volta) ------------
    async def _ws_collect(self, run_seconds: float = 5.0):
        if not self._pairs_all:
            self._discover_pairs()
        if not self._pairs_all:
            return

        chunk_size = 200
        start = time.time()
        async with websockets.connect(_WS_PUBLIC, ping_interval=20, ping_timeout=20) as ws:
            for i in range(0, len(self._pairs_all), chunk_size):
                pairs = self._pairs_all[i : i + chunk_size]
                sub = {"event": "subscribe", "pair": pairs, "subscription": {"name": "ticker"}}
                await ws.send(json.dumps(sub))
                if self.verbose:
                    print(f"[InfoMarket2] WS sottoscritti {len(pairs)} ticker")

            while time.time() - start < run_seconds:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                if isinstance(msg, list) and len(msg) >= 4 and msg[-2] == "ticker":
                    payload = msg[1]
                    pair_human = msg[-1]
                    kr = self._human_to_kr.get(pair_human)
                    if not kr:
                        continue
                    last = float(payload.get("c", [None, None])[0] or 0.0)
                    bid  = float(payload.get("b", [None, None])[0] or 0.0)
                    ask  = float(payload.get("a", [None, None])[0] or 0.0)
                    open_ = float(payload.get("o", [None, None])[0] or 0.0)
                    high = float(payload.get("h", [None, None])[0] or 0.0)
                    low  = float(payload.get("l", [None, None])[0] or 0.0)
                    vol  = float(payload.get("v", [None, None])[0] or 0.0)

                    self._ticker[kr] = {
                        "last": last, "bid": bid, "ask": ask, "open": open_,
                        "high": high, "low": low, "vol": vol, "ts": time.time(),
                    }
        if self.verbose:
            print(f"[InfoMarket2] WS raccolta completata ({len(self._ticker)} coppie aggiornate)")

    # ------------------- REST ranking fallback --------------
    def _collect_ticker_rest(self):
        if not self._pairs_all:
            self._discover_pairs()
        if not self._pairs_all:
            return

        kr_codes = [self._human_to_kr[h] for h in self._pairs_all if h in self._human_to_kr]
        chunk = 100
        for i in range(0, len(kr_codes), chunk):
            sub = kr_codes[i : i + chunk]
            try:
                t = self._rest._public("Ticker", {"pair": ",".join(sub)})
                res = t.get("result", {})
            except Exception:
                res = {}
            for kr, row in res.items():
                try:
                    last = float(row["c"][0]); bid = float(row["b"][0]); ask = float(row["a"][0])
                    o = row.get("o")
                    open_ = float(o) if isinstance(o, str) else float((o or [0])[0])
                    high = float(row["h"][0]) if row.get("h") else None
                    low  = float(row["l"][0]) if row.get("l") else None
                    vol  = float(row["v"][0]) if row.get("v") else None
                except Exception:
                    last = bid = ask = open_ = high = low = vol = None
                self._ticker[kr] = {
                    "last": last, "bid": bid, "ask": ask, "open": open_,
                    "high": high, "low": low, "vol": vol, "ts": time.time(),
                }
        if self.verbose:
            print(f"[InfoMarket2] REST raccolta completata ({len(self._ticker)} coppie aggiornate)")

    # ------------------- ranking -----------------------------
    def _rank_top_movers(self) -> list[str]:
        scored = []
        for kr, t in self._ticker.items():
            o = t.get("open") or 0.0
            last = t.get("last") or 0.0
            chg = (last / o - 1.0) * 100.0 if (o > 0 and last > 0) else -9999.0
            scored.append((chg, kr))
        scored.sort(reverse=True)
        return [kr for _, kr in scored]

    # ------------------- MTF (EMA50/200 1h/4h + bias) -------
    def _mtf_for_block(self, kr_list: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Ritorna: { kr_code: { 'ema50_1h', 'ema200_1h', 'ema50_4h', 'ema200_4h', 'bias_1h', 'bias_4h' } }
        """
        now = int(time.time())
        # 1h: 200 barre ⇒ ~200h (prendo 220h), 4h: 200 barre ⇒ ~800h (prendo ~860h ≈ 36d)
        need = {60: now - 220*3600, 240: now - 860*3600}
        out: Dict[str, Dict[str, float]] = {}
        for kr in kr_list:
            try:
                c_by_iv = self._rest._fetch_ohlc_for_intervals(kr, need, sleep_per_call=0.0)
                mtf = self._rest._mtf_from_candles(c_by_iv)  # {'ema50_1h', 'ema200_1h', 'bias_1h', ...}
            except Exception:
                mtf = {}
            out[kr] = mtf
        return out

    # ------------------- export accurato ---------------------
    def _export_block(self, kr_list: List[str]) -> List[dict]:
        # Mappa kr->human (serve anche per derive base/quote)
        # ap = self._rest._public("AssetPairs")
        # if ap.get("error"):
        #     if self.verbose:
        #         print(f"[InfoMarket2] skip invalid pairs in AssetPairs: {ap['error']}")
        ap_map = self.AssetPair or {}

        def _kr_to_human_local(kr: str) -> str:
            h = self._kr_to_human.get(kr)
            if h: return h
            row = ap_map.get(kr)
            if row and row.get("wsname"):
                b, q = row["wsname"].split("/", 1)
                b = "BTC" if b.upper()=="XBT" else b.upper()
                return f"{b}/{q.upper()}"
            return kr

        human_pairs = [_kr_to_human_local(kr) for kr in kr_list]

        # Hint ranges (coerente con stream)
        ranges_hint = getattr(self, "_ranges_hint", None)
        ranges_use = list(ranges_hint) if ranges_hint else ["NOW","1M","5M","30M","1H","4H","24H","48H"]

        # 1) blocco market da InfoMarket (già con info/liquidity/mtf/OR e 'available' base/quote)
        items_market = self._rest.export_currencies_ws(
            pairs=human_pairs, quote=self.quote, ranges=ranges_use,
            with_liquidity=self.with_liquidity, with_mtf=True, with_or=True,
            max_pairs=len(human_pairs), depth_top_n=25,
            sleep_per_call=0.0, sleep_per_pair=0.0, AssetPair=self.AssetPair
        )
        # Calcolo MTF robusto per il batch e iniezione nei blocchi
        mtf_map = self._mtf_for_block(kr_list)
        # 2) costruiamo arricchimento portfolio da _portfolio_snapshot (se presente)
        snap = self._portfolio_snapshot or {}
        balances = dict(snap.get("balances") or {})
        rows = snap.get("rows") or []            # opzionale: list di righe portafoglio (code, asset, qty, ...)
        trades_all = snap.get("trades") or []    # opzionale: lista trades grezzi Kraken
        ledgers_all = snap.get("ledgers") or []  # opzionale: lista ledgers grezzi Kraken
        open_orders_raw = snap.get("open_orders") or {}  # opzionale: risultato "open" di OpenOrders
        open_positions_raw = snap.get("open_positions") or {}
        account_totals = snap.get("account_totals") or {}
        # indicizza rows per asset code (es. 'XXBT', 'ZEUR', ecc.)
        # portfolio_by_code = {}
        # for r in rows:
        #     code = r.get("code")
        #     if code:
        #         portfolio_by_code[code] = r



        # reverse per normalizzare qualsiasi rappresentazione del pair in kr_code canonico
        rev_by_alt = {}
        rev_by_ws = {}
        # for name, row in ap_map.items():
        #     alt = row.get("altname")
        #     ws  = (row.get("wsname") or "").replace("/", "")
        #     if alt: rev_by_alt[alt] = name
        #     if ws:  rev_by_ws[ws] = name

        portfolio_by_code = {}
        for r in rows:
            if not isinstance(r, dict):   # FIX
                continue
            code = r.get("code")
            if code:
                portfolio_by_code[code] = r

        for it in items_market:
        # Inietta MTF (ema50/200 1h/4h + bias) se mancanti
            kr = it.get("kr_pair")
            mtf = mtf_map.get(kr, {}) if kr else {}
            for rng in ("NOW","1H", "4H", "24H", "48H"):
                blk = it.get("info", {}).get(rng, {})
                if blk is None: continue
                # se i campi sono None, riempi
                for key in ("ema50_1h","ema200_1h","ema50_4h","ema200_4h","bias_1h","bias_4h"):
                    if blk.get(key) is None and (key in mtf):
                        blk[key] = mtf[key]


        def to_kr_pair(s: str | None) -> str | None:
            if not s: return None
            s0 = s.replace("/", "")
            if s0 in ap_map: return s0
            if s0 in rev_by_alt: return rev_by_alt[s0]
            if s0 in rev_by_ws:  return rev_by_ws[s0]
            return s0

        # 2a) open orders → forma semplice + “held” per asset
        held_by_asset = {}
        simple_open_orders = []
        if isinstance(open_orders_raw, dict) and open_orders_raw.get("open"):
            od_map = open_orders_raw["open"]
        elif isinstance(open_orders_raw, dict):
            od_map = open_orders_raw
        else:
            od_map = {}


        # 2a-bis) **NEW**: aggiungi OpenPositions allo stesso array, come “position”
        for pos_txid, p in (open_positions_raw or {}).items():
            # i campi variano; estrai in modo robusto
            pair_field = (p.get("pair") or p.get("pairname") or "").replace("/", "")
            typ   = (p.get("type") or "").lower()           # long/short -> mappo su buy/sell se vuoi
            # alcune risposte Kraken hanno 'cost', 'vol', 'fee', 'terms', 'ordertype', 'price'
            vol   = None
            price = None
            try:
                vol = float(p.get("vol") or p.get("volumen") or 0.0)
            except Exception:
                pass
            try:
                price = float(p.get("price") or p.get("entry_price") or p.get("cost") or 0.0)
            except Exception:
                pass

            base_code = quote_code = None
            if pair_field in ap_map:
                rowp = ap_map[pair_field]
                base_code  = rowp.get("base")
                quote_code = rowp.get("quote")

            simple_open_orders.append({
                "source": "position",
                "kr_pair": pair_field or p.get("pair") or p.get("pairname"),
                "pair": pair_field or p.get("pair") or p.get("pairname"),
                "type": ("buy" if typ == "long" else ("sell" if typ == "short" else typ)),
                "ordertype": "position",
                "price": price,
                "price2": None,
                "vol_rem": vol,  # qui consideriamo tutta la size della posizione
                "base": base_code,
                "quote": quote_code,
                "txid": pos_txid
            })


        for _, od in (od_map or {}).items():
            d = od.get("descr", {})
            raw_pair = d.get("pair") or ""
            kr_pair = to_kr_pair(raw_pair) or to_kr_pair(d.get("pair_short") or "")
            typ = d.get("type")
            ordertype = d.get("ordertype")
            try:
                vol = float(od.get("vol", 0.0)); vol_exec = float(od.get("vol_exec", 0.0) or 0.0)
            except Exception:
                vol = 0.0; vol_exec = 0.0
            vol_rem = max(0.0, vol - vol_exec)
            if vol_rem <= 0:
                continue
            price = None; price2 = None
            try:
                price = float(d.get("price")) if d.get("price") else None
                price2 = float(d.get("price2")) if d.get("price2") else None
            except Exception:
                pass

            base_code = quote_code = None
            if kr_pair in ap_map:
                rowp = ap_map[kr_pair]
                base_code = rowp.get("base")
                quote_code = rowp.get("quote")

            simple_open_orders.append({
                "kr_pair": kr_pair,
                "pair": kr_pair,  # retro-compat
                "type": typ, "ordertype": ordertype,
                "price": price, "price2": price2, "vol_rem": vol_rem,
                "base": base_code, "quote": quote_code
            })

            if typ == "sell" and base_code:
                held_by_asset[base_code] = held_by_asset.get(base_code, 0.0) + vol_rem
            elif typ == "buy" and quote_code:
                px = price or price2
                if px:
                    held_by_asset[quote_code] = held_by_asset.get(quote_code, 0.0) + (px * vol_rem)



        # 2b) available = balances - held
        available_by_asset = {}
        for code, bal in balances.items():
            try:
                fbal = float(bal)
            except Exception:
                continue
            available_by_asset[code] = max(0.0, fbal - held_by_asset.get(code, 0.0))

        # 2c) indicizza trades/ledgers per asset code nel modo più semplice possibile
        trades_by_code = {}
        for t in (trades_all or []):
            pair_raw = (t.get("pair") or "").replace("/", "")
            matched = None
            for code in portfolio_by_code.keys():
                if code and code in pair_raw:
                    matched = code; break
            if not matched:
                # fallback su asset “XBT/EUR” ecc.
                alt = (t.get("asset") or t.get("aclass") or "")
                for code, rowp in portfolio_by_code.items():
                    cand = rowp.get("asset")
                    if isinstance(cand, str) and cand in pair_raw:
                        matched = code; break
            if matched:
                trades_by_code.setdefault(matched, []).append(t)

        ledgers_by_code = {}
        iter_ledgers = ledgers_all.values() if isinstance(ledgers_all, dict) else (ledgers_all or [])
        for l in (iter_ledgers or []):
            asset_code = l.get("asset")
            if isinstance(asset_code, str):
                ledgers_by_code.setdefault(asset_code, []).append(l)

        # 3) arricchisci ogni item di items_market con portfolio + open_orders
        enriched = []
        for it in (items_market or []):
            base = it.get("base"); quote = it.get("quote")
            krp = it.get("kr_pair")
            # deriviamo il base_code dal kr_pair per leggere rows/trades ecc.
            base_code = None
            if isinstance(krp, str):
                qk = self._fiat_to_kr_code(quote)
                if qk and krp.endswith(qk):
                    base_code = krp[:-len(qk)]
                else:
                    # fallback: cerca prefisso che combacia con un asset del portafoglio
                    for code in portfolio_by_code.keys():
                        if krp.startswith(code):
                            base_code = code
                            break

            port_add = {
                "row": portfolio_by_code.get(base_code),
                "trades": trades_by_code.get(base_code, []),
                "ledgers": ledgers_by_code.get(base_code, []),
                "available": {
                    "base": available_by_asset.get(base_code),
                    "quote": available_by_asset.get(self._fiat_to_kr_code(quote)) if quote else None
                }
            }
            oo_this = []
            if simple_open_orders:
                if krp:
                    oo_this = [o for o in simple_open_orders if o["kr_pair"] == krp]
                elif base_code:
                    oo_this = [o for o in simple_open_orders if o.get("base") == base_code]

            merged_item = _merge_one_currency(
                it,
                {
                    "open_orders": oo_this,
                    "portfolio": port_add
                }
            )
            enriched.append(merged_item)

        return enriched


    # ------------------- streaming async (emette mini-batch appena pronti) ---
    async def stream_async(self) -> AsyncIterator[List[dict]]:
        """
        Produce batch di dimensione self.per_run **appena pronti**.
        Internamente avvia un piccolo producer che esporta 1 coppia per volta
        (in thread via _export_block([kr])) e le mette in una queue; il consumer
        raccoglie ogni per_run risultati e li yielda subito.
        """
        import asyncio
        from collections import deque

        self._discover_pairs()
        if not self._pairs_all:
            if self.verbose:
                print("[InfoMarket2] Nessuna coppia da processare dopo discovery/filtro.")
            return

        # Ranking iniziale (via WS se disponibile, altrimenti REST)
        if self._ws_ok:
            await self._ws_collect(run_seconds=5.0)
        else:
            if self.verbose:
                print("[InfoMarket2] websockets non disponibile: uso fallback REST.")
            self._collect_ticker_rest()

        ranked = self._rank_top_movers()

        # --- MUST HAVE robusto (accetta "BTC/EUR" o "XXBTZEUR") ---
        must_have_raw = []   # <- lascia così: più leggibile; puoi cambiarlo da fuori se vuoi
        def _to_kr(x: str) -> str | None:
            x = (x or "").strip().upper()
            if "/" in x:   # formato umano
                # mappa con ciò che conosci già; se assente, normalizza
                return self._human_to_kr.get(x) or self._rest._norm_pair(x)
            # già kr-code
            return x

        must_have_kr = [kr for kr in map(_to_kr, must_have_raw) if kr]

        # Se only_positions è attivo, non voglio perdere il must-have
        if self.only_positions:
            # assicurati che esista il mapping kr->human per il check/merge
            for kr in list(must_have_kr):
                if kr not in self._kr_to_human:
                    # prova a ricavarlo una volta da AssetPairs
                    try:
                        ap = self._rest._public("AssetPairs")
                        row = (ap.get("result") or {}).get(kr)
                        if row and row.get("wsname"):
                            b,q = row["wsname"].split("/",1)
                            b = "BTC" if b.upper()=="XBT" else b.upper()
                            self._kr_to_human[kr] = f"{b}/{q.upper()}"
                            self._human_to_kr[self._kr_to_human[kr]] = kr
                    except Exception:
                        pass

        # De-duplica e metti in testa: must-have + ranked WS/REST
        ranked = list(dict.fromkeys(must_have_kr + ranked))

        if not ranked:
            if self.verbose:
                print("[InfoMarket2] Nessun ticker raccolto per ranking.")
            return

        # --- NON tagliare i must-have anche se only_positions=True ---
        must_have_raw = []  # come già sopra
        def _to_kr(x: str) -> str | None:
            x = (x or "").strip().upper()
            if "/" in x:
                return self._human_to_kr.get(x) or self._rest._norm_pair(x)
            return x
        must_have_kr = {kr for kr in map(_to_kr, must_have_raw) if kr}

        if self.only_positions:
            allowed_humans = set(self._pairs_all)  # es. {"ETH/EUR", "TBTC/EUR", ...}
            keep: list[str] = []
            for kr in ranked:
                human = self._kr_to_human.get(kr)
                if (kr in must_have_kr) or (human in allowed_humans):
                    keep.append(kr)
            ranked = keep


        # Quante coppie in totale voglio produrre?
        to_take = min(int(self.total), len(ranked))
        if to_take <= 0:
            return

        # ---- Producer: esporta UNA coppia per volta e mette il risultato in coda
        queue: asyncio.Queue[list[dict]] = asyncio.Queue(maxsize=max(2, self.per_run * 2))
        stop_sentinel = object()

        async def _producer():
            produced = 0
            for i in range(0, to_take, self.per_run):
                sub = ranked[i:i + self.per_run]
                try:
                    items = await asyncio.to_thread(self._export_block, sub)
                except Exception as e:
                    if self.verbose:
                        print(f"[InfoMarket2] exporter error su {sub}: {e}")
                    items = []
                await queue.put(items)
                produced += len(sub)
            await queue.put(stop_sentinel)


        producer_task = asyncio.create_task(_producer())

        # ---- Consumer: preleva subito dalla queue e raggruppa per_run risultati
        batch_buf: List[dict] = []
        yielded = 0

        while True:
            item = await queue.get()
            if item is stop_sentinel:
                # flush finale se rimane qualcosa nel buffer
                if batch_buf:
                    yield list(batch_buf)
                    yielded += 1
                    batch_buf.clear()
                break

            # item è la lista ritorno di _export_block([kr]) → 0..1 elemento
            if item:
                batch_buf.extend(item)

            # appena raggiungo la dimensione desiderata → YIELD
            if len(batch_buf) >= self.per_run:
                yield list(batch_buf[:self.per_run])
                yielded += 1
                batch_buf = batch_buf[self.per_run:]

        # assicurati che il producer sia chiuso
        try:
            await producer_task
        except Exception:
            pass

        if self.verbose:
            print(f"[InfoMarket2] STREAM COMPLETATO: batches={yielded}, per_run={self.per_run}, total={to_take}")


    # ------------------- compat run() -----------------------
    def run(self) -> List[dict]:
        async def _collect_all():
            out = []
            async for batch in self.stream_async():
                out.extend(batch)
            return out
        try:
            asyncio.get_running_loop()
            raise RuntimeError("InfoMarket2.run() chiamato dentro un event loop: usa await stream_async()")
        except RuntimeError:
            return asyncio.run(_collect_all())



    # === Currency Store / Watch ===

    # --- exporter QUICK solo per short ranges (REST leggerezze) ----------------
    def _export_quick_ranges(self, kr_list: list[str], ranges: tuple[str, ...]) -> list[dict]:
        """
        Export leggero usato dal watcher: aggiorna SOLO NOW/1M/5M/15M/30M/1H.
        **PATCH**: correzione chiamata REST Ticker (parametri nel dict).
        """
        out: list[dict] = []
        ranges = tuple(r.upper() for r in ranges)
        ohlc_map = {"1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60}

        # FIX: passiamo i params come dict, non con 'pair='
        try:
            pairs_arg = ",".join(kr_list)
            tk = self._rest._public("Ticker", {"pair": pairs_arg}).get("result", {})
        except Exception:
            tk = {}

        for kr in kr_list:
            human = self._kr_to_human.get(kr, kr)
            item = {"pair": human, "kr_pair": kr, "info": {}}

            if "NOW" in ranges:
                t = tk.get(kr) or {}
                last = float((t.get("c") or [None, None])[0] or 0.0)
                bid  = float((t.get("b") or [None, None])[0] or 0.0)
                ask  = float((t.get("a") or [None, None])[0] or 0.0)
                high = float((t.get("h") or [None, None])[0] or 0.0)
                low  = float((t.get("l") or [None, None])[0] or 0.0)
                vol  = float((t.get("v") or [None, None])[0] or 0.0)
                item["info"]["NOW"] = {
                    "current_price": last, "last": last, "bid": bid, "ask": ask,
                    "high": high, "low": low, "vol": vol
                }

            for rng, minutes in ohlc_map.items():
                if rng not in ranges:
                    continue
                try:
                    ohlc = self._rest._public("OHLC", {"pair": kr, "interval": minutes}).get("result", {}).get(kr) or []
                except Exception:
                    ohlc = []
                if ohlc:
                    ts, open_, high, low, close, vwap, vol, cnt = ohlc[-1][:8]
                    item["info"][rng] = {
                        "open": float(open_), "high": float(high), "low": float(low),
                        "close": float(close), "vwap": float(vwap) if vwap not in (None, "") else None,
                        "volume": float(vol), "ts": int(ts)
                    }
            out.append(item)

        return out






    async def build_currency_store(self,
                                store_dir: str = "currencyStorePerDay",
                                file_prefix: str = "currencyOfDay",
                                ranges: Tuple[str, ...] = ("NOW","5M","30M","1H","4H","24H","48H"),
                                max_total: Optional[int] = None,
                                overwrite: bool = True) -> str:
        """
        Warm-up: prende tutte le currency con le *stesse* funzioni della classe e salva
        un unico JSON giornaliero (stesso formato dei tuoi input_*.json).

        - store_dir: cartella base (viene creata se non esiste)
        - file_prefix: prefisso del file
        - ranges: suggerimento NON vincolante per ridurre le finestre durante il fetch
                (se la tua pipeline non supporta _ranges_hint, viene ignorato, quindi compatibile)
        - max_total: opzionale per limitare le coppie (altrimenti usa self.total)
        - overwrite: se False e il file del giorno esiste, aggiunge un suffisso incrementale
        """
        os.makedirs(store_dir, exist_ok=True)
        day = _date_str()
        fname = f"{file_prefix}_{day}.json"
        path  = os.path.join(store_dir, fname)
        if (not overwrite) and os.path.exists(path):
            # evita di sovrascrivere: aggiungi indice
            i = 1
            while True:
                alt = os.path.join(store_dir, f"{file_prefix}_{day}_{i}.json")
                if not os.path.exists(alt):
                    path = alt
                    break
                i += 1

        data = await _gather_full_snapshot(self, ranges_hint=ranges, max_total=max_total)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    # --- helper: trova l’ultimo file nel currencyStore ---
    def _latest_store_file(
        self,
        store_dir: str,
        file_prefix: str = "currencyOfDay",
    ) -> Optional[str]:
        try:
            os.makedirs(store_dir, exist_ok=True)
            files = [
                f for f in os.listdir(store_dir)
                if f.startswith(file_prefix) and f.endswith(".json")
            ]
            if not files:
                return None
            files.sort()  # ordinamento lessicografico: *_YYYYMMDD[_i].json
            return os.path.join(store_dir, files[-1])
        except Exception:
            return None

    # --- helper: merge selettivo dei range aggiornati nel JSON esistente ---
    def _merge_ranges_into_store_item(
        self,
        dst: dict,
        src: dict,
        ranges_to_update: Tuple[str, ...]
    ) -> None:
        # aggiorna SOLO i blocchi richiesti
        dst_info = dst.get("info") or {}
        src_info = src.get("info") or {}
        for rng in ranges_to_update:
            if rng in src_info:
                dst_info[rng] = src_info.get(rng)
        dst["info"] = dst_info

        # aggiorna alcuni campi rapidi utili (NOW price, liquidity, portfolio available)
        if "NOW" in src_info:
            now_blk = src_info.get("NOW") or {}
            dst_now = dst_info.get("NOW") or {}
            for k in ("current_price", "last", "bid", "ask"):
                if now_blk.get(k) is not None:
                    dst_now[k] = now_blk[k]
            dst_info["NOW"] = dst_now
            dst["info"] = dst_info

        if "liquidity" in src:
            dst["liquidity"] = src["liquidity"]
        if "pair_limits" in src:
            dst["pair_limits"] = src["pair_limits"]

        # portfolio.available (se l’exporter te li ha messi)
        pf_src = (src.get("portfolio") or {}).get("available") or {}
        if pf_src:
            pf_dst = dst.get("portfolio") or {}
            av_dst = pf_dst.get("available") or {}
            for k in ("base", "quote"):
                if pf_src.get(k) is not None:
                    av_dst[k] = pf_src[k]
            pf_dst["available"] = av_dst
            dst["portfolio"] = pf_dst
   # --- NUOVO watch: aggiorna il file e YIELD batch come stream_async ---
    # --- NUOVO watch: aggiorna store e YIELD batch (senza exporter completo) ---
    # --- dentro class InfoMarket2 ------------------------------------------------
    async def watch_currency_store(
        self,
        *,
        store_dir: str,
        file_prefix: str = "currencyOfDay",
        short_ranges: tuple[str, ...] = ("NOW", "5M", "15M", "30M", "1H"),
        full_ranges: tuple[str, ...] | None = (
            "NOW", "5M", "15M", "30M", "1H", "4H", "24H", "48H", "7D", "30D"
        ),
        full_every_hours: int = 4,
        poll_seconds: float = 2.0,
        batch_size: int | None = None,
        stop_after: int | None = None,
        full_at_start: bool = False,
    ):
        """
        Watcher "veloce": ad ogni giro aggiorna SOLO i prossimi `batch_size` pair
        (usando `short_ranges`) e YIELD-a SUBITO quel mini-batch.
        Ogni `full_every_hours` esegue un giro con `full_ranges`, ma sempre a mini-batch.
        Il file currencyOfDay_YYYYMMDD.json viene aggiornato ad ogni giro.

        Non modifica altri metodi, non chiama export completi: usa _export_block
        su una lista ristretta di kr_pair.
        """
        import os, json, time, asyncio
        from datetime import datetime, timedelta

        # --------- util locali (no nuove dipendenze) ----------
        def _today_tag() -> str:
            return time.strftime("%Y%m%d")

        def _latest_store_path() -> str | None:
            """Trova l'ultimo file currencyOfDay_YYYYMMDD.json in store_dir."""
            if not os.path.isdir(store_dir):
                return None
            pref = f"{file_prefix}_"
            cands = [f for f in os.listdir(store_dir)
                    if f.startswith(pref) and f.endswith(".json")]
            if not cands:
                return None
            cands.sort()
            return os.path.join(store_dir, cands[-1])

        def _ensure_today_store() -> str:
            """Assicura un file per oggi; se non esiste, costruiscilo con tutte le finestre."""
            os.makedirs(store_dir, exist_ok=True)
            path = _latest_store_path()
            today = _today_tag()
            want = os.path.join(store_dir, f"{file_prefix}_{today}.json")
            if path and os.path.basename(path) == os.path.basename(want):
                return want  # già presente quello di oggi
            # serve creare lo snapshot iniziale (full)
            ranges_all = full_ranges or short_ranges
            # usa la routine già esistente senza cambiare firma
            # NB: è sync -> la mando in thread per non bloccare l'event loop
            def _build():
                return self.build_currency_store(
                    store_dir=store_dir,
                    file_prefix=file_prefix,
                    ranges=ranges_all,
                )
            path = asyncio.run(asyncio.to_thread(_build))  # safe: build usa solo I/O
            return path  # dovrebbe essere il file appena creato

        def _load_store(path: str) -> list[dict]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []

        def _save_store(path: str, items: list[dict]):
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

        # --------- setup store (snapshot) ----------
        path = _latest_store_path()
        if not path or full_at_start:
            # creo/aggiorno lo snapshot (full) una volta
            if self.verbose:
                print("[watch] init snapshot …")
            def _build_once():
                return self.build_currency_store(
                    store_dir=store_dir,
                    file_prefix=file_prefix,
                    ranges=(full_ranges or short_ranges),
                )
            path = await asyncio.to_thread(_build_once)
            if self.verbose:
                print("[watch] snapshot creato:", path)

        if not path:
            # ultima difesa
            os.makedirs(store_dir, exist_ok=True)
            path = os.path.join(store_dir, f"{file_prefix}_{_today_tag()}.json")
            _save_store(path, [])

        store_items: list[dict] = _load_store(path)

        # prendi max self.total
        if isinstance(self.total, int) and self.total > 0:
            store_items = store_items[: self.total]

        # estrai human pairs presenti nello store
        pairs_in_store: list[str] = []
        for it in store_items:
            p = it.get("pair") or f"{it.get('base')}/{it.get('quote')}"
            if isinstance(p, str) and "/" in p and p not in pairs_in_store:
                pairs_in_store.append(p)

        if not pairs_in_store:
            if self.verbose:
                print("[watch] store vuoto: nulla da aggiornare (serve build_currency_store)")
            return

        # mappa human->kr SOLO se esiste nel dict (evito self._norm_pair)
        human_to_kr = getattr(self, "_human_to_kr", {}) or {}
        kr_list_all: list[str] = []
        for human in pairs_in_store:
            kr = human_to_kr.get(human)
            if not kr:
                if self.verbose:
                    print(f"[watch] skip {human} (manca mapping kr)")
                continue
            kr_list_all.append(kr)

        if not kr_list_all:
            if self.verbose:
                print("[watch] nessun kr_pair valido — esco")
            return

        # set parametri
        bs = batch_size or self.per_run or 10
        next_full_at = datetime.utcnow()
        cursor = 0
        loops_done = 0

        # funzione di merge "mirato": aggiorno solo i blocchi che ho richiesto
        def _merge_updated(old: list[dict], new_batch: list[dict], ranges: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
            by_pair = { (x.get("pair") or f"{x.get('base')}/{x.get('quote')}"): x for x in old }
            updated_objs: list[dict] = []
            for nb in new_batch:
                hp = nb.get("pair") or f"{nb.get('base')}/{nb.get('quote')}"
                ob = by_pair.get(hp, {})
                # copia shallow
                merged = dict(ob)
                # info: sovrascrivo solo i ranges richiesti
                info_old = dict((ob.get("info") or {}))
                info_new = dict((nb.get("info") or {}))
                for r in ranges:
                    if r in info_new:
                        info_old[r] = info_new[r]
                merged["info"] = info_old
                # pair_limits: la mantengo se manca
                if "pair_limits" not in merged or not merged["pair_limits"]:
                    merged["pair_limits"] = nb.get("pair_limits")
                # portfolio/open_orders si possono aggiornare (sono "freschi")
                if nb.get("portfolio"): merged["portfolio"] = nb["portfolio"]
                if nb.get("open_orders"): merged["open_orders"] = nb["open_orders"]
                by_pair[hp] = merged
                updated_objs.append(merged)
            # riconverti a lista preservando l’ordine originale
            out = []
            for it in old:
                k = it.get("pair") or f"{it.get('base')}/{it.get('quote')}"
                out.append(by_pair.get(k, it))
            return out, updated_objs

        # --- loop principale: YIELD immediato di mini-batch ---
        while True:
            now = datetime.utcnow()
            do_full = now >= next_full_at
            ranges_use = (full_ranges or short_ranges) if do_full else short_ranges
            if do_full:
                next_full_at = now + timedelta(hours=full_every_hours)
                if self.verbose:
                    print(f"[watch] full refresh window @ {now.isoformat()} for {ranges_use}")

            # seleziona il sotto-insieme (rotazione circolare)
            batch_kr = []
            sel_idx = []
            for i in range(bs):
                idx = (cursor + i) % len(kr_list_all)
                batch_kr.append(kr_list_all[idx])
                sel_idx.append(idx)
            cursor = (cursor + bs) % len(kr_list_all)

            # chiama la tua routine "pesante" MA solo sui kr scelti
            async def _call_export():
                # _export_block è sync → sposto in thread
                def _do():
                    return self._export_block(batch_kr)  # NON tocco la firma
                return asyncio.run(asyncio.to_thread(_do))
            try:
                new_items = await _call_export()
            except Exception as e:
                if self.verbose:
                    print(f"[watch] errore export su {len(batch_kr)} pair: {e}")
                new_items = []

            # MERGE nello store e salva
            store_items, updated_objs = _merge_updated(store_items, new_items, ranges_use)
            _save_store(path, store_items)

            # YIELD SUBITO il mini-batch aggiornato
            yield updated_objs

            loops_done += 1
            if stop_after is not None and loops_done >= int(stop_after):
                break
            await asyncio.sleep(max(0.1, float(poll_seconds)))
