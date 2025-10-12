# bot3.py
import time, os, json, math, threading, random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

# # Import delle classi locali
from Class.InfoMarket import InfoMarket
from Class.KrakenPortfolio import KrakenPortfolio
# from Class.StrategyEngine import StrategyEngine
# from Class.ChatGpt import TradeAction, CurrencyAnalyzerForYourPayloadV3
# from Class.Aiensemble import run_ai
# from Class.KrakenOrderRunner import KrakenOrderRunner
# from Pipeline import TradePipeline, PipelineConfig, StageConfig

from Class.OpenOrder import JsonObjectIO, Currency

def _asdict(x):
    return asdict(x) if is_dataclass(x) else x

def _nonnull(x, y):
    """Preferisci y se non None/empty, altrimenti x."""
    return y if (y is not None and y != {}) else x

def _canon_key(obj: Dict[str, Any]) -> Optional[str]:
    """
    Chiave di matching:
    1) kr_pair
    2) pair
    3) open_orders[0].kr_pair
    4) open_orders[0].pair
    Normalizzata in UPPER.
    """
    for path in (
        ("kr_pair",),
        ("pair",),
        ("open_orders", 0, "kr_pair"),
        ("open_orders", 0, "pair"),
    ):
        cur = obj
        try:
            for p in path:
                cur = cur[p]
            if isinstance(cur, str) and cur.strip():
                return cur.strip().upper()
        except Exception:
            pass
    return None

def _count_fields(d: Dict[str, Any]) -> int:
    return sum(1 for v in (d or {}).values() if v not in (None, {}, []))

def _dedupe(items: List[Dict[str, Any]], keyfunc) -> List[Dict[str, Any]]:
    seen = {}
    for it in items or []:
        k = keyfunc(it)
        if k is None:
            # se non ho chiave, mantieni l'ordine e tieni il primo "shape" diverso
            k = json.dumps(it, sort_keys=True, default=str)
        seen[k] = it
    return list(seen.values())

def _merge_info(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for rng, val in (b or {}).items():
        out[rng] = val  # b sovrascrive stesso range
    return out

def _merge_portfolio(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a, b = a or {}, b or {}

    # row: scegli quella più “piena”
    row_a, row_b = a.get("row") or {}, b.get("row") or {}
    row = row_b if _count_fields(row_b) >= _count_fields(row_a) else row_a

    # trades: dedup per ordertxid / trade_id
    def k_trade(t):
        return t.get("ordertxid") or t.get("trade_id")
    trades = _dedupe((a.get("trades") or []) + (b.get("trades") or []), k_trade)

    # ledgers: dedup per refid, fallback tupla stabile
    def k_ledger(l):
        return l.get("refid") or (l.get("time"), l.get("amount"), l.get("asset"))
    ledgers = _dedupe((a.get("ledgers") or []) + (b.get("ledgers") or []), k_ledger)

    # available: merge per chiave
    av_a, av_b = a.get("available") or {}, b.get("available") or {}
    available = {
        "base": _nonnull(av_a.get("base"), av_b.get("base")),
        "quote": _nonnull(av_a.get("quote"), av_b.get("quote")),
    }

    return {"row": row, "trades": trades, "ledgers": ledgers, "available": available}

def _merge_open_orders(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def k(o):
        return (
            (o.get("kr_pair") or o.get("pair") or "").upper(),
            o.get("type"), o.get("ordertype"),
            o.get("price"), o.get("price2"), o.get("vol_rem"),
        )
    return _dedupe((a or []) + (b or []), k)

def _merge_one(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # campi semplici
    for k in ("base", "quote", "pair", "kr_pair"):
        out[k] = _nonnull(a.get(k), b.get(k))

    # info / pair_limits / open_orders / portfolio
    out["info"] = _merge_info(a.get("info"), b.get("info"))
    out["pair_limits"] = _nonnull(a.get("pair_limits"), b.get("pair_limits"))
    out["open_orders"] = _merge_open_orders(a.get("open_orders"), b.get("open_orders"))
    out["portfolio"] = _merge_portfolio(a.get("portfolio"), b.get("portfolio"))

    # copia eventuali altri campi sconosciuti preservando quelli di b
    known = {"base","quote","pair","kr_pair","info","pair_limits","open_orders","portfolio"}
    extra = {k:v for k,v in a.items() if k not in known}
    extra.update({k:v for k,v in b.items() if k not in known})
    out.update(extra)

    return out

def merge_currency_lists(list_a: List[Any], list_b: List[Any]) -> List[Dict[str, Any]]:
    """
    Fonde due liste. Matching su kr_pair/pair (con fallback da open_orders).
    In conflitto vince la seconda lista (b) per i campi non-null.
    Ritorna una lista di dict.
    """
    # normalizza a dict
    la = [_asdict(x) for x in (list_a or [])]
    lb = [_asdict(x) for x in (list_b or [])]

    bucket: Dict[str, Dict[str, Any]] = {}

    # prima carica tutti gli A
    for a in la:
        k = _canon_key(a)
        if not k:
            # se proprio non c'è chiave, crea una "synthetic" unica
            k = f"__NOKEY__:{id(a)}"
        bucket[k] = a

    # poi unisci B sopra A
    for b in lb:
        k = _canon_key(b)
        if not k:
            k = f"__NOKEY__:{id(b)}"
        if k in bucket:
            bucket[k] = _merge_one(bucket[k], b)
        else:
            bucket[k] = b

    return list(bucket.values())


def fnum(x, fmt="{:.6f}", na="n/a"):
    """Formatter sicuro per None/NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return na
    return fmt.format(x)


class Bot:
    """
    Semplice orchestratore che esegue:
      - InfoMarket (più volte con timeframe diversi)
      - KrakenPortfolio (stato portafoglio)
    e salva i risultati negli attributi della classe.
    """
    def __init__(self, pair: str | None = None, quote: str = "EUR"):
        self.pair = pair or os.environ.get("IM_PAIR", "BTC/EUR")
        self.quote = quote

        # Attributi popolati dalle run
        self.info_market_runs: list = []          # lista di dict: {'time':..., 'max':..., 'result': <return getTotalInfo>}
        self.portfolio_rows: list | None = None   # lista di righe portafoglio
        self.portfolio_total: float | None = None
        self.portfolio_trades: list | None = None
        self.portfolio_ledgers: dict | None = None

        # opzionale: cache API e AssetPairs (usate da export_currencies se disponibili)
        self._api = None
        self._pairs_info = None

    # --- ex run_info_market ---
    def run_info_market(self, time: str, max: int = 10):
        """
        Esegue il main di InfoMarket come nell'esempio:
          im = InfoMarket(pair="BTC/EUR", verbose=False)
          tot = im.getTotalInfo(time, quote="EUR", max_pairs=max)
        Salva il risultato in self.info_market_runs e lo ritorna.
        """
        print(f"\n========== InfoMarket: screening su {self.pair} ==========")
        im = InfoMarket(verbose=False, public_qps=1.8)
        ranges = ["NOW","1H","1M","5M","30M", "4H","24H","48H","7D","30D","90D","1Y"]
        # ranges = ["NOW","24H","30D","1Y"]

        # try:
        #     tot = im.getTotalInfo(time, quote=self.quote, max_pairs=max)
        # except TypeError:
        #     tot = im.getTotalInfo(time)
        tot = im.export_currencies_ws(
            pairs=None, quote="EUR", ranges=ranges, max_pairs=max,
            with_liquidity=False,   # lascialo False per velocità
            with_mtf=True, with_or=True,
            sleep_per_call=0.03, sleep_per_pair=0.01
        )

        # salva in attributo elenco esecuzioni
        self.info_market_runs.append({
            "time": time,
            "max": max,
            "result": tot,
        })
        return tot

    # --- ex run_kraken_portfolio ---
    def run_kraken_portfolio(self):
        """
        Esegue il report portafoglio da KrakenPortfolio, stampa e
        salva in attributi: rows/total/trades/ledgers.
        """
        print("\n========== KrakenPortfolio: portafoglio ==========")
        kp = KrakenPortfolio()
        rows, total, trades, ledgers = kp.portfolio_view()

        # salva negli attributi
        self.portfolio_rows = rows
        self.portfolio_total = total
        self.portfolio_trades = trades
        self.portfolio_ledgers = ledgers
        self.portfolioIn = kp.investable_eur()

        return rows, total, trades, ledgers

    # ---------- helper interni opzionali (non richiesti dal main) ----------
    def _get_api(self):
        """Inizializza l'API Kraken solo se sono presenti credenziali ENV; altrimenti None."""
        if self._api is not None:
            return self._api
        api_key = os.environ.get("KRAKEN_API_KEY") or os.environ.get("KRAKEN_KEY")
        api_secret = os.environ.get("KRAKEN_API_SECRET") or os.environ.get("KRAKEN_SECRET")
        if not api_key or not api_secret:
            self._api = None
            return None
        try:
            import krakenex
            self._api = krakenex.API(key=api_key, secret=api_secret)
        except Exception:
            self._api = None
        return self._api

    def _ensure_pairs_info(self, api):
        """Carica AssetPairs e cache in self._pairs_info (se possibile)."""
        if self._pairs_info is not None or not api:
            return
        try:
            resp = api.query_public("AssetPairs")
            if not resp.get("error"):
                self._pairs_info = resp.get("result", {})
        except Exception:
            self._pairs_info = None

    @staticmethod
    def _fiat_to_kr(f):
        return {'EUR': 'ZEUR', 'USD': 'ZUSD', 'GBP': 'ZGBP', 'USDT': 'USDT', 'USDC': 'USDC'}.get(f, f)

    # ----------------------- export_currencies -----------------------
    def export_currencies(self):
        """
        Ritorna una lista di currency aggregate:
        [
          {
            'base': 'BTC',
            'quote': 'EUR',
            'pair': 'BTC/EUR',
            'kr_pair': 'XXBTZEUR',
            'info': { '24H': {...}, '48H': {...}, ... },
            'portfolio': {
                'row': {...} or None,
                'trades': [ ... ],   # trades associati al base asset
                'ledgers': [ ... ],  # ledger associati al base asset
                'available': {'base': <qty>, 'quote': <qty>}  # <-- NEW se API disponibile
            },
            'open_orders': [...],   # <-- NEW se API disponibile
            'pair_limits': {'lot_decimals': int, 'ordermin': float|None}  # <-- NEW se API disponibile
          },
          ...
        ]
        """
        # 1) indicizza il portafoglio per codice (es. 'XXBT' -> row)
        portfolio_by_code = {}
        if self.portfolio_rows:
            for r in self.portfolio_rows:
                portfolio_by_code[r.get('code')] = r

        # 2) raccogli info market per ogni base/quote
        info_by_pair = {}
        krpair_by_pair = {}  # (base,quote) -> kr_pair (es. 'XXBTZEUR')
        time_by_pair = {}    # (base,quote) -> dict timeframe -> info dict

        for run in self.info_market_runs:
            timeframe = run.get("time")
            res = run.get("result")
            entries = None
            if isinstance(res, dict):
                if isinstance(res.get('results'), list):
                    entries = res['results']
                elif isinstance(res.get('currencies'), list):
                    entries = res['currencies']
            if not entries:
                continue

            for e in entries:
                pair_h = e.get('pair') or e.get('pair_human')
                krp    = e.get('kr_pair')
                base   = e.get('base')
                quote  = e.get('quote')
                if not (base and quote) and pair_h and isinstance(pair_h, str) and '/' in pair_h:
                    base, quote = pair_h.split('/', 1)

                key = (base, quote)
                info_by_pair.setdefault(key, {})
                time_by_pair.setdefault(key, {})
                if krp:
                    krpair_by_pair[key] = krp
                time_by_pair[key][timeframe] = e

        # 3) helper per risalire al base_code (tipo 'XXBT') da kr_pair
        def base_code_from_krpair(krp, quote):
            if not isinstance(krp, str):
                return None
            qk = self._fiat_to_kr(quote)
            if qk and krp.endswith(qk):
                return krp[:-len(qk)]
            for code in portfolio_by_code.keys():
                if krp.startswith(code):
                    return code
            return None

        # 4) indicizza trades per base_code
        trades_by_code = {}
        if isinstance(self.portfolio_trades, list):
            for t in self.portfolio_trades:
                pair_raw = t.get('pair') or ''
                pair_norm = pair_raw.replace('/', '')
                matched_code = None
                for code in portfolio_by_code.keys():
                    if code and code in pair_norm:
                        matched_code = code
                        break
                if not matched_code and portfolio_by_code:
                    for code, row in portfolio_by_code.items():
                        alt = row.get('asset')  # 'XBT'
                        if isinstance(alt, str) and alt in pair_norm:
                            matched_code = code
                            break
                if matched_code:
                    trades_by_code.setdefault(matched_code, []).append(t)

        # 5) indicizza ledgers per asset code (es. 'XXBT')
        ledgers_by_code = {}
        if isinstance(self.portfolio_ledgers, dict):
            for _lid, row in self.portfolio_ledgers.items():
                asset_code = row.get('asset')  # es. 'XXBT' | 'ZEUR'
                if asset_code:
                    ledgers_by_code.setdefault(asset_code, []).append(row)

        # ====== (NEW) dati runtime da API Kraken (facoltativi) ======
        balances = {}
        open_orders_raw = {}
        available_by_asset = {}
        self._get_api()  # prova ad inizializzare se possibile
        api = self._api
        if api:
            # balances
            try:
                b = api.query_private("Balance")
                if not b.get("error"):
                    balances = {a: float(v) for a, v in b["result"].items()}
            except Exception:
                balances = {}
            # open orders
            try:
                oo = api.query_private("OpenOrders")
                if not oo.get("error"):
                    open_orders_raw = oo["result"].get("open", {})
            except Exception:
                open_orders_raw = {}
            # pairs info (per lot_decimals / ordermin)
            self._ensure_pairs_info(api)

        # reverse map per normalizzare qualsiasi rappresentazione del pair in kr_pair canonico
        rev_by_alt = {}
        rev_by_ws = {}
        if self._pairs_info:
            for name, row in self._pairs_info.items():
                alt = row.get("altname")
                ws  = (row.get("wsname") or "").replace("/", "")
                if alt:
                    rev_by_alt[alt] = name
                if ws:
                    rev_by_ws[ws] = name

        def to_kr_pair(s: str | None) -> str | None:
            if not s:
                return None
            s0 = s.replace("/", "")
            if self._pairs_info:
                if s0 in self._pairs_info:
                    return s0
                if s0 in rev_by_alt:
                    return rev_by_alt[s0]
                if s0 in rev_by_ws:
                    return rev_by_ws[s0]
            return s0  # best effort

        # calcolo held per asset dai soli ordini aperti
        held_by_asset = {}
        simple_open_orders = []
        if open_orders_raw:
            for _, od in open_orders_raw.items():
                d = od.get("descr", {})
                raw_pair = d.get("pair") or ""              # può essere "XBT/EUR" o "XXBTZEUR"
                kr_pair  = to_kr_pair(raw_pair)             # normalizzo a kr_pair canonico
                typ = d.get("type")
                ordertype = d.get("ordertype")
                try:
                    vol = float(od.get("vol", 0.0))
                    vol_exec = float(od.get("vol_exec", 0.0) or 0.0)
                except Exception:
                    vol = 0.0; vol_exec = 0.0
                vol_rem = max(0.0, vol - vol_exec)
                if vol_rem <= 0:
                    continue
                # prezzi (se presenti)
                price = None; price2 = None
                try:
                    price = float(d.get("price")) if d.get("price") else None
                    price2 = float(d.get("price2")) if d.get("price2") else None
                except Exception:
                    pass

                base_code = quote_code = None
                if self._pairs_info and kr_pair in self._pairs_info:
                    row = self._pairs_info[kr_pair]
                    base_code = row.get("base")
                    quote_code = row.get("quote")

                simple_open_orders.append({
                    "kr_pair": kr_pair,
                    "pair": kr_pair,                 # per retro-compatibilità
                    "type": typ,
                    "ordertype": ordertype,
                    "price": price,
                    "price2": price2,
                    "vol_rem": vol_rem,
                    "base": base_code,
                    "quote": quote_code
                })

                if typ == "sell" and base_code:
                    held_by_asset[base_code] = held_by_asset.get(base_code, 0.0) + vol_rem
                elif typ == "buy" and quote_code:
                    px = price or price2
                    if px:
                        held_by_asset[quote_code] = held_by_asset.get(quote_code, 0.0) + (px * vol_rem)

        # available = balance - held (>=0)
        if balances:
            for code, bal in balances.items():
                available_by_asset[code] = max(0.0, bal - held_by_asset.get(code, 0.0))

        # 6) costruisci l’output unendo info + portafoglio
        out = []

        for (base, quote), tf_map in time_by_pair.items():
            krp = krpair_by_pair.get((base, quote))
            base_code = base_code_from_krpair(krp, quote) if krp else None
            port_row = portfolio_by_code.get(base_code)
            # trades e ledgers coerenti col codice base
            trades = trades_by_code.get(base_code, [])
            ledgs  = ledgers_by_code.get(base_code, [])

            # NEW: available per base/quote (se conosciuti)
            avail_base = available_by_asset.get(base_code) if base_code else None
            avail_quote = available_by_asset.get(self._fiat_to_kr(quote)) if quote else None

            # NEW: open orders pertinenti a questo pair/base
            oo_this = []
            if simple_open_orders:
                if krp:
                    oo_this = [o for o in simple_open_orders if o["kr_pair"] == krp]
                elif base_code:
                    oo_this = [o for o in simple_open_orders if o.get("base") == base_code]

            # NEW: limiti pair
            pair_limits = None
            if self._pairs_info and krp and krp in self._pairs_info:
                pr = self._pairs_info[krp]
                lot_dec = int(pr.get("lot_decimals", 8))
                try:
                    ordermin = float(pr.get("ordermin")) if pr.get("ordermin") else None
                except Exception:
                    ordermin = None
                pair_limits = {"lot_decimals": lot_dec, "ordermin": ordermin}

            out.append({
                'base': base,
                'quote': quote,
                'pair': f"{base}/{quote}",
                'kr_pair': krp,
                'info': tf_map,  # dict: timeframe -> info dict
                'pair_limits': pair_limits,   # NEW
                'open_orders': oo_this,       # NEW
                'portfolio': {
                    'row': port_row,
                    'trades': trades,
                    'ledgers': ledgs,
                    'available': {'base': avail_base, 'quote': avail_quote}  # NEW
                }
            })

        # 7) aggiungi eventuali asset di portafoglio che non compaiono in info_market
        covered_codes = set(
            base_code_from_krpair(krpair_by_pair.get((b, q)), q)
            for (b, q) in time_by_pair.keys()
        )
        for code, row in portfolio_by_code.items():
            if code not in covered_codes:
                base_h = row.get('asset')  # es. 'XBT', 'ZEUR'
                human = base_h
                if human == 'XBT':
                    human = 'BTC'
                elif isinstance(human, str) and human.startswith('X') and len(human) > 3:
                    human = human[1:]  # XETH -> ETH

                avail_base = available_by_asset.get(code) if available_by_asset else None
                oo_this = [o for o in simple_open_orders if o.get("base") == code] if simple_open_orders else []

                out.append({
                    'base': human,
                    'quote': None,
                    'pair': None,
                    'kr_pair': None,
                    'info': {},
                    'pair_limits': None,
                    'open_orders': oo_this,
                    'portfolio': {
                        'row': row,
                        'trades': trades_by_code.get(code, []),
                        'ledgers': ledgers_by_code.get(code, []),
                        'available': {'base': avail_base, 'quote': None}
                    }
                })

        return out


    def save_input_run(self, records, folder: str = "input", filename_prefix: str = "input") -> str:
        """
        Salva 'records' (lista/iterabile di dict/oggetti) in ./<folder>/<prefix>_YYYYmmdd_HHMMSS.json
        La cartella è creata accanto al file di questa classe. Ritorna il path del file scritto.
        """
        # 1) normalizza in una lista
        if records is None:
            items = []
        elif isinstance(records, (list, tuple)):
            items = list(records)
        else:
            items = [records]

        # 2) serializza in oggetti JSON-compatibili
        out = []
        for x in items:
            if isinstance(x, (dict, list, str, int, float, bool)) or x is None:
                out.append(x)
            elif hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
                out.append(x.to_dict())
            else:
                # tenta varie conversioni ragionevoli
                try:
                    from dataclasses import asdict
                    out.append(asdict(x))
                except Exception:
                    try:
                        out.append(dict(x))
                    except Exception:
                        try:
                            out.append(json.loads(x) if isinstance(x, str) else str(x))
                        except Exception:
                            out.append(str(x))

        # 3) path base accanto al file della classe
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        out_dir = os.path.join(base_dir, folder)
        os.makedirs(out_dir, exist_ok=True)

        # 4) nome file con timestamp
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = os.path.join(out_dir, f"{filename_prefix}_{stamp}.json")

        # 5) scrittura JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, separators=(",", ":"), default=str)

        if getattr(self, "verbose", False):
            print(f"[save_input_run] Saved {len(out)} items → {path}")

        return path




# === MAIN ===
if __name__ == "__main__":
    bot = Bot(pair=os.environ.get("IM_PAIR", "BTC/EUR"), quote="EUR")

    arr1 = bot.run_info_market("ALL", 5)

    # Poi stato portafoglio Kraken
    rwest = bot.run_kraken_portfolio()
    currencies = bot.export_currencies()
    merged = merge_currency_lists(arr1, currencies)
    saved_path = bot.save_input_run(merged)

    for curre in currencies:
        print(curre)

    analyzer = CurrencyAnalyzerForYourPayloadV3(
        model="gpt-5",
        request_timeout_s=300,
        reserve_eur=50,
        risk_level=5,
        prefer_limit=True,
        or_min_atr_mult=0.5,
        spread_max_pct=0.5,
        min_volume_label="Medium",
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        enable_shorts=True,
        )
    actions = analyzer.analyze(merged, rischio=6)
    for act in actions:
        print(asdict(act))

    result = run_ai(
        currencies=currencies,
        actions=actions,            # puoi omettere se vuoi solo generare azioni AI
        inputStorico="storico_input",
        outputStorico="storico_output",
        budget_eur=400,            # se None lo calcola dal portafoglio
        per_trade_cap_eur=30      # es. max 100€ per singolo trade
    )

    print(result["scores"])
    print(result["actions_ai"][:2])  # prime due azioni generate

    runner = KrakenOrderRunner(pair_map={"BTC/EUR": "XXBTZEUR", "AEVO/EUR": "AEVO/EUR"})
    bodies = runner.build_bodies(actions, validate=False)
    test = runner.execute_bodies(bodies, timeout=0.5)

    print('bodies')

    for acts in bodies:
        print(acts)

    print('teswt')

    for tests in test:
        print(tests)

