import os
import math
import time
import random
import threading
from time import monotonic
from collections import defaultdict

import krakenex


# ---------------- utilities: throttling, retry, cache ----------------
class RateLimiter:
    """Impone un intervallo minimo (token-bucket semplificato)."""
    def __init__(self, min_interval=1.2):
        self.min_interval = float(min_interval)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self, weight=1.0):
        with self._lock:
            now = monotonic()
            need = self.min_interval * float(weight)
            delta = now - self._last
            if delta < need:
                time.sleep(need - delta)
            self._last = monotonic()


class TTLCache:
    """Cache in-memory con TTL per chiave."""
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def get(self, key, ttl_sec):
        with self._lock:
            v = self._d.get(key)
            if not v:
                return None
            data, ts = v
            if monotonic() - ts <= ttl_sec:
                return data
            return None

    def set(self, key, value):
        with self._lock:
            self._d[key] = (value, monotonic())


class KrakenPortfolio:
    """
    Wrapper portfolio con:
      - rate limit centralizzato (parametrico)
      - retry/backoff su EAPI:Rate limit exceeded
      - cache per balances, trades history e ledgers
      - diagnostica opzionale (per evitare chiamate pesanti ogni run)
    """

    def __init__(
        self,
        key=None,
        secret=None,
        rate_limit_sec=None,
        balances_ttl=30,
        trades_ttl=50,
        ledgers_ttl=50,
    ):
        self.k = krakenex.API(
            key=key or os.environ.get("KRAKEN_KEY"),
            secret=secret or os.environ.get("KRAKEN_SECRET"),
        )

        # Parametri da argomenti o da .env (con default sensati)
        rl = rate_limit_sec or float(os.environ.get("KRAKEN_RATE_LIMIT_SEC", "1.5"))
        self._rl = RateLimiter(min_interval=rl)

        self._balances_ttl = int(os.environ.get("KRAKEN_BALANCES_TTL", balances_ttl))
        self._trades_ttl = int(os.environ.get("KRAKEN_TRADES_TTL", trades_ttl))
        self._ledgers_ttl = int(os.environ.get("KRAKEN_LEDGERS_TTL", ledgers_ttl))

        self._cache = TTLCache()
        self._ticker_cache = {}

        # Metadati (pubblici → peso 1)
        self.assets_info = self._kraken_call("Assets", private=False, weight=1.0)["result"]
        self.pairs_info = self._kraken_call("AssetPairs", private=False, weight=1.0)["result"]

        # Mappe name/alt per i pair
        self.pairs_by_name = self.pairs_info
        self.pairs_by_alt = {}
        for name, p in self.pairs_info.items():
            alt = (p.get("altname") or "").replace("/", "")
            self.pairs_by_alt[alt] = {"name": name, "base": p["base"], "quote": p["quote"]}

    # ---------------- internal: call wrapper con retry/backoff ----------------
    # ---------------- internal: call wrapper con retry/backoff ----------------
    def _is_transient_err(self, msg: str) -> bool:
        if not msg:
            return False
        m = msg.lower()
        transient_markers = [
            "rate limit exceeded", "too many requests",
            "eservice:unavailable", "eapi:timeout", "timeout",
            "read timed out", "connection", "temporar", "gateway", "http 52", "bad gateway",
            "ssl", "proxy", "reset by peer"
        ]
        return any(tok in m for tok in transient_markers)

    def _kraken_call(self, endpoint, data=None, private=False, weight=1.0, max_retries=10):
        """Chiamata con pacing + retry/backoff anche su timeout/52x/connessioni."""
        data = data or {}
        last_err = None
        for attempt in range(max_retries):
            self._rl.wait(weight=weight)
            try:
                resp = self.k.query_private(endpoint, data) if private else self.k.query_public(endpoint, data)
            except Exception as e:
                # eccezioni di rete/timeouts -> retry con backoff
                last_err = str(e)
                sleep_s = min(2 ** attempt, 32) + random.uniform(0.0, 0.5)
                time.sleep(sleep_s)
                continue

            # nessuna risposta o formato inatteso -> ritenta
            if not isinstance(resp, dict):
                last_err = f"invalid response type: {type(resp)}"
                time.sleep(min(2 ** attempt, 32) + random.uniform(0.0, 0.5))
                continue

            # errori Kraken
            errs = resp.get("error") or []
            if errs:
                msg = ",".join(errs)
                if self._is_transient_err(msg):
                    # backoff esponenziale + jitter
                    sleep_s = min(2 ** attempt, 32) + random.uniform(0.0, 0.5)
                    time.sleep(sleep_s)
                    continue
                # errore "duro": esci subito
                raise RuntimeError(msg)

            return resp  # OK

        # esauriti i tentativi
        raise RuntimeError(last_err or "Kraken API: troppi tentativi falliti")


    # ----------------- pair & prezzi -----------------
    def _find_pair(self, base_code: str, quote_code: str):
        for name, p in self.pairs_info.items():
            if p.get("base") == base_code and p.get("quote") == quote_code:
                return name
        return None

    def _find_eur_pair(self, base_code: str):
        return self._find_pair(base_code, "ZEUR")

    def _ticker_last(self, pair_name: str) -> float:
        if pair_name not in self._ticker_cache:
            t = self._kraken_call("Ticker", {"pair": pair_name}, private=False, weight=1.0)["result"]
            self._ticker_cache[pair_name] = float(list(t.values())[0]["c"][0])
        return self._ticker_cache[pair_name]

    def _convert_quote_to_eur(self, amount_in_quote: float, quote_code: str) -> float:
        """Converte un importo espresso nella QUOTE in EUR con i prezzi correnti."""
        if quote_code == "ZEUR":
            return amount_in_quote

        # QUOTE -> EUR diretto
        q2e = self._find_pair(quote_code, "ZEUR")
        if q2e:
            return amount_in_quote * self._ticker_last(q2e)

        # via USD
        amount_in_usd = None
        if quote_code == "ZUSD":
            amount_in_usd = amount_in_quote
        else:
            q2u = self._find_pair(quote_code, "ZUSD")
            if q2u:
                amount_in_usd = amount_in_quote * self._ticker_last(q2u)
        if amount_in_usd is not None:
            usd2eur = self._find_pair("ZUSD", "ZEUR")
            if usd2eur:
                return amount_in_usd * self._ticker_last(usd2eur)

        # via XBT
        q2x = self._find_pair(quote_code, "XXBT")
        x2e = self._find_eur_pair("XXBT")
        if q2x and x2e:
            return amount_in_quote * self._ticker_last(q2x) * self._ticker_last(x2e)

        return float("nan")

    def price_in_eur(self, asset_code: str) -> float:
        if asset_code == "ZEUR":
            return 1.0
        p = self._find_eur_pair(asset_code)
        if p:
            return self._ticker_last(p)

        p_usd = self._find_pair(asset_code, "ZUSD")
        if p_usd:
            usd2eur = self._find_pair("ZUSD", "ZEUR")
            if usd2eur:
                return self._ticker_last(p_usd) * self._ticker_last(usd2eur)

        p_xbt = self._find_pair(asset_code, "XXBT")
        xbt_eur = self._find_eur_pair("XXBT")
        if p_xbt and xbt_eur:
            return self._ticker_last(p_xbt) * self._ticker_last(xbt_eur)

        return float("nan")

    # ----------------- dati account (con cache) -----------------
    def balances(self, ttl_sec: int = None) -> dict:
        ttl = (self._balances_ttl if ttl_sec is None else ttl_sec)
        cached = self._cache.get(("balances",), ttl)
        if cached is not None:
            return cached
        bal = self._kraken_call("Balance", private=True, weight=2.5)  # <-- prima era 1.0
        data = {a: float(v) for a, v in bal["result"].items() if float(v) > 0}
        self._cache.set(("balances",), data)
        return data


    # ----------------- trades history (con cache) -----------------
    def trades_history(self, start=None, ofs=0, max_loops=40) -> list:
        cache_key = ("trades", start, ofs)
        cached = self._cache.get(cache_key, self._trades_ttl)
        if cached is not None:
            return cached

        all_trades, loops = [], 0
        payload = {"type": "all"}
        if start is not None:
            payload["start"] = start

        while loops < max_loops:
            resp = self._kraken_call(
                "TradesHistory",
                {**payload, "ofs": ofs},
                private=True,
                weight=3.0,
            )
            trades = resp["result"].get("trades", {}) or {}
            if not trades:
                break
            items = list(trades.values())
            all_trades.extend(items)
            ofs += len(items)
            loops += 1
            if len(items) < 50:
                break

        all_trades.sort(key=lambda t: t.get("time", 0.0))
        self._cache.set(cache_key, all_trades)
        return all_trades

    # ----------------- ledgers (con cache) -----------------
    def ledgers(self) -> dict:
        cached = self._cache.get(("ledgers",), self._ledgers_ttl)
        if cached is not None:
            return cached
        resp = self._kraken_call("Ledgers", private=True, weight=3.0)
        out = resp["result"].get("ledger", {}) or {}
        self._cache.set(("ledgers",), out)
        return out

    # ----------------- util per interpretare i pair nei trade -----------------
    def _pair_info_from_trade(self, trade_pair_field: str):
        if not trade_pair_field:
            return None
        key = trade_pair_field.replace("/", "")  # gestisce "XBT/EUR"
        p = self.pairs_by_name.get(key)
        if p:
            return {"name": key, "base": p["base"], "quote": p["quote"]}
        p_alt = self.pairs_by_alt.get(key)
        if p_alt:
            return p_alt
        return None

    # ----------------- cost basis da trades -----------------
    def average_costs_from_trades(self) -> dict:
        """Prezzo medio (EUR) per asset base usando SOLO i trade spot."""
        trades = self.trades_history()
        pos = {}  # asset -> {'qty': q, 'cost_eur': c}

        for t in trades:
            pi = self._pair_info_from_trade(t.get("pair"))
            if not pi:
                continue
            base, quote = pi["base"], pi["quote"]
            vol = float(t.get("vol", 0.0))
            cost = float(t.get("cost", 0.0))
            fee = float(t.get("fee", 0.0))
            typ = t.get("type")

            cost_eur = self._convert_quote_to_eur(cost + fee, quote)

            if base not in pos:
                pos[base] = {"qty": 0.0, "cost_eur": 0.0}

            if typ == "buy":
                pos[base]["qty"] += vol
                pos[base]["cost_eur"] += cost_eur
            elif typ == "sell" and pos[base]["qty"] > 0:
                avg = pos[base]["cost_eur"] / pos[base]["qty"]
                sell_q = min(vol, pos[base]["qty"])
                pos[base]["qty"] -= sell_q
                pos[base]["cost_eur"] -= avg * sell_q
                if pos[base]["qty"] < 1e-12:
                    pos[base]["qty"] = 0.0
                    pos[base]["cost_eur"] = 0.0

        out = {}
        for asset, s in pos.items():
            out[asset] = (s["cost_eur"] / s["qty"]) if s["qty"] > 0 else None
        return out

    # ----------------- cost basis da ledgers (per convert/deposit ecc.) -----------------
    def average_costs_from_ledgers(self) -> dict:
        """
        Ricostruisce il prezzo medio in EUR per asset da Ledgers:
          - raggruppa per refid; cerca EUR(-) + ASSET(+) nello stesso gruppo
          - costo = EUR_speso (abs) + fee EUR del gruppo
          - qty   = somma quantità positive dell'asset base
        """
        ldg = self.ledgers()
        by_ref = defaultdict(list)
        for lid, row in ldg.items():
            by_ref[row.get("refid")].append(row)

        pos = defaultdict(lambda: {"qty": 0.0, "cost_eur": 0.0})

        for refid, rows in by_ref.items():
            eur_spent = 0.0
            eur_fee = 0.0
            base_asset = None
            base_qty = 0.0

            for r in rows:
                asset = r.get("asset")
                amt = float(r.get("amount", 0.0))
                fee = float(r.get("fee", 0.0))
                typ = r.get("type")  # 'spend'|'receive'|'transfer'...

                if asset == "ZEUR":
                    if amt < 0:
                        eur_spent += abs(amt)
                    eur_fee += fee
                else:
                    if typ == "receive" and amt > 0:
                        base_asset = asset
                        base_qty += amt

            if base_asset and (eur_spent > 0 or eur_fee > 0) and base_qty > 0:
                pos[base_asset]["qty"] += base_qty
                pos[base_asset]["cost_eur"] += (eur_spent + eur_fee)

        out = {}
        for asset, s in pos.items():
            out[asset] = (s["cost_eur"] / s["qty"]) if s["qty"] > 0 else None
        return out

    # ----------------- merge costi -----------------
    def merged_average_costs(self) -> dict:
        """
        Combina trades e ledgers: usa il prezzo da trades quando disponibile,
        altrimenti quello dai ledger. (Se entrambi, preferisci trades.)
        """
        t = self.average_costs_from_trades()
        l = self.average_costs_from_ledgers()
        merged = {}
        keys = set(t.keys()) | set(l.keys())
        for k in keys:
            at = t.get(k)
            al = l.get(k)
            if at is None and al is None:
                merged[k] = None
            elif at is None:
                merged[k] = al
            elif al is None:
                merged[k] = at
            else:
                merged[k] = at  # preferisci il valore dai trades
        return merged

    # ----------------- vista portafoglio -----------------
    def portfolio_view(self, include_diagnostics: bool = True):
        """
        Ritorna (rows, total_eur, trades, ledgers).
        Di default NON scarica trades/ledgers (pesanti): abilita con include_diagnostics=True.
        """
        self.bals = self.balances(self._balances_ttl)

        avg_costs = {}
        # if include_diagnostics:
        #     # calcoli cost basis solo quando servono
        #     avg_costs_tr = self.average_costs_from_trades()
        #     avg_costs_ld = self.average_costs_from_ledgers()
        #     avg_costs = {
        #         a: (avg_costs_tr.get(a) if avg_costs_tr.get(a) is not None else avg_costs_ld.get(a))
        #         for a in set(list(avg_costs_tr.keys()) + list(avg_costs_ld.keys()) + list(self.bals.keys()))
        #     }

        rows = []
        total_eur = 0.0

        for code, qty in self.bals.items():
            alt = self.assets_info.get(code, {}).get("altname", code)
            px = self.price_in_eur(code)
            val = qty * px if (px is not None and not math.isnan(px)) else None

            # avg_buy = avg_costs.get(code) if include_diagnostics else None
            pnl_pct = None
            # if include_diagnostics and avg_buy not in (None, 0) and px is not None and not math.isnan(px):
            #     pnl_pct = (px - avg_buy) / avg_buy

            if val is not None:
                total_eur += val

            rows.append(
                {
                    "code": code,
                    "asset": alt,
                    "qty": qty,
                    "px_EUR": px,
                    "val_EUR": val,
                    "avg_buy_EUR": None,
                    "pnl_pct": None,
                }
            )

        rows.sort(key=lambda r: (r["val_EUR"] or 0), reverse=True)
        trades = self.trades_history() if include_diagnostics else []
        ledgers = self.ledgers() if include_diagnostics else {}
        return rows, total_eur, trades, ledgers, self.pairs_info

    # ----------------- EUR investibili -----------------
    def investable_eur(self, openOrders = None) -> float:
        """
        ZEUR disponibile meno EUR riservati da ordini BUY aperti su coppie quotate in EUR.
        """
        bals = self.bals if self.bals else self.balances()
        zeur_free = float(bals.get("ZEUR", 0.0))

        resp = self._kraken_call("OpenOrders", private=True, weight=3.0) if openOrders == None else openOrders
        open_orders = resp.get("result", {}).get("open", {}) or {}

        eur_reserved = 0.0
        for oo in open_orders.values():
            descr = oo.get("descr", {}) or {}
            pair_field = (descr.get("pair") or "").replace("/", "")  # es. XXBTZEUR
            pi = self.pairs_by_name.get(pair_field) or self.pairs_by_alt.get(pair_field)
            if not pi or pi.get("quote") != "ZEUR":
                continue
            if (descr.get("type") or "").lower() != "buy":
                continue

            vol_total = float(oo.get("vol", 0.0) or 0.0)
            vol_exec = float(oo.get("vol_exec", 0.0) or 0.0)
            vol_rem = max(vol_total - vol_exec, 0.0)
            if vol_rem <= 0:
                continue

            ordertype = (descr.get("ordertype") or "").lower()
            price = float(descr.get("price", 0.0) or 0.0)
            price2 = float(descr.get("price2", 0.0) or 0.0)
            if ordertype == "stop-loss-limit" and price2 > 0:
                px = price2
            else:
                px = price if price > 0 else self._ticker_last(pi["name"])

            eur_reserved += vol_rem * px

        return max(zeur_free - eur_reserved, 0.0)


# ----------------- runner / test rapido -----------------
if __name__ == "__main__":
    kp = KrakenPortfolio()
    # diagnostica spenta di default per evitare troppe chiamate
    rows, total, trades, ledgers = kp.portfolio_view(include_diagnostics=True)

    def fnum(x, fmt="{:.6f}", na="n/a"):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return na
        return fmt.format(x)

    print(f"{'CODE':6} {'ASSET':8} {'QTY':>18} {'PX_EUR':>14} {'VAL_EUR':>14}")
    for r in rows:
        print(
            f"{r['code']:6} {r['asset']:8} "
            f"{r['qty']:18.8f} "
            f"{fnum(r['px_EUR'], '{:.6f}'):>14} "
            f"{fnum(r['val_EUR'], '{:.2f}'):>14}"
        )
    print(f"\nTotale portafoglio (EUR): {total:,.2f}")
