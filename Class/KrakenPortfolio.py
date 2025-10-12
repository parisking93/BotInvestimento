import os
import math
import krakenex
from collections import defaultdict
import time

class KrakenPortfolio:
    def __init__(self, key=None, secret=None):
        self.k = krakenex.API(
            key=key or os.environ.get("KRAKEN_KEY"),
            secret=secret or os.environ.get("KRAKEN_SECRET"),
        )
        self.assets_info = self.k.query_public('Assets')['result']
        self.pairs_info  = self.k.query_public('AssetPairs')['result']
        self._ticker_cache = {}
        self._budget = os.environ.get("BUDGET")
        # Mappe per risolvere i pair anche quando arrivano in formato alt (es. "XBT/EUR")
        self.pairs_by_name = self.pairs_info
        self.pairs_by_alt  = {}
        for name, p in self.pairs_info.items():
            alt = (p.get('altname') or '').replace('/', '')
            self.pairs_by_alt[alt] = {'name': name, 'base': p['base'], 'quote': p['quote']}

    # ----------------- prezzi -----------------
    def _ticker_last(self, pair_name: str) -> float:
        if pair_name not in self._ticker_cache:
            t = self.k.query_public('Ticker', {'pair': pair_name})['result']
            self._ticker_cache[pair_name] = float(list(t.values())[0]['c'][0])
        return self._ticker_cache[pair_name]

    def _find_pair(self, base_code: str, quote_code: str):
        for name, p in self.pairs_info.items():
            if p.get('base') == base_code and p.get('quote') == quote_code:
                return name
        return None

    def _find_eur_pair(self, base_code: str):
        return self._find_pair(base_code, 'ZEUR')

    def _convert_quote_to_eur(self, amount_in_quote: float, quote_code: str) -> float:
        """Converte un importo espresso nella QUOTE in EUR con i prezzi CORRENTI (stima)."""
        if quote_code == 'ZEUR':
            return amount_in_quote

        # QUOTE -> EUR diretto
        q2e = self._find_pair(quote_code, 'ZEUR')
        if q2e:
            return amount_in_quote * self._ticker_last(q2e)

        # via USD
        amount_in_usd = None
        if quote_code == 'ZUSD':
            amount_in_usd = amount_in_quote
        else:
            q2u = self._find_pair(quote_code, 'ZUSD')
            if q2u:
                amount_in_usd = amount_in_quote * self._ticker_last(q2u)
        if amount_in_usd is not None:
            usd2eur = self._find_pair('ZUSD', 'ZEUR')
            if usd2eur:
                return amount_in_usd * self._ticker_last(usd2eur)

        # via XBT
        q2x = self._find_pair(quote_code, 'XXBT')
        x2e = self._find_eur_pair('XXBT')
        if q2x and x2e:
            return amount_in_quote * self._ticker_last(q2x) * self._ticker_last(x2e)

        return float('nan')

    def price_in_eur(self, asset_code: str) -> float:
        if asset_code == 'ZEUR':
            return 1.0
        p = self._find_eur_pair(asset_code)
        if p:
            return self._ticker_last(p)

        p_usd = self._find_pair(asset_code, 'ZUSD')
        if p_usd:
            usd2eur = self._find_pair('ZUSD', 'ZEUR')
            if usd2eur:
                return self._ticker_last(p_usd) * self._ticker_last(usd2eur)

        p_xbt = self._find_pair(asset_code, 'XXBT')
        xbt_eur = self._find_eur_pair('XXBT')
        if p_xbt and xbt_eur:
            return self._ticker_last(p_xbt) * self._ticker_last(xbt_eur)

        return float('nan')

    # ----------------- dati account -----------------
    def balances(self) -> dict:
        time.sleep(1)
        bal = self.k.query_private('Balance')
        time.sleep(1)

        if bal.get('error'):
            raise RuntimeError(bal['error'])
        return {a: float(v) for a, v in bal['result'].items() if float(v) > 0}

    def trades_history(self, start=None, ofs=0, max_loops=40) -> list:
        all_trades = []
        loops = 0
        payload = {'type': 'all'}
        if start is not None:
            payload['start'] = start
        while loops < max_loops:
            time.sleep(1)
            resp = self.k.query_private('TradesHistory', {**payload, 'ofs': ofs})
            if resp.get('error'):
                raise RuntimeError(resp['error'])
            time.sleep(0.5)
            trades = resp['result'].get('trades', {})
            if not trades:
                break
            items = list(trades.values())
            all_trades.extend(items)
            ofs += len(items)
            loops += 1
            if len(items) < 50:
                break
        all_trades.sort(key=lambda t: t.get('time', 0.0))
        return all_trades

    def ledgers(self) -> dict:
        """Ritorna dict completo dei ledger (id -> entry)."""
        resp = self.k.query_private('Ledgers')
        if resp.get('error'):
            raise RuntimeError(resp['error'])
        return resp['result'].get('ledger', {})

    # ----------------- normalizzazione pair trade -----------------
    def _pair_info_from_trade(self, trade_pair_field: str):
        if not trade_pair_field:
            return None
        key = trade_pair_field.replace('/', '')  # gestisce alt "XBT/EUR"
        p = self.pairs_by_name.get(key)
        if p:
            return {'name': key, 'base': p['base'], 'quote': p['quote']}
        p_alt = self.pairs_by_alt.get(key)
        if p_alt:
            return p_alt
        return None

    # ----------------- cost basis: trades -----------------
    def average_costs_from_trades(self) -> dict:
        """
        Prezzo medio (EUR) per asset base usando solo i trade spot.
        """
        trades = self.trades_history()
        pos = {}   # asset -> {'qty': q, 'cost_eur': c}

        for t in trades:
            pi = self._pair_info_from_trade(t.get('pair'))
            if not pi:
                continue
            base, quote = pi['base'], pi['quote']
            vol   = float(t.get('vol', 0.0))
            cost  = float(t.get('cost', 0.0))
            fee   = float(t.get('fee', 0.0))
            typ   = t.get('type')

            cost_eur = self._convert_quote_to_eur(cost + fee, quote)

            if base not in pos:
                pos[base] = {'qty': 0.0, 'cost_eur': 0.0}

            if typ == 'buy':
                pos[base]['qty']      += vol
                pos[base]['cost_eur'] += cost_eur
            elif typ == 'sell' and pos[base]['qty'] > 0:
                avg = pos[base]['cost_eur'] / pos[base]['qty']
                sell_q = min(vol, pos[base]['qty'])
                pos[base]['qty']      -= sell_q
                pos[base]['cost_eur'] -= avg * sell_q
                if pos[base]['qty'] < 1e-12:
                    pos[base]['qty'] = 0.0
                    pos[base]['cost_eur'] = 0.0

        out = {}
        for asset, s in pos.items():
            out[asset] = (s['cost_eur'] / s['qty']) if s['qty'] > 0 else None
        return out

    # ----------------- cost basis: ledgers (per "Convert", depositi, ecc.) -----------------
    def average_costs_from_ledgers(self) -> dict:
        """
        Ricostruisce il prezzo medio in EUR per asset usando i Ledgers.
        Metodo:
          - raggruppa per refid; per ogni refid cerca coppia EUR(-) + ASSET(+)
          - costo = EUR_speso (abs) + fee EUR del gruppo
          - qty   = somma quantità positive dell'asset base
        """
        ldg = self.ledgers()
        # group by refid
        by_ref = defaultdict(list)
        for lid, row in ldg.items():
            by_ref[row.get('refid')].append(row)

        # accumulatore risultato
        pos = defaultdict(lambda: {'qty': 0.0, 'cost_eur': 0.0})

        for refid, rows in by_ref.items():
            # individua eventuale asset ricevuto e spesa EUR
            eur_spent = 0.0
            eur_fee   = 0.0
            base_asset = None
            base_qty   = 0.0

            for r in rows:
                asset = r.get('asset')
                amt   = float(r.get('amount', 0.0))
                fee   = float(r.get('fee',    0.0))
                typ   = r.get('type')  # 'spend'|'receive'|'transfer'...

                if asset == 'ZEUR':
                    # amount su ledger è già netto, fee separata
                    if amt < 0:
                        eur_spent += abs(amt)
                    eur_fee += fee
                else:
                    # se è un asset cripto ricevuto in questa ref
                    if typ == 'receive' and amt > 0:
                        # attenzione: in ledger asset è già in codice base (XXBT, XETH, ...)
                        base_asset = asset
                        base_qty  += amt

            if base_asset and (eur_spent > 0 or eur_fee > 0) and base_qty > 0:
                pos[base_asset]['qty']      += base_qty
                pos[base_asset]['cost_eur'] += (eur_spent + eur_fee)

        # media
        out = {}
        for asset, s in pos.items():
            out[asset] = (s['cost_eur'] / s['qty']) if s['qty'] > 0 else None
        return out

    # ----------------- merge costi -----------------
    def merged_average_costs(self) -> dict:
        """
        Combina trades e ledgers: usa il prezzo da trades quando disponibile,
        altrimenti quello dai ledger.
        Se entrambi presenti, fa media pesata su qty e cost.
        """
        t = self.average_costs_from_trades()
        l = self.average_costs_from_ledgers()

        # Per avere le quantità, rifacciamo i due accumuli in forma qty/costo
        def rebuild_qty_cost_from_avg(avg_map):
            # solo per ricostruire con i saldi attuali
            # (assunzione: quantità correnti ~ quantità rimaste associate a quel costo medio)
            # più preciso sarebbe rifare gli accumuli quantità/costi come nelle funzioni sopra; qui facciamo merge semplice.
            return avg_map

        merged = {}
        # unisci le chiavi
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
                # se entrambi esistono facciamo media semplice (manca info qty; in pratica at prevale)
                merged[k] = at if not math.isnan(at) else al
        return merged

    # ----------------- vista portafoglio -----------------
    def portfolio_view(self):
        time.sleep(1)
        bals = self.balances()
        time.sleep(1)
        avg_costs_tr = self.average_costs_from_trades()
        avg_costs_ld = self.average_costs_from_ledgers()
        # merge: preferisci trades, poi ledgers
        avg_costs = {a: (avg_costs_tr.get(a) if avg_costs_tr.get(a) is not None else avg_costs_ld.get(a))
                     for a in set(list(avg_costs_tr.keys()) + list(avg_costs_ld.keys()) + list(bals.keys()))}

        rows = []
        total_eur = 0.0

        for code, qty in bals.items():
            alt = self.assets_info.get(code, {}).get('altname', code)
            px  = self.price_in_eur(code)
            val = qty * px if (px is not None and not math.isnan(px)) else None

            avg_buy = avg_costs.get(code)
            pnl_pct = None
            if avg_buy not in (None, 0) and px is not None and not math.isnan(px):
                pnl_pct = (px - avg_buy) / avg_buy

            if val is not None:
                total_eur += val

            rows.append({
                'code': code, 'asset': alt, 'qty': qty,
                'px_EUR': px, 'val_EUR': val,
                'avg_buy_EUR': avg_buy, 'pnl_pct': pnl_pct
            })

        rows.sort(key=lambda r: (r['val_EUR'] or 0), reverse=True)
        # extra diagnostica
        trades = self.trades_history()
        time.sleep(1)
        ledgers = self.ledgers()
        return rows, total_eur, trades, ledgers


    def investable_eur(self) -> float:
        """
        Ritorna gli EUR realmente investibili:
        ZEUR disponibile meno l'EUR riservato da ordini BUY aperti su coppie quotate in EUR.
        """
        # 1) saldo ZEUR libero
        bals = self.balances()
        zeur_free = float(bals.get('ZEUR', 0.0))
        time.sleep(1)
        # 2) EUR riservati da ordini BUY aperti (quote = ZEUR)
        resp = self.k.query_private('OpenOrders')
        if resp.get('error'):
            raise RuntimeError(resp['error'])
        open_orders = resp.get('result', {}).get('open', {}) or {}
        time.sleep(0.5)
        eur_reserved = 0.0
        for oo in open_orders.values():
            descr = oo.get('descr', {}) or {}
            pair_field = (descr.get('pair') or '').replace('/', '')   # es. XXBTZEUR
            pi = self.pairs_by_name.get(pair_field) or self.pairs_by_alt.get(pair_field)
            if not pi:
                continue

            # consideriamo solo coppie quotate in EUR e solo ordini di acquisto
            if pi.get('quote') != 'ZEUR':
                continue
            if (descr.get('type') or '').lower() != 'buy':
                continue

            # quantità residua da eseguire
            vol_total = float(oo.get('vol', 0.0) or 0.0)
            vol_exec  = float(oo.get('vol_exec', 0.0) or 0.0)
            vol_rem   = max(vol_total - vol_exec, 0.0)
            if vol_rem <= 0:
                continue

            # prezzo di riferimento
            ordertype = (descr.get('ordertype') or '').lower()
            price  = float(descr.get('price', 0.0) or 0.0)
            price2 = float(descr.get('price2', 0.0) or 0.0)
            if ordertype == 'stop-loss-limit' and price2 > 0:
                px = price2
            else:
                px = price if price > 0 else self._ticker_last(pi['name'])

            eur_reserved += vol_rem * px

        # 3) EUR investibili
        return max(zeur_free - eur_reserved, 0.0)



# ----------------- runner / stampa -----------------
if __name__ == "__main__":
    kp = KrakenPortfolio()
    rows, total, trades, ledgers = kp.portfolio_view()

    def fnum(x, fmt="{:.6f}", na="n/a"):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return na
        return fmt.format(x)

    print(f"{'CODE':6} {'ASSET':8} {'QTY':>18} {'PX_EUR':>14} {'VAL_EUR':>14} {'AVG_BUY_EUR':>14} {'PNL_%':>10}")
    for r in rows:
        print(
            f"{r['code']:6} {r['asset']:8} "
            f"{r['qty']:18.8f} "
            f"{fnum(r['px_EUR'], '{:.6f}'):>14} "
            f"{fnum(r['val_EUR'], '{:.2f}'):>14} "
            f"{fnum(r['avg_buy_EUR'], '{:.6f}'):>14} "
            f"{(fnum(r['pnl_pct']*100, '{:.2f}%') if r['pnl_pct'] is not None else 'n/a'):>10}"
        )
    print(f"\nTotale portafoglio (EUR): {total:,.2f}")

    # ——— diagnostica come da tue prove ———
    print("\n== TradesHistory ==")
    print(f"conteggio trade: {len(trades)}")
    if trades:
        # mostra i primi 3 trade in forma compatta
        for i, t in enumerate(trades[:3], 1):
            print(f"{i}) pair={t.get('pair')} type={t.get('type')} vol={t.get('vol')} "
                  f"cost={t.get('cost')} fee={t.get('fee')} time={t.get('time')}")

    print("\n== Prime 5 righe di Ledgers ==")
    items = list(ledgers.items())[:5]
    for i, (lid, row) in enumerate(items, 1):
        print(f"{i}) {lid} | asset={row.get('asset')} amount={row.get('amount')} "
              f"fee={row.get('fee')} type={row.get('type')} refid={row.get('refid')} time={row.get('time')}")



