import math
import os   # (serve anche per BUDGET/RISK_PCT se non l’hai già messo)

from Class.Action import Action
from Class.Action import Actions
class StrategyEngine:
    """
    Applica strategie su una lista di currency (export_currencies()) e
    ritorna decisioni minimali in italiano.
    Metodi disponibili: orBreakoutSignal, intradaySignal
    Output per ogni currency:
        {
          'pair': 'BTC/EUR',
          'azione': 'compra' | 'compra_limit' | 'vendi' | 'attendi',
          'prezzo': float | None,
          'stop': float | None,
          'take_profit': float | None,
          'motivo': str,
          'timeframe': '24H' | '48H' | '7D' | '30D' | None
        }
    """
    def __init__(self, currencies: list[dict]):
        self.currencies = currencies  # lista come quella restituita da export_currencies()

    def orBreakoutSignal(self):
        """
        Regole:
          - Se or_ok=True e last > or_high  -> 'compra' (breakout)
          - Se or_ok=True e last < or_low   -> 'vendi' se hai posizione (qty>0), altrimenti 'attendi'
        Stop:   or_low - 0.5*ATR  (se disponibili)   ; fallback: or_low
        TP:     entry + 1.5*(entry - stop)            ; fallback: None
        Ordine: 'compra' = market se spread piccolo e buona liquidità; altrimenti 'compra_limit'
        """
        decisions = []
        for c in self.currencies:
            pair = c.get('pair') or (f"{c.get('base')}/{c.get('quote')}" if c.get('base') and c.get('quote') else None)
            info_tf = c.get('info') or {}
            data = None
            tf_used = None
            # preferenza timeframe: 24H → 48H → 7D → 30D
            for tf in ('24H', '48H', '7D', '30D'):
                if tf in info_tf:
                    data = info_tf[tf]
                    tf_used = tf
                    break
            if not data:
                decisions.append({'pair': pair, 'azione': 'attendi', 'prezzo': None, 'stop': None,
                                  'take_profit': None, 'motivo': 'Dati OR assenti', 'timeframe': None})
                continue

            last = data.get('last') or data.get('close')
            or_ok = data.get('or_ok')
            or_high = data.get('or_high')
            or_low  = data.get('or_low')
            atr = data.get('atr')
            spread = data.get('spread')
            liq_tot = data.get('liquidity_total_sum')
            bid = data.get('bid'); ask = data.get('ask')
            entry = ask or last

            # qty in portafoglio (se presente)
            qty = None
            row = (c.get('portfolio') or {}).get('row')
            if row:
                qty = row.get('qty')

            if or_ok and last is not None and or_high is not None and last > or_high:
                # breakout rialzista
                azione = 'compra'
                # se spread alto o poca liquidità → meglio limit
                if (spread is not None and last and (spread/last) > 0.0003) or (liq_tot is not None and liq_tot < 3):
                    azione = 'compra_limit'
                stop = None
                if or_low is not None:
                    stop = (or_low - 0.5*atr) if (atr is not None) else or_low
                take = None
                if stop and entry and entry > stop:
                    take = entry + 1.5*(entry - stop)
                decisions.append({
                    'pair': pair, 'azione': azione, 'prezzo': float(entry) if entry else None,
                    'stop': float(stop) if stop else None, 'take_profit': float(take) if take else None,
                    'motivo': f"Breakout OR sopra {or_high}", 'timeframe': tf_used
                })
            elif or_ok and last is not None and or_low is not None and last < or_low:
                # breakdown ribassista
                if qty and qty > 0:
                    stop = (or_high + 0.5*atr) if (atr is not None and or_high is not None) else or_high
                    take = None
                    if stop and entry:
                        risk = abs(entry - stop)
                        take = entry - 1.5*risk
                    decisions.append({
                        'pair': pair, 'azione': 'vendi', 'prezzo': float(bid or last),
                        'stop': float(stop) if stop else None, 'take_profit': float(take) if take else None,
                        'motivo': f"Breakdown OR sotto {or_low} e posizione aperta", 'timeframe': tf_used
                    })
                else:
                    decisions.append({
                        'pair': pair, 'azione': 'attendi', 'prezzo': None, 'stop': None, 'take_profit': None,
                        'motivo': "Breakdown OR ma nessuna posizione", 'timeframe': tf_used
                    })
            else:
                decisions.append({
                    'pair': pair, 'azione': 'attendi', 'prezzo': None, 'stop': None, 'take_profit': None,
                    'motivo': 'Nessun breakout OR valido', 'timeframe': tf_used
                })

        return decisions

    def intradaySignal(self):
        """
        Regole intraday (semplici):
          - COMPRA se: bias_1h=='UP' o (close>ema50_1h>ema200_1h) e bias_4h non 'DOWN'
            → 'compra_limit' a ~ mid*(1-0.05%); stop = close - 1.2*ATR ; TP = close + 2*ATR
          - VENDI se: bias_1h=='DOWN' e (close<ema50_1h<ema200_1h) e hai posizione
            → 'vendi'; stop = close + 1.2*ATR ; TP = close - 2*ATR
          - Altrimenti 'attendi'
        """
        decisions = []
        for c in self.currencies:
            pair = c.get('pair') or (f"{c.get('base')}/{c.get('quote')}" if c.get('base') and c.get('quote') else None)
            info_tf = c.get('info') or {}
            data = None
            tf_used = None
            # per intraday preferiamo 24H; fallback 48H
            for tf in ('24H', '48H'):
                if tf in info_tf:
                    data = info_tf[tf]
                    tf_used = tf
                    break
            if not data:
                decisions.append({'pair': pair, 'azione': 'attendi', 'prezzo': None, 'stop': None,
                                  'take_profit': None, 'motivo': 'Dati intraday assenti', 'timeframe': None})
                continue

            close = data.get('close') or data.get('last')
            mid = data.get('mid') or close
            last = data.get('last') or close
            ema50_1h = data.get('ema50_1h')
            ema200_1h = data.get('ema200_1h')
            bias_1h = data.get('bias_1h')
            bias_4h = data.get('bias_4h')
            atr = data.get('atr')

            # posizione esistente?
            qty = None
            row = (c.get('portfolio') or {}).get('row')
            if row:
                qty = row.get('qty')

            long_cond = ((bias_1h == 'UP') or (close and ema50_1h and ema200_1h and close > ema50_1h > ema200_1h)) and (bias_4h != 'DOWN')
            short_cond = ((bias_1h == 'DOWN') and (close and ema50_1h and ema200_1h and close < ema50_1h < ema200_1h))

            if long_cond:
                entry = (mid or last) * (1 - 0.0005) if (mid or last) else None
                stop = (close - 1.2*atr) if (close and atr) else None
                take = (close + 2*atr) if (close and atr) else None
                decisions.append({
                    'pair': pair, 'azione': 'compra_limit', 'prezzo': float(entry) if entry else None,
                    'stop': float(stop) if stop else None, 'take_profit': float(take) if take else None,
                    'motivo': 'Trend intraday UP (bias/EMA 1h)', 'timeframe': tf_used
                })
            elif short_cond and qty and qty > 0:
                entry = mid or last
                stop = (close + 1.2*atr) if (close and atr) else None
                take = (close - 2*atr) if (close and atr) else None
                decisions.append({
                    'pair': pair, 'azione': 'vendi', 'prezzo': float(entry) if entry else None,
                    'stop': float(stop) if stop else None, 'take_profit': float(take) if take else None,
                    'motivo': 'Trend intraday DOWN (bias/EMA 1h) con posizione aperta', 'timeframe': tf_used
                })
            else:
                decisions.append({
                    'pair': pair, 'azione': 'attendi', 'prezzo': None, 'stop': None, 'take_profit': None,
                    'motivo': 'Nessun segnale intraday valido', 'timeframe': tf_used
                })

        return decisions


    def build_actions(self, dec_or: list[dict], dec_intraday: list[dict],
                        allow_short: bool = True,
                        size_mode: str = "position",   # 'position' usa la qty posseduta; 'fixed' usa default_qty
                        default_qty: float | None = None) -> Actions:
            """
            Combina segnali OR + Intraday con lo stato del portafoglio e produce una lista di 'Action'.
            Regole principali (semplici):
            - Se OR dice 'vendi' e possiedi qty>0 -> Action SELL 'adesso' per chiudere/ridurre.
                + (se allow_short) aggiungi Action 'order' per aprire SHORT: 'adesso' se già sotto OR-low, altrimenti 'al_break' con break=OR-low
            - Se OR dice 'compra'/'compra_limit' e NON possiedi -> Action BUY (market o limit).
            - Intraday genera azioni solo come fallback se non c'è già una OR per quel pair.
            """
            # indicizza segnali per pair
            by_pair_or = {d.get("pair"): d for d in dec_or if d.get("pair")}
            by_pair_itd = {d.get("pair"): d for d in dec_intraday if d.get("pair")}
            out = Actions()

            # indice portafoglio: pair -> row (best effort via base/quote)
            def portfolio_row_for(c: dict):
                row = (c.get("portfolio") or {}).get("row")
                return row

            def info24_for(c: dict):
                info = c.get("info") or {}
                return info.get("24H") or info.get("48H") or {}

            for c in self.currencies:
                pair = c.get('pair') or (f"{c.get('base')}/{c.get('quote')}" if c.get('base') and c.get('quote') else None)
                if not pair:
                    continue  # non tradabile

                row = portfolio_row_for(c)
                qty_have = (row or {}).get("qty") or 0.0

                info24 = info24_for(c)
                or_low = info24.get("or_low")
                or_high = info24.get("or_high")
                last = info24.get("last") or info24.get("close")
                bid = info24.get("bid"); ask = info24.get("ask")

                # grandezza dell'ordine
                if size_mode == "position" and qty_have:
                    qty_order = qty_have
                else:
                    qty_order = default_qty  # può restare None: l'esecutore deciderà

                used = False  # se generiamo già un'azione forte da OR, evitiamo doppioni dall'intraday

                # ----- regole OR -----
                d_or = by_pair_or.get(pair)
                if d_or:
                    az = d_or.get("azione")
                    stop = d_or.get("stop")
                    take = d_or.get("take_profit")
                    prezzo = d_or.get("prezzo")
                    tf = d_or.get("timeframe")
                    motivo = d_or.get("motivo") or "Segnale OR"

                    # 1) chiusura long su 'vendi'
                    if az == "vendi" and qty_have and qty_have > 0:
                        out.add(Action(
                            pair=pair, tipo="sell", quando="adesso", lato="long",
                            prezzo=float(bid or last) if (bid or last) else None,
                            stop_loss=None, take_profit=None,
                            quantita=qty_order, timeframe=tf,
                            motivo=motivo + " - chiudi posizione"
                        ))
                        used = True

                        # 2) eventuale apertura short
                        if allow_short:
                            if last is not None and or_low is not None and last < or_low:
                                # già sotto il break -> entra adesso
                                out.add(Action(
                                    pair=pair, tipo="order", quando="adesso", lato="short",
                                    prezzo=float(bid or last) if (bid or last) else None,
                                    stop_loss=float(stop) if stop else None,
                                    take_profit=float(take) if take else None,
                                    quantita=default_qty, timeframe=tf,
                                    motivo="Apri SHORT su breakdown OR"
                                ))
                            elif or_low is not None:
                                # prepara ordine a stop (break)
                                out.add(Action(
                                    pair=pair, tipo="order", quando="al_break", lato="short",
                                    break_price=float(or_low),
                                    stop_loss=float(stop) if stop else None,
                                    take_profit=float(take) if take else None,
                                    quantita=default_qty, timeframe=tf,
                                    motivo="Apri SHORT alla rottura di OR-low"
                                ))

                    # 3) apertura long se non possiedi
                    elif az in ("compra", "compra_limit") and (not qty_have or qty_have == 0):
                        if az == "compra":
                            out.add(Action(
                                pair=pair, tipo="buy", quando="adesso", lato="long",
                                prezzo=float(ask or last) if (ask or last) else None,
                                stop_loss=float(stop) if stop else None,
                                take_profit=float(take) if take else None,
                                quantita=default_qty, timeframe=tf,
                                motivo=motivo
                            ))
                        else:  # compra_limit
                            out.add(Action(
                                pair=pair, tipo="buy", quando="al_limite", lato="long",
                                limite=float(prezzo) if prezzo else None,
                                stop_loss=float(stop) if stop else None,
                                take_profit=float(take) if take else None,
                                quantita=default_qty, timeframe=tf,
                                motivo=motivo + " (limit)"
                            ))
                        used = True

                # ----- fallback Intraday se non abbiamo già creato azioni forti -----
                if not used:
                    d_itd = by_pair_itd.get(pair)
                    if d_itd:
                        az = d_itd.get("azione")
                        tf = d_itd.get("timeframe")
                        motivo = d_itd.get("motivo") or "Segnale intraday"
                        prezzo = d_itd.get("prezzo")
                        stop = d_itd.get("stop"); take = d_itd.get("take_profit")

                        if az == "vendi" and qty_have and qty_have > 0:
                            out.add(Action(
                                pair=pair, tipo="sell", quando="adesso", lato="long",
                                prezzo=float(bid or last) if (bid or last) else None,
                                stop_loss=None, take_profit=None,
                                quantita=qty_order, timeframe=tf,
                                motivo=motivo + " - chiudi posizione"
                            ))
                        elif az == "compra_limit" and (not qty_have or qty_have == 0):
                            out.add(Action(
                                pair=pair, tipo="buy", quando="al_limite", lato="long",
                                limite=float(prezzo) if prezzo else None,
                                stop_loss=float(stop) if stop else None,
                                take_profit=float(take) if take else None,
                                quantita=default_qty, timeframe=tf,
                                motivo=motivo
                            ))
                        # altri 'attendi' vengono ignorati

            return out


    def build_actions_autoconfig(self) -> Actions:
        """
        Ricalcola i segnali (OR + Intraday), costruisce le Action e
        aggiunge automaticamente il campo 'leverage' quando serve.

        Regole leverage:
          - SELL di posizione long esistente -> leverage = None
          - BUY/BUY_LIMIT long               -> leverage = None
          - SHORT (tipo='order' e lato='short') -> leverage = DEFAULT_SHORT_LEVERAGE (es. '2:1')
        """
        # default configurabili via env (facoltativi)
        default_qty = 0.01
        default_short_leverage = os.getenv("DEFAULT_SHORT_LEVERAGE", "2:1")

        # 1) segnali
        dec_or = self.orBreakoutSignal()
        dec_itd = self.intradaySignal()

        # 2) build azioni base (chiusure long, eventuali short, acquisti)
        actions = self.build_actions(
            dec_or, dec_itd,
            allow_short=True,
            size_mode="position",     # chiusura usa qty posseduta
            default_qty=default_qty,  # per nuovi ingressi
        )

        # 3) mappa pair -> qty posseduta (per chiarezza, può servirti altrove)
        qty_have_by_pair = {}
        for c in self.currencies:
            pair = c.get('pair') or (f"{c.get('base')}/{c.get('quote')}"
                                     if c.get('base') and c.get('quote') else None)
            if not pair:
                continue
            row = (c.get('portfolio') or {}).get('row') or {}
            qty_have_by_pair[pair] = row.get('qty') or 0.0

        # 4) arricchisci le action con 'leverage' dove necessario
        for a in actions.items:
            # chiusura long: nessuna leva
            if a.tipo == "sell" and a.lato == "long":
                a.leverage = None
                continue

            # aperture long (buy / buy-limit): normalmente spot, nessuna leva
            if a.lato == "long":
                a.leverage = None
                continue

            # aperture short: richiedono margin/leverage
            if a.lato == "short":
                a.leverage = default_short_leverage

        return actions


    def allocate_quantities(self, actions: Actions) -> Actions:
        """
        Calcola e scrive la 'quantita' da eseguire per ciascuna Action
        in base al budget disponibile, al rischio per trade e allo stop.

        Env supportati (tutti opzionali):
          - BUDGET:                 capitale totale disponibile (EUR). es: "200"
          - RISK_PCT:               rischio per trade sul budget (def. 0.01 = 1%)
          - MAX_CASH_PCT_TRADE:     tetto max di cassa per singolo trade (def. 0.25 = 25%)
          - DEFAULT_STOP_PCT:       stop% di fallback se manca lo stop_loss (def. 0.01 = 1%)
          - MIN_NOTIONAL_EUR:       nozionale minimo per ordine (def. 5)
        Regole:
          qty = min( risk_amount / |entry - stop| , cash_cap / entry )
          con gestione leva: cash_required ≈ (entry * qty) / max(leverage, 1)
          e aggiornamento del cash rimanente trade dopo trade.
        """
        # ---- 1) parametri da env (con default ragionevoli)
        budget = float(os.getenv("BUDGET", "0") or 0)
        risk_pct = float(os.getenv("RISK_PCT", "70") or 0.01)                 # 1% del budget
        max_cash_pct_trade = float(os.getenv("MAX_CASH_PCT_TRADE", "0.25") or 0.25)
        default_stop_pct = float(os.getenv("DEFAULT_STOP_PCT", "0.01") or 0.01)  # 1% fallback
        min_notional = float(os.getenv("MIN_NOTIONAL_EUR", "5") or 5.0)

        # ---- 2) capitale già impegnato (stima): somma valore EUR delle posizioni non-EUR
        invested_eur = 0.0
        cash_eur = 0.0
        for c in self.currencies:
            row = (c.get('portfolio') or {}).get('row') or {}
            asset = row.get('asset')
            val_eur = row.get('val_EUR') or 0.0
            if not val_eur:
                continue
            if asset == 'EUR' or c.get('base') == 'EUR':
                cash_eur += float(val_eur)
            else:
                invested_eur += float(val_eur)

        # Disponibile = budget - investito (se negativo, mettiamo zero).
        # Se preferisci usare la cassa reale (ZEUR) sostituisci con: available = cash_eur
        available = max(0.0, float(budget) - float(invested_eur))

        # ---- 3) loop azioni: alloca qty
        per_trade_risk_eur = budget * risk_pct
        max_cash_per_trade = budget * max_cash_pct_trade

        def _is_num(x):
            return isinstance(x, (int, float)) and math.isfinite(float(x))

        def _entry_for(a: Action) -> float | None:
            # entry dalla natura dell'azione
            if a.quando == "adesso":
                return a.prezzo
            if a.quando == "al_limite":
                return a.limite if _is_num(a.limite) else a.prezzo
            if a.quando == "al_break":
                return a.break_price
            return a.prezzo

        for a in actions.items:
            # salta azioni senza pair o senza prezzo di ingresso
            entry = _entry_for(a)
            if not _is_num(entry) or entry <= 0:
                a.quantita = a.quantita or None
                continue

            # stop e rischio unitario
            stop = a.stop_loss
            if not _is_num(stop) or stop <= 0:
                # fallback su stop percentuale
                stop = entry * (1.0 - default_stop_pct if (a.lato == "long") else 1.0 + default_stop_pct)

            risk_per_unit = abs(entry - stop)
            if risk_per_unit <= 0:
                a.quantita = a.quantita or None
                continue

            # cappiamo la cassa utilizzabile per questo trade
            cash_cap = min(available, max_cash_per_trade)
            if cash_cap <= 0:
                a.quantita = a.quantita or None
                continue

            # qty per rischio e per cassa
            qty_by_risk = per_trade_risk_eur / risk_per_unit
            qty_by_cash = cash_cap / entry
            qty = max(0.0, min(qty_by_risk, qty_by_cash))

            # leva (se presente) riduce il cash richiesto (margine)
            lev = None
            if hasattr(a, "leverage") and a.leverage:
                # valori tipo "2:1" oppure "3"
                try:
                    s = str(a.leverage)
                    lev = float(s.split(":")[0]) if ":" in s else float(s)
                    if lev <= 0:
                        lev = None
                except Exception:
                    lev = None

            # cash richiesto (margine) e adeguamento qty se serve
            cash_required = (entry * qty) / (lev if lev else 1.0)
            if cash_required > cash_cap and entry > 0:
                qty = (cash_cap * (lev if lev else 1.0)) / entry
                cash_required = (entry * qty) / (lev if lev else 1.0)

            # rispetto nozionale minimo
            if entry * qty < min_notional:
                # prova ad alzare al minimo possibile se c'è cassa
                target_qty = min_notional / entry
                if (entry * target_qty) / (lev if lev else 1.0) <= cash_cap:
                    qty = target_qty
                    cash_required = (entry * qty) / (lev if lev else 1.0)
                else:
                    # non raggiungibile: salta
                    a.quantita = a.quantita or None
                    continue

            # scrivi quantità sull'azione
            a.quantita = float(qty)

            # aggiorna cassa rimanente
            available = max(0.0, available - cash_required)

        return actions


    def convert_quantities_from_eur(self, actions: Actions) -> Actions:
        """
        Interpreta 'quantita' (attuale) come EUR, la sposta in 'quantita_eur'
        e popola 'quantita' in asset base, usando un prezzo di ingresso coerente.
        Ritorna lo stesso oggetto Actions modificato.
        """

        # indicizza info per pair per trovare prezzi di fallback
        info_by_pair: dict[str, dict] = {}
        for c in self.currencies:
            pair = c.get('pair') or (f"{c.get('base')}/{c.get('quote')}"
                                     if c.get('base') and c.get('quote') else None)
            if not pair:
                continue
            info24 = (c.get('info') or {}).get('24H') or (c.get('info') or {}).get('48H') or {}
            info_by_pair[pair] = info24

        def _is_num(x) -> bool:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return False
            return math.isfinite(xf)

        def _entry_price(a: Action) -> float | None:
            """Sceglie il miglior prezzo di ingresso disponibile per l'Action."""
            # 1) in base al 'quando'
            if a.quando == "adesso" and _is_num(a.prezzo):
                return float(a.prezzo)
            if a.quando == "al_limite" and _is_num(a.limite):
                return float(a.limite)
            if a.quando == "al_break" and _is_num(a.break_price):
                return float(a.break_price)

            # 2) fallback: dai dati della currency
            mkt = info_by_pair.get(a.pair, {}) if a.pair else {}
            # BUY/LONG preferisce ask; SELL/SHORT preferisce bid
            if a.tipo in ("buy", "order") and (a.lato == "long" or a.tipo == "buy"):
                for k in ("ask", "last", "mid", "close"):
                    if _is_num(mkt.get(k)):
                        return float(mkt[k])
            else:
                for k in ("bid", "last", "mid", "close"):
                    if _is_num(mkt.get(k)):
                        return float(mkt[k])

            return None

        for a in actions.items:
            eur = a.quantita
            if not _is_num(eur) or float(eur) <= 0:
                # niente da convertire
                a.quantita_eur = None  # opzionale: tener traccia che non c'era un valore valido
                continue

            px = _entry_price(a)
            if not _is_num(px) or float(px) <= 0:
                # non abbiamo un prezzo per convertire: lascio quantita_eur e azzero quantita
                a.quantita_eur = float(eur)
                a.quantita = None
                continue

            qty_asset = float(eur) / float(px)
            a.quantita_eur = float(eur)
            a.quantita = qty_asset  # il runner invierà questo come 'volume'

        return actions
