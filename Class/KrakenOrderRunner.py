from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import asdict
import time
import os

try:
    import krakenex  # opzionale a runtime
except Exception:
    krakenex = None

if TYPE_CHECKING:
    from krakenex import API as KrakenAPI
else:
    KrakenAPI = Any


class KrakenOrderRunner:
    """
    Runner che traduce le action del planner in payload Kraken e le esegue.

    Nuova action attesa (chiavi principali):
      action = {
        "pair": pair,
        "tipo": type,               # "buy" | "sell" | "hold" | "sell/close" | "buy/close"
        "prezzo": price,            # riferimento o trigger
        "quando": quando,           # "market" | "limit" | "stop" (o "al_break")
        "quantita": qty,            # base size desiderata
        "quantita_eur": qty_eur,    # informativa
        "stop_loss": stop_px,       # prezzo assoluto (non %)
        "take_profit": take_px,     # prezzo assoluto (non %)
        "timeframe": "24H",
        "lato": type,               # duplicato di tipo per compat
        "break_price": None,        # trigger stop, se diverso da prezzo
        "limite": limite,           # prezzo limite per ordini LIMIT o price2 per stop-limit
        "leverage": leverage,       # "2:1" ecc. (opzionale, >1 soltanto)
        "motivo": reason,
        "meta": {"reason": motivo},
        "reduce_only": reduce_only, # True/False
        "blend": blended,
        "cancel_order_id": ""       # se valorizzato -> genera body di cancel
      }
    """

    def __init__(self, api: Optional[KrakenAPI] = None, pair_map: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """
        api: krakenex.API opzionale
        pair_map: mapping opzionale alias->pair Kraken (retro-compat con vecchi chiamanti)
        kwargs: parametri legacy ignorati per non rompere l'interfaccia
        """
        self.api: Optional[KrakenAPI] = api
        self.pair_map: Dict[str, str] = dict(pair_map or {})
        self._pairs_cache: Dict[str, dict] = {}

        # sicurezza: cuscinetto sul disponibile (2% per evitare rifiuti per fee/slippage)
        self.safety_buffer: float = 0.98
        # default leva short se richiesto implicitamente
        self.default_short_lev: str = "2:1"

    # ----------------- utility mapping action -----------------
    @staticmethod
    def _as_action_dict(a: Any) -> Dict[str, Any]:
        if isinstance(a, dict):
            return a
        try:
            return asdict(a)  # dataclass
        except Exception:
            keys = ["tipo","pair","prezzo","quando","quantita","quantita_eur",
                    "stop_loss","take_profit","timeframe","lato","break_price",
                    "limite","leverage","motivo","meta","reduce_only","blend","cancel_order_id"]
            return {k: getattr(a, k, None) for k in keys}

    @staticmethod
    def _side_from_action(tipo: Optional[str], lato: Optional[str]) -> Optional[str]:
        if not tipo and not lato:
            return None
        t = (tipo or "").strip().lower()
        l = (lato or "").strip().lower()
        if t in {"sell", "sell/close", "close-sell", "chiudi-sell"}: return "sell"
        if t in {"buy", "buy/close", "close-buy", "chiudi-buy"}:     return "buy"
        if t == "order":
            if l in {"sell", "sell/close", "close-sell"}: return "sell"
            if l in {"buy", "buy/close", "close-buy"}:     return "buy"
        return None

    @staticmethod
    def _opposite_side(side: str) -> str:
        return "sell" if side == "buy" else "buy"

    @staticmethod
    def _normalize_leverage(val: Optional[str]) -> Optional[str]:
        """
        Ritorna 'X:1' SOLO se X>1. Se None o <=1 -> None (non includere il campo leverage).
        """
        if not val:
            return None
        s = str(val).strip()
        # forme esplicite di 1x
        if s in {"1", "1:1", "1.0", "1.0:1"}:
            return None
        if ":" in s:
            left = s.split(":", 1)[0].strip()
            try:
                f = float(left)
            except Exception:
                return None
            return (f"{int(f)}:1" if abs(f - int(f)) < 1e-9 else f"{f}:1") if f > 1 else None
        try:
            f = float(s)
        except Exception:
            return None
        if f <= 1:
            return None
        return f"{int(f)}:1" if abs(f - int(f)) < 1e-9 else f"{f}:1"

    # ----------------- corpo principale: build -----------------
    def _resolve_pair(self, h: str) -> Optional[str]:
        """Mappa eventuale alias -> pair Kraken"""
        if not h:
            return None
        return self.pair_map.get(h, h)

    def build_bodies(
        self,
        actions: List[Any],
        *,
        pair_map: Optional[Dict[str, str]] = None,
        leverage: Optional[str] = None,
        validate: bool = False,
        post_only: bool = False,
        time_in_force: Optional[str] = None,
        auto_brackets: bool = False,  # PER REGOLA: non creiamo più ordini SL/TP separati
    ) -> List[dict]:
        if pair_map:
            self.pair_map.update(pair_map)

        bodies: List[dict] = []

        for a_raw in actions:
            a = self._as_action_dict(a_raw)

            # CANCEL: genera body di annullamento e passa oltre
            cancel_id = a.get("cancel_order_id")
            if isinstance(cancel_id, str):
                cancel_id = cancel_id.strip()
            if cancel_id:
                bodies.append({"_op": "cancel", "txid": cancel_id, "_meta": {"from": "Action/cancel"}})
                continue

            if (a.get("tipo") == "hold") or not a.get("pair"):
                continue

            pair_h = a.get("pair")
            kr_pair = self._resolve_pair(pair_h)
            if not kr_pair:
                continue

            tipo        = a.get("tipo")
            quando      = a.get("quando")
            lato        = a.get("lato")
            prezzo      = a.get("prezzo")
            limite      = a.get("limite")
            break_price = a.get("break_price")
            stop_loss   = a.get("stop_loss")
            take_profit = a.get("take_profit")
            volume      = a.get("quantita")
            tf          = a.get("timeframe")
            motivo      = a.get("motivo")
            lev         = self._normalize_leverage(a.get("leverage") or leverage)

            side = self._side_from_action(tipo, lato)
            if side is None:
                continue

            # close condition: vince sempre lo stop_loss se presenti entrambi
            close_ordertype = None
            close_price = None
            if stop_loss is not None:
                close_ordertype = "stop-loss"
                close_price = stop_loss
            elif take_profit is not None:
                close_ordertype = "take-profit"
                close_price = take_profit

            body = {"pair": kr_pair, "type": side}
            if lev:
                body["leverage"] = lev
            if validate:
                body["validate"] = True
            if post_only:
                body["oflags"] = "post"
            if time_in_force:
                body["timeinforce"] = time_in_force

            q = (str(quando or "")).lower()
            if q in ("market", "adesso"):
                body["ordertype"] = "market"
            elif q in ("limit", "al_limite"):
                body["ordertype"] = "limit"
                px = limite if (limite is not None) else prezzo
                if px is None:
                    continue
                body["price"] = px
            elif q in ("stop", "al_break"):
                trig = break_price if (break_price is not None) else prezzo
                if trig is None:
                    continue
                if limite is not None:
                    body["ordertype"] = "stop-loss-limit"
                    body["price"] = trig      # trigger
                    body["price2"] = limite   # limite
                else:
                    body["ordertype"] = "stop-loss"
                    body["price"] = trig
            else:
                continue

            # volume: NON modifichiamo il valore
            if volume is not None:
                body["volume"] = volume

            # reduce-only
            reduce_only = a.get("reduce_only")
            if not reduce_only and isinstance(a.get("meta"), dict):
                reduce_only = a["meta"].get("reduce_only")
            if reduce_only:
                body["reduce_only"] = True

            # Conditional close: UNA SOLA, attaccata all'ordine principale
            if close_ordertype and close_price is not None:
                body["close[ordertype]"] = close_ordertype
                body["close[price]"] = close_price
                # se mai userai stop-limit/take-profit-limit, potrai aggiungere anche close[price2]

            body["_meta"] = {"from": "Action", "timeframe": tf, "motivo": motivo}
            # Se l'azione porta già un decision id (es. creato dal TRM/planner), propagalo
            dec_id = a.get("decision_id")
            if not dec_id and isinstance(a.get("meta"), dict):
                dec_id = a["meta"].get("decision_id")
            if dec_id:
                body["_decision_id"] = str(dec_id)
            bodies.append(body)

        return bodies

    # ----------------- helper API/Pairs -----------------
    def _get_api_from_env(self):
        api_key = os.environ.get("KRAKEN_KEY")
        api_secret = os.environ.get("KRAKEN_SECRET")
        if krakenex is None:
            raise RuntimeError("krakenex non disponibile")
        k = krakenex.API()
        k.key = api_key
        k.secret = api_secret
        return k

    def _ensure_pairs_info(self, k: KrakenAPI) -> None:
        if self._pairs_cache:
            return
        try:
            resp = k.query_public("AssetPairs")
            self._pairs_cache = (resp.get("result") or {})
        except Exception:
            self._pairs_cache = {}

    def _split_base_quote(self, pair: str) -> Optional[tuple[str,str]]:
        d = self._pairs_cache.get(pair)
        if not d:
            return None
        return d.get("base"), d.get("quote")

    def _pair_min_volume(self, pair: str) -> float:
        d = self._pairs_cache.get(pair) or {}
        try:
            return float(d.get("ordermin") or d.get("lot_mininum") or 0.0)
        except Exception:
            return 0.0

    def _pair_min_cost(self, pair: str) -> Optional[float]:
        d = self._pairs_cache.get(pair) or {}
        try:
            c = d.get("costmin")
            return float(c) if c is not None else None
        except Exception:
            return None

    def _lot_step(self, pair: str) -> float:
        d = self._pairs_cache.get(pair) or {}
        prec = int(d.get("lot_decimals", 8))
        return 10 ** (-prec)

    def _pair_max_leverage(self, pair: str, side: str) -> Optional[float]:
        d = self._pairs_cache.get(pair) or {}
        key = "leverage_sell" if side == "sell" else "leverage_buy"
        arr = d.get(key) or []
        try:
            return max(float(x) for x in arr) if arr else None
        except Exception:
            return None

    def _available_amounts(self, k: KrakenAPI) -> Dict[str, float]:
        try:
            resp = k.query_private("Balance")
            bals = (resp.get("result") or {})
            return {k: float(v) for k, v in bals.items()}
        except Exception:
            return {}

    def _ticker_mid_price(self, k: KrakenAPI, pair: str) -> Optional[float]:
        try:
            r = k.query_public("Ticker", {"pair": pair}) or {}
            t = (r.get("result") or {})
            v = list(t.values())[0]
            a = float(v["a"][0]); b = float(v["b"][0])
            return (a + b) / 2.0
        except Exception:
            return None

    # ----------------- esecuzione -----------------
    def execute_bodies(self, bodies: list[dict], *, timeout: float = 0.5) -> list[dict]:
        """
        Esegue soltanto: CancelOrder/AddOrder.
        Nessuna logica extra.
        """
        k = getattr(self, "api", None)
        if k is None:
            k = self._get_api_from_env()
            self.api = k

        results: list[dict] = []

        for i, body in enumerate(bodies, 1):
            try:
                # CANCEL
                if body.get("_op") == "cancel":
                    txid = body.get("txid")
                    if not txid:
                        results.append({"error": ["missing txid for cancel"], "result": None})
                    else:
                        try:
                            resp = k.query_private("CancelOrder", {"txid": txid})
                        except Exception as e:
                            resp = {"error": [str(e)], "result": None}
                        results.append(resp)
                    if i < len(bodies) and timeout and timeout > 0:
                        time.sleep(timeout)
                    continue

                # Pulisci meta e invia esattamente i valori passati (senza cast/rounding)
                payload = {kk: vv for kk, vv in body.items() if not kk.startswith("_")}

                try:
                    resp = k.query_private("AddOrder", payload)
                except Exception as e:
                    resp = {"error": [str(e)], "result": None}

                echo = {
                        "pair": body.get("pair"),
                        "type": body.get("type"),
                        "ordertype": body.get("ordertype"),
                        "volume": body.get("volume"),
                        "price": body.get("price"),
                        "price2": body.get("price2"),
                        "reduce_only": body.get("reduce_only"),
                        "_decision_id": body.get("_decision_id"),
                }
                # prova ad estrarre un order id/txid dalla risposta Kraken
                order_id = None
                try:
                    r = resp.get("result") or {}
                    tx = r.get("txid")
                    if isinstance(tx, list) and tx:
                        order_id = tx[0]
                    elif isinstance(tx, str):
                        order_id = tx
                except Exception:
                    pass
                resp = {
                    "error": resp.get("error") or [],
                    "result": resp.get("result"),
                    "_echo": {**echo, "order_id": order_id},
                }
                results.append(resp)

                if i < len(bodies) and timeout and timeout > 0:
                    time.sleep(timeout)

            except Exception as e:
                results.append({"error": [f"runner exception: {e!s}"], "result": None})

        return results
