# adapter_pipeline_main.py
import asyncio
import time, os, json, math, threading, random, sys
from dataclasses import asdict, is_dataclass
from typing import Sequence, Tuple, Dict, Any, List, Optional
from Class.Util import reconcile_shadow_with_query_orders
# # root del progetto = cartella padre della cartella in cui si trova main.py
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# PKG  = os.path.join(ROOT, "_packages")

# if PKG not in sys.path:
#     sys.path.insert(0, PKG)
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# print("HAS _packages:", any(p.endswith("\BotInvestimentoNuovo\_packages") for p in sys.path))
# for p in sys.path:
#     if p.endswith("\BotInvestimentoNuovo\_packages"): print("PKG:", p)


# importa le TUE classi/funzioni
from bot3 import Bot                              # <-- il tuo Bot
# from Class.ChatGpt import CurrencyAnalyzerForYourPayloadV3 as Analyzer

# Import delle classi locali
from Class.InfoMarket import InfoMarket, InfoMarket2
from Class.KrakenPortfolio import KrakenPortfolio
from Class.StrategyEngine import StrategyEngine
# from Class.ChatGpt import TradeAction
from Class.Aiensemble import AIEnsembleTrader
from Class.KrakenOrderRunner import KrakenOrderRunner
from Pipeline import TradePipeline, PipelineConfig, StageConfig

from Class.OpenOrder import JsonObjectIO, Currency


def _asdict(x):
    return asdict(x) if is_dataclass(x) else x

def _nonnull(x, y):
    """Preferisci y se non None/empty, altrimenti x."""
    return y if (y is not None and y != {}) else x

def _merge_info(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for rng, val in (b or {}).items():
        out[rng] = val  # b sovrascrive stesso range
    return out

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

def _dedupe(items: List[Dict[str, Any]], keyfunc) -> List[Dict[str, Any]]:
    seen = {}
    for it in items or []:
        k = keyfunc(it)
        if k is None:
            # se non ho chiave, mantieni l'ordine e tieni il primo "shape" diverso
            k = json.dumps(it, sort_keys=True, default=str)
        seen[k] = it
    return list(seen.values())

def _merge_open_orders(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def k(o):
        return (
            (o.get("kr_pair") or o.get("pair") or "").upper(),
            o.get("type"), o.get("ordertype"),
            o.get("price"), o.get("price2"), o.get("vol_rem"),
        )
    return _dedupe((a or []) + (b or []), k)


def _count_fields(d: Dict[str, Any]) -> int:
    return sum(1 for v in (d or {}).values() if v not in (None, {}, []))

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

# ========================
# SETUP OGGETTI "GLOBALI"
# ========================
# Nota: tienili fuori dalle funzioni per riutilizzare connessioni/cache
bot = Bot(pair=("IM_PAIR", "BTC/EUR"), quote="EUR")

# analyzer = Analyzer(
#     model="gpt-5-nano",  #gpt-5,gpt-5-nano,gpt-4.1-mini
#     request_timeout_s=300,
#     reserve_eur=50,
#     risk_level=5,
#     prefer_limit=True,
#     or_min_atr_mult=0.5,
#     spread_max_pct=0.5,
#     min_volume_label="Medium",
#     sl_atr_mult=1.5,
#     tp_atr_mult=2.5,
#     enable_shorts=True,
# )

# Se hai già un portafoglio / lista valute:
# arr1 = bot.run_info_market("ALL", 5)         # (esempio tuo)
# nwest = bot.run_kraken_portfolio()            # (se ti serve)
# currencies = bot.export_currencies()          # lista dei tuoi asset/valute
# merged = merge_currency_lists(arr1, currencies)   # payload “grande” (dict/list)

# im2 = InfoMarket2(per_run=15, total=45, quote="EUR", verbose=True)
# merged = im2.run()  # lista di oggetti nel formato compatibile col tuo JSON


# Indice rapido per simbolo → dato merged (comodo per batch)
# Adatta la chiave in base a come sono strutturati i tuoi item.
def _key_of(x: Dict[str, Any]) -> str:
    # es.: "AEVO/EUR" oppure campo "symbol"
    return x.get("pair") or x.get("symbol") or x["name"]

# MERGED_BY_SYMBOL: Dict[str, Any] = {
#     _key_of(x): x for x in merged
# }

# ========================
#  STADIO 1: FETCH (50)
# ========================
async def fetch_batch_fn(symbols: Sequence[str]) -> Dict[str, Any]:
    """
    Ottiene i dati necessari per ANALYZE per i soli simboli del batch.
    - Se usi WebSocket con cache, leggi da lì.
    - In alternativa, filtra dal 'merged' già pronto.
    RITORNA: dict {symbol: payload_mercato}
    """
    # Se hai un MarketCache live: return market_cache.snapshot(symbols)
    # Qui, per semplicità, ricostruiamo dal MERGED_BY_SYMBOL
    out = {}
    for s in symbols:
        if s in MERGED_BY_SYMBOL:
            out[s] = MERGED_BY_SYMBOL[s]
    return out


# ========================
#  STADIO 3: DELIVER (AI)
# ========================
# Contenitori aggregati sull’intero run
ALL_ACTIONS = []
ALL_SCORES: Dict[str, float] = {}
ALL_RUN_AI_OUTPUTS = []
ALL_EXEC_REPORTS = []

# ========================
#  AVVIO PIPELINE
# ========================
# ... (setup Analyzer ecc.)

# 1) Sorgente STREAMING da InfoMarket2
# istanzia InfoMarket2 in modalità streaming
im2 = InfoMarket2(per_run=1, total=300, quote="EUR", verbose=True, only_positions=False)
source_aiter = im2.stream_async()   # <--- async iterator di batch da 15 oggetti

from dataclasses import asdict

async def judge_fn(item):
    batch_id, batch_objs, payload = item  # payload == batch_objs
    symbols = [obj.get("pair") for obj in batch_objs]
    print(f"[judge_fn#{batch_id}] start su {len(symbols)}: {symbols}")
    # ai.refresh_portfolio()

    # analyzer.analyze è sincrona? Spostala in thread per non bloccare l'event loop
    res_ai = ai.run(currencies=batch_objs, replace=True)
    actions = res_ai["actions_ai"]
    # actions = await asyncio.to_thread(analyzer.analyze, batch_objs, 6)
    # actions =[]
    # print(res_ai["scores"])
    # print("[TRM] azioni:", [ (a.get("pair"), a.get("action"), a.get("price"), a.get("size")) for a in actions ])
    # actions_json = [asdict(a) for a in actions]
    actions_json = []

    scores = {a.get("symbol") or a.get("pair"): a.get("score") for a in actions_json if isinstance(a, dict)}
    # for a in actions_json:
    #     print(a)

    # print(f"[judge_fn#{batch_id}] done → {len(actions_json)} azioni")
    return {"batch_id": batch_id, "symbols": symbols, "actions": actions, "actions_json": actions_json, "scores": scores}

ALL_ACTIONS = []; ALL_SCORES = {}; ALL_RUN_AI_OUTPUTS = []; ALL_EXEC_REPORTS = []

ai = AIEnsembleTrader(
    inputStorico="storico_input",
    outputStorico="storico_output",
    per_trade_cap_eur=25.0,
    debug_signals=True,
    cash_floor_pct=0.07,     # viene accettato ma non usato (compatibilità)
    budget_eur=im2.portfolioIn
)

async def deliver_fn(item):
    batch_id, batch_objs, judged = item
    symbols = [obj.get("pair") for obj in batch_objs]
    print(f"[deliver_fn#{batch_id}] start per {symbols}")

    actions = judged["actions"]
    neural = next(s for s in ai.strategies if s.name == "Neural")

    runner = KrakenOrderRunner(pair_map={s: s for s in symbols})
    bodies = runner.build_bodies(actions, validate=True, auto_brackets=False)
    test = runner.execute_bodies(bodies, timeout=0.8)


    # NUOVO: logga le execution nel JSONL delle shadow actions
    ai.log_execution_to_shadow(actions, test)

    ai.learn_price_size_from_results(bodies, test)  # <-- NOVITÀ

    # === NEW: aggiorna goal/pesi in modo idempotente leggendo SOLO i SELL nuovi di oggi ===
    pnl_delta = ai.update_goal_from_trades_incremental(batch_objs,actions)
    print(f"[deliver_fn#{batch_id}] pnl_delta_goal = {pnl_delta:.2f} EUR (incrementale)")
    report = ai.update_weights_from_kraken(actions_ai=actions, lookback_hours=72)



    for acts in actions:
        print(acts) if acts.get('tipo') != 'hold' else print(f"pair: {acts.get('pair')}, side: {acts.get('side')}")


    for tests in test:
        print(tests)

    ALL_ACTIONS.extend(judged["actions_json"])
    # ALL_RUN_AI_OUTPUTS.append(res_ai)
    ALL_EXEC_REPORTS.append({"batch_id": batch_id, "bodies": bodies, "test": test})
    for k, v in (judged.get("scores") or {}).items():
        if k: ALL_SCORES[k] = v
    print(f"[deliver_fn#{batch_id}] done — bodies={len(bodies)}")

cfg = PipelineConfig(
    batch_size=1,
    max_in_flight_batches=4,
    judge_cfg=StageConfig(concurrency=2, timeout=200, retries=0),
    deliver_cfg=StageConfig(concurrency=2, timeout=90, retries=0),
    logger=print,
)


pipe = TradePipeline(cfg)

async def main():

    await pipe.run_streaming(
        source_aiter=source_aiter,
        judge_fn=judge_fn,
        deliver_fn=deliver_fn,
    )
    krakeapi=ai._get_kraken_api_from_env()
    base_dir = os.path.dirname(__file__)            # directory dove si trova main.py
    log_path = os.path.join(base_dir, "storico_output", "trm_log", "shadow_actions.jsonl")
    n = reconcile_shadow_with_query_orders(log_path, krakeapi, sleep_s=0.1)
    print(f"[reconcile] aggiornati {n} record")


    print("\n[pipeline] COMPLETATA")
    print("Tot actions:", len(ALL_ACTIONS))
    print("Tot batch AI:", len(ALL_RUN_AI_OUTPUTS))

if __name__ == "__main__":
    asyncio.run(main())




