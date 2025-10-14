# adapter_pipeline_main.py
import asyncio
import time, os, json, math, threading, random
from dataclasses import asdict, is_dataclass
from typing import Sequence, Tuple, Dict, Any, List, Optional
from pathlib import Path
from typing import Iterator, Tuple
import pandas as pd
# from darts.models import TFTModel
# from darts.models import XGBModel
# from darts.models import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastCroston
# importa le TUE classi/funzioni
from bot3 import Bot
import torch
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass                             # <-- il tuo Bot
# from Class.ChatGpt import CurrencyAnalyzerForYourPayloadV3 as Analyzer


try:
    import torch
    import time
    from Aiensemble import TORCH_DEVICE   # se in altro modulo, importa il tuo helper

    x = torch.randn(8192, 8192)
    t0 = time.time()
    y = (x @ x).sum()
    y.backward() if y.requires_grad else None
    dt_cpu = time.time() - t0

    x = x.to(TORCH_DEVICE)
    t0 = time.time()
    y = (x @ x).sum()
    y.backward() if y.requires_grad else None
    dt_gpu = time.time() - t0

    print(f"[check] CPU matmul ~ {dt_cpu:.3f}s  |  {TORCH_DEVICE} matmul ~ {dt_gpu:.3f}s")
except Exception as e:
    print("[check] skip:", e)

# Import delle classi locali
from Class.InfoMarket import InfoMarket, InfoMarket2
from Class.KrakenPortfolio import KrakenPortfolio
from Class.StrategyEngine import StrategyEngine
# from Class.ChatGpt import TradeAction
from Class.Aiensemble import AIEnsembleTrader, train_lgbm_offline, NeuralStrategy, Strategy, TFTStrategy, train_tft_offline
from Class.KrakenOrderRunner import KrakenOrderRunner
from Pipeline import TradePipeline, PipelineConfig, StageConfig

from Class.OpenOrder import JsonObjectIO, Currency
RES_STATE_PATH = os.path.join(os.getcwd(), "ai_features_state.json")
_RES_IDX = None  # cache

_RES_RANGES = ["6H", "24H", "48H", "7D", "1M", "6M", "1Y"]
_RES_FEATNAMES = sum([[f"dist_top_{r}", f"dist_bot_{r}", f"bandw_{r}", f"pos_in_band_{r}"] for r in _RES_RANGES], [])
def _asdict(x):
    return asdict(x) if is_dataclass(x) else x



import numpy as np  # se non già importato

def _index_res_state(state: dict) -> dict:
    """state['per_currency'][pair] -> list di {ts, max_resistance, min_support}  ==>  dict[pair][ts]=(top,bot)"""
    out = {}
    for pair, items in (state.get("per_currency") or {}).items():
        d = {}
        for it in items or []:
            ts = it.get("ts")
            if not ts:
                continue
            d[ts] = (it.get("max_resistance"), it.get("min_support"))
        out[pair] = d
    return out

def _ensure_res_index(path: str = RES_STATE_PATH) -> dict:
    global _RES_IDX
    if _RES_IDX is not None:
        return _RES_IDX
    try:
        state = safe_read_json(path)
    except Exception:
        state = {}
    _RES_IDX = _index_res_state(state)
    return _RES_IDX

def _pick_pair_from_row_like(row: dict) -> str:
    p = row.get("pair")
    if p:
        return p
    b = row.get("base"); q = row.get("quote")
    if b and q:
        return f"{b}/{q}"
    return str(p or "")

def _pick_price_now_from_row_like(row: dict) -> float:
    # prova varie colonne tipiche del tuo dataset offline
    for k in ("price_now","current_price","last","close","open","mid","px"):
        v = row.get(k)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except Exception:
            continue
    return 0.0

def add_resistance_features_to_df(df: "pd.DataFrame", res_state_path: str = RES_STATE_PATH) -> "pd.DataFrame":
    """Aggiunge (se mancanti) le colonne di resistenza al DataFrame e le calcola riga-per-riga usando ai_features_state.json."""
    idx = _ensure_res_index(res_state_path)

    # crea colonne se mancano
    for r in _RES_RANGES:
        for c in ("dist_top","dist_bot","bandw","pos_in_band"):
            col = f"{c}_{r}"
            if col not in df.columns:
                df[col] = 0.0

    # calcolo per riga
    def _calc_row(row):
        pair = _pick_pair_from_row_like(row)
        lvls = idx.get(pair, {})
        px = _pick_price_now_from_row_like(row)
        out = {}
        for R in _RES_RANGES:
            top, bot = (None, None)
            if R in lvls:
                t, b = lvls[R]
                top = float(t) if isinstance(t, (int, float)) else None
                bot = float(b) if isinstance(b, (int, float)) else None
            band = (top - bot) if (top is not None and bot is not None) else None

            dist_top = (px - top) / px if (top is not None and px) else np.nan
            dist_bot = (px - bot) / px if (bot is not None and px) else np.nan
            bandw    = (band / px)    if (band is not None and px) else np.nan
            pos      = ((px - bot) / (band + 1e-12)) if (band is not None and bot is not None) else np.nan

            out[f"dist_top_{R}"] = 0.0 if (dist_top is None or np.isnan(dist_top)) else float(dist_top)
            out[f"dist_bot_{R}"] = 0.0 if (dist_bot is None or np.isnan(dist_bot)) else float(dist_bot)
            out[f"bandw_{R}"]    = 0.0 if (bandw is None or np.isnan(bandw)) else float(bandw)
            out[f"pos_in_band_{R}"] = 0.0 if (pos is None or np.isnan(pos)) else float(pos)
        return pd.Series(out)

    # applica in blocco (veloce)
    df[_RES_FEATNAMES] = df.apply(_calc_row, axis=1)
    return df

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
#  STADIO 2: JUDGE (LLM)
# ========================
# async def judge_fn(item: Tuple[int, Sequence[str], Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Riceve: (batch_id, symbols, market_payload_per_symbol)
#     Chiama il tuo Analyzer (ChatGPT) sul sotto-payload e torna un risultato
#     compatto: es. {"actions": [...], "scores": {...}}.
#     """
#     batch_id, symbols, per_symbol_payload = item

#     # Costruisci il “merged parziale” per l’analyzer
#     merged_partial = [per_symbol_payload[s] for s in symbols if s in per_symbol_payload]

#     # Usa il tuo analyzer come nel main
#     # actions = analyzer.analyze(merged_partial, rischio=6)     # lista di Action
#     actions =[]     # lista di Action

#     # (facoltativo) converti per logging / serializzazione
#     actions_json = [asdict(a) for a in actions]

#     # Se vuoi anche “scores” (dipende dal tuo Analyzer)
#     scores = {a["symbol"]: a.get("score") for a in actions_json if "symbol" in a}

    # return {
    #     "batch_id": batch_id,
    #     "symbols": list(symbols),
    #     "actions": actions,           # mantieni oggetti originali per run_ai
    #     "actions_json": actions_json, # comodo per debug
    #     "scores": scores
    # }

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
im2 = InfoMarket2(per_run=10, total=400, quote="EUR", verbose=True, only_positions=False)
source_aiter = im2.stream_async()   # <--- async iterator di batch da 15 oggetti

from dataclasses import asdict

async def judge_fn(item):
    batch_id, batch_objs, payload = item  # payload == batch_objs
    symbols = [obj.get("pair") for obj in batch_objs]
    print(f"[judge_fn#{batch_id}] start su {len(symbols)}: {symbols}")

    # analyzer.analyze è sincrona? Spostala in thread per non bloccare l'event loop
    # actions = await asyncio.to_thread(analyzer.analyze, batch_objs, 6)
    actions =[]
    actions_json = [asdict(a) for a in actions]
    scores = {a.get("symbol") or a.get("pair"): a.get("score") for a in actions_json if isinstance(a, dict)}
    for a in actions_json:
        print(a)

    print(f"[judge_fn#{batch_id}] done → {len(actions_json)} azioni")
    return {"batch_id": batch_id, "symbols": symbols, "actions": actions, "actions_json": actions_json, "scores": scores}

ALL_ACTIONS = []; ALL_SCORES = {}; ALL_RUN_AI_OUTPUTS = []; ALL_EXEC_REPORTS = []

ai = AIEnsembleTrader(
    inputStorico="storico_input",
    outputStorico="storico_output",
    per_trade_cap_eur=20.0,
    debug_signals=True,
    cash_floor_pct=0.07,     # viene accettato ma non usato (compatibilità)
)

async def deliver_fn(item):
    batch_id, batch_objs, judged = item
    symbols = [obj.get("pair") for obj in batch_objs]
    print(f"[deliver_fn#{batch_id}] start per {symbols}")

    ai.refresh_portfolio()
    res_ai = ai.run(currencies=batch_objs, actions=judged["actions"], replace=True)
    actions = res_ai["actions_ai"]

    print(res_ai["scores"])

    runner = KrakenOrderRunner(pair_map={s: s for s in symbols})
    bodies = runner.build_bodies(res_ai["actions_ai"], validate=False)
    test = runner.execute_bodies(bodies, timeout=0.8)
    report = ai.update_weights_from_kraken(actions_ai=actions, lookback_hours=72)

    print('report')

    print(report)
    print('report end')

    print('bodies')

    for acts in bodies:
        print(acts)

    print('teswt')

    for tests in test:
        print(tests)

    ALL_ACTIONS.extend(judged["actions_json"])
    ALL_RUN_AI_OUTPUTS.append(res_ai)
    ALL_EXEC_REPORTS.append({"batch_id": batch_id, "bodies": bodies, "test": test})
    for k, v in (judged.get("scores") or {}).items():
        if k: ALL_SCORES[k] = v
    print(f"[deliver_fn#{batch_id}] done — bodies={len(bodies)}")

cfg = PipelineConfig(
    batch_size=10,
    max_in_flight_batches=4,
    judge_cfg=StageConfig(concurrency=2, timeout=200, retries=1),
    deliver_cfg=StageConfig(concurrency=2, timeout=90, retries=1),
    logger=print,
)



# ------ utils: JSON folder walker (oldest -> newest) ------

def safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_sibling_folder(rel_folder: str) -> str:
    """
    Restituisce il percorso assoluto di una cartella 'rel_folder' allo stesso
    livello del file eseguìto (cwd), creandola se non esiste.
    Esempio: ensure_sibling_folder("aiConfig")
    """
    base = Path(os.getcwd())
    p = base / rel_folder
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def iter_json_oldest_to_newest(folder: str, pattern: str = "*.json"
                               ) -> Iterator[Tuple[str, object]]:
    """
    Scorre TUTTI i file JSON in 'folder' in ordine dal meno recente al più recente,
    facendo il parse di ciascuno. Ritorna (path, data). I file illeggibili vengono
    semplicemente saltati.
    """
    p = Path(folder)
    if not p.exists():
        # se non esiste la creo per aderenza alla tua richiesta
        p.mkdir(parents=True, exist_ok=True)
        return iter(())  # cartella vuota

    files = sorted(p.glob(pattern), key=lambda fp: fp.stat().st_mtime)
    for fp in files:
        try:
            data = safe_read_json(str(fp))
        except Exception:
            continue  # salta file corrotti/non leggibili
        yield str(fp), data

def load_all_json_oldest_to_newest(folder: str, pattern: str = "*.json") -> list:
    """Comodo se vuoi direttamente una lista dei payload JSON in ordine temporale."""
    return [data for _, data in iter_json_oldest_to_newest(folder, pattern)]

def _make_regular_timeseries_from_df(df, freq="min"):
    import pandas as pd
    from darts import TimeSeries

    # tieni solo le colonne utili
    df = df[["ts", "price"]].copy()

    # 1) numerico -> datetime UTC e poi rimuovi tz (naive)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"])
    dt = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    df["ts"] = dt.dt.tz_convert(None)     # niente timezone

    # 2) ordina, allinea alla griglia (floor) e de-duplica
    df = df.sort_values("ts")
    df["ts"] = df["ts"].dt.floor(freq)
    df = df.groupby("ts", as_index=False).last()

    # 3) ricostruisci indice continuo con la stessa freq e riempi i buchi (ffill)
    idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq)
    s = df.set_index("ts").reindex(idx)
    s["price"] = s["price"].astype(float).ffill().bfill()
    s = s.reset_index().rename(columns={"index": "ts"})
    print(type(dt), getattr(dt, 'dtype', None))

    # 4) crea la TimeSeries già regolare (freq implicita nell’indice)
    return TimeSeries.from_dataframe(s, time_col="ts", value_cols="price", fill_missing_dates=False)

pipe = TradePipeline(cfg)

if __name__ == "__main__":

    cfg_dir = ensure_sibling_folder("currency")  # crea se manca
    neural = next(s for s in ai.strategies if s.name=="Neural")
    for path, snap in iter_json_oldest_to_newest(cfg_dir, pattern="*.json"):
        # fai le tue elaborazioni con 'payload'
        neural.warmup_scaler_from_currencies(snap, features="mtf")
        pairs = [c.get("pair") for c in (snap or []) if isinstance(c, dict)]
        pass

    # training lgbm

    train_lgbm_offline("aiConfig/dataset_lgbm.csv", "aiConfig/lgbm_model.pkl")

    # training Neural

    df = pd.read_csv("aiConfig/dataset_lgbm.csv")
    # 1) target
    if "ret24h" not in df and "ret_h" in df:
        df["ret24h"] = df["ret_h"]
    # 2) f_mtf_* dalle 13 feature
    feat = ["ch24","ch48","dev_1h","dev_4h","vol_dev","atr1h","b1","b4",
            "spread_ratio","slip_avg","or_ok","or_pos","or_w"]
    for i, c in enumerate(feat):
        df[f"f_mtf_{i:02d}"] = df[c].astype(float)
    df.to_csv("aiConfig/dataset_for_neural.csv", index=False)
    s = NeuralStrategy(hidden=32, epochs=80, lr=1e-3)
    print(s.nn)
    s.fit(pd.read_csv("aiConfig/dataset_for_neural.csv"))


    # === AUGMENT DATASET LGBM CON FEATURE DI RESISTENZE ===
    ds_path = "aiConfig/dataset_lgbm.csv"
    if os.path.exists(ds_path):
        df_ds = pd.read_csv(ds_path)
        df_ds = add_resistance_features_to_df(df_ds, res_state_path=RES_STATE_PATH)
        # opzionale: assicurati che tutte le colonne base esistano (fallback)
        base_feat = ["ch24","ch48","dev_1h","dev_4h","vol_dev","atr1h","b1","b4",
                    "spread_ratio","slip_avg","or_ok","or_pos","or_w"]
        for col in base_feat:
            if col not in df_ds.columns:
                df_ds[col] = 0.0
        # salva aggiornato: il trainer ora vedrà anche le nuove colonne
        df_ds.to_csv(ds_path, index=False)
    else:
        print(f"[WARN] Dataset LGBM non trovato: {ds_path} — salto l'augment.")


    train_lgbm_offline("aiConfig/dataset_lgbm.csv", "aiConfig/lgbm_model.pkl")
    from pandas.errors import EmptyDataError
    import os, pandas as pd

    # import os, torch
    # torch.set_num_threads(max(1,  os.cpu_count()//2))
    # torch.set_num_interop_threads(2)
    # os.environ["OMP_NUM_THREADS"]=str(max(1, os.cpu_count()//2))
    # os.environ["MKL_NUM_THREADS"]=os.environ["OMP_NUM_THREADS"]
    print(os.cpu_count() )
    hist_dir = "aiConfig/tft_hist"
    for fn in os.listdir(hist_dir):
        if not fn.endswith(".csv"):
            continue
        p = os.path.join(hist_dir, fn)

        # se il file è vuoto → inizializza con header corretto e passa oltre
        if os.path.getsize(p) == 0:
            pd.DataFrame(columns=["ts", "price"]).to_csv(p, index=False)
            continue

        try:
            df = pd.read_csv(p, dtype=str)
        except EmptyDataError:
            pd.DataFrame(columns=["ts", "price"]).to_csv(p, index=False)
            continue

        # normalizzazione base
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["ts", "price"])
        df.to_csv(p, index=False)

    rep = ai.training(
        input_folder="storico_input",
        adjust_pct=0.25,
        apply=True,
        log_path="storico_output/training_log.json",
        verbose=True
    )



    # train_tft_offline("aiConfig/tft_hist", "aiConfig/tft_models", horizon=48, n_epochs=50)
    # calibrate weight



