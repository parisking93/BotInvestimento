# TimeSfm.py
# Forecast BTC/EUR con Google TimesFM 2.5 (PyTorch) + salvataggio JSON per TRM.
# Output: ../storico_output/timesfm/BTC_EUR.json
# Contiene:
#   meta: pair, ts_unix, last_price, horizon, dates[T+1..T+H], p10/p50/p90
#   features: edge/uncert/signal (numerici, nessun riempitivo)

from __future__ import annotations
import os, json, time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import timesfm  # google-research/timesfm (PyTorch)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import shutil
# --- ADD: in cima ai import ---
try:
    from InfoMarket import InfoMarket
except Exception:
    try:
        from .InfoMarket import InfoMarket
    except Exception:
        from Class.InfoMarket import InfoMarket

import pandas as pd
import time
# --- in cima ---
from typing import Optional


# --- RE-ADD: reader per AIEnsemble ---
from typing import Optional
from pathlib import Path
import os, json



# --- SAFE PUBLIC + DISK CACHE PER ASSETPAIRS ---
from functools import lru_cache
import time, json
from pathlib import Path

_AP_CACHE_PATH = Path(__file__).resolve().parent.parent / "storico_output" / "_cache" / "kr_assetpairs.json"
_AP_CACHE_TTL_S = 6 * 3600  # 6h

def _safe_public(im, method: str, params: dict | None = None, retries: int = 4, base_sleep: float = 1.0) -> dict:
    params = params or {}
    last_err = None
    for i in range(retries):
        try:
            return im._public(method, params)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # backoff aggressivo su 403/forbidden
            if "403" in msg or "forbidden" in msg:
                time.sleep(base_sleep * (2 ** i) + (0.2 * i))
                continue
            time.sleep(0.5 * (i + 1))
    raise last_err

def _load_ap_cache() -> dict | None:
    try:
        if not _AP_CACHE_PATH.exists():
            return None
        with open(_AP_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - data.get("ts", 0) <= _AP_CACHE_TTL_S and "map" in data:
            return data
    except Exception:
        return None
    return None

def _save_ap_cache(mapping: dict) -> None:
    try:
        _AP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AP_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), "map": mapping}, f)
    except Exception:
        pass

@lru_cache(maxsize=1)
def _get_assetpairs_map(im) -> dict:
    """Mappa 'HUMAN wsname' -> 'kr-code' (es. 'BTC/EUR' -> 'XXBTZEUR'), con cache process+disco."""
    # 1) prova cache su disco
    cached = _load_ap_cache()
    if cached:
        return cached["map"]

    # 2) una sola chiamata a AssetPairs, con retry/backoff
    res = _safe_public(im, "AssetPairs", {})
    rows = res.get("result") or {}
    mapping = {}
    for code, row in rows.items():
        ws = row.get("wsname")
        if not ws or "/" not in ws:
            continue
        b, q = ws.split("/", 1)
        b = "BTC" if b.upper() == "XBT" else b.upper()
        mapping[f"{b}/{q.upper()}"] = code

    if mapping:
        _save_ap_cache(mapping)
    return mapping


def load_latest(pair: str, base_dir: Optional[str] = None) -> Optional[dict]:
    """
    Carica l'ultimo JSON TimesFM per la coppia indicata.
    - pair: es. "BTC/EUR" (accetta anche "BTC_EUR")
    - base_dir: cartella che contiene i file .json; se None, usa <repo_root>/storico_output/timesfm
    Ritorna: dict con chiavi {"meta": {...}, "features": {...}} oppure None se non trovato.
    """
    name = (pair or "").replace("/", "_")
    if not name:
        return None

    # default coerente con lo writer di TimeSfm
    search_roots: list[Path] = []
    if base_dir:
        search_roots.append(Path(base_dir))
    # percorso usato dallo script che scrive i file (repo_root/storico_output/timesfm)
    try:
        search_roots.append(Path(__file__).resolve().parent.parent / "storico_output" / "timesfm")
    except Exception:
        pass
    # fallback: cwd/storico_output/timesfm (utile in ambienti diversi)
    try:
        search_roots.append(Path(os.getcwd()) / "storico_output" / "timesfm")
    except Exception:
        pass

    candidates: list[Path] = []
    for root in search_roots:
        try:
            if not root or not root.exists():
                continue
            # match esatto
            f = root / f"{name}.json"
            if f.exists():
                candidates.append(f)
            # eventuali snapshot timestamped tipo BTC_EUR_2025-10-31.json
            candidates.extend(sorted(root.glob(f"{name}_*.json")))
        except Exception:
            continue

    if not candidates:
        return None

    # prendi il più recente per mtime
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with open(best, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            data.setdefault("meta", {})
            data.setdefault("features", {})
        return data
    except Exception:
        return None


def _kr_code_from_assetpairs(im, human: str) -> Optional[str]:
    mp = _get_assetpairs_map(im)
    return mp.get(human.upper())

# --- ADD: helper Kraken -> Serie Close giornaliera ---
# --- fallback Kraken corretto: usa kr-code da AssetPairs, non _normalize_pair ---
def _kraken_close_series(pair: str, days: int = 5*365) -> Optional[pd.Series]:
    im = InfoMarket(pair, verbose=False, public_qps=1.6)
    since = int(time.time()) - int(days * 86400)
    kr_code = _kr_code_from_assetpairs(im, pair)
    if not kr_code:
        # la coppia in EUR non è su Kraken → None (verrà skippata a monte)
        return None
    data = _safe_public(im, "OHLC", {"pair": kr_code, "interval": 1440, "since": since})
    if data.get("error"):
        return None
    rows = data.get("result", {}).get(kr_code) or []
    if not rows:
        return None
    ts  = [int(r[0]) for r in rows]
    cls = [float(r[4]) for r in rows]
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    s = pd.Series(cls, index=idx).sort_index()
    return s[~s.index.duplicated(keep="last")]

# --- ADD: downloader con fallback ---
def _download_close(PAIR: str, SYM: str) -> pd.Series:
    # 1) tentativo Yahoo
    try:
        df = yf.download(SYM, period="5y", interval="1d", auto_adjust=True, group_by="ticker", progress=False)
        s = _get_close(df, SYM).dropna().astype(float)
        if len(s) >= 16:
            return s
    except Exception:
        pass
    # 2) fallback Kraken
    s = _kraken_close_series(PAIR)
    if s is None or len(s) < 16:
        # segnala a chi chiama che questa pair va skippata
        raise RuntimeError(f"SKIP {PAIR}: non disponibile su Yahoo/Kraken")
    return s
# =============== util ===============
def _finite(x) -> bool:
    a = np.asarray(x, dtype=float)
    return np.isfinite(a).all()

def _dates_Tplus(last_idx: pd.Timestamp, horizon: int) -> List[str]:
    start = (pd.to_datetime(last_idx) + pd.Timedelta(days=1)).normalize()
    return [d.date().isoformat() for d in pd.date_range(start=start, periods=horizon, freq="D")]

def _get_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Selezione robusta della Close (gestisce MultiIndex/variazioni yfinance)."""
    if hasattr(df.columns, "levels"):
        lv0 = set(df.columns.get_level_values(0))
        if "Close" in lv0:
            s = df["Close"]
            return s[ticker] if hasattr(s, "columns") and ticker in s.columns else s.squeeze()
        if "Price" in lv0 and "Close" in getattr(df.get("Price", pd.DataFrame()), "columns", []):
            return df["Price"]["Close"]
        lv1 = set(df.columns.get_level_values(1))
        if ticker in lv0 and "Close" in lv1:
            return df[ticker]["Close"]
        # fallback generico: prendi 'Close' all’ultimo livello
        return df.xs("Close", axis=1, level=df.columns.nlevels - 1)
    return df["Close"]

def _compile(model, ctx_len: int, horizon: int, normalize_inputs: bool, infer_is_positive: bool, flip: bool, qhead: bool):
    model.compile(timesfm.ForecastConfig(
        max_context=ctx_len,
        max_horizon=horizon + 1,
        normalize_inputs=normalize_inputs,
        use_continuous_quantile_head=qhead,
        force_flip_invariance=flip,
        infer_is_positive=infer_is_positive,
        fix_quantile_crossing=True,
    ))

def _pick_qcols(qn: np.ndarray) -> Tuple[int, int]:
    """
    Sceglie colonne ~10% e ~90% senza assumere indici fissi:
    ordina le colonne per media e prende i rank 10% e 90%.
    """
    m = qn.shape[1]
    means = qn.mean(axis=0)
    order = np.argsort(means)  # dal più basso al più alto
    i_p10 = int(round(0.10 * (m - 1)))
    i_p90 = int(round(0.90 * (m - 1)))
    return int(order[i_p10]), int(order[i_p90])

def _as_np(x):
    # timesfm può dare torch.Tensor; riportiamo sempre a np.ndarray
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# =============== core ===============
def _run_timesfm_all(ctx: np.ndarray, horizon: int, ctx_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prova più combinazioni sicure (Windows/CPU-friendly).
    Ritorna (p50, qmat) in PREZZI, dove qmat ha colonne [p10, p50, p90].
    """
    torch.set_float32_matmul_precision("medium")  # più sicuro su CPU
    try:
        torch.set_num_threads(max(1, int(os.environ.get("T_ORCH_THREADS", "1"))))
        torch.backends.mkldnn.enabled = False  # evita edge-case NaN su alcune build
    except Exception:
        pass

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    trials = [
        # (norm_self, norm_flag, pos, flip, qhead)
        (True,  False, False, True,  True),   # A) standardizzato da noi (può avere negativi) -> pos=False
        (False, True,  True,  True,  True),   # B) RAW positiva, normalizza TimesFM
        (True,  False, False, False, True),   # C) come A senza flip invariance
        (False, True,  True,  False, True),   # D) come B senza flip invariance
        (True,  False, False, True,  False),  # E) come A senza quantile head continua
        (False, True,  True,  True,  False),  # F) come B senza quantile head continua
    ]

    scaler = StandardScaler()
    norm = scaler.fit_transform(ctx.reshape(-1, 1)).flatten().astype(np.float32)

    for norm_self, norm_flag, pos, flip, qhead in trials:
        _compile(model, ctx_len, horizon, normalize_inputs=norm_flag, infer_is_positive=pos, flip=flip, qhead=qhead)
        try:
            with torch.no_grad():
                if norm_self:
                    point, quants = model.forecast(horizon=horizon, inputs=[norm])
                    p50n = _as_np(point[0])    # (H,)
                    qn   = _as_np(quants[0])   # (H, Q)
                    j10, j90 = _pick_qcols(qn)
                    p50 = scaler.inverse_transform(p50n.reshape(-1, 1)).flatten()
                    p10 = scaler.inverse_transform(qn[:, j10].reshape(-1, 1)).flatten()
                    p90 = scaler.inverse_transform(qn[:, j90].reshape(-1, 1)).flatten()
                else:
                    point, quants = model.forecast(horizon=horizon, inputs=[ctx.astype(np.float32)])
                    p50 = _as_np(point[0])
                    qn  = _as_np(quants[0])
                    j10, j90 = _pick_qcols(qn)
                    p10 = qn[:, j10]; p90 = qn[:, j90]
        except Exception:
            continue

        if _finite(p50) and _finite(p10) and _finite(p90):
            # garantisci monotonicità quantili (p10 <= p50 <= p90) per ogni t
            p10c = np.minimum.reduce([p10, p50, p90])
            p90c = np.maximum.reduce([p10, p50, p90])
            p50c = np.clip(p50, p10c, p90c)
            qmat = np.vstack([p10c, p50c, p90c]).T
            return p50c, qmat

    raise RuntimeError("TimesFM ha prodotto valori non finiti in tutte le prove.")

# =============== main ===============
def main(pairInput, symInput):
    PAIR = pairInput
    SYM  = symInput
    H    = 7
    CTX  = 2048  # 2048 va bene, ma 1024 è più stabile su alcune CPU

    # 1) dati
    # df = yf.download(SYM, period="5y", interval="1d", auto_adjust=True, group_by="ticker")
    # close = _get_close(df, SYM).dropna().astype(float)
    close = _download_close(PAIR, SYM)
    if len(close) < 16:
        raise ValueError("Serie troppo corta per TimesFM (>=16 punti).")

    ctx = close.values[-CTX:] if len(close) > CTX else close.values
    # last = float(ctx[-1])
    last  = float(close.values[-1])
    # 2) forecast (robusto)
    p50, qmat = _run_timesfm_all(ctx, horizon=H, ctx_len=CTX)  # qmat: [p10,p50,p90]
    p10 = qmat[:, 0]; p90 = qmat[:, 2]

    # 3) metriche e meta
    dates = _dates_Tplus(close.index[-1], H)
    edge_1d = (p50[0]  - last) / max(last, 1e-12)
    edge_h  = (p50[-1] - last) / max(last, 1e-12)
    u1      = (p90[0]  - p10[0]) / max(p50[0], 1e-12)
    uh      = float(np.mean((p90 - p10) / np.maximum(p50, 1e-12)))
    signal  = float(np.clip(edge_1d / max(u1, 1e-6), -1.0, 1.0))

    meta: Dict[str, Any] = {
        "pair": PAIR,
        "ts_unix": int(time.time()),
        "last_price": last,
        "horizon": H,
        "dates": dates,          # T+1..T+H
        "p50": p50.tolist(),     # mediana prevista (prezzi futuri)
        "p10": p10.tolist(),     # quantile 10%
        "p90": p90.tolist(),     # quantile 90%
        "edge_1d": float(edge_1d),
        "edge_h": float(edge_h),
        "uncert_1d": float(u1),
        "uncert_h": float(uh),
        "signal": signal,
    }

    features = {
        "timesfm_edge_1d": meta["edge_1d"],
        "timesfm_edge_h":  meta["edge_h"],
        "timesfm_uncert_1d": meta["uncert_1d"],
        "timesfm_uncert_h":  meta["uncert_h"],
        "timesfm_signal":    meta["signal"],
        "timesfm_h": float(H),
    }

        # 4) salva


    # cartella .../<repo_root>/storico_output/timesfm
    # Use current working directory as the base
    base_dir = (Path(__file__).resolve().parent.parent / "storico_output" / "timesfm")
    base_dir.mkdir(parents=True, exist_ok=True)

    out_path = (base_dir / f"{PAIR.replace('/','_')}.json").resolve()  # path assoluto canonicalizzato
    tmp_path = out_path.with_suffix(".json.tmp")

    payload = {"meta": meta, "features": features}

    # 1) scrivi su tmp e forza su disco
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    # 2) sostituzione atomica del file di destinazione
    os.replace(tmp_path, out_path)

    # 3) rileggi dal disco e stampa conferma (deve coincidere con la preview)
    with open(out_path, "r", encoding="utf-8") as f:
        check = json.load(f)

    print("WROTE:", str(out_path))
    print("CHECK p50 (from file):", [round(x, 2) for x in check["meta"]["p50"]])

    # # 4) (opzionale) crea anche uno snapshot datato per ispezione manuale
    # snap_path = base_dir / f"{PAIR.replace('/','_')}.json"
    # shutil.copy2(out_path, snap_path)
    # print("SNAPSHOT:", str(snap_path))

if __name__ == "__main__":
    pairs=[{"pair":"BTC/EUR"},{"pair":"ETH/EUR"},{"pair":"BANANAS31/EUR"},{"pair":"BAND/EUR"},{"pair":"BAT/EUR"},{"pair":"BCH/EUR"},{"pair":"BEAM/EUR"},{"pair":"BERA/EUR"},{"pair":"BERT/EUR"},{"pair":"BICO/EUR"},{"pair":"BIGTIME/EUR"},{"pair":"BILLY/EUR"},{"pair":"BIO/EUR"},{"pair":"BIT/EUR"},{"pair":"BLUAI/EUR"},{"pair":"BLUR/EUR"},{"pair":"BMT/EUR"},{"pair":"BNB/EUR"},{"pair":"BNC/EUR"},{"pair":"BNT/EUR"},{"pair":"BOBA/EUR"},{"pair":"BODEN/EUR"},{"pair":"BOND/EUR"},{"pair":"BONK/EUR"},{"pair":"BOS/EUR"},{"pair":"BRICK/EUR"},{"pair":"BTR/EUR"},{"pair":"BTT/EUR"},{"pair":"C98/EUR"},{"pair":"CAKE/EUR"},{"pair":"CAMP/EUR"},{"pair":"CARV/EUR"},{"pair":"CAT/EUR"},{"pair":"CCD/EUR"},{"pair":"CELO/EUR"},{"pair":"CELR/EUR"},{"pair":"CFG/EUR"},{"pair":"CGN/EUR"},{"pair":"CHEEMS/EUR"},{"pair":"CHILLHOUSE/EUR"},{"pair":"CHR/EUR"},{"pair":"CLANKER/EUR"},{"pair":"CLOUD/EUR"},{"pair":"CLV/EUR"},{"pair":"COMP/EUR"},{"pair":"COOKIE/EUR"},{"pair":"COQ/EUR"},{"pair":"CORN/EUR"},{"pair":"COTI/EUR"},{"pair":"COW/EUR"},{"pair":"CPOOL/EUR"},{"pair":"CQT/EUR"},{"pair":"CRO/EUR"},{"pair":"CRV/EUR"},{"pair":"CSM/EUR"},{"pair":"CTC/EUR"},{"pair":"CTSI/EUR"},{"pair":"CVC/EUR"},{"pair":"CYBER/EUR"},{"pair":"DAI/EUR"},{"pair":"DASH/EUR"},{"pair":"DBR/EUR"},{"pair":"DEEP/EUR"},{"pair":"DEGEN/EUR"},{"pair":"DENT/EUR"},{"pair":"DMC/EUR"},{"pair":"DOG/EUR"},{"pair":"DOGS/EUR"},{"pair":"DOLO/EUR"},{"pair":"DOT/EUR"},{"pair":"DRIFT/EUR"},{"pair":"DRV/EUR"},{"pair":"DUCK/EUR"},{"pair":"DYM/EUR"},{"pair":"EAT/EUR"},{"pair":"EDGE/EUR"},{"pair":"EGLD/EUR"},{"pair":"EIGEN/EUR"},{"pair":"ENA/EUR"},{"pair":"ENJ/EUR"},{"pair":"ENS/EUR"},{"pair":"ENSO/EUR"},{"pair":"EPT/EUR"},{"pair":"ES/EUR"},{"pair":"ETC/EUR"},{"pair":"ETHFI/EUR"},{"pair":"ETHW/EUR"},{"pair":"EUL/EUR"},{"pair":"EURC/EUR"},{"pair":"EUROP/EUR"},{"pair":"EURQ/EUR"},{"pair":"EURR/EUR"},{"pair":"EWT/EUR"},{"pair":"FARM/EUR"},{"pair":"FARTCOIN/EUR"},{"pair":"FET/EUR"},{"pair":"FF/EUR"},{"pair":"FHE/EUR"},{"pair":"FIDA/EUR"},{"pair":"FIL/EUR"},{"pair":"FIS/EUR"},{"pair":"FLOKI/EUR"},{"pair":"FLOW/EUR"},{"pair":"FLR/EUR"},{"pair":"FLY/EUR"},{"pair":"FORTH/EUR"},{"pair":"FWOG/EUR"},{"pair":"G/EUR"},{"pair":"GAIA/EUR"},{"pair":"GAL/EUR"},{"pair":"GALA/EUR"},{"pair":"GARI/EUR"},{"pair":"GFI/EUR"},{"pair":"GHIBLI/EUR"},{"pair":"GHST/EUR"},{"pair":"GIGA/EUR"},{"pair":"GLMR/EUR"},{"pair":"GMT/EUR"},{"pair":"GNO/EUR"},{"pair":"GOAT/EUR"},{"pair":"GOMINING/EUR"},{"pair":"GRASS/EUR"},{"pair":"GRIFFAIN/EUR"},{"pair":"GRT/EUR"},{"pair":"GST/EUR"},{"pair":"GTC/EUR"},{"pair":"GUN/EUR"},{"pair":"H/EUR"},{"pair":"HBAR/EUR"},{"pair":"HFT/EUR"},{"pair":"HIPPO/EUR"},{"pair":"HMSTR/EUR"},{"pair":"HNT/EUR"},{"pair":"HONEY/EUR"},{"pair":"HOUSE/EUR"},{"pair":"HPOS10I/EUR"},{"pair":"ICNT/EUR"},{"pair":"ICP/EUR"},{"pair":"INIT/EUR"},{"pair":"INJ/EUR"},{"pair":"INTR/EUR"},{"pair":"IP/EUR"},{"pair":"JAILSTOOL/EUR"},{"pair":"JASMY/EUR"},{"pair":"JITOSOL/EUR"},{"pair":"JOE/EUR"},{"pair":"JST/EUR"},{"pair":"JTO/EUR"},{"pair":"JUNO/EUR"},{"pair":"JUP/EUR"},{"pair":"KAITO/EUR"},{"pair":"KAR/EUR"},{"pair":"KAS/EUR"},{"pair":"KAVA/EUR"},{"pair":"KEEP/EUR"},{"pair":"KERNEL/EUR"},{"pair":"KET/EUR"},{"pair":"KEY/EUR"},{"pair":"KGEN/EUR"},{"pair":"KIN/EUR"},{"pair":"KINT/EUR"},{"pair":"KMNO/EUR"},{"pair":"KNC/EUR"},{"pair":"KOBAN/EUR"},{"pair":"KP3R/EUR"},{"pair":"KSM/EUR"},{"pair":"KTA/EUR"},{"pair":"L3/EUR"},{"pair":"LAYER/EUR"},{"pair":"LCAP/EUR"},{"pair":"LDO/EUR"},{"pair":"LINEA/EUR"},{"pair":"LINK/EUR"},{"pair":"LION/EUR"},{"pair":"LIT/EUR"},{"pair":"LMWR/EUR"},{"pair":"LOBO/EUR"},{"pair":"LOCKIN/EUR"},{"pair":"LOFI/EUR"},{"pair":"LPT/EUR"},{"pair":"LQTY/EUR"},{"pair":"LRC/EUR"},{"pair":"LSETH/EUR"},{"pair":"LSK/EUR"},{"pair":"LSSOL/EUR"},{"pair":"LTC/EUR"},{"pair":"LUNA/EUR"},{"pair":"LUNA2/EUR"},{"pair":"M/EUR"},{"pair":"MANA/EUR"},{"pair":"MASK/EUR"},{"pair":"MAT/EUR"},{"pair":"MC/EUR"},{"pair":"ME/EUR"},{"pair":"MELANIA/EUR"},{"pair":"MEME/EUR"},{"pair":"MERL/EUR"},{"pair":"METIS/EUR"},{"pair":"MEW/EUR"},{"pair":"MF/EUR"},{"pair":"MICHI/EUR"},{"pair":"MIM/EUR"},{"pair":"MINA/EUR"},{"pair":"MIR/EUR"},{"pair":"MIRA/EUR"},{"pair":"MIRROR/EUR"},{"pair":"MLN/EUR"},{"pair":"MNGO/EUR"},{"pair":"MNT/EUR"},{"pair":"MOCA/EUR"},{"pair":"MOG/EUR"},{"pair":"MOODENG/EUR"},{"pair":"MOON/EUR"},{"pair":"MORPHO/EUR"},{"pair":"MOVE/EUR"},{"pair":"MOVR/EUR"},{"pair":"MSOL/EUR"},{"pair":"MUBARAK/EUR"},{"pair":"MULTI/EUR"},{"pair":"MV/EUR"},{"pair":"NANO/EUR"},{"pair":"NEAR/EUR"},{"pair":"NEIRO/EUR"},{"pair":"NIL/EUR"},{"pair":"NMR/EUR"},{"pair":"NOBODY/EUR"},{"pair":"NODE/EUR"},{"pair":"NODL/EUR"},{"pair":"NOS/EUR"},{"pair":"NOT/EUR"},{"pair":"NPC/EUR"},{"pair":"NTRN/EUR"},{"pair":"NYM/EUR"},{"pair":"OCEAN/EUR"},{"pair":"ODOS/EUR"},{"pair":"OGN/EUR"},{"pair":"OM/EUR"},{"pair":"OMG/EUR"},{"pair":"OMNI/EUR"},{"pair":"ONDO/EUR"},{"pair":"OP/EUR"},{"pair":"OPEN/EUR"},{"pair":"ORCA/EUR"},{"pair":"ORDER/EUR"},{"pair":"OSMO/EUR"},{"pair":"PARTI/EUR"},{"pair":"PDA/EUR"},{"pair":"PEAQ/EUR"},{"pair":"PENDLE/EUR"},{"pair":"PENGU/EUR"},{"pair":"PEPE/EUR"},{"pair":"PERP/EUR"},{"pair":"PHA/EUR"},{"pair":"PIPE/EUR"},{"pair":"PLAY/EUR"},{"pair":"PLUME/EUR"},{"pair":"PNUT/EUR"},{"pair":"POL/EUR"},{"pair":"POLIS/EUR"},{"pair":"POLS/EUR"},{"pair":"POND/EUR"},{"pair":"PONKE/EUR"},{"pair":"POPCAT/EUR"},{"pair":"PORTAL/EUR"},{"pair":"POWR/EUR"},{"pair":"PRCL/EUR"},{"pair":"PRIME/EUR"},{"pair":"PRO/EUR"},{"pair":"PROMPT/EUR"},{"pair":"PROVE/EUR"},{"pair":"PSTAKE/EUR"},{"pair":"PTB/EUR"},{"pair":"PUFFER/EUR"},{"pair":"PUMP/EUR"},{"pair":"PUPS/EUR"},{"pair":"PYTH/EUR"},{"pair":"PYUSD/EUR"},{"pair":"Q/EUR"},{"pair":"QI/EUR"},{"pair":"QNT/EUR"},{"pair":"QTUM/EUR"},{"pair":"RAD/EUR"},{"pair":"RAIIN/EUR"},{"pair":"RARE/EUR"},{"pair":"RARI/EUR"},{"pair":"RAY/EUR"},{"pair":"RBC/EUR"},{"pair":"RED/EUR"},{"pair":"REKT/EUR"},{"pair":"REN/EUR"},{"pair":"RENDER/EUR"},{"pair":"REP/EUR"},{"pair":"REPV2/EUR"},{"pair":"REQ/EUR"},{"pair":"RETARDIO/EUR"},{"pair":"RHEA/EUR"},{"pair":"RLC/EUR"},{"pair":"RLUSD/EUR"},{"pair":"ROOK/EUR"},{"pair":"RPL/EUR"},{"pair":"RSR/EUR"},{"pair":"RUJI/EUR"},{"pair":"RUNE/EUR"},{"pair":"S/EUR"},{"pair":"SAFE/EUR"},{"pair":"SAGA/EUR"},{"pair":"SAHARA/EUR"},{"pair":"SAMO/EUR"},{"pair":"SAND/EUR"},{"pair":"SAPIEN/EUR"},{"pair":"SAROS/EUR"},{"pair":"SBR/EUR"},{"pair":"SC/EUR"},{"pair":"SCA/EUR"},{"pair":"SCRT/EUR"},{"pair":"SDN/EUR"},{"pair":"SEI/EUR"},{"pair":"SGB/EUR"},{"pair":"SHIB/EUR"},{"pair":"SIDEKICK/EUR"},{"pair":"SIGMA/EUR"},{"pair":"SKY/EUR"},{"pair":"SLAY/EUR"},{"pair":"SNEK/EUR"},{"pair":"SOGNI/EUR"},{"pair":"SOL/EUR"},{"pair":"SOLV/EUR"},{"pair":"SOMI/EUR"},{"pair":"SONIC/EUR"},{"pair":"SOON/EUR"},{"pair":"SOSO/EUR"},{"pair":"SPELL/EUR"},{"pair":"SPICE/EUR"},{"pair":"SPK/EUR"},{"pair":"SRM/EUR"},{"pair":"SSV/EUR"},{"pair":"STBL/EUR"},{"pair":"STEP/EUR"},{"pair":"STG/EUR"},{"pair":"STORJ/EUR"},{"pair":"STRD/EUR"},{"pair":"STRK/EUR"},{"pair":"SUI/EUR"},{"pair":"SUKU/EUR"},{"pair":"SUN/EUR"},{"pair":"SUNDOG/EUR"},{"pair":"SUPER/EUR"},{"pair":"SUSHI/EUR"},{"pair":"SWARMS/EUR"},{"pair":"SWEAT/EUR"},{"pair":"SWELL/EUR"},{"pair":"SYN/EUR"},{"pair":"SYND/EUR"},{"pair":"SYRUP/EUR"},{"pair":"T/EUR"},{"pair":"TAC/EUR"},{"pair":"TANSSI/EUR"},{"pair":"TAO/EUR"},{"pair":"TBTC/EUR"},{"pair":"TEER/EUR"},{"pair":"TERM/EUR"},{"pair":"TIA/EUR"},{"pair":"TITCOIN/EUR"},{"pair":"TLM/EUR"},{"pair":"TNSR/EUR"},{"pair":"TOKE/EUR"},{"pair":"TOKEN/EUR"},{"pair":"TON/EUR"},{"pair":"TOSHI/EUR"},{"pair":"TRAC/EUR"},{"pair":"TREE/EUR"},{"pair":"TREMP/EUR"},{"pair":"TRU/EUR"},{"pair":"TRUMP/EUR"},{"pair":"TURBO/EUR"},{"pair":"TUSD/EUR"},{"pair":"TVK/EUR"},{"pair":"U/EUR"},{"pair":"U2U/EUR"},{"pair":"UFD/EUR"},{"pair":"UMA/EUR"},{"pair":"UNFI/EUR"},{"pair":"UNI/EUR"},{"pair":"UNITE/EUR"},{"pair":"USDC/EUR"},{"pair":"USDD/EUR"},{"pair":"USDE/EUR"},{"pair":"USDG/EUR"},{"pair":"USDQ/EUR"},{"pair":"USDR/EUR"},{"pair":"USDS/EUR"},{"pair":"USDT/EUR"},{"pair":"USDUC/EUR"},{"pair":"USELESS/EUR"},{"pair":"UST/EUR"},{"pair":"USUAL/EUR"},{"pair":"VANRY/EUR"},{"pair":"VELODROME/EUR"},{"pair":"VELVET/EUR"},{"pair":"VERSE/EUR"},{"pair":"VFY/EUR"},{"pair":"VINE/EUR"},{"pair":"VIRTUAL/EUR"},{"pair":"VSN/EUR"},{"pair":"VULT/EUR"},{"pair":"VVV/EUR"},{"pair":"W/EUR"},{"pair":"WAL/EUR"},{"pair":"WBTC/EUR"},{"pair":"WCT/EUR"},{"pair":"WELL/EUR"},{"pair":"WEN/EUR"},{"pair":"WIF/EUR"},{"pair":"WIN/EUR"},{"pair":"WLD/EUR"},{"pair":"WLFI/EUR"},{"pair":"WOO/EUR"},{"pair":"YALA/EUR"},{"pair":"YB/EUR"},{"pair":"YFI/EUR"},{"pair":"YGG/EUR"}]

    for pair in pairs:
        try:
            main(pair["pair"], pair["pair"].replace("/", "-"))
        except Exception as e:
            print(e)
            continue
