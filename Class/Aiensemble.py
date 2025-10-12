from __future__ import annotations
import os, json, glob, math, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import re
import math
# from darts.models import TFTModel
# from darts.models import XGBModel
# from darts.models import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastCroston

try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

# Darts per TFT
try:
    from darts import TimeSeries  # type: ignore
    from darts.models import TFTModel, NHiTSModel  # type: ignore
    from darts.utils.likelihood_models import QuantileRegression  # type: ignore
    _has_darts = True
except Exception:
    _has_darts = False

# opzionale .env (per le API Kraken)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =============================================================
# AIEnsembleTrader – ensemble con pesi adattivi + capital manager
# (STATEFUL: istanzi una volta, poi passi currencies/actions a run()).
# =============================================================

# --------------------- utils ---------------------
def fee_in_crypto(qty, fee_rate):
    return float(qty) * float(fee_rate)

def floor_dec(x: float, n: int) -> float:
    f = 10 ** n
    return math.floor(x * f) / f

# --- helper: da mettere in alto nel file Aiensemble.py (vicino ad altri util) ---
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
    df["ts"] = dt.dt.tz_convert(None)
    # 2) ordina, allinea alla griglia (floor) e de-duplica
    df = df.sort_values("ts")
    df["ts"] = df["ts"].dt.floor(freq)
    df = df.groupby("ts", as_index=False).last()

    # 3) ricostruisci indice continuo con la stessa freq e riempi i buchi (ffill)
    idx = pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq)
    s = df.set_index("ts").reindex(idx)
    s["price"] = s["price"].astype(float).ffill().bfill()
    s = s.reset_index().rename(columns={"index": "ts"})

    # 4) crea la TimeSeries già regolare (freq implicita nell’indice)
    return TimeSeries.from_dataframe(s, time_col="ts", value_cols="price", fill_missing_dates=False)

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float)
    if temp <= 0: temp = 1e-9
    x = x / temp
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def to_score01(x: float) -> float:
    x = float(np.clip(x, 0, 1))
    return min(0.99999, max(0.0, x * 0.99999))

def get_now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    base = os.getcwd()
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    os.makedirs(path, exist_ok=True)
    return path

# --- utilità leggere (riuso del formato scaler della tua Neural) ---

def _load_neural_scaler(kind: str = "mtf") -> Optional[tuple]:
    p = os.path.join(os.getcwd(), "aiConfig", f"neural_scaler_{kind}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        mu = np.array(d.get("mu"), dtype=float).reshape(1, -1)
        sd = np.array(d.get("sd"), dtype=float).reshape(1, -1)
        return (mu, sd)
    except Exception:
        return None


def _standardize(x: np.ndarray, scaler: Optional[tuple]) -> np.ndarray:
    if scaler is None:  # niente scaler
        return x
    mu, sd = scaler
    xd = x.reshape(1, -1)
    # adatta lunghezze
    if xd.shape[1] != mu.shape[1]:
        d = min(xd.shape[1], mu.shape[1])
        xd = xd[:, :d]
        mu = mu[:, :d]
        sd = sd[:, :d]
    sd_safe = np.where(sd == 0.0, 1.0, sd)
    return (xd - mu) / sd_safe


# --- estrazione feature essenziale (coerente con la tua Neural MTF) ---
# NB: è un sottoinsieme robusto (no dipendenza diretta dalla classe Neural)

def features_mtf_from_row(row: Dict[str, Any]) -> np.ndarray:
    info = (row.get("info") or {})
    now = (info.get("NOW") or {})
    px = float(now.get("current_price") or now.get("last") or now.get("close") or now.get("open") or 0.0)
    # qualità mercato
    spread = float(now.get("spread") or 0.0)
    spread_ratio = spread / (abs(px) + 1e-12)
    slip_b = float(now.get("slippage_buy_pct") or 0.0)
    slip_s = float(now.get("slippage_sell_pct") or 0.0)
    slip_avg = 0.5 * (slip_b + slip_s)
    # OR
    or_ok = 1.0 if bool(now.get("or_ok")) else 0.0
    or_high = now.get("or_high")
    or_low = now.get("or_low")
    if or_high is not None and or_low is not None and (or_high - or_low) != 0:
        or_mid = 0.5 * (float(or_high) + float(or_low))
        or_rng = float(or_high) - float(or_low)
        or_pos = math.tanh((px - or_mid) / (abs(or_rng) + 1e-12))
        or_w = or_rng / (abs(px) + 1e-12)
    else:
        or_pos, or_w = 0.0, 0.0
    # bias
    b1 = 1.0 if now.get("bias_1h") == "UP" else (-1.0 if now.get("bias_1h") == "DOWN" else 0.0)
    b4 = 1.0 if now.get("bias_4h") == "UP" else (-1.0 if now.get("bias_4h") == "DOWN" else 0.0)
    # trend locale
    ema50_1h = float(now.get("ema50_1h") or 0.0)
    ema200_1h = float(now.get("ema200_1h") or 0.0)
    ema50_4h = float(now.get("ema50_4h") or 0.0)
    ema200_4h = float(now.get("ema200_4h") or 0.0)
    dev_1h = math.tanh((ema50_1h - ema200_1h) / (abs(ema200_1h) + 1e-9)) if ema200_1h else 0.0
    dev_4h = math.tanh((ema50_4h - ema200_4h) / (abs(ema200_4h) + 1e-9)) if ema200_4h else 0.0
    # ritorni recenti
    b24 = (info.get("24H") or {})
    b48 = (info.get("48H") or {})
    ch24 = float(b24.get("change_pct") or 0.0)
    ch48 = float(b48.get("change_pct") or 0.0)
    # volatilità
    b1h = (info.get("1H") or info.get("60M") or {})
    atr1h = float(b1h.get("atr") or 0.0)
    # deviazione da EMA lenta
    ema_slow = float(now.get("ema_slow") or 0.0)
    vol_dev = math.tanh((px - ema_slow) / (abs(ema_slow) + 1e-9)) if ema_slow else 0.0
    vec = np.array([
        ch24, ch48, dev_1h, dev_4h, vol_dev, atr1h, b1, b4,
        spread_ratio, slip_avg, or_ok, or_pos, or_w
    ], dtype=float)
    return vec


# ============================
#   TRAINER DI ESEMPIO (OFFLINE)
# ============================

def train_lgbm_offline(csv_path: str, model_out: str,
                        target_h: int = 12,
                        max_rows: Optional[int] = None) -> None:
    """Esempio minimo:
    - `csv_path` con colonne: [pair, ts, price, ch24, ch48, dev_1h, dev_4h, vol_dev, atr1h, b1, b4, spread_ratio, slip_avg, or_ok, or_pos, or_w]
    - target: label 0/1 su orizzonte `target_h` con soglia su rendimento futuro
    Crea `model_out` (joblib pkl).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

    if lgb is None or joblib is None:
        raise RuntimeError("lightgbm/joblib non disponibili. `pip install lightgbm joblib`")

    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.tail(max_rows)
    # esempio: feature da colonne (stesso ordine della funzione features_mtf_from_row)
    feat_cols = [
        "ch24","ch48","dev_1h","dev_4h","vol_dev","atr1h","b1","b4",
        "spread_ratio","slip_avg","or_ok","or_pos","or_w"
    ]
    X = df[feat_cols].astype(float).values
    # target: 1 se return futuro > 0 (puoi mettere barriera/fee)
    y = (df["ret_h"].astype(float).values > 0.0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.02,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    print("AUC:", roc_auc_score(y_te, p))

    # calibrazione opzionale
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    # cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X_te, y_te)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(cal, model_out.replace(".pkl", "_cal.pkl"))


def train_tft_offline(hist_dir: str, model_dir: str,
                      horizon: int = 12,
                      n_epochs: int = 50,
                      use_nhits: bool = False) -> None:
    """
    Trainer 'resumable':
      - scansiona hist_dir
      - per ogni <PAIR>.csv controlla se il modello è 'up-to-date'
      - se sì -> SKIP; altrimenti -> TRAIN e salva <PAIR>.pt + <PAIR>.meta.json
    Criteri di 'up-to-date':
      - esiste il .pt e il .meta.json
      - meta.horizon == horizon e meta.use_nhits == use_nhits e meta.n_epochs == n_epochs
      - meta.last_ts == ultimo ts del CSV e meta.rows == n righe del CSV
      - meta.csv_mtime >= mtime attuale del CSV
    """
    if not _has_darts:
        raise RuntimeError("Darts non disponibile. `pip install darts[u]`")
    import pandas as pd

    os.makedirs(model_dir, exist_ok=True)

    def _slug_pair(basename_csv: str) -> str:
        # basename senza .csv (già slug in TFTStrategy), ma ricontrollo per sicurezza
        base = os.path.splitext(basename_csv)[0]
        return re.sub(r'[^A-Za-z0-9._-]+', '_', base)

    def _meta_path(base_slug: str) -> str:
        return os.path.join(model_dir, f"{base_slug}.meta.json")

    def _model_path(base_slug: str) -> str:
        return os.path.join(model_dir, f"{base_slug}.pt")

    def _read_meta(path: str) -> Optional[dict]:
        if not os.path.exists(path): return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_meta(path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # --- scan ordinato per stabilità, così "riprende" naturalmente
    csv_files = sorted([fn for fn in os.listdir(hist_dir) if fn.endswith(".csv")])

    if not csv_files:
        print(f"[train_tft_offline] nessun CSV in {hist_dir}")
        return

    tot = len(csv_files)
    done = 0
    skipped = 0
    retrained = 0

    for i, fn in enumerate(csv_files, 1):
        csv_path = os.path.join(hist_dir, fn)
        base = _slug_pair(fn)
        mpath = _model_path(base)
        jpath = _meta_path(base)

        try:
            # --- leggi CSV e prepara firma
            df = pd.read_csv(csv_path)
            if df is None or df.empty or len(df) < 256:
                print(f"[{i}/{tot}] {fn}: pochi punti ({len(df) if df is not None else 0}) -> SKIP")
                skipped += 1
                continue

            # metriche “firma”
            rows = int(len(df))
            last_ts = int(pd.to_numeric(df["ts"], errors="coerce").dropna().astype("int64").max())
            csv_mtime = int(os.path.getmtime(csv_path))

            # --- controlla meta esistente
            meta = _read_meta(jpath)

            up_to_date = False
            if meta and os.path.exists(mpath):
                up_to_date = (
                    int(meta.get("horizon", -1)) == int(horizon) and
                    bool(meta.get("use_nhits", False)) == bool(use_nhits) and
                    int(meta.get("n_epochs", -1)) == int(n_epochs) and
                    int(meta.get("rows", -1)) == rows and
                    int(meta.get("last_ts", -1)) == last_ts and
                    int(meta.get("csv_mtime", -1)) >= csv_mtime
                )

            if up_to_date:
                print(f"[{i}/{tot}] {fn}: up-to-date -> SKIP")
                skipped += 1
                continue

            # --- prepara serie regolare
            series = _make_regular_timeseries_from_df(df, freq="min")

            # --- crea modello
            if use_nhits:
                model = NHiTSModel(
                    input_chunk_length=128,
                    output_chunk_length=horizon,
                    n_epochs=n_epochs,
                    random_state=42
                )
                used = "NHiTS"
                qlist = None
            else:
                qlist = [0.1, 0.5, 0.9]
                model = TFTModel(
                    input_chunk_length=128,
                    output_chunk_length=horizon,
                    hidden_size=64,
                    lstm_layers=1,
                    num_attention_heads=4,
                    dropout=0.1,
                    batch_size=64,
                    n_epochs=n_epochs,
                    random_state=42,
                    likelihood=QuantileRegression(quantiles=qlist),
                    add_encoders={
                        "cyclic": {"future": ["hour", "dayofweek"]},
                        "datetime_attribute": {"future": ["month"]},
                        "position": {"future": ["relative"]}
                    }
                )
                used = "TFT"

            # --- fit + save
            print(f"[{i}/{tot}] {fn}: TRAIN {used} (h={horizon}, epochs={n_epochs}) ...")
            model.fit(series)
            model.save(mpath)

            # --- meta
            meta_out = {
                "model": used,
                "horizon": int(horizon),
                "n_epochs": int(n_epochs),
                "use_nhits": bool(use_nhits),
                "quantiles": qlist,
                "rows": rows,
                "last_ts": last_ts,
                "csv_mtime": csv_mtime,
                "saved_at": get_now_ts(),
            }
            _write_meta(jpath, meta_out)
            print(f"  -> Salvato {mpath}")
            retrained += 1

        except KeyboardInterrupt:
            print("\n[train_tft_offline] Interrotto manualmente. Il prossimo run riprenderà.")
            break
        except Exception as e:
            print(f"[{i}/{tot}] {fn}: ERRORE {e!r} -> SKIP")
            # continua il giro senza bloccare tutta la sessione
            continue
        finally:
            done += 1

    print(f"[train_tft_offline] completato: tot={tot}, skip={skipped}, trained={retrained}")


# ------------------- strategie -------------------

class Strategy:
    name: str = "Base"
    def __init__(self, **kwargs): self.params = kwargs
    def fit(self, df: pd.DataFrame) -> "Strategy": return self
    def signal(self, row: Dict[str, Any]) -> float: raise NotImplementedError


# --- ActionPlanner: config & plan ---


@dataclass
class PlannerConfig:
    daily_pnl_target_eur: float = 20.0    # esempio
    weekly_pnl_target_eur: float = 100.0
    tp_pct: float = 0.8                    # TP a 0.8 * ATR multipliers o usa percentuale fissa
    sl_atr_mult: float = 2.5               # SL = 2.5 * ATR
    reduce_pct_default: float = 0.33       # frazione di reduce_only
    use_margin: bool = False
    default_leverage: str = "2:1"          # per margin (se attivo)

# ======== Price/Size/Feasibility policy (ONLINE, STATEFUL) ========

@dataclass
class _PSFState:
    alpha: float = 1.0      # 0..1 aggressività prezzo (0 prudente, 1 aggressivo)
    gamma: float = 1.0      # >=0 moltiplicatore size (1=come prima)
    succ: int = 0           # successi esecuzione
    fail: int = 0           # fallimenti (minsize/funds/perm/altro)
    last_err: str = ""      # ultimo errore visto (debug)

class PriceSizeFeasPolicy:
    """
    Impara 3 cose per (pair, side):
    - p(feasible) ≈ Beta(succ, fail) -> decide HOLD in caso di bassa fattibilità osservata
    - gamma (size): cresce se incontriamo spesso ordemin; cala se spesso 'insufficient funds'
    - alpha (prezzo): si adatta a spread/slippage e outcome (se ordini rimangono indietro)
    """
    def __init__(self, state_path: Optional[str]):
        self.state_path = state_path
        self._st: Dict[str, _PSFState] = {}
        self._load()

    def _k(self, pair: str, side: str) -> str:
        return f"{(pair or '').upper()}|{(side or '').lower()}"

    def _get(self, pair: str, side: str) -> _PSFState:
        return self._st.setdefault(self._k(pair, side), _PSFState())

    def _save(self):
        if not self.state_path: return
        try:
            payload = {k: asdict(v) for k, v in self._st.items()}
            safe_write_json(self.state_path, payload)
        except Exception:
            pass

    def _load(self):
        if not self.state_path or not os.path.exists(self.state_path): return
        try:
            d = safe_read_json(self.state_path) or {}
            for k, row in d.items():
                self._st[k] = _PSFState(**{**_PSFState().__dict__, **(row or {})})
        except Exception:
            self._st = {}

    # ---------- PREDICT ----------
    def predict(self, pair: str, side: str, now: Dict[str, Any], limits: Dict[str, Any]) -> Dict[str, Any]:
        s = self._get(pair, side)
        # stima p_ok = (succ+1)/(succ+fail+2) (Beta(1,1) prior)
        p_ok = (s.succ + 1.0) / (s.succ + s.fail + 2.0)
        # soglia HOLD dinamica (più alto spread/slippage, più tolleranza)
        slip = float(now.get("slippage_buy_pct" if side=="buy" else "slippage_sell_pct") or 0.0)
        spread = float(now.get("spread") or 0.0)
        hold_thresh = 0.25 - min(0.10, 0.5 * slip) - min(0.10, 0.5 * (spread or 0.0))
        should_hold = (p_ok < hold_thresh)

        # alpha default dalla situazione corrente (se non “imparato”)
        alpha_ctx = max(0.0, min(1.0, s.alpha))  # stato appreso
        if alpha_ctx == 0.0:
            # heuristic dolce alla prima run: più slippage → più aggressivo
            alpha_ctx = max(0.0, min(0.5, 0.5 * (slip or 0.0)))

        # gamma dallo stato (cap size). Clamp per sicurezza
        gamma_ctx = max(0.1, min(10.0, s.gamma))
        return {"p_ok": float(p_ok), "hold": bool(should_hold), "alpha": float(alpha_ctx), "gamma": float(gamma_ctx)}

    # ---------- UPDATE ----------
    def update_from_exec(self,
                         pair: str,
                         side: str,
                         *,
                         error: Optional[str],
                         desired_adj_qty: Optional[float],
                         ordermin: Optional[float]):
        s = self._get(pair, side)
        err = (error or "").lower()

        if not err:
            # successo
            s.succ += 1
            # verso maker: se troppo aggressivo (alpha>0.8) riduci un pelo, altrimenti lenta crescita
            s.alpha = float(max(0.0, min(1.0, s.alpha * 0.98 + 0.02)))
        else:
            s.fail += 1
            s.last_err = error or ""
            # MIN SIZE dopo rounding → alza gamma per superare ordemin la prossima
            if "min size" in err or "ordermin" in err:
                if desired_adj_qty and ordermin:
                    need = float(ordermin) / max(float(desired_adj_qty), 1e-12)
                    s.gamma = float(min(10.0, max(s.gamma, 1.05 * need)))
                else:
                    s.gamma = float(min(10.0, s.gamma * 1.25))
            # INSUFFICIENT FUNDS → riduci gamma
            elif "insufficient funds" in err:
                s.gamma = float(max(0.1, s.gamma * 0.8))
            # PERMISSIONS → metti in hold “strutturale” (p_ok giù)
            elif "invalid permissions" in err:
                # boosta i fail così p_ok scende e scatta hold
                s.fail += 3
            else:
                # errore generico → leggera riduzione gamma e alpha prudente
                s.gamma = float(max(0.1, s.gamma * 0.95))
                s.alpha = float(max(0.0, s.alpha * 0.95))

        self._save()


# ============================
#   LGBMStrategy (triple-barrier)
# ============================
class LGBMStrategy(Strategy):
    name = "LGBM"
    def __init__(self,
                 model_path: str = os.path.join(os.getcwd(), "aiConfig", "lgbm_model.pkl"),
                 calibrator_path: Optional[str] = None,
                 feature_kind: str = "mtf",
                 scaler_kind: str = "mtf",
                 proba_clip: float = 0.02,
                 score_gain: float = 1.0):
        super().__init__(model_path=model_path, calibrator_path=calibrator_path,
                         feature_kind=feature_kind, scaler_kind=scaler_kind,
                         proba_clip=proba_clip, score_gain=score_gain)
        self.model = None
        self.cal = None
        self.scaler = _load_neural_scaler(scaler_kind)

    def _lazy_load(self):
        if self.model is not None:
            return
        if joblib is None:
            return
        if os.path.exists(self.params["model_path"]):
            try:
                self.model = joblib.load(self.params["model_path"])  # type: ignore
            except Exception:
                self.model = None
        calp = self.params.get("calibrator_path")
        if calp and os.path.exists(calp):
            try:
                self.cal = joblib.load(calp)  # type: ignore
            except Exception:
                self.cal = None

    def fit(self, df):
        # allenamento offline (vedi trainer sotto)
        return self

    def signal(self, row: Dict[str, Any]) -> float:
        import pandas as pd
        self._lazy_load()
        if self.model is None:
            return 0.0

        # 1) features + standardizzazione → array 2D
        x = features_mtf_from_row(row)
        xz = _standardize(x, self.scaler)
        xz = np.asarray(xz, dtype=float).reshape(1, -1)

        # 2) Se il modello è stato addestrato con nomi colonna,
        #    ricrea un DataFrame con gli stessi nomi (evita il warning)
        X_in = xz
        try:
            if hasattr(self.model, "feature_names_in_"):
                fnames = list(self.model.feature_names_in_)
                n = len(fnames)
                # allinea dimensioni (clip/pad se necessario)
                if xz.shape[1] > n:
                    xz = xz[:, :n]
                elif xz.shape[1] < n:
                    xz = np.pad(xz, ((0, 0), (0, n - xz.shape[1])), mode="constant")
                X_in = pd.DataFrame(xz, columns=fnames)

            # 3) predizione + calibrazione opzionale
            if hasattr(self.model, "predict_proba"):
                p = float(self.model.predict_proba(X_in)[0, 1])
            else:
                raw = float(self.model.predict(X_in)[0])
                p = 1.0 / (1.0 + math.exp(-raw))

            if self.cal is not None and hasattr(self.cal, "predict_proba"):
                p = float(self.cal.predict_proba([[p]])[0, 1])
        except Exception:
            return 0.0

        # 4) clip e mappa in score [-1, 1]
        eps = float(self.params.get("proba_clip", 0.02))
        p = min(max(p, eps), 1.0 - eps)
        score = (p - 0.5) * 2.0
        return float(np.tanh(score * float(self.params.get("score_gain", 1.0))))


# ============================
#   TFTStrategy (forecast quantile → edge)
# ============================
class TFTStrategy(Strategy):
    def __init__(
        self,
        params: dict | None = None,
        *,
        model_dir: str | None = None,
        hist_dir: str | None = None,
        horizon: int | None = None,
        min_points: int | None = None,
        quantiles: list[float] | None = None,
        edge_gain: float | None = None,
        multi_horizon: bool | None = None,
        horizons: list[int] | None = None,
        flush_interval: float | None = None,
    ):
        # 1) base params dict
        p = dict(params or {})

        # 2) override con argomenti nominati (retro-compatibile)
        if model_dir is not None:   p["model_dir"] = model_dir
        if hist_dir is not None:    p["hist_dir"] = hist_dir
        if horizon is not None:     p["horizon"] = horizon
        if min_points is not None:  p["min_points"] = min_points
        if quantiles is not None:   p["quantiles"] = quantiles
        if edge_gain is not None:   p["edge_gain"] = edge_gain
        if multi_horizon is not None: p["multi_horizon"] = multi_horizon
        if horizons is not None:    p["horizons"] = horizons
        if flush_interval is not None: p["flush_interval"] = flush_interval

        # 3) default sensati
        self.params = p
        self.model_dir   = p.get("model_dir", "../aiConfig/tft_models")
        self.hist_dir    = p.get("hist_dir",  "../aiConfig/tft_hist")
        self.horizon     = int(p.get("horizon", 12))
        self.min_points  = int(p.get("min_points", 128))
        self.quantiles   = list(p.get("quantiles", [0.1, 0.5, 0.9]))  # 👈 restano!
        self.edge_gain   = float(p.get("edge_gain", 8.0))
        self.multi_horizon = bool(p.get("multi_horizon", True))        # default True
        self.horizons    = list(p.get("horizons", [12, 24, 48]))
        self.flush_interval = float(p.get("flush_interval", 60.0))

        # buffer/dirs come prima
        self.last_flush = 0.0
        self.buffer = {}
        os.makedirs(self.hist_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


    def _buffer_append(self, pair: str, ts: int, price: float):
        dq = self.buffer.setdefault(pair, deque(maxlen=5000))
        if dq and ts <= dq[-1][0]:
            return  # evita duplicati/out-of-order
        dq.append((ts, price))
        now = time.time()
        if now - self.last_flush > self.flush_interval:
            self._flush_buffers()
            self.last_flush = now

    def _flush_buffers(self):
        """Scrittura atomica dei buffer su file"""
        for pair, dq in self.buffer.items():
            if not dq:
                continue
            p = self._hist_path(pair)  # usa slug sicuro es. YGG_EUR.csv
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            df_new = pd.DataFrame(dq, columns=["ts", "price"])
            if os.path.exists(p):
                try:
                    df_old = pd.read_csv(p)
                    df = pd.concat([df_old, df_new], ignore_index=True)
                    df = df.drop_duplicates(subset="ts", keep="last").tail(5000)
                except Exception:
                    df = df_new
            else:
                df = df_new
            df.to_csv(tmp, index=False)
            os.replace(tmp, p)

    def _ensure_dirs(self):
        for d in (self.model_dir, self.hist_dir):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass


    def _slug_pair(self, pair: str) -> str:
        # tieni solo lettere, numeri, ., -, _ ; sostituisci tutto il resto con _
        return re.sub(r'[^A-Za-z0-9._-]+', '_', pair)

    def _hist_path(self, pair: str) -> str:
        fn = self._slug_pair(pair) + ".csv"
        return os.path.join(self.hist_dir, fn)

    def _model_path(self, pair: str) -> str:
        fn = self._slug_pair(pair) + ".pt"
        return os.path.join(self.model_dir, fn)
    # def _model_path(self, pair: str) -> str:
    #     fn = pair.replace("/", "_") + ".pt"
    #     return os.path.join(self.params["model_dir"], fn)

    def _update_local_history(self, pair: str, ts: float, price: float) -> int:
        """Append in CSV (rolling) e ritorna il numero di righe."""
        p = self._hist_path(pair)
        try:
            import pandas as pd  # lazy import
            df = None
            if os.path.exists(p):
                df = pd.read_csv(p)
            else:
                df = pd.DataFrame(columns=["ts", "price"])  # type: ignore
            df.loc[len(df)] = {"ts": int(ts), "price": float(price)}
            # roll a 5000 punti massimo
            # roll a 5000 punti massimo
            if len(df) > 5000:
                df = df.iloc[-5000:]
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, p)
            return len(df)
        except Exception:
            return 0

    def _load_series(self, pair: str):
        p = self._hist_path(pair)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        if len(df) < self.min_points:
            return None
        # staleness guard
        if (time.time() - int(df["ts"].iloc[-1])) > 300:  # >5 min senza update
            return None
        df = df.dropna()
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
        df = df.set_index("ts").asfreq("min")
        return TimeSeries.from_dataframe(df, value_cols=["price"])


    def _predict_edge(self, series: TimeSeries, row: dict) -> float:
        """Predice edge (%) con TFT o fallback"""
        last = float(series.univariate_values()[-1])
        path = self._model_path(row["pair"])

        # ---- fallback naive se manca modello
        if not os.path.exists(path):
            vals = series.univariate_values()
            if len(vals) < 30:
                return 0.0
            slope = np.polyfit(np.arange(30), vals[-30:], 1)[0] / (abs(vals[-1]) + 1e-12)
            return float(np.tanh(slope * 10.0) * 0.25)

        try:
            model = TFTModel.load(path)
        except Exception:
            # se fallisce il load, ritorna neutro
            return 0.0

        # multi-horizon o singolo
        horizons = self.horizons if self.multi_horizon else [self.horizons[-1]]
        edges = []
        for h in horizons:
            try:
                pred = model.predict(h)

                # scegli i quantili secondo configurazione
                q_med = 0.5 if 0.5 in self.quantiles else 0.5
                q_lo  = min(self.quantiles, default=0.1)
                q_hi  = max(self.quantiles, default=0.9)

                # median (fallback a deterministic)
                try:
                    p50 = float(pred.quantile_timeseries(q_med).univariate_values()[-1])
                except Exception:
                    p50 = float(pred.univariate_values()[-1])

                # incertezza (se il modello non ha quantili -> unc=0)
                try:
                    qlo = float(pred.quantile_timeseries(q_lo).univariate_values()[-1])
                    qhi = float(pred.quantile_timeseries(q_hi).univariate_values()[-1])
                    unc = (qhi - qlo) / (abs(last) + 1e-12)
                except Exception:
                    unc = 0.0

                edge = (p50 - last) / (abs(last) + 1e-12)
                conf = 1.0 / (1.0 + 12.0 * unc)
                edges.append(edge * conf)
            except Exception:
                continue

        if not edges:
            return 0.0
        edge = np.average(edges, weights=np.linspace(1, 0.3, len(edges)))

        # cost-aware
        info = row.get("info", {}).get("NOW", {})
        sl = info.get("slippage_buy_pct") or 0.0
        fee = info.get("fee_taker_pct") or 0.0
        cost = sl + fee
        edge_net = edge - cost * 1.2
        if edge_net <= 0:
            return 0.0

        return float(np.tanh(edge_net * self.edge_gain))


    # def _predict_edge(self, pair: str, series: Any) -> Optional[float]:
    #     if not _has_darts:
    #         return None
    #     mpath = self._model_path(pair)
    #     if not os.path.exists(mpath):
    #         return None
    #     try:
    #         # Proviamo TFT; in fallback N-HiTS
    #         model = Nones
    #         try:
    #             model = TFTModel.load(mpath)
    #         except Exception:
    #             try:
    #                 model = NHiTSModel.load(mpath)
    #             except Exception:
    #                 model = None
    #         if model is None:
    #             return None
    #         # Se il modello è probabilistico → prendi p50
    #         pred = model.predict(int(self.params["horizon"]))
    #         last = float(series.univariate_values()[-1])
    #         try:
    #             p50 = float(pred.quantile_timeseries(0.5).univariate_values()[-1])
    #         except Exception:
    #             p50 = float(pred.univariate_values()[-1])
    #         exp_ret = (p50 - last) / (abs(last) + 1e-12)
    #         # edge→score compresso
    #         return float(np.tanh(exp_ret * float(self.params.get("edge_gain", 8.0))))
    #     except Exception:
    #         return None


    def signal(self, row: dict) -> float:
        """Main entry: riceve un blocco row con 'pair', 'info.NOW', ecc."""


        now  = (row.get("info") or {}).get("NOW", {}) or {}
        pair = str(row.get("pair") or now.get("pair") or "?")

        # usa 'or' per evitare None nei cast
        ts    = int(now.get("timestamp") or time.time())
        price = float(now.get("current_price") or now.get("last") or 0.0)
        if price <= 0.0 or pair is None:
            print(pair)
            print(price)
            return 0.0

        # now = row.get("info", {}).get("NOW", {})
        # pair = row.get("pair")
        # ts = int(now.get("timestamp", time.time()))
        # price = float(now.get("current_price", 0.0))
        # if price <= 0:
        #     return 0.0

        # aggiorna buffer (RAM); flusha periodicamente
        self._buffer_append(pair, ts, price)

        # (opzionale) forza un flush prima di leggere, se vuoi includere l'ultimo punto
        # self._flush_buffers()

        # carica serie se disponibile
        series = self._load_series(pair)

        if series is None:
            return 0.0

        try:
            edge = self._predict_edge(series, row)
        except Exception as e:
            print(f"[TFTStrategy] errore predizione {pair}: {e}")
            edge = 0.0
        return float(np.clip(edge, -1.0, 1.0))

    # def signal(self, row: Dict[str, Any]) -> float:
    #     # 1) aggiorna storia locale
    #     info = (row.get("info") or {})
    #     now = (info.get("NOW") or {})
    #     pair = str(now.get("pair") or row.get("pair") or "?")
    #     px = float(now.get("current_price") or now.get("last") or 0.0)
    #     ts = float(now.get("since") or time.time())
    #     self._update_local_history(pair, ts, px)
    #     # 2) prova a predire se abbiamo serie e modello
    #     series = self._load_series(pair)
    #     if series is None:
    #         return 0.0
    #     edge = self._predict_edge(pair, series)
    #     return float(edge) if edge is not None else 0.0



class MomentumStrategy(Strategy):
    name = "Momentum"

    # Parametri facilmente tunabili (possono anche arrivare da config esterna)
    timeframes_weights = [
        ("NOW", 0.10),
        ("1H",  0.20),
        ("4H",  0.15),
        ("24H", 0.35),
        ("48H", 0.20)
    ]
    beta_vol: float = 1.2     # quanta penalizzazione dare alla volatilità
    vol_floor: float = 0.5    # evita divisioni troppo aggressive quando vol è bassa
    agree_boost: float = 0.15 # bonus/malus in base all’accordo fra timeframe
    downtrend_dampen: float = 0.70  # freno ai long se il 30D è short
    clip_abs: float = 1.0

    @staticmethod
    def _ch(info: dict, tf: str) -> float:
        """Ritorna change_pct per il timeframe tf se presente, altrimenti 0.0."""
        blk = info.get(tf, {})
        ch = blk.get("change_pct", 0.0)
        try:
            return float(ch)
        except Exception:
            return 0.0

    def signal(self, row: Dict[str, Any]) -> float:
        info = row.get("info", {}) or {}

        # ===== 1) Momentum pesato su più timeframe =====
        changes, weights = [], []
        for tf, w in self.timeframes_weights:
            c = self._ch(info, tf)
            if c is None:
                c = 0.0
            changes.append(c)
            weights.append(w)

        if sum(weights) > 0:
            m = float(np.dot(changes, weights) / sum(weights))  # momentum medio
        else:
            m = 0.0

        # ===== 2) Proxy di volatilità e normalizzazione "Sharpe-like" =====
        # uso la media degli assoluti delle variazioni disponibili come vol
        nonzero = [abs(c) for c in changes if c is not None]
        vol = float(np.mean(nonzero)) if len(nonzero) else 0.0
        denom = max(self.vol_floor, vol * self.beta_vol)
        score = m / denom

        # ===== 3) Bonus/malus di accordo tra timeframe =====
        if len(nonzero):
            sgn_m = np.sign(m) if m != 0 else 0.0
            if sgn_m != 0:
                agree = np.mean([1.0 if np.sign(c) == sgn_m and c != 0 else 0.0 for c in changes])
                # scala in ~[0.85, 1.15] con agree_boost=0.15
                score *= (1.0 + self.agree_boost * (agree - 0.5) * 2.0)

        # ===== 4) Freno di regime: long più prudenti se 30D è negativo =====
        ch30 = self._ch(info, "30D")
        if score > 0 and ch30 < 0:
            score *= self.downtrend_dampen

        # ===== 5) Squash morbido e clip finale =====
        score = float(np.tanh(score))                # mantiene dinamica senza saturare troppo
        return float(np.clip(score, -self.clip_abs, self.clip_abs))


class ValueStrategy(Strategy):
    name = "Value"
    def signal(self, row: Dict[str, Any]) -> float:
        info = row.get("info", {})
        ref = info.get("30D", {}) or info.get("24H", {})
        now = info.get("NOW", {})
        vwap = ref.get("vwap")
        px = now.get("current_price") or now.get("last")
        if vwap is None or px is None or vwap == 0: return 0.0
        gap = (float(vwap) - float(px)) / float(vwap)
        return float(np.clip(gap * 2.0, -1.0, 1.0))

class PairsStrategy(Strategy):
    name = "Pairs"
    def __init__(self, universe: Optional[Dict[str, Dict[str, Any]]] = None, **kw):
        super().__init__(**kw); self.universe = universe or {}
    def signal(self, row: Dict[str, Any]) -> float:
        if row.get("base") == "BTC": return 0.0
        this = row.get("info", {}).get("24H", {}).get("change_pct")
        btc  = self.universe.get("BTC/EUR", {}).get("info", {}).get("24H", {}).get("change_pct")
        if this is None or btc is None: return 0.0
        spread = float(this) - float(btc)
        return float(np.clip(-np.tanh(spread / 2.0), -1.0, 1.0))

class EarningsEventStrategy(Strategy):
    name = "Earnings"
    def signal(self, row: Dict[str, Any]) -> float:
        now = row.get("info", {}).get("NOW", {})
        b1, b4 = now.get("bias_1h"), now.get("bias_4h")
        s = 0.0
        if b1 == "UP": s += 0.4
        if b4 == "UP": s += 0.3
        if b1 == "DOWN": s -= 0.4
        if b4 == "DOWN": s -= 0.3
        return float(np.clip(s, -1.0, 1.0))

class MLStrategy(Strategy):
    name = "ML"
    def fit(self, df: pd.DataFrame) -> "Strategy": return self
    def signal(self, row: Dict[str, Any]) -> float:
        info = row.get("info", {})
        a = info.get("NOW", {}).get("ema50_1h")
        b = info.get("NOW", {}).get("ema200_1h")
        c = info.get("NOW", {}).get("ema50_4h")
        d = info.get("NOW", {}).get("ema200_4h")
        vals = [a,b,c,d]
        if any(v is None for v in vals): return 0.0
        s = 0.5*np.tanh((float(a)-float(b))/(abs(float(b))+1e-9)) + \
            0.5*np.tanh((float(c)-float(d))/(abs(float(d))+1e-9))
        return float(np.clip(s, -1.0, 1.0))

class AutoStrategy(Strategy):
    name = "Auto"
    def __init__(self, seed: int = 13, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)
        self.best = {"w1": 0.5, "w2": 0.5, "thr": 0.0}
        self.best_score = -1e9
    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        a = pd.to_numeric(a, errors="coerce").astype(float)
        b = pd.to_numeric(b, errors="coerce").astype(float)
        a = a.replace([np.inf, -np.inf], np.nan)
        b = b.replace([np.inf, -np.inf], np.nan)
        a = a.fillna(0.0)
        b = b.fillna(0.0)
        sa = float(a.std()); sb = float(b.std())
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(a.corr(b))
    def fit(self, df: pd.DataFrame) -> "Strategy":
        if df.empty or "ret24h" not in df.columns:
            return self
        mom24 = pd.to_numeric(df.get("mom24h", 0.0), errors="coerce").fillna(0.0)
        mom48 = pd.to_numeric(df.get("mom48h", 0.0), errors="coerce").fillna(0.0)
        target = pd.to_numeric(df["ret24h"], errors="coerce").fillna(0.0)
        for _ in range(80):
            w1 = float(self.rng.uniform(-2, 2))
            w2 = float(self.rng.uniform(-2, 2))
            thr = float(self.rng.uniform(-0.01, 0.01))
            pred = np.tanh(w1 * mom24 + w2 * mom48 - thr)
            sc = self._safe_corr(pd.Series(pred), target)
            if sc > self.best_score:
                self.best_score = sc
                self.best = {"w1": w1, "w2": w2, "thr": thr}
        return self
    def signal(self, row: Dict[str, Any]) -> float:
        info = row.get("info", {})
        ch24 = info.get("24H", {}).get("change_pct") or 0.0
        ch48 = info.get("48H", {}).get("change_pct") or 0.0
        z = self.best["w1"] * float(ch24) + self.best["w2"] * float(ch48) - self.best["thr"]
        return float(np.clip(np.tanh(z), -1.0, 1.0))

class MeanReversionSRStrategy(Strategy):
    """Mean-reversion vs EMA con trigger S/R multi-timeframe (1m/15m/30m compatibili)."""
    name = "MeanRevSR"
    def __init__(self, k_atr=1.5, sr_buffer_pct=0.002, bias_boost=0.15, bias_penalty=0.10):
        super().__init__(k_atr=k_atr, sr_buffer_pct=sr_buffer_pct, bias_boost=bias_boost, bias_penalty=bias_penalty)
    @staticmethod
    def _get_block(info: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        for k in keys:
            if k in info: return info.get(k) or {}
            for cand in list(info.keys()):
                if str(cand).upper()==k.upper(): return info.get(cand) or {}
        return {}
    @staticmethod
    def _get_num(d: Dict[str, Any], key: str) -> Optional[float]:
        v = d.get(key)
        if v is None: return None
        try: return float(v)
        except Exception: return None
    def signal(self, row: Dict[str, Any]) -> float:
        info = row.get("info", {}) or {}; now = info.get("NOW", {}) or {}
        px = self._get_num(now, "current_price") or self._get_num(now, "last")
        if px is None: return 0.0
        blk24 = self._get_block(info, ["24H"]); blk4h = self._get_block(info, ["4H"])
        blk1h = self._get_block(info, ["1H","60M","60m"]); blk30 = self._get_block(info, ["30M","30m"])
        blk15 = self._get_block(info, ["15M","15m"])
        mean = (self._get_num(blk24,"ema_slow") or self._get_num(blk4h,"ema_slow")
                or self._get_num(blk1h,"ema_slow") or self._get_num(blk30,"ema_slow")
                or self._get_num(blk15,"ema_slow") or self._get_num(now,"ema_slow")
                or self._get_num(now,"ema_fast"))
        if mean is None: return 0.0
        atr = (self._get_num(blk24,"atr") or self._get_num(blk4h,"atr")
               or self._get_num(blk1h,"atr") or self._get_num(blk30,"atr")
               or self._get_num(blk15,"atr"))
        if atr is None or atr<=0: atr = abs(mean)*0.0025
        if bool(now.get("or_ok")):
            support = self._get_num(now,"or_low"); resist = self._get_num(now,"or_high")
        else:
            sr_blk = blk1h or blk30 or blk24 or blk4h or blk15 or now
            support = self._get_num(sr_blk,"low"); resist = self._get_num(sr_blk,"high")
        k = float(self.params.get("k_atr",1.5)); dev = (px-mean)/(atr+1e-12)
        buf = float(self.params.get("sr_buffer_pct",0.002))
        s_long=s_short=0.0
        near_sup = bool(support) and px <= float(support)*(1.0+buf)
        if dev < -k or near_sup:
            s_long = float(np.clip(np.tanh((-dev-k)/2.0),0.0,1.0))
            if near_sup: s_long = max(s_long, 0.35)
        near_res = bool(resist) and px >= float(resist)*(1.0-buf)
        if dev > k or near_res:
            s_short = float(np.clip(np.tanh((dev-k)/2.0),0.0,1.0))
            if near_res: s_short = max(s_short, 0.35)
        b1 = now.get("bias_1h"); b4 = now.get("bias_4h")
        boost = float(self.params.get("bias_boost",0.15)); pen = float(self.params.get("bias_penalty",0.10))
        for b in (b1,b4):
            if b=="UP": s_long+=boost; s_short-=pen
            elif b=="DOWN": s_long-=pen; s_short+=boost
        score = float(np.clip(s_long,0,1)-np.clip(s_short,0,1))
        return float(np.clip(score,-1,1))

# --- NeuralStrategy (MTF-aware) ---
class NeuralStrategy(Strategy):
    """
    Strategia neurale compatta con supporto MTF:
    - Input flessibile: 2 feature (ch24,ch48), 8 feature "basic" oppure MTF estese.
    - Warmup scaler da snapshot (senza storico) anche per MTF.
    - Guard-rails su ATR piatto e OR non valido; bias 1h/4h come boost/penalty.
    """
    name = "Neural"

    def __init__(self, hidden=32, epochs=60, lr=1e-3, seed=42,
                 snapshot_scaler: bool = True,
                 flat_atr_floor: float = 1e-8,
                 bias_boost: float = 0.08,
                 bias_penalty: float = 0.06):
        super().__init__(hidden=hidden, epochs=epochs, lr=lr, seed=seed,
                         snapshot_scaler=snapshot_scaler,
                         flat_atr_floor=flat_atr_floor,
                         bias_boost=bias_boost, bias_penalty=bias_penalty)
        self.model = None
        self.scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (mu, sd)
        self._snapshot_scaler_ready: bool = False
        self._use_torch = False
        self.d_in: int = 2
        self.feature_kind: str = "2"  # "2" | "8" | "mtf"

# --- scaler persistence in ./aiConfig ---
        self._scaler_dir = os.path.join(os.getcwd(), "aiConfig")
        try:
            os.makedirs(self._scaler_dir, exist_ok=True)
        except Exception:
            pass
        self._scaler_loaded_kind: Optional[str] = None

        def _scaler_path_for(kind: str) -> str:
            return os.path.join(self._scaler_dir, f"neural_scaler_{kind}.json")

        self._scaler_path_for = _scaler_path_for  # bind as attribute for later use

        # tentativo di load in ordine di preferenza (mtf -> 8 -> 2)
        try:
            # print('thorch before')
            for _kind in ("mtf", "8", "2"):
                p = self._scaler_path_for(_kind)
                if os.path.exists(p):
                    d = safe_read_json(p)
                    mu = d.get("mu"); sd = d.get("sd"); dim = int(d.get("dim") or 0)
                    if mu is not None and sd is not None and dim > 0:
                        self.scaler = (np.array(mu, dtype=float).reshape(1,-1),
                                    np.array(sd, dtype=float).reshape(1,-1))
                        self.d_in = dim
                        self.feature_kind = _kind
                        self._scaler_loaded_kind = _kind
                        break
        except Exception:
            # print('no-thorch before')
            self._scaler_loaded_kind = None
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            self._use_torch = True
        except Exception:
            self.torch = None
            self.nn = None
            self._use_torch = False

        # try:
        #     import torch, torch.nn as nn  # type: ignore
        #     self.torch = __import__("torch")
        #     self.nn = __import__("torch.nn", fromlist=["nn"]).nn
        #     self._use_torch = True
        # except Exception:
        #     self._use_torch = False
        print(nn)
    # ---------- helpers ----------
    @staticmethod
    def _get_block(info: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        """Match case-insensitive: restituisce il primo blocco tra keys presente in info."""
        for k in keys:
            if k in info: return info.get(k) or {}
            for cand in list(info.keys()):
                if str(cand).upper() == str(k).upper():
                    return info.get(cand) or {}
        return {}

    @staticmethod
    def _num(d: Dict[str, Any], key: str) -> Optional[float]:
        v = None if d is None else d.get(key)
        if v is None: return None
        try: return float(v)
        except Exception: return None

    # ---------- feature engineering (basic 8D) ----------
    @staticmethod
    def _row_features_basic(row: Dict[str, Any]) -> np.ndarray:
        info = (row.get("info") or {})
        now = (info.get("NOW") or {})
        b24 = (info.get("24H") or {}); b48 = (info.get("48H") or {})
        b1h = (info.get("1H") or info.get("60M") or {}) or {}
        ema50_1h = float(now.get("ema50_1h") or 0.0)
        ema200_1h = float(now.get("ema200_1h") or 0.0)
        ema50_4h = float(now.get("ema50_4h") or 0.0)
        ema200_4h = float(now.get("ema200_4h") or 0.0)
        dev_1h = np.tanh((ema50_1h - ema200_1h) / (abs(ema200_1h)+1e-9))
        dev_4h = np.tanh((ema50_4h - ema200_4h) / (abs(ema200_4h)+1e-9))
        ch24 = float(b24.get("change_pct") or 0.0)
        ch48 = float(b48.get("change_pct") or 0.0)
        atr1h = float(b1h.get("atr") or 0.0)
        price = float((now.get("current_price") or now.get("last") or 0.0))
        ema_slow = float(now.get("ema_slow") or 0.0)
        vol_dev = np.tanh((price - ema_slow) / (abs(ema_slow)+1e-9)) if ema_slow else 0.0
        b1 = 1.0 if now.get("bias_1h") == "UP" else (-1.0 if now.get("bias_1h") == "DOWN" else 0.0)
        b4 = 1.0 if now.get("bias_4h") == "UP" else (-1.0 if now.get("bias_4h") == "DOWN" else 0.0)
        return np.array([ch24, ch48, dev_1h, dev_4h, vol_dev, atr1h, b1, b4], dtype=float)

    # ---------- feature engineering (MTF estese ~36D) ----------
    def _row_features_mtf(self, row: Dict[str, Any]) -> np.ndarray:
        info = (row.get("info") or {})
        now  = self._get_block(info, ["NOW"])
        px   = self._num(now, "current_price") or self._num(now, "last") or \
               self._num(now, "close") or self._num(now, "open") or 0.0

        # generali (costi/qualità mercato)
        spread = self._num(now, "spread") or 0.0
        spread_ratio = spread / (abs(px)+1e-12)
        slip_b = self._num(now, "slippage_buy_pct") or 0.0
        slip_s = self._num(now, "slippage_sell_pct") or 0.0
        slip_avg = (slip_b + slip_s) / 2.0
        bid_sum = self._num(now, "liquidity_bid_sum") or 0.0
        ask_sum = self._num(now, "liquidity_ask_sum") or 0.0
        liq_skew = 0.0
        if (bid_sum + ask_sum) > 0:
            liq_skew = (bid_sum - ask_sum) / (bid_sum + ask_sum)

        # Opening Range
        or_ok   = 1.0 if bool(now.get("or_ok")) else 0.0
        or_high = self._num(now, "or_high")
        or_low  = self._num(now, "or_low")
        or_rng  = self._num(now, "or_range") or ((or_high - or_low) if (or_high is not None and or_low is not None) else 0.0)
        or_pos  = 0.0
        if or_high is not None and or_low is not None and (or_rng or 0.0) > 0:
            or_mid = 0.5*(or_high+or_low)
            or_pos = np.tanh((px - or_mid) / (abs(or_rng)+1e-12))
        or_w = (or_rng or 0.0) / (abs(px)+1e-12)

        # bias
        b1 = now.get("bias_1h"); b4 = now.get("bias_4h")
        bias1 = 1.0 if b1=="UP" else (-1.0 if b1=="DOWN" else 0.0)
        bias4 = 1.0 if b4=="UP" else (-1.0 if b4=="DOWN" else 0.0)

        # per timeframe: momentum + dev EMA + px-dev EMA + ATR-ratio
        feats = []
        tfs = [("5M", 1.5), ("15M", 2.0), ("30M", 2.5),
               ("1H", 3.0), ("4H", 4.0), ("24H", 6.0), ("48H", 8.0)]
        for key, scale in tfs:
            blk = self._get_block(info, [key])
            ch  = blk.get("change_pct")
            if ch is None:
                op = self._num(blk, "open"); cl = self._num(blk, "close")
                ch = ((cl - op) / (abs(op)+1e-12) * 100.0) if (op is not None and cl is not None) else 0.0
            mom = np.tanh(float(ch) / float(scale))
            ema_f = self._num(blk, "ema_fast") or 0.0
            ema_s = self._num(blk, "ema_slow") or 0.0
            dev = np.tanh((ema_f - ema_s) / (abs(ema_s)+1e-12)) if ema_s else 0.0
            cp  = self._num(blk, "current_price") or px
            px_dev = np.tanh((cp - ema_s) / (abs(ema_s)+1e-12)) if ema_s else 0.0
            atr = self._num(blk, "atr") or 0.0
            atr_ratio = np.tanh(atr / (abs(ema_s)+1e-12)) if ema_s else 0.0
            feats.extend([mom, dev, px_dev, atr_ratio])

        # features generali in coda (segnate con segno "naturale")
        general = [
            -np.tanh(spread_ratio*10.0),      # spread alto => penalità
            -np.tanh(slip_avg*5.0),           # slippage medio => penalità
            float(liq_skew),                   # >0 più bid che ask
            float(or_pos),                     # posizionamento nell'OR
            float(or_w),                       # ampiezza OR relativa
            float(or_ok),                      # 1 se OR valido
            float(bias1), float(bias4)        # bias 1h/4h
        ]

        return np.array(feats + general, dtype=float)

    # ---------- warmup scaler (compat con vecchia firma) ----------
    def warmup_scaler_from_currencies(self,
                                      currencies: List[Dict[str, Any]],
                                      use_full: Optional[bool] = None,
                                      features: str = "mtf") -> None:
        """
        Calcola (mu, sd) dallo snapshot:
        - features="2"   -> [ch24, ch48]
        - features="8"   -> basic 8D
        - features="mtf" -> MTF estese (~36D)
        Nota: 'use_full' resta per compatibilità (True~"8", False~"2").
        """
        if use_full is not None:
            features = "8" if use_full else "2"
        feats = []
        for cur in currencies or []:
            if features == "mtf":
                f = self._row_features_mtf(cur); self.feature_kind = "mtf"
            elif features == "8":
                f = self._row_features_basic(cur); self.feature_kind = "8"
            else:
                b = self._row_features_basic(cur); f = b[:2]; self.feature_kind = "2"
            feats.append(f)
        if not feats:
            return
        X = np.asarray(feats, dtype=float)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-9
        self.scaler = (mu, sd)
        self.d_in = X.shape[1]
        self._snapshot_scaler_ready = True

        try:
            self._save_scaler()
        except Exception:
            print('error scaler')
            pass

    def _save_scaler(self) -> None:
        """Persisti scaler su ./aiConfig/neural_scaler_{kind}.json"""
        try:
            kind = self.feature_kind or "mtf"
            p = self._scaler_path_for(kind) if hasattr(self, "_scaler_path_for") else os.path.join(os.getcwd(), "aiConfig", f"neural_scaler_{kind}.json")
            mu, sd = self.scaler if self.scaler is not None else (None, None)
            if mu is None or sd is None:
                return
            payload = {
                "version": 1,
                "feature_kind": kind,
                "dim": int(mu.shape[1]),
                "mu": mu.reshape(-1).tolist(),
                "sd": sd.reshape(-1).tolist(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            safe_write_json(p, payload)
        except Exception:
            print('error save scaler')
            pass


    # ---------- training ----------
    def fit(self, df: pd.DataFrame) -> "Strategy":
        if df is None or df.empty or "ret24h" not in df.columns:
            return self

        # Scelta colonne di input:
        feat_cols_2 = ["mom24h", "mom48h"]
        feat_cols_8 = ["f_ch24","f_ch48","f_dev1h","f_dev4h","f_voldev","f_atr1h","f_b1","f_b4"]
        feat_cols_mtf = [c for c in df.columns if str(c).startswith("f_mtf_")]  # opzionale

        if feat_cols_mtf:
            use_cols = feat_cols_mtf; self.feature_kind = "mtf"
        elif all(c in df.columns for c in feat_cols_8):
            use_cols = feat_cols_8;   self.feature_kind = "8"
        elif all(c in df.columns for c in feat_cols_2):
            use_cols = feat_cols_2;   self.feature_kind = "2"
        else:
            return self

        X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in use_cols}).values.astype(float)
        y = pd.to_numeric(df["ret24h"], errors="coerce").fillna(0.0).values.astype(float)
        if len(X) < 20:
            return self

        self.d_in = X.shape[1]
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-9
        self.scaler = (mu, sd)
        try:
            self._save_scaler()
        except Exception:
            pass
        Xn = (X - mu) / sd
        yt = np.tanh(y / 5.0)
        if self._use_torch and self.torch is not None and self.nn is not None:
            # print('thorch')
            torch, nn = self.torch, self.nn
            torch.manual_seed(int(self.params.get("seed", 42)))
            Xtn = torch.tensor(Xn, dtype=torch.float32)
            ytn = torch.tensor(yt, dtype=torch.float32).view(-1, 1)

            class MLP(nn.Module):
                def __init__(self, d_in, d_h):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(d_in, d_h), nn.ReLU(),
                        nn.Linear(d_h, d_h),  nn.ReLU(),
                        nn.Linear(d_h, 1),    nn.Tanh()
                    )
                def forward(self, x): return self.net(x)

            self.model = MLP(d_in=Xtn.shape[1], d_h=int(self.params.get("hidden", 32)))
            opt = torch.optim.Adam(self.model.parameters(), lr=float(self.params.get("lr",1e-3)))
            loss_fn = nn.MSELoss()

            self.model.train()
            for _ in range(int(self.params.get("epochs", 60))):
                opt.zero_grad(); pred = self.model(Xtn)
                loss = loss_fn(pred, ytn); loss.backward(); opt.step()
        else:
            # print('no thorch')
            # fallback: tanh-regression con GD
            rng = np.random.default_rng(int(self.params.get("seed", 42)))
            W = rng.normal(scale=0.1, size=(Xn.shape[1], 1)); b = 0.0
            lr = float(self.params.get("lr", 1e-2))
            for _ in range(int(self.params.get("epochs", 120))):
                z = Xn @ W + b
                pred = np.tanh(z)
                grad = (pred - yt.reshape(-1,1)) * (1 - pred**2)
                W -= lr * (Xn.T @ grad / len(Xn)); b -= lr * float(grad.mean())
            self.model = (W, b)

        return self

    # ---------- inference utilities ----------
    def _build_runtime_vector(self, row: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[np.ndarray,np.ndarray]]:
        """
        Crea X coerente con lo scaler disponibile (2D, 8D, MTF).
        Ordine: scaler addestrato > scaler snapshot > fallback μ=0, σ=1.
        """
        # scegli vettore in base al kind
        def vec():
            if self.feature_kind == "mtf": return self._row_features_mtf(row)
            if self.feature_kind == "8":   return self._row_features_basic(row)
            b = self._row_features_basic(row); return b[:2]

        f_now = vec()

        # scaler da fit o da warmup
        if self.scaler is not None:
            mu, sd = self.scaler
            # adatta automaticamente lunghezze (taglia/espande se serve)
            d = mu.shape[1]
            if len(f_now) != d:
                # riduci o prendi subset sensato
                if d == 2:
                    base = self._row_features_basic(row)[:2]
                    X = np.asarray(base, dtype=float)[None, :]
                elif d == 8:
                    base = self._row_features_basic(row)
                    X = np.asarray(base, dtype=float)[None, :]
                else:
                    base = self._row_features_mtf(row)
                    # se mf più corto, pad con zeri (raro)
                    if len(base) < d:
                        base = np.pad(base, (0, d-len(base)))
                    X = np.asarray(base[:d], dtype=float)[None, :]
                self.d_in = d
                return X, (mu, sd)
            X = np.asarray(f_now, dtype=float)[None, :]
            self.d_in = d
            return X, (mu, sd)

        # scaler snapshot
        if bool(self.params.get("snapshot_scaler", True)) and self._snapshot_scaler_ready and self.scaler is not None:
            mu, sd = self.scaler
            d = mu.shape[1]
            base = f_now
            if len(base) < d:
                base = np.pad(base, (0, d-len(base)))
            X = np.asarray(base[:d], dtype=float)[None, :]
            self.d_in = d
            return X, (mu, sd)

        # fallback: μ=0, σ=1 con vettore coerente al kind
        X = np.asarray(f_now, dtype=float)[None, :]
        mu = np.zeros((1, X.shape[1]), dtype=float)
        sd = np.ones((1, X.shape[1]), dtype=float)
        self.d_in = X.shape[1]
        return X, (mu, sd)

    def _apply_guards_and_bias(self, score: float, row: Dict[str, Any]) -> float:
        """Smorza su ATR piatto/OR non valido e aggiusta per bias 1h/4h."""
        info = (row.get("info") or {}); now = (info.get("NOW") or {})
        b1h = (info.get("1H") or info.get("60M") or {}) or {}
        atr1h = float(b1h.get("atr") or 0.0)
        if atr1h <= float(self.params.get("flat_atr_floor", 1e-8)):
            score *= 0.2
        if not bool(now.get("or_ok")):
            score *= 0.7
        b1 = now.get("bias_1h"); b4 = now.get("bias_4h")
        boost = float(self.params.get("bias_boost", 0.08))
        pen   = float(self.params.get("bias_penalty", 0.06))
        for b in (b1, b4):
            if b == "UP":   score += boost
            elif b == "DOWN": score -= pen
        return float(np.clip(score, -1.0, 1.0))

    def _heuristic_score_mtf(self, row: Dict[str, Any]) -> float:
        """Combinazione euristica se non c'è un modello addestrato."""
        info = (row.get("info") or {}); now = self._get_block(info, ["NOW"])
        # momentum multi-TF
        def mom_of(tf, scale):
            blk = self._get_block(info, [tf]); ch = blk.get("change_pct")
            if ch is None:
                op = self._num(blk,"open"); cl = self._num(blk,"close")
                ch = ((cl-op)/(abs(op)+1e-12)*100.0) if (op is not None and cl is not None) else 0.0
            return np.tanh(float(ch)/scale)
        m = (
            0.30*mom_of("1H",3.0) + 0.20*mom_of("4H",4.0) +
            0.15*mom_of("30M",2.5) + 0.10*mom_of("15M",2.0) +
            0.10*mom_of("24H",6.0) + 0.10*mom_of("48H",8.0) +
            0.05*mom_of("5M",1.5)
        )

        # posizione nell'OR e qualità mercato
        px   = self._num(now,"current_price") or self._num(now,"last") or 0.0
        spread = self._num(now,"spread") or 0.0
        spread_ratio = spread/(abs(px)+1e-12)
        slip = ((self._num(now,"slippage_buy_pct") or 0.0) + (self._num(now,"slippage_sell_pct") or 0.0))/2.0
        or_high = self._num(now,"or_high"); or_low=self._num(now,"or_low"); or_rng=self._num(now,"or_range") or 0.0
        or_pos=0.0
        if or_high is not None and or_low is not None and or_rng>0:
            or_mid=0.5*(or_high+or_low); or_pos=np.tanh((px-or_mid)/(or_rng+1e-12))

        # dev EMA 1H/4H
        ema50_1h = self._num(now,"ema50_1h") or 0.0
        ema200_1h= self._num(now,"ema200_1h") or 0.0
        ema50_4h = self._num(now,"ema50_4h") or 0.0
        ema200_4h= self._num(now,"ema200_4h") or 0.0
        d1 = np.tanh((ema50_1h-ema200_1h)/(abs(ema200_1h)+1e-12)) if ema200_1h else 0.0
        d4 = np.tanh((ema50_4h-ema200_4h)/(abs(ema200_4h)+1e-12)) if ema200_4h else 0.0

        z = 0.6*m + 0.2*(0.5*d1 + 0.5*d4) + 0.2*or_pos \
            - 0.25*np.tanh(spread_ratio*10.0) - 0.35*np.tanh(slip*5.0)
        return float(np.clip(np.tanh(z), -1.0, 1.0))

    # ---------- inference ----------
    def signal(self, row: Dict[str, Any]) -> float:
        X, (mu, sd) = self._build_runtime_vector(row)
        Xn = (X - mu) / sd

        # 1) modello PyTorch
        if self._use_torch and hasattr(self, "torch") and self.model is not None and not isinstance(self.model, tuple):
            self.model.eval()
            with self.torch.no_grad():
                out = self.model(self.torch.tensor(Xn, dtype=self.torch.float32))
            raw = float(np.clip(float(out.cpu().numpy().ravel()[0]), -1.0, 1.0))
            return self._apply_guards_and_bias(raw, row)

        # 2) modello numpy (W,b)
        if self.model is not None and isinstance(self.model, tuple):
            W, b = self.model
            if Xn.shape[1] != W.shape[0]:
                Xn = Xn[:, :W.shape[0]]
            z = Xn @ W + b
            raw = float(np.clip(np.tanh(float(z.ravel()[0])), -1.0, 1.0))
            return self._apply_guards_and_bias(raw, row)

        # 3) nessun training: fallback
        if self.feature_kind == "mtf":
            raw = self._heuristic_score_mtf(row)
        else:
            # semplice combinazione a 2D
            z = float(Xn.ravel()[0]) * 0.8 + float(Xn.ravel()[1]) * 0.6
            raw = float(np.clip(np.tanh(z), -1.0, 1.0))
        return self._apply_guards_and_bias(raw, row)

class InventorStrategy(Strategy):
    """Genera formule a blocchi e ne seleziona la migliore via ricerca stocastica."""
    name = "Inventor"

    def __init__(self, seed: int = 17, iters: int = 120):
        super().__init__(seed=seed, iters=iters)
        self.rng = np.random.default_rng(seed)
        self.best = None
        self.best_score = -1e9

    @staticmethod
    def _s(x):  # squash
        x = np.clip(x, -10, 10)
        return np.tanh(x)

    @staticmethod
    def _safe_series(x):
        x = pd.to_numeric(x, errors="coerce").astype(float)
        return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _eval(self, df: pd.DataFrame, p: Dict[str, float]) -> float:
        m24 = self._safe_series(df.get("mom24h", 0.0))
        m48 = self._safe_series(df.get("mom48h", 0.0))
        tgt = self._safe_series(df.get("ret24h", 0.0))
        # blocchi
        f1 = self._s(p["a1"] * m24 + p["a2"] * m48 - p["t1"])
        f2 = self._s(p["b1"] * (m24 - m48) - p["t2"])
        f3 = self._s(p["c1"] * (m24.rolling(3, min_periods=1).mean()) - p["t3"])
        pred = self._s(p["w1"] * f1 + p["w2"] * f2 + p["w3"] * f3)
        # score = correlazione con il target (robusta a scala)
        if float(pred.std()) < 1e-12 or float(tgt.std()) < 1e-12:
            return -1e9
        return float(np.nan_to_num(pd.Series(pred).corr(tgt), nan=-1e9))

    def fit(self, df: pd.DataFrame) -> "Strategy":
        if df is None or df.empty or "ret24h" not in df.columns:
            return self
        for _ in range(int(self.params.get("iters", 120))):
            cand = {
                "a1": self.rng.uniform(-3, 3), "a2": self.rng.uniform(-3, 3), "t1": self.rng.uniform(-0.02, 0.02),
                "b1": self.rng.uniform(-3, 3), "t2": self.rng.uniform(-0.02, 0.02),
                "c1": self.rng.uniform(-3, 3), "t3": self.rng.uniform(-0.02, 0.02),
                "w1": self.rng.uniform(-2, 2), "w2": self.rng.uniform(-2, 2), "w3": self.rng.uniform(-2, 2),
            }
            sc = self._eval(df, cand)
            if sc > self.best_score:
                self.best_score, self.best = sc, cand
        return self

    def signal(self, row: Dict[str, Any]) -> float:
        info = (row.get("info") or {})
        m24 = float((info.get("24H") or {}).get("change_pct") or 0.0)
        m48 = float((info.get("48H") or {}).get("change_pct") or 0.0)
        p = self.best or {"a1": 1.0, "a2": 0.0, "t1": 0.0, "b1": 1.0, "t2": 0.0,
                          "c1": 0.5, "t3": 0.0, "w1": 0.6, "w2": 0.3, "w3": 0.1}
        f1 = np.tanh(p["a1"]*m24 + p["a2"]*m48 - p["t1"])
        f2 = np.tanh(p["b1"]*(m24 - m48) - p["t2"])
        f3 = np.tanh(p["c1"]*((m24 + m48*0.5)) - p["t3"])
        score = np.tanh(p["w1"]*f1 + p["w2"]*f2 + p["w3"]*f3)
        return float(np.clip(score, -1.0, 1.0))

class MicrostructureStrategy(Strategy):
    name = "Micro"
    def __init__(self, spread_max=0.003, slip_max=0.02, liq_w=1.0):
        super().__init__(spread_max=spread_max, slip_max=slip_max, liq_w=liq_w)

    def signal(self, row: Dict[str, Any]) -> float:
        now = (row.get("info") or {}).get("NOW", {}) or {}
        px = now.get("current_price") or now.get("last")
        if not px: return 0.0
        px = float(px)

        spread = float(now.get("spread") or 0.0) / max(px, 1e-12)
        if spread > float(self.params.get("spread_max", 0.003)):
            return 0.0  # ambiente costoso → niente segnale

        bid_sum = float(now.get("liquidity_bid_sum") or 0.0)
        ask_sum = float(now.get("liquidity_ask_sum") or 0.0)
        skew = (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-12)  # [-1,1]

        slip_b = float(now.get("slippage_buy_pct") or 0.0)
        slip_s = float(now.get("slippage_sell_pct") or 0.0)
        slip = 0.5 * (slip_b + slip_s)

        # qualità mercato: 1→ok, 0→pessimo
        q = (1.0 - np.tanh(slip / float(self.params.get("slip_max", 0.02))))
        s = np.tanh(float(self.params.get("liq_w", 1.0)) * skew) * q
        return float(np.clip(s, -1.0, 1.0))


class TrendPullbackStrategy(Strategy):
    name = "TrendPB"
    def __init__(self, pull=0.004, thr=0.10):
        super().__init__(pull=pull, thr=thr)

    def signal(self, row: Dict[str, Any]) -> float:
        now = (row.get("info") or {}).get("NOW", {}) or {}
        px = now.get("current_price") or now.get("last")
        e50 = now.get("ema50_1h"); e200 = now.get("ema200_1h")
        if px is None or e50 is None or e200 is None: return 0.0
        px = float(px); e50=float(e50); e200=float(e200)

        trend = np.tanh((e50 - e200) / (abs(e200) + 1e-12))  # >0 uptrend
        dev = (px - e50) / (abs(e50) + 1e-12)                # deviazione da EMA50

        s_long  = (trend > float(self.params["thr"]))  and (dev <  0) and (abs(dev) <= float(self.params["pull"]))
        s_short = (trend < -float(self.params["thr"])) and (dev >  0) and (abs(dev) <= float(self.params["pull"]))
        score = (1.0 if s_long else 0.0) - (1.0 if s_short else 0.0)
        return float(score)

class SqueezeBreakoutStrategy(Strategy):
    name = "SqueezeBO"
    def __init__(self, atr_low_pct=0.0012, or_buf=0.0008):
        super().__init__(atr_low_pct=atr_low_pct, or_buf=or_buf)

    def signal(self, row: Dict[str, Any]) -> float:
        info = (row.get("info") or {}); now = info.get("NOW", {}) or {}
        px = now.get("current_price") or now.get("last")
        if not px: return 0.0
        px = float(px)
        # usa ATR 1h oppure 24h come proxy
        a1h = (info.get("1H") or {}).get("atr"); a24 = (info.get("24H") or {}).get("atr")
        atr = float(a1h or a24 or 0.0)
        if atr <= 0: return 0.0

        or_ok = bool(now.get("or_ok"))
        hi = now.get("or_high"); lo = now.get("or_low")
        if not or_ok or hi is None or lo is None: return 0.0

        low_vol = atr / max(px, 1e-12) < float(self.params["atr_low_pct"])  # compressione
        buf = float(self.params["or_buf"])
        long_trg  = low_vol and px >= float(hi) * (1.0 + buf)
        short_trg = low_vol and px <= float(lo) * (1.0 - buf)
        return float((1.0 if long_trg else 0.0) - (1.0 if short_trg else 0.0))

# ---------------------- pesi ----------------------

# --- aggiungi campi opzionali ---
@dataclass
class WeightConfig:
    lr: float = 0.48
    decay: float = 0.97
    temp: float = 1.6
    jitter: float = 1e-3
    # NUOVI (opzionali, hanno default):
    reward_scale: float = 50.0   # scala del PnL prima della tanh
    min_raw: float = 1e-4        # floor più alto per evitare saturazione
    max_raw: float = 1e4         # cap simmetrico

class AdaptiveWeights:
    def __init__(self, names: List[str], cfg: WeightConfig, state_path: Optional[str] = None):
        self.names = names
        self.cfg = cfg
        self.state_path = state_path
        self.raw = np.ones(len(names), dtype=float) + np.random.default_rng(7).uniform(-cfg.jitter, cfg.jitter, len(names))
        self.perf_ema = np.zeros(len(names), dtype=float)
        self._load()

    # --- utility nuove ---
    def reset(self):
        """Riparte da pesi uniformi (softmax ~ 1/N)."""
        self.raw[:] = 1.0
        self.perf_ema[:] = 0.0
        self._save()

    def _repair_if_degenerate(self):
        """Se i pesi sono tutti al floor e il softmax è piatto, resetta."""
        w = softmax(self.raw, temp=self.cfg.temp)
        if (np.all(self.raw <= self.cfg.min_raw * 1.01) and
            np.allclose(w, np.ones_like(w) / len(w), atol=1e-9)):
            self.reset()

    def _save(self):
        if not self.state_path:
            return
        safe_write_json(self.state_path, {
            "names": self.names,
            "raw": self.raw.tolist(),
            "perf_ema": self.perf_ema.tolist()
        })

    def scores(self) -> Dict[str, float]:
        """
        Ritorna i pesi normalizzati (softmax) in forma 0..1 (quasi-1 escluso),
        comodo per logging/telemetria e compatibile con il vecchio flow.
        """
        w_soft = softmax(self.raw, temp=self.cfg.temp)
        return {name: to_score01(float(x)) for name, x in zip(self.names, w_soft)}

    @property
    def weights_softmax(self) -> np.ndarray:
        return softmax(self.raw, temp=self.cfg.temp)

    # --- invariata ---
    def _load(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            d = safe_read_json(self.state_path)
            if d.get("names") == self.names:
                self.raw = np.array(d.get("raw", self.raw), dtype=float)
                self.perf_ema = np.array(d.get("perf_ema", self.perf_ema), dtype=float)
        except Exception:
            pass

    def blend(self, signals: List[float]) -> Tuple[float, np.ndarray]:
        w = softmax(self.raw, temp=self.cfg.temp)
        s = np.array(signals, dtype=float)
        comp = float(np.sum(w * np.sign(s) * (np.abs(s) ** 1.25)))
        return float(np.clip(comp, -1.0, 1.0)), w

    # --- PATCH DELL’UPDATE ---
    def update(self, contrib_w: np.ndarray, realized_pnl: float):
        """
        Update 'relativo' + reward shaping:
        - normalizza i contributi
        - reward = tanh(PnL/scala)
        - centra la performance (meno la media) per premiare i migliori *relativi*
        - moltiplicativo con clip + autoriparazione in caso di degeneracy
        """
        if contrib_w is None:
            return
        s = float(np.sum(contrib_w))
        if s <= 1e-12:
            return

        # normalizza contributi (>=0, somma=1)
        alloc = contrib_w / s

        # reward shaping per stabilità ([-1,+1])
        r = float(np.tanh(float(realized_pnl or 0.0) / self.cfg.reward_scale))
        if abs(r) < 1e-6:
            # PnL quasi nullo → nessun update (rumore)
            return

        perf = alloc * r
        perf_centered = perf - perf.mean()  # relativo ai pari

        # EMA delle performance
        self.perf_ema = self.cfg.decay * self.perf_ema + (1 - self.cfg.decay) * perf_centered
        self.perf_ema = np.clip(self.perf_ema, -1.0, 1.0)

        # update moltiplicativo con clip più “alto”
        self.raw *= np.exp(self.cfg.lr * self.perf_ema)
        self.raw = np.clip(self.raw, self.cfg.min_raw, self.cfg.max_raw)

        # se saturato al floor e piatto → reset
        self._repair_if_degenerate()
        self._save()




# ================ AIEnsembleTrader (STATEFUL) ================

class AIEnsembleTrader:
    """
    - Istanzia UNA VOLTA (passi path/config).
    - Chiama run(currencies, actions, replace=True/False) per generare azioni.
    - Output come prima: {"scores": {...}, "actions_ai": [...]}
    """

    # -------------------- INIT --------------------
    def __init__(self,
                 currencies: Optional[List[Dict[str, Any]]] = None,
                 actions: Optional[List[Dict[str, Any]]] = None,
                 inputStorico: Optional[str] = None,
                 outputStorico: Optional[str] = None,
                 budget_eur: Optional[float] = None,
                 per_trade_cap_eur: Optional[float] = None,
                 cfg: Optional[WeightConfig] = None,
                 debug_signals: bool = False,
                 # policy allocator
                 capital_manager: bool = True,
                 enter_thr: float = 0.42,
                 exit_thr: float = -0.32,
                 strong_thr: float = 0.6,
                 use_market_for_strong: bool = True,
                 min_order_eur: float = 35.0,
                 respect_pair_limits: bool = True,
                 safety_buffer_pct: float = 0.005,  # 0.5% per evitare insufficient funds
                 # tollero extra kwargs (es. cash_floor_pct) senza usarli
                 **_unused_kwargs):
        self._currencies: List[Dict[str, Any]] = list(currencies or [])
        self._actions_in: List[Dict[str, Any]] = list(actions or [])

        self.input_dir = ensure_dir(inputStorico) if inputStorico else None
        self.output_dir = ensure_dir(outputStorico) if outputStorico else None
        self.per_trade_cap_eur = per_trade_cap_eur
        self.debug_signals = debug_signals

        self.capital_manager = capital_manager
        self.enter_thr = float(enter_thr)
        self.exit_thr = float(exit_thr)
        self.strong_thr = float(strong_thr)
        self.use_market_for_strong = bool(use_market_for_strong)
        self.min_order_eur = float(min_order_eur)
        self.respect_pair_limits = bool(respect_pair_limits)
        self.safety_buffer_pct = float(safety_buffer_pct)

        # --- operational risk params (PATCH) ---
        self.allow_margin_short = True   # short su margine: attivo di default
        self.leverage = 2              # rischio medio-basso

        # livelli dinamici SL/TP e trailing
        self.atr_mult_sl = 1.2           # SL ≈ 1.2 * ATR(1h) minimo
        self.atr_mult_tp = 2.0           # TP ≈ 2.0 * ATR(1h) minimo
        self.min_stop_pct = 0.006        # 0.6% se ATR è molto basso
        self.min_take_pct = 0.012        # 1.2% minimo
        self.trail_mult = 1.5            # trailing aggressivo ≈ 1.5 * ATR
        # --- tracking del blend per pair (EMA + last) ---
        self.blend_track: Dict[str, Dict[str, float]] = {}

        # persistenza last/ema blend
        self.blend_state_path = os.path.join(self.output_dir, "blend_state.json") if self.output_dir else None
        self._blend_state_last_save_ts = 0.0
        self._load_blend_state()

        # iperparametri “debolezza”
        self.blend_ema_alpha: float = 0.25   # EMA del blend (0.1–0.3 tipico)
        self.weak_drop: float = 0.15         # quanto sotto l’EMA considero “debolezza”
        self.weak_reduce: float = 0.33       # frazione di posizione da chiudere parzialmente
        # stato per trailing/entry per pair
        self.pos_state: Dict[str, Dict[str, Any]] = {}
        self.universe_map: Dict[str, Dict[str, Any]] = {}

        # --- execution & cost controls ---
        self.fee_taker_pct = 0.0026  # 0.26% stimato (taker)
        self.fee_maker_pct = 0.0016  # 0.16% stimato (maker)
        self.min_order_eur = 20.0    # notional minimo per evitare micro-ordini
        self.cooldown_s = 120         # antirimbalzo per pair
        self.max_sells_rebalance = 3 # cap al numero di sell nel rebalance
        self.edge_over_cost_mult = 2.7  # quanto il segnale deve superare i costi
        self._last_order_ts: Dict[str, float] = {}

        # tracking della quota EUR "committata" nel batch corrente
        self._session_quote_left: Optional[float] = None


        # path file di stato per la policy (nella stessa cartella dei weights)
        self.policy_state_path = os.path.join(self.output_dir or ".", "policy_state.json")
        self.policy = PriceSizeFeasPolicy(self.policy_state_path)

        neural = NeuralStrategy(hidden=32, epochs=80, lr=1e-3)
        neural._use_torch = True
        self.planner_cfg = PlannerConfig()
        self._run_memory = {}   # per ricordare azioni già emesse nella run (pair -> dict)

        self.enable_margin_short = getattr(self, "enable_margin_short", True)  # short OFF di default
        self.take_profit_pct     = getattr(self, "take_profit_pct", 0.01)       # TP 1% sopra il prezzo medio (esempio)
        self.exit_frac_default   = getattr(self, "exit_frac_default", 0.33)     # frazione di uscita spot

        self.strategies: List[Strategy] = [
            MomentumStrategy(),
            ValueStrategy(),
            PairsStrategy(universe=self.universe_map),
            # EarningsEventStrategy(),
            MLStrategy(),
            MeanReversionSRStrategy(),
            neural,
            InventorStrategy(seed=15),
            TrendPullbackStrategy(),
            SqueezeBreakoutStrategy(),
            MicrostructureStrategy(),
            # TFTStrategy(),
            LGBMStrategy()
        ]

        weights_state = os.path.join(self.output_dir, "weights.json") if self.output_dir else None
        self.weights = AdaptiveWeights([s.name for s in self.strategies], cfg or WeightConfig(), state_path=weights_state)
        # self.weights.reset()
        self.budget_eur = budget_eur if budget_eur is not None else self._infer_budget(self._currencies)
        if self.per_trade_cap_eur is None:
            self.per_trade_cap_eur = max(0.0, float(self.budget_eur) * 0.10 if self.budget_eur is not None else 0.0)

        self.portfolio_snapshot: Dict[str, Any] = {}
        self._last_kraken_pnl_eur: Optional[float] = None

        if self.input_dir and self._currencies:
            self.export_input(self._currencies)

        self.planner = ActionPlanner(self)

    # -------------------- gestione stato dati --------------------

    def attach_data(self,
                    currencies: Optional[List[Dict[str, Any]]] = None,
                    actions: Optional[List[Dict[str, Any]]] = None,
                    *,
                    replace: bool = True) -> None:
        if currencies:
            if replace: self._currencies = list(currencies)
            else:       self._currencies.extend(list(currencies))
        if actions:
            if replace: self._actions_in = list(actions)
            else:       self._actions_in.extend(list(actions))

        self.universe_map.clear()
        for c in self._currencies:
            key = c.get("pair") or f"{c.get('base')}/{c.get('quote')}"
            self.universe_map[key] = c
        for s in self.strategies:
            if isinstance(s, PairsStrategy):
                s.universe = self.universe_map

        self.budget_eur = self._infer_budget(self._currencies)

    # -------------------- Kraken helpers --------------------

    @staticmethod
    def _get_kraken_api_from_env():
        key = os.environ.get("KRAKEN_API_KEY") or os.environ.get("KRAKEN_KEY")
        sec = os.environ.get("KRAKEN_API_SECRET") or os.environ.get("KRAKEN_SECRET")
        if not key or not sec:
            raise RuntimeError("Mancano le credenziali Kraken nell'ENV.")
        try:
            import krakenex  # type: ignore
            return krakenex.API(key=key, secret=sec)
        except Exception as e:
            raise RuntimeError("Impossibile inizializzare krakenex.API") from e

    def refresh_portfolio(self) -> Dict[str, Any]:
        k = self._get_kraken_api_from_env()
        snap: Dict[str, Any] = {}

        try:
            bal = k.query_private("Balance")
            snap["balance"] = bal.get("result", {})
        except Exception as e:
            snap["balance_error"] = str(e)

        since = int(time.time()) - 3*24*3600
        try:
            led = k.query_private("Ledgers", {"start": since})
            snap["ledgers"] = led.get("result", {}).get("ledger", {})
        except Exception as e:
            snap["ledgers_error"] = str(e)

        try:
            tr = k.query_private("TradesHistory", {"start": since})
            snap["trades"] = tr.get("result", {}).get("trades", {})
        except Exception as e:
            snap["trades_error"] = str(e)

        self.portfolio_snapshot = snap
        return snap

    # -------------------- history IO --------------------

    def _list_json(self, folder: Optional[str], prefix: str) -> List[str]:
        if not folder: return []
        return sorted(glob.glob(os.path.join(folder, f"{prefix}*.json")))

    def load_history_inputs(self) -> List[List[Dict[str, Any]]]:
        return [safe_read_json(p) for p in self._list_json(self.input_dir, "input_")]

    def load_history_outputs(self) -> List[List[Dict[str, Any]]]:
        return [safe_read_json(p) for p in self._list_json(self.output_dir, "output_")]

    # -------------------- features/precision --------------------
    def _day_start_ts(self) -> float:
        # mezzanotte locale/UTC coerente con i tuoi NOW.day_start
        return float(((time.time() // 86400) * 86400))

    def _pnl_realized_since(self, cur: dict, ts_from: float) -> float:
        trds = ((cur.get("portfolio") or {}).get("trades") or [])
        pnl = 0.0
        for t in trds:
            try:
                # eur out/in: usa fee e cost e side; semplifichiamo: maker/swap ignorati
                side = (t.get("type") or "").lower()
                cost = float(t.get("cost") or 0.0)
                fee  = float(t.get("fee") or 0.0)
                # segno: sell incassa, buy spende (gross)
                pnl += (cost if side == "sell" else -cost) - fee
            except Exception:
                pass
        # se hai già avg_buy/pnl_pct in portfolio.row puoi combinare, qui basico
        return pnl  # eur approx


    @staticmethod
    def _round_qty_to_step(qty: float, lot_decimals: int) -> float:
        """Arrotonda **per difetto** alla granularità di volume consentita (lot_decimals)."""
        if qty <= 0: return 0.0
        step = 10.0 ** (-int(max(0, lot_decimals)))
        return math.floor(qty / step) * step

    @staticmethod
    def _pair_limits_of(cur: Dict[str, Any]) -> Tuple[int, float]:
        """Ritorna (lot_decimals, ordermin) dal campo 'pair_limits' dello snapshot, con fallback prudente."""
        lim = (cur.get("pair_limits") or {})
        lot_decimals = int(lim.get("lot_decimals", 5))
        ordermin = float(lim.get("ordermin", 0.0) or 0.0)
        return lot_decimals, ordermin

    def _affordable_cap_eur(self, cur: Dict[str, Any]) -> float:
        """
        Riduce il cap per trade per non sforare la disponibilità del quote (con safety buffer)
        e non superare il budget già "prenotato" nel batch (sessione).
        """
        cap_cfg = float(self.per_trade_cap_eur)  # valore target per singolo trade
        avail = float(((cur.get("portfolio") or {}).get("available") or {}).get("quote") or cap_cfg)
        avail_net = max(0.0, avail * (1.0 - self.safety_buffer_pct))
        base_cap = float(min(cap_cfg, avail_net))
        # anche il budget di sessione: se è None non limita
        return self._session_affordable_eur(base_cap)


    def _session_affordable_eur(self, fallback_cap: float) -> float:
        """
        Quanto EUR posso davvero usare in questo istante, considerando la sessione?
        Non rompe se _session_quote_left è None.
        """
        if self._session_quote_left is None:
            return float(fallback_cap)
        return float(max(0.0, min(self._session_quote_left, fallback_cap)))

    def _session_commit(self, eur_used: float) -> bool:
        """
        Scala la quota di sessione; ritorna False se non c'era abbastanza spazio (non committare).
        Safe se _session_quote_left è None (in quel caso lascia passare).
        """
        try:
            x = float(eur_used)
        except Exception:
            return False
        if self._session_quote_left is None:
            return True
        if x <= 0.0:
            return True
        if self._session_quote_left + 1e-12 < x:
            return False
        self._session_quote_left -= x
        return True



    def _enforce_min_constraints(self, cur: Dict[str, Any], price: float, raw_qty: float) -> Tuple[bool, float, str]:
        """
        Applica SOLO:
        - granularità di volume (lot_decimals) con floor
        - check di ordemin (se presente)
        Niente controlli/cap/notional in EUR e nessun auto-upsize alla ordemin.
        """
        lot_decimals, ordermin = self._pair_limits_of(cur)
        q = self._round_qty_to_step(max(0.0, float(raw_qty)), lot_decimals)
        if ordermin > 0.0 and q + 1e-18 < float(ordermin):
            return (False, 0.0, f"qty {q:.8f} < ordemin {float(ordermin)}")
        return (q > 0.0, float(q), "")


    @staticmethod
    def _infer_budget(currencies: List[Dict[str, Any]]) -> float:
        quotes = []
        for c in currencies or []:
            av = ((c.get("portfolio") or {}).get("available") or {}).get("quote")
            if av is not None:
                try: quotes.append(float(av))
                except Exception: pass
        return float(max(quotes) if quotes else 0.0)

    def _flatten_df(self) -> pd.DataFrame:
        rows = []
        for c in self._currencies:
            info = c.get("info", {})
            rows.append({
                "pair": c.get("pair") or f"{c.get('base')}/{c.get('quote')}",
                "mom24h": (info.get("24H", {}) or {}).get("change_pct", 0.0),
                "mom48h": (info.get("48H", {}) or {}).get("change_pct", 0.0),
                "ret24h": (info.get("24H", {}) or {}).get("change_pct", 0.0),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _infer_price_decimals(price: float) -> int:
        if price is None or price <= 0: return 2
        if price >= 1: return 2
        if price >= 0.1: return 4
        return 6

    @staticmethod
    def _round_price(pair_limits: Dict[str, Any], price: Optional[float]) -> Optional[float]:
        if price is None: return None
        tick = pair_limits.get("tick_size") or pair_limits.get("tick")
        if tick:
            try:
                q = Decimal(str(tick)); p = Decimal(str(price))
                return float((p / q).to_integral_value(rounding=ROUND_FLOOR) * q)
            except Exception: pass
        dec = pair_limits.get("price_decimals") or pair_limits.get("pair_decimals")
        if dec is None: dec = AIEnsembleTrader._infer_price_decimals(float(price))
        try:
            p = Decimal(str(price)); quant = Decimal(1).scaleb(-int(dec))
            return float(p.quantize(quant, rounding=ROUND_HALF_UP))
        except Exception:
            return float(round(price, int(dec)))

    @staticmethod
    def _calc_qty(cap_eur: float, price: float, lot_decimals: int, ordermin: Optional[float]) -> float:
        if price is None or price <= 0 or cap_eur <= 0: return 0.0
        raw_qty = Decimal(str(cap_eur)) / Decimal(str(price))
        step = Decimal(1).scaleb(-int(lot_decimals))
        qty = (raw_qty / step).to_integral_value(rounding=ROUND_FLOOR) * step
        if ordermin is not None and float(qty) < float(ordermin): return 0.0
        return float(qty)

    def _best_quotes(self, cur: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        now = (cur.get("info") or {}).get("NOW", {}) or {}
        def f(k):
            try:
                v = now.get(k)
                return None if v is None else float(v)
            except Exception:
                return None
        return f("bid"), f("ask"), f("last")

    def _choose_limit_price(self, cur: Dict[str, Any], side: str) -> Optional[float]:
        """
        Sceglie un prezzo LIMIT che tenda a fare maker:
        - BUY -> al best bid (arrotondato), se non c'è uso last/ mid se presenti
        - SELL -> al best ask (arrotondato)
        Non esplode se mancano dati.
        """
        limits = (cur.get("pair_limits") or {})
        bid, ask, last = self._best_quotes(cur)
        px = None
        if side == "buy":
            px = bid if bid is not None else (last if last is not None else None)
        else:
            px = ask if ask is not None else (last if last is not None else None)
        return self._round_price(limits, px)


    # -------------------- signals/blend --------------------

    # def _signals_for_currency(self, cur: Dict[str, Any]):
    #     sigs = []
    #     for s in self.strategies:
    #         try:
    #             if isinstance(s, AutoStrategy):
    #                 s.fit(self._flatten_df())
    #             v = float(s.signal(cur))
    #         except Exception:
    #             v = 0.0
    #         if not np.isfinite(v):  # NaN/Inf -> 0
    #             v = 0.0
    #         sigs.append(v)

    #     w_soft = self.weights.weights_softmax
    #     return self._blend_masked_floor(sigs, w_soft, eps=0.02)


    # def _blend_masked_floor(self, sigs: List[float], w_soft: np.ndarray,
    #                         eps: float = 0.02) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    #     s = np.array(sigs, dtype=float)
    #     w = np.array(w_soft, dtype=float)

    #     # --- SANITY ---
    #     s[~np.isfinite(s)] = 0.0                    # NaN/Inf -> 0 (segnale neutro)
    #     w[~np.isfinite(w)] = 0.0                    # NaN/Inf -> 0
    #     if w.sum() <= 0:                            # evita divisione per 0
    #         w = np.ones_like(w) / max(1, len(w))

    #     valid = (np.isfinite(s) & (np.abs(s) > 1e-6))
    #     if valid.any():
    #         w_eff = w * valid
    #         w_eff = w_eff / (w_eff.sum() + 1e-12)
    #     else:
    #         w_eff = w.copy()

    #     K = len(w_eff)
    #     w_eff = (w_eff + eps) / (w_eff.sum() + eps * K)

    #     contribs = w_eff * np.sign(s) * (np.abs(s) ** 1.25)
    #     # sanifica eventuali residui
    #     contribs[~np.isfinite(contribs)] = 0.0

    #     blend = float(np.clip(np.nansum(contribs), -1.0, 1.0))
    #     return blend, w, w_eff, contribs

    def _signals_for_currency(self, cur: Dict[str, Any]) -> Tuple[List[float], float, np.ndarray, np.ndarray, np.ndarray]:
        sigs: List[float] = []
        for s in self.strategies:
            if isinstance(s, AutoStrategy):
                s.fit(self._flatten_df())
            sigs.append(s.signal(cur))

        w_soft = self.weights.weights_softmax
        # # blended, w = self.weights.blend(sigs)
        blended, w_soft, w_eff, contribs = self._blend_masked_floor(sigs, w_soft, eps=0.02)
        return  sigs, blended, w_soft, w_eff, contribs

    def _blend_masked_floor(self, sigs: List[float], w_soft: np.ndarray,
                        eps: float = 0.02) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applica:
        1) maschera strategie 'valide' (|signal|>1e-6)
        2) rinormalizzazione su valide
        3) floor epsilon su tutti i pesi
        Ritorna: (blend, w_soft, w_eff, contribs)
        """
        s = np.array(sigs, dtype=float)
        w = np.array(w_soft, dtype=float)

        valid = (np.abs(s) > 1e-6)
        if valid.any():
            w_eff = w * valid
            w_eff = w_eff / (w_eff.sum() + 1e-12)
        else:
            w_eff = w.copy()

        # epsilon-floor (es. 2%)
        K = len(w_eff)
        w_eff = (w_eff + eps) / (w_eff.sum() + eps * K)

        # contributi (stessa non-linearità del tuo weights.blend)
        contribs = w_eff * np.sign(s) * (np.abs(s) ** 1.25)
        blend = float(np.clip(np.sum(contribs), -1.0, 1.0))
        return blend, w, w_eff, contribs




    # -------------------- action helpers --------------------

    def _fee_pct_from_now(self, now: Dict[str, Any], *, taker: bool) -> float:
        key = "fee_taker_pct" if taker else "fee_maker_pct"
        try:
            v = now.get(key)
            return float(v) if v is not None else (self.fee_taker_pct if taker else self.fee_maker_pct)
        except Exception:
            return self.fee_taker_pct if taker else self.fee_maker_pct

    def _est_fee_eur(self, qty: float, price: float, pct: float) -> float:
        return float(max(0.0, qty) * max(0.0, price) * max(0.0, pct))

    def _notional_ok(self, qty: float, price: float) -> bool:
        return (qty or 0.0) * (price or 0.0) >= self.min_order_eur

    def _throttle_ok(self, pair: str) -> bool:
        t = time.time()
        last = self._last_order_ts.get(pair, 0.0)
        if t - last < self.cooldown_s:
            return False
        self._last_order_ts[pair] = t
        return True


    def _action_schema(self, tipo, pair, prezzo, quando, quantita, quantita_eur,
                       stop_loss, take_profit, timeframe, lato, break_price,
                       limite, leverage, motivo, meta=None) -> Dict[str, Any]:
        return {
            "tipo": tipo, "pair": pair, "prezzo": prezzo, "quando": quando,
            "quantita": quantita, "quantita_eur": quantita_eur,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "timeframe": timeframe, "lato": lato, "break_price": break_price,
            "limite": limite, "leverage": leverage, "motivo": motivo,
            "meta": meta or {},
        }

    @staticmethod
    def _read_available(cur: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        av = ((cur.get("portfolio") or {}).get("available") or {})
        base = av.get("base"); quote = av.get("quote")
        try: base = None if base is None else float(base)
        except Exception: base = None
        try: quote = None if quote is None else float(quote)
        except Exception: quote = None
        return base, quote



    # subito sotto _read_available(...)
    @staticmethod
    def _read_position_qty(cur: Dict[str, Any]) -> float:
        pos = ((cur.get("portfolio") or {}).get("position") or {})
        try:
            return float(pos.get("base") or 0.0)
        except Exception:
            return 0.0


    @staticmethod
    def _read_position_cost(cur):
        """
        Ritorna (qty_base, avg_price). Se non disponibili, 0.0, 0.0
        """
        pos = ((cur.get("portfolio") or {}).get("position") or {})
        try:
            q = float(pos.get("base") or 0.0)
        except Exception:
            q = 0.0
        try:
            ap = float(pos.get("avg_price") or 0.0)
        except Exception:
            ap = 0.0
        return q, ap

    @staticmethod
    def _read_short_capacity(cur):
        lim = (cur.get("pair_limits") or {})
        lev_max_short = float(lim.get("leverage_short_max") or lim.get("leverage_max") or 1.0)
        _, eur_av = AIEnsembleTrader._read_available(cur)
        return float(lev_max_short), float(eur_av or 0.0)


    def _score_to_cap(self, blend: float) -> float:
        b = float(abs(blend))
        lo, hi = max(1e-6, self.enter_thr), max(self.enter_thr + 1e-6, self.strong_thr)
        x = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
        scale = float(1 / (1 + np.exp(-6 * (x - 0.5))))
        return float(self.per_trade_cap_eur * max(0.25, scale))

    def _mk_market_action(self, pair, side, qty, price_now, tf, motivo, meta):
        # ritorna il VECCHIO SCHEMA completo
        return self._action_schema(
            side,           # tipo (buy/sell)
            pair,           # pair
            float(price_now) if price_now is not None else None,  # prezzo (solo per UI/log)
            "market",       # quando
            float(qty),     # quantita (BASE)
            None,           # quantita_eur
            None,           # stop_loss
            None,           # take_profit
            tf,             # timeframe
            side,           # lato (buy/sell)
            None,           # break_price
            None,           # limite
            None,           # leverage
            motivo,         # motivo
            meta or {},     # meta
        )


    def _mk_limit_action(self, pair, side, qty, px, tf, motivo, meta,
                        stop=None, take=None, break_px=None, limit_px=None):
        # VECCHIO SCHEMA con "limit" e campi opzionali
        return self._action_schema(
            side,                  # tipo
            pair,
            float(px) if px is not None else None,   # prezzo (UI)
            "limit",               # quando
            float(qty),            # quantita
            None,                  # quantita_eur
            float(stop) if stop is not None else None,          # stop_loss (eventuale)
            float(take) if take is not None else None,          # take_profit (eventuale)
            tf,
            side,                  # lato
            float(break_px) if break_px is not None else None,  # break_price (se vuoi log)
            float(limit_px) if limit_px is not None else None,  # limite (se vuoi log)
            None,                  # leverage (lo gestisce il runner)
            motivo,
            meta or {},
        )


    # --- NEW: quantité helper robusto ---
    def _qty_floor(self, qty: float, lot_decimals_or_step) -> float:
        """
        Arrotonda per difetto la quantità al tick del pair.
        - Se lot_decimals_or_step è int -> numero di decimali (es. 8)
        - Se è float -> step di lotto (es. 0.001)
        """
        try:
            q = float(qty)
            if q <= 0:
                return 0.0
            # int => decimali
            if isinstance(lot_decimals_or_step, int):
                dec = max(0, lot_decimals_or_step)
                step = 10.0 ** (-dec)
            else:
                step = float(lot_decimals_or_step)
                if step <= 0:
                    step = 1e-8
                # prova a dedurre i decimali dallo step
                dec = max(0, min(12, str(step)[::-1].find('.')))
            import math
            qf = math.floor(q / step) * step
            # normalizza ai decimali dedotti
            return float(round(qf, dec))
        except Exception:
            return float(qty or 0.0)

    # --- NEW: builders per stop/take espliciti ---
    def _mk_stop_action(
            self,
            pair: str,
            side: str,
            qty: float,
            stop_px: float,
            tf: str,
            motivo: str,
            meta: dict | None = None,
            limit_px: float | None = None,
            reduce_only: bool = True,
            leverage: str | None = None,
        ):
        m = dict(meta or {})
        if reduce_only:
            # reduce_only rimane **solo** dentro meta; non nei campi top-level
            m["reduce_only"] = True

        # NB: "quando" = "al_break" e mettiamo anche break_price
        return self._action_schema(
            side,                               # tipo (buy/sell)
            pair,
            float(stop_px),                     # prezzo (UI)
            "al_break",                         # quando -> il runner crea stop/stop-limit
            float(qty),
            None,                               # quantita_eur
            None,                               # stop_loss (non usato dal runner per STOP)
            None,                               # take_profit
            tf,
            side,                               # lato
            float(stop_px),                     # break_price -> TRIGGER
            float(limit_px) if limit_px is not None else None,  # limite (per stop-limit)
            leverage,                           # (pass-through, ma può restare None)
            motivo,
            m,
        )


    def _mk_take_action(
        self,
        pair: str,
        side: str,
        qty: float,
        take_px: float,
        tf: str,
        motivo: str,
        meta: dict | None = None,
        reduce_only: bool = True,
        leverage: str | None = None,
    ):
        """
        Crea un'azione di tipo TAKE-PROFIT come ordine LIMIT dalla parte opposta.
        - side: 'sell' per chiudere long; 'buy' per chiudere short
        - reduce_only per close-only
        """
        meta = dict(meta or {})
        if reduce_only:
            meta["reduce_only"] = True
        return self._action_schema(
            side,                # tipo
            pair,
            take_px,             # prezzo (UI)
            "limit",             # quando
            qty,                 # quantita
            None,                # quantita_eur
            None,                # stop_loss
            None,                # take_profit
            tf,
            side,                # lato
            None,                # break_price
            None,                # limite aggiuntivo (non usato qui)
            leverage,
            motivo,
            meta,
        )


    # --- PATCH: ATR/SL/TP/trailing helpers ---
    def _read_atr(self, cur: Dict[str, Any]) -> float:
        info = (cur.get("info") or {})
        b1h = (info.get("1H") or info.get("60M") or {})
        try:
            return float(b1h.get("atr") or 0.0)
        except Exception:
            return 0.0

    def _load_blend_state(self) -> None:
        """Carica da disco la memoria del blend (ema/last) per pair, se esiste."""
        path = getattr(self, "blend_state_path", None)
        if not path or not os.path.exists(path):
            return
        try:
            d = safe_read_json(path)
            payload = d.get("data", d) if isinstance(d, dict) else {}
            if isinstance(payload, dict):
                clean = {}
                for k, v in payload.items():
                    if isinstance(v, dict):
                        ema  = float(v.get("ema", 0.0))
                        last = float(v.get("last", ema))
                        clean[k] = {"ema": ema, "last": last}
                if clean:
                    self.blend_track.update(clean)
        except Exception:
            pass

    def _save_blend_state(self, force: bool = False) -> None:
        """Salva su disco la memoria del blend con un piccolo throttle."""
        path = getattr(self, "blend_state_path", None)
        if not path:
            return
        import time as _t
        now = _t.time()
        if (not force) and (now - getattr(self, "_blend_state_last_save_ts", 0.0) < 10.0):
            return
        try:
            safe_write_json(path, {"ts": now, "data": self.blend_track})
            self._blend_state_last_save_ts = now
        except Exception:
            pass


    def _update_blend_track(self, pair: str, blend: float) -> Dict[str, float]:
        """Aggiorna/storicizza EMA e last del blend per pair e ritorna lo stato."""
        st = self.blend_track.get(pair)
        if not st:
            st = {"ema": float(blend), "last": float(blend)}
        else:
            st["ema"] = (1.0 - self.blend_ema_alpha) * float(st["ema"]) + self.blend_ema_alpha * float(blend)
            st["last"] = float(blend)
        self.blend_track[pair] = st
        self._save_blend_state(force=False)
        return st


    def _dyn_levels(self, side: str, price: float, atr: float):
        if not price or price <= 0:
            return (None, None)
        pct_sl = max(self.min_stop_pct, (self.atr_mult_sl * atr) / max(price, 1e-12))
        pct_tp = max(self.min_take_pct, (self.atr_mult_tp * atr) / max(price, 1e-12))
        if side == "long":
            return (price * (1 - pct_sl), price * (1 + pct_tp))
        else:
            return (price * (1 + pct_sl), price * (1 - pct_tp))

    def _remember_entry(self, pair: str, side: str, qty: float, entry_px: float, sl, tp) -> None:
        self.pos_state[pair] = {"side": side, "qty": float(qty), "entry": float(entry_px),
                                "sl": (float(sl) if sl else None), "tp": (float(tp) if tp else None),
                                "trail_px": (float(sl) if sl else None)}

    def _manage_open_position(self, cur: Dict[str, Any], price: float, base_av: float):
        pair = cur.get("pair") or f"{cur.get('base')}/{cur.get('quote')}"
        st = self.pos_state.get(pair)
        if not st or not price:
            return None
        limits = (cur.get("pair_limits") or {})
        meta = {"position": st}

        atr = self._read_atr(cur)

        # LONG: trailing/SL/TP
        if st["side"] == "long":
            trail_candidate = price * (1 - max(self.min_stop_pct, (self.trail_mult * atr) / max(price, 1e-12)))
            if st.get("trail_px") is not None:
                st["trail_px"] = max(st["trail_px"], trail_candidate)
            stop_px = max(st.get("sl") or 0.0, st.get("trail_px") or 0.0)
            if stop_px and price <= stop_px and base_av > 0:
                qty = min(base_av, st["qty"])
                return self._mk_market_action(pair, "sell", qty, price, "NOW", "Stop/Trail hit", meta)
            if st.get("tp") and price >= st["tp"] and base_av > 0:
                qty = min(base_av, st["qty"])
                px = float(price)
                return self._mk_limit_action(pair, "sell", qty, px, "24H", "Take profit", meta)

        # SHORT: trailing/SL/TP (simmetrico)
        else:
            trail_candidate = price * (1 + max(self.min_stop_pct, (self.trail_mult * atr) / max(price, 1e-12)))
            if st.get("trail_px") is not None:
                st["trail_px"] = min(st["trail_px"], trail_candidate) if st["trail_px"] else trail_candidate
            stop_px = None
            if (st.get("sl") or st.get("trail_px")):
                vals = [p for p in [st.get("sl"), st.get("trail_px")] if p]
                stop_px = min(vals) if vals else None
            if stop_px and price >= stop_px:
                qty = st["qty"]
                return self._mk_market_action(pair, "buy", qty, price, "NOW", "Short stop/trail hit", meta)
            if st.get("tp") and price <= st["tp"]:
                qty = st["qty"]
                px = float(price)
                return self._mk_limit_action(pair, "buy", qty, px, "24H", "Short take profit", meta)

        return None
    # def _rebalance_if_needed(self, target_cap_eur: float, available_eur: float,
    #                             sell_pool: List[Tuple[float, Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    #         actions: List[Dict[str, Any]] = []
    #         need = max(0.0, target_cap_eur - max(0.0, available_eur or 0.0))
    #         if need <= 1e-9 or not sell_pool: return actions
    #         sell_pool = sorted(sell_pool, key=lambda x: x[0])  # blend più negativo prima
    #         remaining = need
    #         for blend, cur, price in sell_pool:
    #             base_av, _ = self._read_available(cur)
    #             if not base_av or base_av <= 0 or not price: continue
    #             lot_dec = int((cur.get("pair_limits") or {}).get("lot_decimals", 6))
    #             qty = float(min(base_av, remaining / max(price, 1e-12)))
    #             step = Decimal(1).scaleb(-lot_dec)
    #             qty = float((Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_FLOOR) * step)
    #             if qty <= 0: continue
    #             pair = cur.get("pair"); motivo = f"Rebalance: libero EUR per opportunità (need≈{need:.2f})."
    #             meta = {"blend": blend, "allocator": "rebalance"}
    #             actions.append(self._mk_market_action(pair, "sell", qty, price, "NOW", motivo, meta))
    #             remaining -= qty * price
    #             if remaining <= 1e-6: break
    #         return actions

    def _rebalance_if_needed(self, target_cap_eur: float, available_eur: float,
                            sell_pool: List[Tuple[float, Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        need = max(0.0, target_cap_eur - max(0.0, available_eur or 0.0))
        if need <= 1e-9 or not sell_pool:
            return actions

        sell_pool = sorted(sell_pool, key=lambda x: x[0])  # prima i blend più negativi
        remaining = need
        sells = 0

        for blend, cur, price in sell_pool:
            if sells >= self.max_sells_rebalance:
                break
            base_av, _ = self._read_available(cur)
            if not base_av or base_av <= 0 or not price:
                continue

            pair = cur.get("pair") or f"{cur.get('base')}/{cur.get('quote')}"
            now  = (cur.get("info") or {}).get("NOW", {}) or {}
            lot_dec = int((cur.get("pair_limits") or {}).get("lot_decimals", 6))

            qty = float(min(base_av, remaining / max(price, 1e-12)))
            step = Decimal(1).scaleb(-lot_dec)
            qty = float((Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_FLOOR) * step)
            if qty <= 0 or not self._notional_ok(qty, float(price)) or not self._throttle_ok(pair):
                continue

            fee_pct = self._fee_pct_from_now(now, taker=True)
            freed_eur = qty * float(price) - self._est_fee_eur(qty, float(price), fee_pct)
            if freed_eur <= 0:
                continue

            motivo = f"Rebalance: libero EUR (need≈{need:.2f}, fee≈{fee_pct*100:.2f}%)."
            meta = {"blend": float(blend), "allocator": "rebalance", "fee_pct": fee_pct}
            actions.append(self._mk_market_action(pair, "sell", qty, price, "NOW", motivo, meta))

            remaining -= freed_eur
            sells += 1
            if remaining <= self.min_order_eur * 0.5:
                break

        return actions

    def _suggest_action(self, cur: Dict[str, Any]) -> Dict[str, Any]:
        pair = cur.get("pair") or f"{cur.get('base')}/{cur.get('quote')}"
        limits = cur.get("pair_limits") or {}
        now = cur.get("info", {}).get("NOW", {})
        price = now.get("current_price") or now.get("last")

        sigs, blended, w_soft, w_eff, contribs = self._signals_for_currency(cur)


        btrk = self._update_blend_track(pair, float(blended))
        conf = float(abs(blended))
        if self.debug_signals:
            print(f"[signals] {pair} -> blend={round(blended,3)}")

        tf = "24H"
        meta = {
            "signals": sigs,
            "weights": w_soft.tolist(),
            "weights_eff": w_eff.tolist(),
            "contribs": contribs.tolist(),
            "w_names": [s.name for s in self.strategies],
            "blend": float(blended),
            "conf": conf,
        }

        if self.capital_manager and price:
            base_av, quote_av = self._read_available(cur)
            lot_dec = int((limits or {}).get("lot_decimals", 6))
            ordermin = limits.get("ordermin")

            # --- Gestione trailing / SL / TP su posizione aperta ---
            mgmt = self._manage_open_position(cur, float(price), float(base_av or 0.0))
            if mgmt:
                return mgmt

            # --- Uscita parziale dinamica su debolezza blend vs EMA ---
            st_pos = self.pos_state.get(pair)
            if st_pos and price:
                lot_dec = int((limits or {}).get("lot_decimals", 6))
                step = Decimal(1).scaleb(-lot_dec)

                if st_pos["side"] == "long":
                    strength = max(0.0, float(blended))
                    ema_strength = max(0.0, float(btrk["ema"]))
                    delta = ema_strength - strength
                    if (0.0 < strength < self.enter_thr) and (delta >= self.weak_drop):
                        base_av, _ = self._read_available(cur)
                        if base_av and base_av > 0:
                            qty = float(base_av * self.weak_reduce)
                            qty = float((Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_FLOOR) * step)
                            if qty > 0:
                                motivo = f"Uscita parziale LONG: debolezza blend vs EMA (Δ={delta:.2f})."
                                meta2 = {**meta, "weak_exit": True, "blend_ema": float(btrk["ema"]), "delta": float(delta)}
                                return self._mk_market_action(pair, "sell", qty, price, "NOW", motivo, meta2)

                else:  # SHORT aperto
                    strength = max(0.0, -float(blended))
                    ema_strength = max(0.0, -float(btrk["ema"]))
                    delta = ema_strength - strength
                    if (0.0 < strength < self.enter_thr) and (delta >= self.weak_drop):
                        qty = float(st_pos.get("qty") or 0.0) * float(self.weak_reduce)
                        qty = float((Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_FLOOR) * step)
                        if qty > 0:
                            motivo = f"Cover parziale SHORT: debolezza blend vs EMA (Δ={delta:.2f})."
                            meta2 = {**meta, "weak_exit": True, "blend_ema": float(btrk["ema"]), "delta": float(delta)}
                            return self._mk_market_action(pair, "buy", qty, price, "NOW", motivo, meta2)

            # --- Uscita SELL se blend < -exit_thr ---
            if blended <= -self.exit_thr and base_av and base_av > 0:
                frac = np.clip((abs(blended) - self.exit_thr) /
                            max(1e-6, self.strong_thr - self.exit_thr), 0.25, 1.0)
                qty = float(base_av * float(frac))
                step = Decimal(1).scaleb(-lot_dec)
                qty = float((Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_FLOOR) * step)
                if qty > 0 and self._notional_ok(qty, float(price)) and self._throttle_ok(pair):
                    fee_pct = self._fee_pct_from_now(now, taker=True)
                    motivo = f"Blend={blended:.3f} (<= -{self.exit_thr}); SELL (fee≈{fee_pct*100:.2f}%)."
                    meta.update({"fee_pct": fee_pct})
                    return self._mk_market_action(pair, "sell", qty, price, "NOW", motivo, meta)

            # --- INGRESSO LONG ---
            if blended >= self.enter_thr:
                # 1) Policy: alpha/gamma/hold (fallback BUY)
                now_now   = (cur.get("info") or {}).get("NOW") or {}
                limits_now = cur.get("pair_limits") or {}
                pol = self.policy.predict(pair, "buy", now_now, limits_now)

                if pol.get("hold"):
                    # stop “intelligente”: poca fattibilità osservata
                    return self._action_schema(
                        "hold", pair, None, "NOW", None, None, None, None,
                        tf, None, None, None, None,
                        f"Policy HOLD: feasible_p≈{pol.get('p_ok', 0.0):.2f} sotto soglia.", meta
                    )

                # 2) gamma → ridimensiona cap
                base_cap = self._affordable_cap_eur(cur)
                cap_eur  = float(max(0.0, base_cap * float(pol.get("gamma", 1.0))))

                # Scegli prezzo LIMIT "maker" e arrotonda
                px = self._choose_limit_price(cur, "buy")
                if not px or px <= 0:
                    motivo2 = "Dati prezzo incompleti; hold."
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            tf, None, None, None, None, motivo2, meta)

                # 3) alpha → aggiusta il prezzo (più aggressivo se slippage alto)
                alpha = float(pol.get("alpha", 0.0))
                slip  = float(now_now.get("slippage_buy_pct") or 0.0)
                bump  = max(0.0001, min(0.01, slip)) * alpha     # 1–100 bps tipico
                px    = self._round_price(limits_now, float(px) * (1.0 + bump))
                if not px or px <= 0:
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            tf, None, None, None, None, "Prezzo non disponibile dopo bump; hold.", meta)

                # qty dalla notional realmente affordable + lot_decimals/ordermin
                lot_dec, ordermin = self._pair_limits_of(cur)
                qty_raw = self._calc_qty(cap_eur, px, lot_dec, ordermin)
                ok, qty_final, why = (True, qty_raw, "")
                if self.respect_pair_limits:
                    ok, qty_final, why = self._enforce_min_constraints(cur, float(px), float(qty_raw))
                if (not ok) or qty_final <= 0:
                    motivo2 = f"Hold per limiti: {why or 'qty<=0'} (px={px})"
                    return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                            tf, None, None, None, None, motivo2, meta)

                qty = float(qty_final)

                # Verifica notional minima in EUR (se definita)
                if not self._notional_ok(qty, float(px)):
                    motivo2 = "Notional troppo bassa; hold."
                    return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                            tf, None, None, None, None, motivo2, meta)

                # SL/TP dinamici
                atr = self._read_atr(cur)
                sl, tp = self._dyn_levels("long", float(px), atr)

                # Costi stimati e edge check
                fee_buy = self._fee_pct_from_now(now, taker=(conf >= self.strong_thr and self.use_market_for_strong))
                slip = float((now or {}).get("slippage_buy_pct") or 0.0)
                cost = fee_buy + slip
                edge = float(blended - self.enter_thr)
                if edge < self.edge_over_cost_mult * cost:
                    motivo2 = "Edge inferiore ai costi stimati; hold."
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            tf, None, None, None, None, motivo2, meta)

                # Commit del budget di sessione: previeni insufficient funds cross-pair
                eur_needed = float(qty * px) * (1.0 + fee_buy + self.safety_buffer_pct)
                if not self._session_commit(eur_needed):
                    motivo2 = "Budget di sessione esaurito; hold."
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            tf, None, None, None, None, motivo2, meta)

                # MARKET per segnali molto forti (come già avevi), altrimenti LIMIT maker
                if conf >= self.strong_thr and self.use_market_for_strong:
                    motivo = f"Blend={blended:.3f} forte; ingresso MARKET con cap≈{cap_eur:.2f} EUR."
                    self._remember_entry(pair, "long", qty, float(px), sl, tp)
                    return self._mk_market_action(pair, "buy", qty, float(px), "NOW", motivo, meta)
                else:
                    motivo = f"Blend={blended:.3f}; ingresso LIMIT maker con cap≈{cap_eur:.2f} EUR."
                    self._remember_entry(pair, "long", qty, float(px), sl, tp)
                    return self._mk_limit_action(pair, "buy", qty, float(px), tf, motivo, meta, stop=sl, take=tp)


            # --- INGRESSO SHORT MARGINE ---
            if self.allow_margin_short and blended <= -self.enter_thr and (not base_av or base_av <= 0):
                cap_eur = self._affordable_cap_eur(cur)
                qty_raw = cap_eur / float(price)
                ok, qty_final, why = (True, qty_raw, "")
                if self.respect_pair_limits:
                    ok, qty_final, why = self._enforce_min_constraints(cur, float(price), float(qty_raw))
                if not ok:
                    motivo = f"Hold (short) per limiti: {why} (price={price})"
                    return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                            tf, None, None, None, None, motivo, meta)
                qty = qty_final

                atr = self._read_atr(cur)
                sl, tp = self._dyn_levels("short", float(price), atr)
                meta_short = {**meta, "margin": True, "short": True, "leverage": self.leverage}
                self._remember_entry(pair, "short", qty, float(price), sl, tp)

                px = self._choose_limit_price(cur, "sell") or float(price or 0.0)
                if not px or px <= 0:
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            tf, None, None, None, None, "Dati prezzo incompleti per SHORT; hold.", meta_short)
                return self._mk_limit_action(pair, "sell", qty, float(px), tf, "Ingresso SHORT LIMIT", meta_short, stop=sl, take=tp)


            # --- fallback HOLD se conf bassa o prezzo assente ---
            if conf < 0.10 or not price:
                return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                        "24H", None, None, None, None,
                                        motivo="Confidenza bassa o dati incompleti; nessuna azione.", meta=meta)

            # --- fallback ingresso semplice ---
            lot_dec = int((limits or {}).get("lot_decimals", 6))
            ordermin = limits.get("ordermin")

            if blended > 0:
                # 1) Policy: alpha/gamma/hold anche nel fallback BUY
                now_now   = (cur.get("info") or {}).get("NOW") or {}
                limits_now = cur.get("pair_limits") or {}
                pol = self.policy.predict(pair, "buy", now_now, limits_now)

                if pol.get("hold"):
                    return self._action_schema(
                        "hold", pair, None, "NOW", None, None, None, None,
                        "24H", None, None, None, None,
                        f"Policy HOLD: feasible_p≈{pol.get('p_ok', 0.0):.2f} sotto soglia.", meta
                    )

                base_cap = self._affordable_cap_eur(cur)
                cap_eur  = float(max(0.0, base_cap * float(pol.get("gamma", 1.0))))
                qty_raw  = cap_eur / float(price)

                if self.respect_pair_limits:
                    ok, qty_final, why = self._enforce_min_constraints(cur, float(price), float(qty_raw))
                if not ok:
                    motivo = f"Hold per limiti: {why} (price={price})"
                    return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                            "24H", None, None, None, None, motivo, meta)
                qty = qty_final
                px = self._choose_limit_price(cur, "buy")
                alpha = float(pol.get("alpha", 0.0))
                slip  = float(now_now.get("slippage_buy_pct") or 0.0)
                bump  = max(0.0001, min(0.01, slip)) * alpha
                px    = self._round_price(limits_now, float(px) * (1.0 + bump))
                if not px or px <= 0:
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            "24H", None, None, None, None, "Prezzo non disponibile dopo bump; hold.", meta)
                lot_dec, ordermin = self._pair_limits_of(cur)
                qty_raw = self._calc_qty(cap_eur, float(px), lot_dec, ordermin)
                ok, qty_final, why = (True, qty_raw, "")
                if self.respect_pair_limits:
                    ok, qty_final, why = self._enforce_min_constraints(cur, float(px), float(qty_raw))
                if (not ok) or qty_final <= 0 or (not self._notional_ok(qty_final, float(px))):
                    motivo = f"Hold per limiti: {why or 'notional'}"
                    return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                            "24H", None, None, None, None, motivo, meta)
                qty = float(qty_final)
                eur_needed = float(qty * float(px)) * (1.0 + self._fee_pct_from_now(now, taker=False) + self.safety_buffer_pct)
                if not self._session_commit(eur_needed):
                    return self._action_schema("hold", pair, None, "NOW", None, None, None, None,
                                            "24H", None, None, None, None, "Budget di sessione esaurito; hold.", meta)
                motivo = f"Ingresso LIMIT semplice con cap≈{cap_eur:.2f} EUR."
                return self._mk_limit_action(pair, "buy", qty, float(px), "24H", motivo, meta)

            else:
                base_av, _ = self._read_available(cur)
                if base_av and base_av > 0:
                    qty = float(base_av)
                    px = float(price)
                    return self._mk_limit_action(pair, "sell", qty, px, "24H",
                                                f"Blend={blended:.3f} (conf={conf:.2f}); uscita prudente su disponibilità.", meta)
                return self._action_schema("hold", pair, None, "limit", None, None, None, None,
                                        "24H", None, None, None, None,
                                        motivo="Segnale sell ma nessuna disponibilità base; hold.", meta=meta)


    # -------------------- orchestration --------------------

    def suggest_all(self) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        if self.capital_manager:
            port = self.refresh_portfolio()
            scored = []; sells_pool = []
            for cur in self._currencies:
                # PRIMA:  _, blend, _ = self._signals_for_currency(cur)
                _, blend, *_ = self._signals_for_currency(cur)   # prende solo il 2° elemento
                now = cur.get("info", {}).get("NOW", {})
                price = now.get("current_price") or now.get("last") or None
                base_av, _ = self._read_available(cur)
                if base_av and base_av > 0 and price:
                    sells_pool.append((float(blend), cur, float(price)))
                scored.append((float(blend), cur))

            scored.sort(key=lambda x: x[0], reverse=True)
            for blend, cur in scored:
                now = cur.get("info", {}).get("NOW", {})
                price = now.get("current_price") or now.get("last")
                _, eur = self._read_available(cur)
                if blend >= self.enter_thr and price and (eur or 0.0) < 1e-9:
                    cap_need = self._score_to_cap(blend)
                    actions.extend(self._rebalance_if_needed(cap_need, eur or 0.0, sells_pool))

                # res = self._suggest_action(cur)
                try:
                    # sigs, blended, w_soft, w_eff, contribs = self._signals_for_currency(cur)
                    planned = self.planner.plan_for_pair(cur, port)  # <— tutto qui dentro
                    if planned:
                        actions.extend(planned)
                except Exception as e:
                    if getattr(self, "verbose", False):
                        print("[planner] error on", cur.get("pair"), e)
        else:
            print('test')


        return actions

    def learn_price_size_from_results(self,
                                    actions_ai: List[Dict[str, Any]],
                                    exec_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collega esiti runner alla policy:
        - min size -> aumenta gamma
        - insufficient funds -> riduci gamma
        - invalid permissions -> abbassa p_ok (fail)
        - successo -> alza p_ok e micro-ottimizza alpha
        """
        report = []
        for a, r in zip(actions_ai or [], exec_results or []):
            pair = a.get("pair") or f"{a.get('base')}/{a.get('quote')}"
            side = (a.get("type") or a.get("tipo") or "").lower()
            if side not in ("buy","sell"):
                continue
            err_list = r.get("error") or []
            err = err_list[0] if (isinstance(err_list, list) and err_list) else None

            # recupera il contesto per l'update (quanto volevamo comprare dopo rounding "adj"?)
            lim = (next((c.get("pair_limits") for c in (self._currencies or [])
                        if (c.get("pair") or f"{c.get('base')}/{c.get('quote')}")==pair), {}) or {})
            ordermin = float(lim.get("ordermin") or 0.0)
            desired_adj = a.get("volume")  # qty mandata nel body (già arrotondata)

            self.policy.update_from_exec(pair, side,
                                        error=err,
                                        desired_adj_qty=(float(desired_adj) if desired_adj is not None else None),
                                        ordermin=ordermin)

            report.append({
                "pair": pair, "side": side, "error": err,
                "gamma_now": self.policy._get(pair, side).gamma,
                "alpha_now": self.policy._get(pair, side).alpha,
                "succ": self.policy._get(pair, side).succ,
                "fail": self.policy._get(pair, side).fail,
            })
        return {"updated": report}

    # -------------------- export IO --------------------

    def export_input(self, payload: Any):
        if not self.input_dir: return
        safe_write_json(os.path.join(self.input_dir, f"input_{get_now_ts()}.json"), payload)

    def export_output(self, payload: Any):
        if not self.output_dir: return
        safe_write_json(os.path.join(self.output_dir, f"output_{get_now_ts()}.json"), payload)


        # --- NEW: builder HOLD (nessun ordine, solo traccia decisionale) ---
    def _mk_hold(
        self,
        cur_or_pair,
        motivo: str = "Hold",
        meta: dict | None = None,
        tf: str = "24H",
    ):
        """
        Crea una 'azione' di tipo HOLD (non genera body per Kraken).
        Accetta:
        - cur (dict) con 'pair' / 'kr_pair' nel payload, oppure
        - pair (str)
        """
        meta = dict(meta or {})
        meta.setdefault("planner", "hold")

        # accetta sia il dict 'cur' che la stringa pair
        if isinstance(cur_or_pair, dict):
            pair = cur_or_pair.get("pair") or cur_or_pair.get("kr_pair")
        else:
            pair = cur_or_pair

        # NB: usiamo _action_schema come gli altri builder, ma:
        #  - tipo: "hold"
        #  - quando: "hold" (il runner deve ignorare queste azioni)
        #  - campi prezzo/qty/stop/take/leverage sono None/0
        return self._action_schema(
            "hold",          # tipo
            pair,            # pair
            None,            # prezzo
            "hold",          # quando -> il runner deve fare 'continue' su queste
            0.0,             # quantita
            None,            # quantita_eur
            None,            # stop_loss
            None,            # take_profit
            tf,              # timeframe
            None,            # lato
            None,            # break_price
            None,            # limite
            None,            # leverage
            motivo,          # motivo
            meta,            # _meta
        )


    # -------------------- API: run --------------------

    def run(self,
            currencies: Optional[List[Dict[str, Any]]] = None,
            actions: Optional[List[Dict[str, Any]]] = None,
            *,
            replace: bool = True) -> Dict[str, Any]:
        if currencies or actions:
            self.attach_data(currencies, actions, replace=replace)

        if self.input_dir:
            self.export_input(currencies)

        # --- inizializza il budget di sessione: prendi dal primo currency che lo espone, altrimenti fallback ---
        self._session_quote_left = None
        try:
            for cur in (self._currencies or []):
                q = (((cur.get("portfolio") or {}).get("available") or {}).get("quote"))
                if q is not None:
                    self._session_quote_left = float(q)
                    break
        except Exception:
            self._session_quote_left = None

        # Se non trovato, prova con budget/per_trade settings (non rompere se mancano)
        if self._session_quote_left is None:
            if self.budget_eur is not None:
                self._session_quote_left = float(self.budget_eur)
            else:
                # ultima difesa: non bloccare la run, ma limita cap a per_trade_cap se presente
                self._session_quote_left = float(self.per_trade_cap_eur or 0.0)


        actions_out = self.suggest_all()
        out = {"scores": self.weights.scores(), "actions_ai": actions_out}
        if self.output_dir: self.export_output(actions_out)
        return out

    # -------------------- >>> CREDIT ASSIGNMENT helper --------------------
    def _aggregate_strategy_contrib(self, actions_ai: List[Dict[str, Any]]) -> np.ndarray:
        """
        Costruisce un vettore di contributi per strategia (>=0, somma=1) usando
        i segnali delle azioni:
          contrib_i = sum_over_actions( w_i * |signal_i|^1.25 )
        Se non disponibile nulla, fallback a softmax(raw).
        """
        n = len(self.strategies)
        acc = np.zeros(n, dtype=float)
        for a in actions_ai or []:
            meta = a.get("meta") or {}
            sigs = meta.get("signals") or []
            w    = meta.get("weights") or []
            if len(sigs) != n or len(w) != n:
                continue
            v = np.array([max(0.0, float(wi) * (abs(float(si)) ** 1.25))
                          for wi, si in zip(w, sigs)], dtype=float)
            acc += v
        tot = float(acc.sum())
        if tot <= 1e-12:
            return softmax(self.weights.raw, temp=self.weights.cfg.temp)
        return acc / tot

    # -------------------- FEEDBACK --------------------

    def update_weights_from_orders(self,
                                   actions_ai: List[Dict[str, Any]],
                                   exec_results: List[Dict[str, Any]],
                                   realized_pnl_eur: Optional[float]) -> Dict[str, Any]:
        """
        Aggiorna i pesi con un PnL già calcolato; il 'merito' è assegnato
        alle strategie in base ai segnali usati dalle azioni del batch.
        """
        contrib = self._aggregate_strategy_contrib(actions_ai)
        self.weights.update(contrib_w=contrib, realized_pnl=float(realized_pnl_eur or 0.0))
        return {
            "names": [s.name for s in self.strategies],
            "contrib_vector": contrib.tolist(),
            "pnl_total_eur": float(realized_pnl_eur or 0.0),
            "weights_softmax_after": softmax(self.weights.raw, temp=self.weights.cfg.temp).tolist(),
        }

    def update_weights_from_kraken(self,
                                   actions_ai: List[Dict[str, Any]],
                                   lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Calcola PnL EUR sugli ultimi N hours dai Ledgers (cash flow) e
        aggiorna i pesi usando un vettore di contributi derivato dalle azioni_ai.
        """
        k = self._get_kraken_api_from_env()
        end_ts = int(time.time()); start_ts = end_ts - int(lookback_hours * 3600)

        pnl_total = 0.0
        try:
            led = k.query_private("Ledgers", {"start": start_ts, "end": end_ts})
            ledger = (led.get("result") or {}).get("ledger") or {}
            for _, row in (ledger or {}).items():
                asset = (row.get("asset") or "").upper()
                try: amt = float(row.get("amount") or 0.0)
                except Exception: amt = 0.0
                if asset in {"ZEUR", "EUR"}:
                    pnl_total += amt
        except Exception:
            pass

        self._last_kraken_pnl_eur = float(pnl_total)

        # >>> qui la differenza: contributi NON uniformi
        contrib = self._aggregate_strategy_contrib(actions_ai)
        self.weights.update(contrib_w=contrib, realized_pnl=pnl_total)
        w_after = softmax(self.weights.raw, temp=self.weights.cfg.temp)

        return {
            "names": [s.name for s in self.strategies],
            "contrib_vector": contrib.tolist(),
            "pnl_total_eur": float(pnl_total),
            "weights_softmax_after": w_after.tolist(),
            "kraken_window": {"start": start_ts, "end": end_ts},
            "kraken_total_eur": float(pnl_total),
        }

    # -------------------- TRAINING OFFLINE (immutato salvo ranking) --------------------
    @staticmethod
    def _parse_ts_from_filename(path: str) -> Optional[int]:
        base = os.path.basename(path)
        try:
            token = base.split("_", 1)[1].split(".")[0]
            num = int(token.replace("_", ""))
            return num
        except Exception:
            return None

    def _filter_files_by_date(self, files: List[str],
                              start: Optional[str],
                              end: Optional[str]) -> List[str]:
        def to_num(s: Optional[str]) -> Optional[int]:
            if not s: return None
            try: return int(s.replace("_", ""))
            except Exception: return None
        s = to_num(start); e = to_num(end)
        items = []
        for p in files:
            t = self._parse_ts_from_filename(p)
            if t is None: continue
            if s is not None and t < s: continue
            if e is not None and t > e: continue
            items.append((t, p))
        items.sort(key=lambda x: x[0])
        return [p for _, p in items]

    def training(self,
                 input_folder: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 *,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 adjust_pct: float = 0.25,
                 apply: bool = True,
                 log_path: Optional[str] = None,
                 verbose: bool = False) -> Dict[str, Any]:

        in_dir = input_folder or self.input_dir
        input_files = self._list_json(in_dir, "input_") if in_dir else []
        input_files = self._filter_files_by_date(input_files, start, end)

        if len(input_files) < 2:
            return {"files_used": 0, "pnl_total_eur": 0.0, "note": "Servono almeno 2 input_* per fare una proiezione."}

        w_before = softmax(self.weights.raw, temp=self.weights.cfg.temp).tolist()
        strat_names = [s.name for s in self.strategies]

        total_pnl = 0.0
        pnl_by_pair: Dict[str, float] = {}
        pnl_by_strategy: Dict[str, float] = {name: 0.0 for name in strat_names}
        decisions_log: List[Dict[str, Any]] = []

        for idx in range(len(input_files) - 1):
            snap_now = safe_read_json(input_files[idx])
            snap_next = safe_read_json(input_files[idx + 1])

            self.attach_data(currencies=snap_now, replace=True)
            acts = self.suggest_all()

            # cintura di sicurezza: tieni solo dict con 'pair'
            acts = [x for x in acts if isinstance(x, dict) and (x.get("pair") or x.get("base"))]

            price_now: Dict[str, float] = {}
            price_next: Dict[str, float] = {}

            for cur in snap_now:
                pair = cur.get("pair") or f"{cur.get('base')}/{cur.get('quote')}"
                now = (cur.get("info") or {}).get("NOW") or {}
                p = now.get("current_price") or now.get("last")
                try: price_now[pair] = float(p)
                except Exception: pass
            for cur in snap_next:
                pair = cur.get("pair") or f"{cur.get('base')}/{cur.get('quote')}"
                now = (cur.get("info") or {}).get("NOW") or {}
                p = now.get("current_price") or now.get("last")
                try: price_next[pair] = float(p)
                except Exception: pass
            # dopo price_now/price_next
            lookahead = 30  # o parametro
            future_prices = [{} for _ in range(lookahead)]
            for j in range(2, min(lookahead+2, len(input_files)-idx)):
                snap_j = safe_read_json(input_files[idx + j])
                mp = {}
                for cur in snap_j:
                    pair = cur.get("pair") or f"{cur.get('base')}/{cur.get('quote')}"
                    now = (cur.get("info") or {}).get("NOW") or {}
                    p = now.get("current_price") or now.get("last")
                    try: mp[pair] = float(p)
                    except: pass
                future_prices[j-2] = mp

                # nel loop sugli 'acts'
                # p0 = price_now.get(pair); p1 = price_next.get(pair)


            step_pnl = 0.0
            step_items: List[Dict[str, Any]] = []
            for a in acts:
                pair = a.get("pair") or (f"{a.get('base')}/{a.get('quote')}" if a.get('base') and a.get('quote') else None)
                if not pair: continue
                p0 = price_now.get(pair); p1 = price_next.get(pair)
                if (p0 is None or p1 is None or p0 <= 0):
                    # prova a cercare nei prossimi file
                    for mp in future_prices:
                        p1 = mp.get(pair)
                        if p1 is not None: break
                if p0 is None or p1 is None or p0 <= 0:
                    continue  # ancora niente: salta

                meta = a.get("meta") or {}
                blend = float((meta.get("blend") or 0.0))
                signals = list(meta.get("signals") or [])
                w_soft = list(meta.get("weights") or [])

                cap = float(self._score_to_cap(blend))
                ret = (p1 / p0) - 1.0

                direction = (a.get("tipo") or "").lower()
                pnl = 0.0
                if direction == "buy": pnl = cap * ret
                elif direction == "sell": pnl = cap * (-ret)

                step_pnl += pnl
                pnl_by_pair[pair] = pnl_by_pair.get(pair, 0.0) + pnl

                if pnl != 0.0 and signals and w_soft and len(signals) == len(w_soft) == len(strat_names):
                    contrib = np.array([max(0.0, float(wi) * (abs(float(si)) ** 1.25))
                                        for wi, si in zip(w_soft, signals)], dtype=float)
                    tot = float(contrib.sum())
                    if tot > 1e-12:
                        share = (contrib / tot).tolist()
                        for name, frac in zip(strat_names, share):
                            pnl_by_strategy[name] = pnl_by_strategy.get(name, 0.0) + (pnl * float(frac))

                step_items.append({
                    "pair": pair,
                    "tipo": direction,
                    "blend": blend,
                    "cap_eur": cap,
                    "p0": p0, "p1": p1, "ret": ret,
                    "pnl_eur": pnl,
                })

            total_pnl += step_pnl
            decisions_log.append({
                "window": {
                    "from_file": os.path.basename(input_files[idx]),
                    "to_file": os.path.basename(input_files[idx+1]),
                },
                "pnl_step_eur": step_pnl,
                "items": step_items,
            })
            if verbose:
                print(f"[training] {os.path.basename(input_files[idx])} -> {os.path.basename(input_files[idx+1])}  step_pnl={step_pnl:.2f} EUR")

        w_after = w_before
        if apply:
            w_soft_now = softmax(self.weights.raw, temp=self.weights.cfg.temp)
            self.weights.update(contrib_w=w_soft_now, realized_pnl=float(total_pnl * float(adjust_pct)))
            w_after = softmax(self.weights.raw, temp=self.weights.cfg.temp).tolist()

        report = {
            "files_used": len(input_files),
            "pnl_total_eur": float(total_pnl),
            "pnl_by_pair": {k: float(v) for k, v in sorted(pnl_by_pair.items(), key=lambda kv: -abs(kv[1]))},
            "pnl_by_strategy": {k: float(v) for k, v in sorted(pnl_by_strategy.items(), key=lambda kv: -kv[1])},
            "weights_before": w_before,
            "weights_after": w_after,
            "adjust_pct": float(adjust_pct),
            "applied": bool(apply),
        }
        if log_path:
            safe_write_json(log_path, {
                "report": report,
                "decisions": decisions_log,
            })
        return report


    def _classify_sell_intent(self, cur, blended, price):
        """
        Ritorna una delle stringhe: "SPOT_TP", "SPOT_REDUCE", "SHORT_ENTRY", oppure None.
        Logica:
        - Se HO posizione (pos_qty>0): prima provo TP, poi reduce/stop.
        - Se NON ho posizione: posso valutare SHORT ENTRY solo se abilitato.
        """
        pos_qty = self._read_position_qty(cur)
        near_daily  = bool(cur.get("near_daily_target"))
        near_weekly = bool(cur.get("near_weekly_target"))

        if pos_qty > 0:
            # --- SPOT: TAKE PROFIT se profitto o target vicini, o segnale molto negativo
            _, avg_price = self._read_position_cost(cur)
            pnl_ok = False
            if price and avg_price:
                pnl = (price - avg_price) / avg_price
                pnl_ok = pnl >= float(self.take_profit_pct)

            if pnl_ok or near_daily or near_weekly or (blended <= -self.exit_thr):
                return "SPOT_TP"

            # altrimenti possibile riduzione leggera se blended un po' negativo
            if blended <= -self.exit_thr/2:
                return "SPOT_REDUCE"

            return None

        # --- NIENTE POSIZIONE: valuta SHORT ENTRY
        if not self.enable_margin_short:
            return None

        if blended <= -self.exit_thr:
            # (opzionale: puoi mettere qui ulteriori gate di regime)
            return "SHORT_ENTRY"

        return None



# --- PATCH: ActionPlanner (decisioni multi-azione, sostituisce _suggest_action) ---
class ActionPlanner:
    def __init__(self, ai: "AIEnsembleTrader"):
        self.ai = ai
        self.actions_ai = []
        self.market_enter = 0.7
        self.margin_sell_enter = -0.6
        self.lev_average = 0.2
        self.market_enter_lev = self.market_enter + self.lev_average
        self.margin_sell_enter_lev = self.margin_sell_enter - self.lev_average

        self.limit_atr_mult = 0.25   # porzione di ATR usata per la distanza base
        self.limit_bps_base = 8.0    # bps extra per tenere un cuscinetto (8 bps = 0.08%)
        self.limit_aggr_max = 0.65   # quanto accorciare la distanza quando il segnale è forte (0..1)
        self.budget = ai.budget_eur

    def _compute_signals(self, cur: dict):
        """
        Punto unico per calcolare segnali/blend.
        Se hai già _signals_for_currency(cur) usala qui.
        """
        return self.ai._signals_for_currency(cur)  # sigs, blended, w_soft, w_eff, contribs




    # def _req_min_qty(self, cur, price: float) -> float:
    #     limits = (cur.get("pair_limits") or {}) or {}
    #     lot_dec  = int(limits.get("lot_decimals") or 6)
    #     step     = 10.0 ** (-lot_dec)
    #     ordermin = float(limits.get("ordermin") or 0.0)          # MIN in BASE
    #     # se usi un minimo nozionale (es. self.ai.min_notional_eur), inseriscilo qui:
    #     costmin  = float(getattr(self.ai, "min_notional_eur", 0.0) or 0.0)
    #     req = max(ordermin, step)
    #     if price > 0 and costmin > 0:
    #         req = max(req, costmin / price)                      # MIN EUR → BASE
    #     # allinea allo step (floor up “di sicurezza”)
    #     k = max(1.0, math.ceil(req / step))
    #     return k * step

    # def _qty_from_blend(self, req_min: float, blend: float, lot_dec: int) -> float:
    #     # piccola spinta sopra l’ordermin proporzionale al |blend|
    #     # es: +0.0..+10% dell’ordermin
    #     bump_pct = min(0.10, max(0.0, abs(float(blend))))  # clamp 0..10%
    #     step = 10.0 ** (-lot_dec)
    #     raw = req_min * (1.0 + bump_pct)
    #     # floor allo step
    #     n = max(1.0, math.floor(raw / step))
    #     return n * step


    # def plan_for_pair(self, cur: dict) -> list[dict]:
    #     import math

    #     # ---------- helpers locali (no dipendenze esterne) ----------
    #     def _req_min_qty(cur, price: float) -> tuple[float, int, float]:
    #         """
    #         Quantità minima eseguibile in BASE, rispettando:
    #         - pair_limits.ordermin (BASE)
    #         - lot_decimals (step BASE)
    #         - (opzionale) min notional in EUR se self.ai.min_notional_eur è impostato
    #         Ritorna: (req_min_base, lot_dec, step)
    #         """
    #         limits   = (cur.get("pair_limits") or {}) or {}
    #         lot_dec  = int(limits.get("lot_decimals") or 6)
    #         step     = 10.0 ** (-lot_dec)
    #         ordermin = float(limits.get("ordermin") or 0.0)  # BASE
    #         req = max(ordermin, step)

    #         costmin = float(getattr(self.ai, "min_notional_eur", 0.0) or 0.0)
    #         if price > 0 and costmin > 0:
    #             req = max(req, costmin / price)              # converto il min EUR in BASE

    #         # allineo verso l'alto allo step (ceil → non rischio di stare sotto)
    #         k = max(1, math.ceil(req / step))
    #         req_min = k * step
    #         # normalizzo precisione
    #         req_min = float(round(req_min, lot_dec))
    #         return req_min, lot_dec, step

    #     def _qty_from_blend(req_min: float, blend: float, lot_dec: int, step: float) -> float:
    #         """
    #         Aggiunge una piccola spinta sopra il minimo proporzionale al |blend|.
    #         Es.: bump 0..10% dell'ordermin.
    #         """
    #         bump_pct = min(0.10, max(0.0, abs(float(blend))))  # 0..10%
    #         raw = req_min * (1.0 + bump_pct)
    #         n = max(1, math.floor(raw / step))
    #         qty = n * step
    #         return float(round(qty, lot_dec))

    #     # ---------- 1) segnali/blend ----------
    #     sigs, blended, w_soft, w_eff, contribs = self._compute_signals(cur)

    #     actions = []
    #     now = ((cur.get("info") or {}).get("NOW") or {})
    #     price = float(now.get("current_price") or 0.0)
    #     if price <= 0:
    #         return actions

    #     base_av, eur_av = self.ai._read_available(cur)
    #     pos_qty = self.ai._read_position_qty(cur)  # size reale posizione aperta (BASE)
    #     intent = self.ai._classify_sell_intent(cur, blended, float(price or 0.0))
    #     pair = cur.get("pair") or cur.get("kr_pair")
    #     tf   = "24H"

    #     # min tecnico per questa pair
    #     req_min, lot_dec, step = _req_min_qty(cur, price)
    #     # ---- metriche utili (target / risk) ----
    #     day_pnl  = self.ai._pnl_realized_since(cur, self.ai._day_start_ts())
    #     wstart   = self.ai._day_start_ts() - 6*86400
    #     week_pnl = self.ai._pnl_realized_since(cur, wstart)
    #     near_daily_target  = (day_pnl  >= 0.75 * self.ai.planner_cfg.daily_pnl_target_eur)
    #     near_weekly_target = (week_pnl >= 0.75 * self.ai.planner_cfg.weekly_pnl_target_eur)
    #     limits   = (cur.get("pair_limits") or {}) or {}
    #     atr  = float(now.get("atr") or 0.0)
    #     atr  = atr if atr > 0 else price * 0.01


    #     # --- tracking EMA del blend + gestione posizione aperta ---
    #     btrk = self.ai._update_blend_track(pair, float(blended))  # salva ema/last del blend per il pair
    #     mgmt = self.ai._manage_open_position(cur, float(price or 0.0), float(pos_qty or 0.0))
    #     if self.ai.debug_signals:
    #         print(f"[signals] {pair} -> blend={round(blended,3)}")
    #     if mgmt:
    #         # se ha triggerato stop/trail/take, rientra subito (1 azione sola)
    #         return [mgmt]

    #     # --- weak partial exit: se forza < enter_thr e EMA-blend segnala debolezza ---
    #     st_pos = self.ai.pos_state.get(pair)
    #     if st_pos and price:
    #         lot_dec2 = int((limits or {}).get("lot_decimals", 6))
    #         step2 = Decimal(1).scaleb(-lot_dec2)
    #         if st_pos["side"] == "long":
    #             strength = max(0.0, float(blended))
    #             ema_strength = max(0.0, float(btrk["ema"]))
    #             delta = ema_strength - strength
    #             if (0.0 < strength < self.ai.enter_thr) and (delta >= self.ai.weak_drop):
    #                 if st_pos and float(st_pos.get("qty") or 0.0) > 0:
    #                     qty = float(st_pos["qty"]) * float(self.ai.weak_reduce)
    #                     # floor allo step già presente nelle righe successive
    #                     qty = float((Decimal(str(qty)) / step2).to_integral_value(rounding=ROUND_FLOOR) * step2)
    #                     if qty > 0:
    #                         motivo = f"Uscita parziale LONG: debolezza blend vs EMA (Δ={delta:.2f})."
    #                         meta2 = {"weak_exit": True, "blend_ema": float(btrk["ema"]), "delta": float(delta)}
    #                         return [self.ai._mk_market_action(pair, "sell", qty, float(price), "NOW", motivo, meta2)]
    #         else:  # short aperta
    #             strength = max(0.0, -float(blended))
    #             ema_strength = max(0.0, -float(btrk["ema"]))
    #             delta = ema_strength - strength
    #             if (0.0 < strength < self.ai.enter_thr) and (delta >= self.ai.weak_drop):
    #                 qty = float(st_pos.get("qty") or 0.0) * float(self.ai.weak_reduce)
    #                 qty = float((Decimal(str(qty)) / step2).to_integral_value(rounding=ROUND_FLOOR) * step2)
    #                 if qty > 0:
    #                     motivo = f"Cover parziale SHORT: debolezza blend vs EMA (Δ={delta:.2f})."
    #                     meta2 = {"weak_exit": True, "blend_ema": float(btrk["ema"]), "delta": float(delta)}
    #                     return [self.ai._mk_market_action(pair, "buy", qty, float(price), "NOW", motivo, meta2)]


    #     # ---------------- (A) ENTRY LONG + BRACKET ----------------
    #     if blended >= self.ai.enter_thr and eur_av > 0:
    #         # qty di base: ordemin + piccolo delta dal blend
    #         qty_des = _qty_from_blend(req_min, blended, lot_dec, step)

    #         # fondi massimi traducibili in BASE
    #         max_buyable = eur_av / price
    #         if max_buyable + 1e-18 < req_min:
    #             # non raggiunge il minimo → HOLD esplicito se disponibile
    #             if hasattr(self.ai, "_mk_hold"):
    #                 actions.append(self.ai._mk_hold(cur, motivo="Fondi EUR insufficienti a superare MIN"))
    #             return actions

    #         qty = min(qty_des, max_buyable)
    #         # floor allo step (verso il basso) e check di sicurezza
    #         qty = float(round(math.floor(qty / step) * step, lot_dec))
    #         if qty + 1e-18 < req_min:
    #             if hasattr(self.ai, "_mk_hold"):
    #                 actions.append(self.ai._mk_hold(cur, motivo="Qty BUY < MIN dopo floor"))
    #             return actions

    #         # prezzo limit “maker-friendly”
    #         px = self.ai._choose_limit_price(cur, side="buy") or price

    #         # 1) ENTRY
    #         actions.append(self.ai._mk_limit_action(
    #             pair, "buy", qty, float(px), tf,
    #             motivo=f"Entry blend={blended:.3f}",
    #             meta={"planner": "entry", "sigs": list(map(float, sigs)), "contribs": list(map(float, contribs))}
    #         ))

    #         # 2) STOP (reduce_only=True)
    #         sl = max(0.0, px - self.ai.planner_cfg.sl_atr_mult * atr)
    #         actions.append(self.ai._mk_stop_action(
    #             pair, "sell", qty, float(sl), tf,
    #             motivo="Protective SL",
    #             meta={"planner": "protect"},
    #             limit_px=None,
    #             reduce_only=True,
    #             leverage=None
    #         ))

    #         # 3) TAKE (reduce_only=True)
    #         tp = px + self.ai.planner_cfg.tp_pct * atr
    #         actions.append(self.ai._mk_take_action(
    #             pair, "sell", qty, float(tp), tf,
    #             motivo="Take profit",
    #             meta={"planner": "take"},
    #             reduce_only=True,
    #             leverage=None
    #         ))

    #     # ---------------- (B) REDUCE-ONLY (target / housekeeping) ----------------
    #     # non emettere SELL se la posizione non raggiunge il MIN
    #     if pos_qty > 0 and (near_daily_target or near_weekly_target or blended <= -self.ai.exit_thr/2):
    #         if pos_qty + 1e-18 < req_min:
    #             if hasattr(self.ai, "_mk_hold"):
    #                 actions.append(self.ai._mk_hold(cur, motivo="Posizione < MIN: skip reduce_only"))
    #             # niente ordini
    #         else:
    #             frac = float(self.ai.planner_cfg.reduce_pct_default or 0.33)
    #             qty_des = max(req_min, pos_qty * frac)
    #             qty = min(pos_qty, qty_des)
    #             qty = float(round(math.floor(qty / step) * step, lot_dec))
    #             if qty + 1e-18 >= req_min and self.ai._notional_ok(qty, price):
    #                 px_s = self.ai._choose_limit_price(cur, side="sell") or price
    #                 actions.append(self.ai._mk_limit_action(
    #                     pair, "sell", qty, float(px_s), tf,
    #                     motivo="Reduce-only (target/housekeeping)",
    #                     meta={"planner": "reduce", "reduce_only": True}
    #                 ))
    #             else:
    #                 if hasattr(self.ai, "_mk_hold"):
    #                     actions.append(self.ai._mk_hold(cur, motivo="Qty SELL < MIN dopo floor/notional"))

    #     # ---------------- (C) FUNDING reduce_only (libera capitale) ----------------
    #     # libera capitale solo se ho almeno il MIN di posizione
    #     # (minimo richiesto in EUR per una prossima entry)
    #     need_eur_min = (req_min * price)
    #     if pos_qty > 0 and eur_av < need_eur_min and blended < 0.0:
    #         if base_av + 1e-18 >= req_min:
    #             # frazione proporzionale alla carenza, ma non oltre il 50%
    #             gap = (need_eur_min - eur_av)
    #             frac = min(0.5, max(0.25, gap / max(need_eur_min, 1e-9)))
    #             qty_des = max(req_min, pos_qty * frac)
    #             qty = min(pos_qty, qty_des)
    #             qty = float(round(math.floor(qty / step) * step, lot_dec))
    #             if qty + 1e-18 >= req_min and self.ai._notional_ok(qty, price):
    #                 px_f = self.ai._choose_limit_price(cur, side="sell") or price
    #                 actions.append(self.ai._mk_limit_action(
    #                     pair, "sell", qty, float(px_f), tf,
    #                     motivo="Funding reduce_only",
    #                     meta={"planner":"funding", "reduce_only": True}
    #                 ))
    #             else:
    #                 if hasattr(self.ai, "_mk_hold"):
    #                     actions.append(self.ai._mk_hold(cur, motivo="Qty funding < MIN dopo floor/notional"))

    #     # ---------------- (D) STOP refresh se rischio alto ----------------
    #     # crea stop ONLY se ho almeno il MIN in posizione
    #     if pos_qty > 0 and blended < -self.ai.exit_thr:
    #         if pos_qty + 1e-18 >= req_min:
    #             sl2 = price - self.ai.planner_cfg.sl_atr_mult * atr
    #             qty = float(round(math.floor(pos_qty / step) * step, lot_dec))
    #             # clamp a base_av ma mai sotto req_min
    #             if qty + 1e-18 >= req_min and self.ai._notional_ok(qty, price):
    #                 actions.append(self.ai._mk_stop_action(
    #                     pair, "sell", qty, float(sl2), tf,
    #                     motivo="Stop refresh", meta={"planner":"stop_refresh"},
    #                     limit_px=None, reduce_only=True, leverage=None
    #                 ))
    #         else:
    #             if hasattr(self.ai, "_mk_hold"):
    #                 actions.append(self.ai._mk_hold(cur, motivo="Posizione < MIN: skip stop refresh"))


    #     if intent == "SHORT_ENTRY":
    #         lev_max, eur_collat = self.ai._read_short_capacity(cur)
    #         if price and price > 0 and eur_collat > 0:
    #             # leva effettiva conservativa: usa init MR se lo usi, o limita con lev_max
    #             lev_eff = min(float(lev_max), 1.0 / max(getattr(self.ai, "margin_init_mr", 0.2), 1e-9))
    #             lev_eff *= float(getattr(self.ai, "margin_safety", 0.9))

    #             cap_qty = (eur_collat * lev_eff) / price
    #             qty_des = max(req_min, cap_qty * float(getattr(self.ai, "size_mult", 1.0)))
    #             qty = min(cap_qty, qty_des)
    #             qty = float(round(math.floor(qty / step) * step, lot_dec))

    #             if qty >= req_min and self.ai._notional_ok(qty, price):
    #                 actions.append(self.ai._mk_market_action(
    #                     pair=pair,
    #                     side="sell",
    #                     qty=qty,
    #                     price_now=price,   # o now price
    #                     tf="24H",
    #                     motivo="short_entry",
    #                     meta={"reason": "short_entry"}  # opzionale; qui puoi mettere flag informativi
    #                 ))


    #     return actions

    #     # ---------- helpers locali (no dipendenze esterne) ----------
    def _req_min_qty(self, cur, price: float) -> tuple[float, int, float]:
        """
        Quantità minima eseguibile in BASE, rispettando:
        - pair_limits.ordermin (BASE)
        - lot_decimals (step BASE)
        - (opzionale) min notional in EUR se self.ai.min_notional_eur è impostato
        Ritorna: (req_min_base, lot_dec, step)
        """
        limits   = (cur.get("pair_limits") or {}) or {}
        lot_dec  = int(limits.get("lot_decimals") or 6)
        step     = 10.0 ** (-lot_dec)
        ordermin = float(limits.get("ordermin") or 0.0)  # BASE
        req = max(ordermin, step)

        costmin = float(getattr(self.ai, "min_notional_eur", 0.0) or 0.0)
        if price > 0 and costmin > 0:
            req = max(req, costmin / price)              # converto il min EUR in BASE

        # allineo verso l'alto allo step (ceil → non rischio di stare sotto)
        k = max(1, math.ceil(req / step))
        req_min = k * step
        # normalizzo precisione
        req_min = float(round(req_min, lot_dec))
        return req_min, lot_dec, step


    def _levels_from_blend(self, entry_side: str, ref_price: float, atr_value: float, blend_value: float) -> tuple[float | None, float | None]:
        """
        Calcola (stop_loss, take_profit) a partire da:
        - prezzo corrente (ref_price)
        - ATR della pair (atr_value)
        - forza del segnale (blend_value ∈ [-1, 1])
        - parametri già presenti: min_stop_pct, min_take_pct, atr_mult_sl, atr_mult_tp, enter_thr, strong_thr

        Logica:
        1) Costruisco distanze base SL/TP da ATR con guardrail percentuali minime.
        2) Traduco la 'forza' del blend in [0..1] mappando enter_thr → 0 e strong_thr → 1 (clamp).
        3) Modulo le distanze:
            - segnale più forte ⇒ SL più stretto (riduce %), TP più ampio (aumenta %).
        4) Ritorno livelli prezzo per long o short.
        """
        # Validazioni essenziali
        if not ref_price or ref_price <= 0:
            return (None, None)

        # 1) Distanze base (percentuali) da ATR con guardrail minimi
        #    NB: restano coerenti con la tua _dyn_levels
        sl_pct_base = max(self.ai.min_stop_pct, (self.ai.atr_mult_sl * float(atr_value or 0.0)) / max(ref_price, 1e-12))
        tp_pct_base = max(self.ai.min_take_pct, (self.ai.atr_mult_tp * float(atr_value or 0.0)) / max(ref_price, 1e-12))

        # 2) Forza del segnale in [0..1]
        #    - sotto enter_thr ⇒ 0
        #    - sopra strong_thr ⇒ 1
        b_abs = abs(float(blend_value))
        lo_thr = max(1e-6, float(self.ai.enter_thr))        # evita divisioni per zero
        hi_thr = max(lo_thr + 1e-6, float(self.ai.strong_thr))
        blend_strength = (b_abs - lo_thr) / (hi_thr - lo_thr)
        if blend_strength < 0.0: blend_strength = 0.0
        if blend_strength > 1.0: blend_strength = 1.0

        # 3) Modulazione aggressività (parametri nuovi, con default “ragionevoli”)
        #    - più grande sl_tighten_max ⇒ SL più vicino quando il segnale è forte
        #    - più grande tp_widen_max   ⇒ TP più lontano quando il segnale è forte
        sl_tighten_max = getattr(self, "sl_tighten_max", 0.35)  # fino a -35% distanza SL
        tp_widen_max   = getattr(self, "tp_widen_max",  0.50)  # fino a +50% distanza TP

        sl_pct = sl_pct_base * (1.0 - sl_tighten_max * blend_strength)
        tp_pct = tp_pct_base * (1.0 + tp_widen_max   * blend_strength)

        # 4) Converti percentuali in livelli prezzo
        if entry_side == "buy":
            stop_loss  = ref_price * (1.0 - sl_pct)
            take_profit= ref_price * (1.0 + tp_pct)
        else:  # "short"
            stop_loss  = ref_price * (1.0 + sl_pct)
            take_profit= ref_price * (1.0 - tp_pct)

        return float(stop_loss), float(take_profit)


    def _compute_limit_price_from_signals(
        self,
        cur: dict,
        side: str,                  # "buy" | "sell"
        last_price: float,
        sigs: list[float],
        blended: float,
        w_eff: "np.ndarray | list[float]" = None,
        contribs: "np.ndarray | list[float]" = None,
        step: float = 0.0           # tick/step della price; se 0 prova a stimarlo
    ) -> float:
        """
        Calcola un prezzo LIMIT “maker-friendly” usando blend/ATR e segnali.
        Idea: quando il segnale è più forte ⇒ prezzo più vicino al mercato (più aggressivo);
            quando è debole ⇒ più paziente (più distante).

        Fonti e fallback:
        - ATR dalla pair (fallback 1% del prezzo).
        - bid/ask da info.NOW se presenti, altrimenti usa il last_price.
        - step di prezzo per agganciarsi al tick corretto (floor/ceil a seconda del lato).

        Parametri (config consigliata, puoi metterli nel costruttore):
        - self.limit_atr_mult: quanta ATR usare come distanza base (default 0.25)
        - self.limit_bps_base: base in bps (1 bps = 0.01%) da sommare (default 8)
        - self.limit_aggr_max: quanto “accorciare” la distanza quando il segnale è forte (default 0.65)
        - self.enter_thr / self.strong_thr: soglie già presenti per normalizzare la forza
        """
        now = ((cur.get("info") or {}).get("NOW") or {})
        bid = float(now.get("best_bid") or now.get("bid") or 0.0) or last_price
        ask = float(now.get("best_ask") or now.get("ask") or 0.0) or last_price

        # 1) Lettura ATR (fallback 1% del prezzo se non disponibile)
        atr_value = self.ai._read_atr(cur) or (last_price * 0.01)

        # 2) Forza normalizzata del blend in [0..1] (enter_thr → 0, strong_thr → 1)
        enter_thr  = float(self.ai.enter_thr)
        strong_thr = float(self.ai.strong_thr)
        lo = max(1e-6, enter_thr)
        hi = max(lo + 1e-6, strong_thr)
        strength = (abs(float(blended)) - lo) / (hi - lo)
        if strength < 0.0: strength = 0.0
        if strength > 1.0: strength = 1.0

        # 3) Distanza “grezza” dalla migliore controparte:
        #    somma di componente ATR e una base in bps del prezzo
        limit_atr_mult = getattr(self, "limit_atr_mult", 0.25)       # 25% di ATR
        limit_bps_base = getattr(self, "limit_bps_base", 8.0)        # 8 bps = 0.08%
        base_dist = (limit_atr_mult * atr_value) + (last_price * (limit_bps_base / 10000.0))

        # 4) Aggressività: più forte il segnale ⇒ avvicino il prezzo al mercato
        #    Esempio: con strength=1 e limit_aggr_max=0.65 riduco la distanza del 65%
        limit_aggr_max = getattr(self, "limit_aggr_max", 0.65)
        dist = base_dist * (1.0 - limit_aggr_max * strength)

        # 5) Micro-aggiustamenti dai contributi (opzionale, piccolo tocco)
        #    Se i contributi sono molto concordi col lato, leggermente più aggressivo
        try:
            if contribs is not None and w_eff is not None:
                # concordanza pesata: somma(|w_eff[i]*contribs[i]|) già direzionale dal segno del blend
                conc = float(np.sum(np.abs(np.array(w_eff, dtype=float) * np.array(contribs, dtype=float))))
                conc = float(np.clip(conc, 0.0, 1.0))
                dist *= (1.0 - 0.10 * conc)   # fino a -10% di distanza se c'è forte accordo
        except Exception:
            pass

        # 6) Scegli base bid/ask e applica la distanza nel verso giusto
        if side == "buy":
            raw_px = max(0.0, bid - dist)
        else:  # "sell"
            raw_px = ask + dist

        # 7) Snap al tick/step corretto (floor per buy, ceil per sell)
        #    Usa lo "step" passato (da _req_min_qty) oppure prova a dedurlo dal pair_limits
        if not step or step <= 0:
            try:
                # se hai già un helper, meglio: step = self._price_step_for(cur)  # es.
                step = 0.0
            except Exception:
                step = 0.0

        if step and step > 0:
            # round “maker-friendly”
            n = math.floor(raw_px / step) if side == "buy" else math.ceil(raw_px / step)
            limit_px = float(n * step)
        else:
            # fallback: nessun tick noto → usa raw
            limit_px = float(raw_px)

        # 8) Guardrail per non finire dall’altra parte dello spread
        #    (cioè evitare di diventare “market” per errore)
        if side == "buy":
            limit_px = min(limit_px, bid)        # non oltre il bid
            limit_px = max(0.0, limit_px)
        else:
            limit_px = max(limit_px, ask)        # non sotto l’ask

        return float(limit_px)


    def plan_for_pair(self, cur: dict, port) -> list[dict]:
        import math
        self.port = port
        actions = []
        now = ((cur.get("info") or {}).get("NOW") or {})
        price = float(now.get("current_price") or 0.0)
        pair = cur.get("pair") or cur.get("kr_pair")
        openOrders = cur.get("open_orders") or []
        tf   = "24H"
        type = 'hold'   # hold/buy/sell
        quando="limit"  # limit/market/
        reason= "looking"
        motivo= "Meglio Attendere Momenti Migliori"
        qty = 0
        limite = None
        wallet = cur.get("portfolio")
        decimal = int(cur.get("pair_limits")['lot_decimals']) if cur.get("pair_limits") else 4
        pair_decimals = int(cur.get("pair_limits")['pair_decimals']) if cur.get("pair_limits") else 4
        leverage_sell = cur.get("pair_limits")['levarage_sell'] if cur.get("pair_limits") else False
        leverage_buy = cur.get("pair_limits")['levarage_buy'] if cur.get("pair_limits") else False
        fees = (cur.get("pair_limits")['fees_maker'] if cur.get("pair_limits") else [{'0': 0, '1': 0.2},{'0': 10000, '1': 0.3}])
        reduce_only = False
        avalaible_in_portfolio = None
        if wallet:
            avalaible_in_portfolio = wallet['available']['base']

        if pair == "YFI/EUR":
            TESTO = 2
        try:
            sigs, blended, w_soft, w_eff, contribs = self.ai._signals_for_currency(cur)
            req_min, lot_dec, step = self._req_min_qty(cur, price)
            btrk = self.ai._update_blend_track(pair, float(blended))  # salva ema/last del blend per il pair
            leverage = 1
        except Exception as e:
            print(f"ERRORE {e!r} -> SKIP")

        qty = req_min
        limitBool = False
        if self.ai.debug_signals:
            print(f"[signals] {pair} -> blend={round(blended,3)}")

        # BUY
        if blended >= self.ai.enter_thr:
            type = "buy"
            totVov = 0
            for o in openOrders:
                ord_type = (o.get("ordertype") or "").lower()
                typeTrans = (o.get("type") or "").lower()
                vol = float(o.get("vol_rem") or 0)
                if typeTrans == 'sell' and ord_type == 'position':
                    totVov += vol
            if blended >= self.market_enter:
                quando = "market"
                motivo="Compro subito che mo scoppia!!"
                reason = "buy market"
                if totVov > 0:
                    qty = totVov
                    reduce_only = True
                if blended >= self.market_enter_lev:
                    leverage = self.ai.leverage if leverage_buy == True else 1
                    motivo="Compro subito che mo scoppia!! e glie do forte!!"
                    reason = "buy market lev"
            else:
                limitBool = True
                motivo="Entrata con prudenza"
                reason = "buy limite"
        # SELL
        if abs(blended) > abs(self.ai.exit_thr) and blended < 0:
            type = "sell"
            if blended <= self.margin_sell_enter:
                quando = "market"
                motivo= "shorto brutto per blend alto"
                reason = "short market"
                if blended <= self.margin_sell_enter_lev:
                    leverage = self.ai.leverage if leverage_sell == True else 1
                    motivo="blend alto venduto tutto con leverage"
                    reason = "short lev market"
                else:
                    if avalaible_in_portfolio and avalaible_in_portfolio > req_min:
                        reason = "Take Profit"
                        if avalaible_in_portfolio > (req_min * 2):
                            qty = avalaible_in_portfolio * 0.33
                        else:
                            qty = avalaible_in_portfolio
                    else:
                        if avalaible_in_portfolio:
                            type="hold"
                            motivo="Nessuna operazione possibile"
                            reason = "dust"
                        else:
                            type = 'sell' if leverage_sell == True else 'hold'
                            leverage = self.ai.leverage if leverage_sell == True else 1
                            motivo= "shorto blando"
                            reason = "short market"
            else:
                limitBool = True
                if avalaible_in_portfolio and avalaible_in_portfolio > req_min:
                    if avalaible_in_portfolio > (req_min * 2):
                        qty = avalaible_in_portfolio * 0.33
                        motivo="Prendo profitto parziale e libero quantità"
                        reason = "Take Profit"
                    else:
                        qty = avalaible_in_portfolio
                        motivo="Prendo profitto"
                        reason = "Take All Profit"
                else:
                    if avalaible_in_portfolio:
                        type="hold"
                        motivo="Nessuna operazione possibile"
                        reason = "dust"
                    else:
                        motivo= "shorto blando"
                        reason = "short market"
                        type = 'sell' if leverage_sell == True else 'hold'
                        leverage = self.ai.leverage if leverage_sell == True else 1


        atr_value = price * 0.01  # fallback 1% se ATR mancante
        if pair == "WIN/EUR":
            testo = 2
        # apply fee
        if qty < fees[1][0]:
            fee = fee_in_crypto(qty,fees[0][1])
        else:
            fee = fee_in_crypto(qty,fees[1][1])
        qty += fee
        #

        qty_eur = float(price * qty)
        checkBudget = self.budget - qty_eur
        if type == 'buy':
            if checkBudget < 0 and reduce_only == False:
                type = 'hold'
            else:
                self.budget -= qty_eur
        if type == 'sell':
            if checkBudget < 0 and reason != "Take Profit" and  reason != "Take All Profit":
                type = 'hold'

            else:
                if reason != "Take Profit" and  reason != "Take All Profit":
                    self.budget -= qty_eur

        if limitBool and type != "hold":
            limite = self._compute_limit_price_from_signals(
                cur=cur,
                side=type,                # "buy" o "sell"
                last_price=price,
                sigs=sigs,
                blended=blended,
                w_eff=w_eff,
                contribs=contribs,
                step=step                 # dallo stesso _req_min_qty
            )
            stop_px, take_px = self._levels_from_blend(type, limite, atr_value, blended)
        else:
            stop_px, take_px = self._levels_from_blend(type, price, atr_value, blended)

        if type != "hold":
            self.ai._remember_entry(pair, ("long" if type == 'buy' else 'short'), qty, float((limite if limite else price)), stop_px, take_px)

        qty = floor_dec(qty, (decimal)) if decimal > 0 else int(qty)
        stop_px = floor_dec(stop_px, (pair_decimals)) if pair_decimals > 0 else int(stop_px)
        take_px = floor_dec(take_px, (pair_decimals)) if pair_decimals > 0 else int(take_px)
        price = floor_dec(price, (pair_decimals)) if pair_decimals > 0 else int(price)
        if limite:
            limite = floor_dec(limite, (pair_decimals))
        try:
            action = {
                "pair": pair,
                "tipo": type,
                "prezzo": price,
                "quando": quando,
                "quantita": qty,
                "quantita_eur": qty_eur,
                "stop_loss": stop_px,
                "take_profit": take_px,
                "timeframe": "24H",
                "lato": type,
                "break_price": None,
                "limite": limite,
                "leverage": leverage,
                "motivo": reason,
                "meta": {
                    "reason": motivo
                },
                "reduce_only" : reduce_only,
                "blend": blended,
                "cancel_order_id": ""
            }
            actions.append(action)
            self.actions_ai.append(action)
            return actions
        except Exception as e:
            print(e)
            return []
# --- END PATCH ---
