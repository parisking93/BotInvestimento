# -*- coding: utf-8 -*-
"""
Incolla questo file **alla fine** del tuo `Aiensemble.py` (oppure importalo come modulo
separato) e aggiungi le due strategie all'elenco in `AIEnsembleTrader.__init__`.

Dipendenze opzionali:
  - lightgbm, scikit-learn, joblib (solo per LGBMStrategy)
  - darts[u] (TFTStrategy), torch

Nota importante:
  - Le strategie **non allenano** online: caricano modelli salvati.
  - Per TFT, viene mantenuta una piccola storia locale per pair su `./aiConfig/tft_hist/`.
  - Sotto trovi due funzioni "trainer" di esempio da eseguire offline per creare i modelli.
"""

from __future__ import annotations
import os, json, math, time
from typing import Any, Dict, Optional, List
import numpy as np
# Prova import opzionali (non obbligatori per l'avvio del bot)
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
        self._lazy_load()
        if self.model is None:
            return 0.0
        x = features_mtf_from_row(row)
        xz = _standardize(x, self.scaler)
        try:
            if hasattr(self.model, "predict_proba"):
                p = float(self.model.predict_proba(xz)[0, 1])
            else:
                # se regressione → mappa a [0,1]
                raw = float(self.model.predict(xz)[0])
                p = 1.0/(1.0 + math.exp(-raw))
            if self.cal is not None and hasattr(self.cal, "predict_proba"):
                p = float(self.cal.predict_proba([[p]])[0, 1])
        except Exception:
            return 0.0
        # clip e conversione a score simmetrico
        eps = float(self.params.get("proba_clip", 0.02))
        p = min(max(p, eps), 1.0 - eps)
        score = (p - 0.5) * 2.0  # [-1,1]
        return float(np.tanh(score * float(self.params.get("score_gain", 1.0))))


# ============================
#   TFTStrategy (forecast quantile → edge)
# ============================
class TFTStrategy(Strategy):
    name = "TFT"
    def __init__(self,
                 model_dir: str = os.path.join(os.getcwd(), "aiConfig", "tft_models"),
                 hist_dir: str = os.path.join(os.getcwd(), "aiConfig", "tft_hist"),
                 horizon: int = 12,
                 min_points: int = 128,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 edge_gain: float = 8.0):
        super().__init__(model_dir=model_dir, hist_dir=hist_dir, horizon=horizon,
                         min_points=min_points, quantiles=quantiles, edge_gain=edge_gain)
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in (self.params["model_dir"], self.params["hist_dir"]):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

    def _hist_path(self, pair: str) -> str:
        fn = pair.replace("/", "_") + ".csv"
        return os.path.join(self.params["hist_dir"], fn)

    def _model_path(self, pair: str) -> str:
        fn = pair.replace("/", "_") + ".pt"
        return os.path.join(self.params["model_dir"], fn)

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
            df.loc[len(df)] = {"ts": float(ts), "price": float(price)}
            # roll a 5000 punti massimo
            if len(df) > 5000:
                df = df.iloc[-5000:]
            df.to_csv(p, index=False)
            return len(df)
        except Exception:
            return 0

    def _load_series(self, pair: str) -> Optional[Any]:
        if not _has_darts:
            return None
        try:
            import pandas as pd
            p = self._hist_path(pair)
            if not os.path.exists(p):
                return None
            df = pd.read_csv(p)
            if len(df) < int(self.params["min_points"]):
                return None
            return TimeSeries.from_dataframe(df, time_col="ts", value_cols="price", fill_missing_dates=False)
        except Exception:
            return None

    def _predict_edge(self, pair: str, series: Any) -> Optional[float]:
        if not _has_darts:
            return None
        mpath = self._model_path(pair)
        if not os.path.exists(mpath):
            return None
        try:
            # Proviamo TFT; in fallback N-HiTS
            model = None
            try:
                model = TFTModel.load(mpath)
            except Exception:
                try:
                    model = NHiTSModel.load(mpath)
                except Exception:
                    model = None
            if model is None:
                return None
            # Se il modello è probabilistico → prendi p50
            pred = model.predict(int(self.params["horizon"]))
            last = float(series.univariate_values()[-1])
            try:
                p50 = float(pred.quantile_timeseries(0.5).univariate_values()[-1])
            except Exception:
                p50 = float(pred.univariate_values()[-1])
            exp_ret = (p50 - last) / (abs(last) + 1e-12)
            # edge→score compresso
            return float(np.tanh(exp_ret * float(self.params.get("edge_gain", 8.0))))
        except Exception:
            return None

    def signal(self, row: Dict[str, Any]) -> float:
        # 1) aggiorna storia locale
        info = (row.get("info") or {})
        now = (info.get("NOW") or {})
        pair = str(now.get("pair") or row.get("pair") or "?")
        px = float(now.get("current_price") or now.get("last") or 0.0)
        ts = float(now.get("since") or time.time())
        self._update_local_history(pair, ts, px)
        # 2) prova a predire se abbiamo serie e modello
        series = self._load_series(pair)
        if series is None:
            return 0.0
        edge = self._predict_edge(pair, series)
        return float(edge) if edge is not None else 0.0


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
    from sklearn.calibration import CalibratedClassifierCV

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
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X_te, y_te)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(cal, model_out.replace(".pkl", "_cal.pkl"))


def train_tft_offline(hist_dir: str, model_dir: str,
                      horizon: int = 12,
                      n_epochs: int = 50,
                      use_nhits: bool = False) -> None:
    """Allena un modello per **ogni pair** con storia locale in `hist_dir`.
    Salva i pesi in `model_dir` come `<PAIR>.pt`.
    """
    if not _has_darts:
        raise RuntimeError("Darts non disponibile. `pip install darts[u]`")
    import pandas as pd
    os.makedirs(model_dir, exist_ok=True)
    for fn in os.listdir(hist_dir):
        if not fn.endswith(".csv"): continue
        pair = fn[:-4].replace("_", "/")
        df = pd.read_csv(os.path.join(hist_dir, fn))
        if len(df) < 256:  # minimo
            continue
        series = TimeSeries.from_dataframe(df, time_col="ts", value_cols="price", fill_missing_dates=False)
        # Modello probabilistico (quantili)
        if use_nhits:
            model = NHiTSModel(input_chunk_length=128, output_chunk_length=horizon,
                               n_epochs=n_epochs, random_state=42)
        else:
            model = TFTModel(input_chunk_length=128, output_chunk_length=horizon,
                             hidden_size=64, lstm_layers=1, num_attention_heads=4,
                             dropout=0.1, batch_size=64, n_epochs=n_epochs, random_state=42,
                             likelihood=QuantileRegression(quantiles=[0.1,0.5,0.9]))
        model.fit(series)
        outp = os.path.join(model_dir, fn.replace(".csv", ".pt"))
        model.save(outp)
        print(f"Salvato {outp}")


# ============================
#  Come agganciarle all'Ensemble
# ============================
"""
Nel tuo `AIEnsembleTrader.__init__`, aggiungi **dopo** la definizione delle altre strategie:

    self.strategies: List[Strategy] = [
        # ... le tue strategie esistenti ...
        SqueezeBreakoutStrategy(),
        # ↓↓↓ NUOVE ↓↓↓
        LGBMStrategy(model_path=os.path.join(os.getcwd(), "aiConfig", "lgbm_model.pkl"),
                     calibrator_path=os.path.join(os.getcwd(), "aiConfig", "lgbm_model_cal.pkl")),
        TFTStrategy(model_dir=os.path.join(os.getcwd(), "aiConfig", "tft_models"),
                    hist_dir=os.path.join(os.getcwd(), "aiConfig", "tft_hist"),
                    horizon=12),
    ]

Suggerimenti:
- All'inizio le due strategie restituiranno 0.0 finché non esistono i modelli.
- Fai girare il bot qualche ora per riempire `aiConfig/tft_hist/` (si popola automaticamente).
- Poi esegui offline:
    train_tft_offline("aiConfig/tft_hist", "aiConfig/tft_models", horizon=12, n_epochs=50)
- Per LGBM, genera un CSV supervisionato dal tuo storico (o da `export_input`) e lancia:
    train_lgbm_offline("dataset_lgbm.csv", "aiConfig/lgbm_model.pkl", target_h=12)
"""
