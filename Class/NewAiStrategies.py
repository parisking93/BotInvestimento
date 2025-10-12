# -*- coding: utf-8 -*-
"""
Crea un dataset CSV per LGBM partendo dai file JSON in `storico_input/`.

Modalità supportate:
1) **Singolo CSV** (default): unisce tutti i JSON e salva in `--output`.
2) **CSV per ogni JSON** (`--per-file`): per ogni `input_*.json` genera un CSV separato
   dentro `--output-dir` (usa il nome del JSON per comporre il CSV). Utile se i file
   diventano troppo grandi.

Ogni riga = (pair, timestamp, prezzo, feature MTF) e include la colonna `ret_h`
(ritorno futuro) calcolata cercando il primo punto futuro per lo stesso pair
oltre `horizon_hours` **entro lo stesso perimetro di dati** (globale o singolo file).

Uso da terminale (singolo CSV):
    python make_dataset_lgbm_from_storico_input.py \
        --input-dir ./storico_input \
        --output ./aiConfig/dataset_lgbm.csv \
        --horizon-hours 12

Uso per **CSV per file**:
    python make_dataset_lgbm_from_storico_input.py \
        --input-dir ./storico_input \
        --per-file \
        --output-dir ./aiConfig/datasets_per_file \
        --horizon-hours 12

Dipendenze: pandas, numpy (solo lato script).
Le colonne generate sono compatibili con `train_lgbm_offline`:
    pair,ts,price,ch24,ch48,dev_1h,dev_4h,vol_dev,atr1h,b1,b4,spread_ratio,slip_avg,or_ok,or_pos,or_w,ret_h

Suggerimenti:
- In modalità `--per-file`, il `ret_h` usa solo dati **di quel JSON**: le ultime righe
  potrebbero non avere futuro sufficiente (verranno con `ret_h` NaN e poi droppate).
- Per un dataset unico ma non enorme, puoi creare CSV per file e poi concatenarli.
"""

from __future__ import annotations
import os, json, math, argparse, glob
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _bias_to_num(bias: Any) -> float:
    if isinstance(bias, str):
        b = bias.upper()
        if b == "UP":
            return 1.0
        if b == "DOWN":
            return -1.0
    if isinstance(bias, (int, float)):
        return float(bias)
    return 0.0


def extract_rows_from_file(path: str) -> List[Dict[str, Any]]:
    """Estrae righe (dict) dall'array JSON `input_*.json`.
    Ritorna lista di record con feature + price + ts + pair.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for row in arr:
        try:
            info = (row.get("info") or {})
            now = (info.get("NOW") or {})
            pair = str(now.get("pair") or row.get("pair") or "?")
            ts = _safe_float(now.get("since"))  # epoch seconds
            px = _safe_float(now.get("current_price") or now.get("last") or now.get("close") or now.get("open"))

            # qualità mercato
            spread = _safe_float(now.get("spread"))
            spread_ratio = spread / (abs(px) + 1e-12) if px else 0.0
            slip_b = _safe_float(now.get("slippage_buy_pct"))
            slip_s = _safe_float(now.get("slippage_sell_pct"))
            slip_avg = 0.5 * (slip_b + slip_s)

            # opening range
            or_ok = 1.0 if bool(now.get("or_ok")) else 0.0
            or_high = now.get("or_high")
            or_low = now.get("or_low")
            if or_high is not None and or_low is not None and (float(or_high) - float(or_low)) != 0.0:
                or_mid = 0.5 * (float(or_high) + float(or_low))
                or_rng = float(or_high) - float(or_low)
                or_pos = math.tanh((px - or_mid) / (abs(or_rng) + 1e-12))
                or_w = or_rng / (abs(px) + 1e-12) if px else 0.0
            else:
                or_pos, or_w = 0.0, 0.0

            # bias
            b1 = _bias_to_num(now.get("bias_1h"))
            b4 = _bias_to_num(now.get("bias_4h"))

            # trend locale via EMA
            ema50_1h = _safe_float(now.get("ema50_1h"))
            ema200_1h = _safe_float(now.get("ema200_1h"))
            ema50_4h = _safe_float(now.get("ema50_4h"))
            ema200_4h = _safe_float(now.get("ema200_4h"))
            dev_1h = math.tanh((ema50_1h - ema200_1h) / (abs(ema200_1h) + 1e-9)) if ema200_1h else 0.0
            dev_4h = math.tanh((ema50_4h - ema200_4h) / (abs(ema200_4h) + 1e-9)) if ema200_4h else 0.0

            # returns recenti
            b24 = (info.get("24H") or {})
            b48 = (info.get("48H") or {})
            ch24 = _safe_float(b24.get("change_pct"))
            ch48 = _safe_float(b48.get("change_pct"))

            # volatilità
            b1h = (info.get("1H") or info.get("60M") or {})
            atr1h = _safe_float(b1h.get("atr"))

            # deviazione da ema lenta
            ema_slow = _safe_float(now.get("ema_slow"))
            vol_dev = math.tanh((px - ema_slow) / (abs(ema_slow) + 1e-9)) if ema_slow else 0.0

            out.append({
                "pair": pair,
                "ts": ts,
                "price": px,
                "ch24": ch24,
                "ch48": ch48,
                "dev_1h": dev_1h,
                "dev_4h": dev_4h,
                "vol_dev": vol_dev,
                "atr1h": atr1h,
                "b1": b1,
                "b4": b4,
                "spread_ratio": spread_ratio,
                "slip_avg": slip_avg,
                "or_ok": or_ok,
                "or_pos": or_pos,
                "or_w": or_w,
            })
        except Exception:
            continue
    return out


def compute_future_returns(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Per ogni pair calcola ret_h cercando il primo punto con ts >= ts0 + H.
    Usa merge_asof per efficienza.
    """
    H = float(horizon_hours) * 3600.0
    out_list = []
    for pair, g in df.groupby("pair", sort=False):
        g2 = g.sort_values("ts").reset_index(drop=True)
        fut = g2[["ts", "price"]].copy()
        fut["ts"] = fut["ts"] - H  # shift indietro per asof-merge su stesso timestamp
        merged = pd.merge_asof(g2.sort_values("ts"),
                               fut.sort_values("ts").rename(columns={"price": "price_fut"}),
                               on="ts", direction="forward")
        merged["ret_h"] = (merged["price_fut"] - merged["price"]) / (merged["price"].abs() + 1e-12)
        out_list.append(merged)
    out = pd.concat(out_list, ignore_index=True)
    return out


def build_single_csv(files: List[str], out_path: str, horizon_hours: int) -> None:
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows.extend(extract_rows_from_file(fp))
    if not rows:
        raise SystemExit("Nessun record estratto dai JSON")
    df = pd.DataFrame(rows).dropna(subset=["pair","ts","price"]).copy()
    df2 = compute_future_returns(df, horizon_hours)
    df2 = df2.dropna(subset=["ret_h"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df2.to_csv(out_path, index=False)
    print(f"Salvato: {out_path}  |  righe: {len(df2)}  | pair: {df2['pair'].nunique()}")


def build_many_csv(files: List[str], out_dir: str, horizon_hours: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]  # es. input_20251002_210643
        out_path = os.path.join(out_dir, f"dataset_{base}_H{horizon_hours}.csv")
        rows = extract_rows_from_file(fp)
        if not rows:
            print(f"[skip] {fp}: nessuna riga")
            continue
        df = pd.DataFrame(rows).dropna(subset=["pair","ts","price"]).copy()
        df2 = compute_future_returns(df, horizon_hours)
        df2 = df2.dropna(subset=["ret_h"]).reset_index(drop=True)
        df2.to_csv(out_path, index=False)
        print(f"Salvato: {out_path}  |  righe: {len(df2)}  | pair: {df2['pair'].nunique()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Cartella con i file input_*.json")
    ap.add_argument("--output", help="CSV unico di output (modalità default)")
    ap.add_argument("--per-file", action="store_true", help="Crea un CSV per ogni JSON")
    ap.add_argument("--output-dir", help="Cartella per i CSV per-file (richiesta con --per-file)")
    ap.add_argument("--horizon-hours", type=int, default=12, help="Orizzonte futuro per il target (ore)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "input_*.json")))
    if not files:
        raise SystemExit("Nessun file input_*.json trovato nella cartella indicata")

    if args.per_file:
        if not args.output_dir:
            raise SystemExit("Con --per-file devi specificare --output-dir")
        build_many_csv(files, args.output_dir, args.horizon_hours)
    else:
        if not args.output:
            raise SystemExit("Specificare --output per la modalità singolo CSV")
        build_single_csv(files, args.output, args.horizon_hours)


if __name__ == "__main__":
    main()




# BEST OPTION
# python .\NewAiStrategies.py --input-dir ..\storico_input --output ..\aiConfig\dataset_lgbm.csv --horizon-hours 12
# python .\NewAiStrategies.py --input-dir ..\storico_input --per-file --output-dir ..\aiConfig\datasets_per_file --horizon-hours 12


# Due modalità disponibili

# Singolo CSV (default) – unisce tutto:

# python make_dataset_lgbm_from_storico_input.py \
#   --input-dir ./storico_input \
#   --output ./aiConfig/dataset_lgbm.csv \
#   --horizon-hours 12


# CSV per ogni file – crea un CSV per ciascun input_*.json:

# python make_dataset_lgbm_from_storico_input.py \
#   --input-dir ./storico_input \
#   --per-file \
#   --output-dir ./aiConfig/datasets_per_file \
#   --horizon-hours 12


# Output tipo: ./aiConfig/datasets_per_file/dataset_input_20251002_210643_H12.csv
