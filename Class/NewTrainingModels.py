# -*- coding: utf-8 -*-
"""
Genera un dataset "policy_outcome" unendo **storico_output** (azioni del bot)
con **storico_input** (snapshots di mercato) per valutare l'esito a H ore.

Per ogni azione salva: pair, ts_action, side, price_action, sl, tp, conf, blend,
feasibility, signals/weights (flatten), e calcola:
  - price_future (≈ primo punto ≥ ts_action + H)
  - ret_h (ritorno segnato per la direzione dell'azione)
  - outcome_win (ret_h > fee_totale)
  - tp_hit_possible / sl_hit_possible (stima grossolana vs tp/sl)

USO (CSV unico):
    python make_policy_outcome_from_logs.py \
        --input-dir ./storico_input \
        --output-dir ./storico_output \
        --out ./aiConfig/policy_outcome_H12.csv \
        --horizon-hours 12 \
        --fee-bps 15 --slip-bps 5

USO (per-file: un CSV per ogni output_*.json):
    python make_policy_outcome_from_logs.py \
        --input-dir ./storico_input \
        --output-dir ./storico_output \
        --per-file \
        --out-dir ./aiConfig/policy_outcome_files \
        --horizon-hours 12

Dipendenze: pandas, numpy
Note:
- I JSON possono avere strutture leggermente diverse; lo script è robusto
  alle chiavi mancanti (riempie 0/NaN).
- La stima tp/sl è **approssimata** (non backtesta il path intermedio).
"""

from __future__ import annotations
import os, json, glob, math, argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# -------------------- utils --------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _read_json(path: str) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _bias_to_num(bias: Any) -> float:
    if isinstance(bias, str):
        b = bias.upper()
        if b == 'UP':
            return 1.0
        if b == 'DOWN':
            return -1.0
    if isinstance(bias, (int, float)):
        return float(bias)
    return 0.0

# ---------------- load INPUT timeline (price per pair) ----------------

def extract_input_rows(path: str) -> List[Dict[str, Any]]:
    arr = _read_json(path)
    if not isinstance(arr, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in arr:
        info = (row.get('info') or {})
        now = (info.get('NOW') or {})
        pair = str(now.get('pair') or row.get('pair') or '?')
        ts = _safe_float(now.get('since'))
        px = _safe_float(now.get('current_price') or now.get('last') or now.get('close') or now.get('open'))
        if not pair or ts == 0 or px == 0:
            continue
        out.append({'pair': pair, 'ts': ts, 'price': px})
    return out


def build_price_timeline(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, 'input_*.json')))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows.extend(extract_input_rows(fp))
    if not rows:
        raise SystemExit('Nessun dato in input per costruire la timeline prezzi')
    df = pd.DataFrame(rows).dropna(subset=['pair', 'ts', 'price']).copy()
    # Assicura tipi
    df['ts'] = df['ts'].astype(float)
    df['price'] = df['price'].astype(float)
    return df

# ---------------- load OUTPUT actions ----------------

def _flatten_signals(sig: Any, prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(sig, dict):
        for k, v in sig.items():
            key = f'{prefix}_{k}'.replace(' ', '_')
            if isinstance(v, (int, float)):
                out[key] = float(v)
            else:
                try:
                    out[key] = float(v)
                except Exception:
                    pass
    return out


def extract_actions_from_output(path: str) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if data is None:
        return []
    # Alcuni output possono essere dict con chiavi tipo 'actions' o direttamente una lista
    if isinstance(data, dict) and 'actions' in data:
        items = data['actions']
    elif isinstance(data, list):
        items = data
    else:
        # prova altre chiavi comuni
        items = data.get('results') if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    for a in items:
        try:
            meta = (a.get('meta') or {})
            ts = _safe_float(a.get('ts') or meta.get('ts') or meta.get('timestamp'))
            pair = str(a.get('pair') or meta.get('pair') or (a.get('now') or {}).get('pair') or '?')
            side = str(a.get('side') or a.get('action') or meta.get('action') or '').upper()
            # normalizza lato: BUY/SELL/HOLD
            if side not in ('BUY', 'SELL', 'HOLD'):
                # se contiene sell/buy in testo
                if 'SELL' in side:
                    side = 'SELL'
                elif 'BUY' in side:
                    side = 'BUY'
                elif 'HOLD' in side or 'WAIT' in side:
                    side = 'HOLD'
                else:
                    side = 'HOLD'
            price = _safe_float(a.get('price') or a.get('planned_price') or (a.get('order') or {}).get('price'))
            sl = _safe_float(a.get('sl') or (a.get('order') or {}).get('sl'))
            tp = _safe_float(a.get('tp') or (a.get('order') or {}).get('tp'))
            conf = _safe_float(a.get('conf') or meta.get('conf'))
            blend = _safe_float(a.get('blend') or meta.get('blend'))
            feas = a.get('feasibility') or meta.get('feasibility') or {}
            feas_ok = bool(feas.get('ok')) if isinstance(feas, dict) else False
            feas_why = str(feas.get('why') or '') if isinstance(feas, dict) else ''
            # signals/weights
            sigs = a.get('signals') or meta.get('signals') or {}
            wts = a.get('weights') or a.get('weights_eff') or meta.get('weights_eff') or {}
            flat = {
                'pair': pair, 'ts_action': ts, 'side': side,
                'price_action': price, 'sl': sl, 'tp': tp,
                'conf': conf, 'blend': blend,
                'feas_ok': feas_ok, 'feas_why': feas_why,
            }
            flat.update(_flatten_signals(sigs, 'sig'))
            flat.update(_flatten_signals(wts, 'w'))
            out.append(flat)
        except Exception:
            continue
    return out


# --------------- join actions with future price -----------------

def join_with_future(df_prices: pd.DataFrame, actions: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    H = float(horizon_hours) * 3600.0
    res = []
    for pair, gA in actions.groupby('pair', sort=False):
        gP = df_prices[df_prices['pair'] == pair].sort_values('ts')
        if gP.empty:
            continue
        # future timeline: shift indietro per usare asof verso forward
        fut = gP[['ts', 'price']].copy()
        fut['ts'] = fut['ts'] - H
        m = pd.merge_asof(gA.sort_values('ts_action'),
                          fut.sort_values('ts').rename(columns={'price': 'price_future'}),
                          left_on='ts_action', right_on='ts', direction='forward')
        # aggiungi prezzo "al momento dell'azione" più vicino (<= ts_action)
        m2 = pd.merge_asof(m.sort_values('ts_action'),
                           gP[['ts', 'price']].rename(columns={'price': 'price_at_ts'}).sort_values('ts'),
                           left_on='ts_action', right_on='ts', direction='backward')
        res.append(m2)
    if not res:
        return pd.DataFrame()
    out = pd.concat(res, ignore_index=True)
    # calcolo ritorno segnato per side
    pa = out['price_action'].fillna(out['price_at_ts'])
    pf = out['price_future']
    ret = (pf - pa) / (pa.abs() + 1e-12)
    side = out['side'].fillna('HOLD').str.upper()
    sign = np.where(side == 'SELL', -1.0, 1.0)
    out['ret_h'] = ret * sign
    return out


def mark_outcomes(df: pd.DataFrame, fee_bps: float, slip_bps: float) -> pd.DataFrame:
    fee = (float(fee_bps) + float(slip_bps)) / 10000.0
    df['outcome_win'] = (df['ret_h'] > fee).astype(int)
    # tp/sl possibili (stima statica)
    # BUY vince su tp se price_future >= tp; SELL se price_future <= tp
    buy = df['side'].str.upper() == 'BUY'
    sell = df['side'].str.upper() == 'SELL'
    df['tp_hit_possible'] = np.where(buy & df['tp'].notna(), (df['price_future'] >= df['tp']).astype(int),
                                     np.where(sell & df['tp'].notna(), (df['price_future'] <= df['tp']).astype(int), 0))
    df['sl_hit_possible'] = np.where(buy & df['sl'].notna(), (df['price_future'] <= df['sl']).astype(int),
                                     np.where(sell & df['sl'].notna(), (df['price_future'] >= df['sl']).astype(int), 0))
    return df

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True, help='Cartella con input_*.json (timeline prezzi)')
    ap.add_argument('--output-dir', required=True, help='Cartella con output_*.json (azioni)')
    ap.add_argument('--horizon-hours', type=int, default=12)
    ap.add_argument('--fee-bps', type=float, default=15.0)
    ap.add_argument('--slip-bps', type=float, default=5.0)
    ap.add_argument('--out', help='Percorso CSV unico di output')
    ap.add_argument('--per-file', action='store_true', help='Emetti un CSV per ogni file di output')
    ap.add_argument('--out-dir', help='Cartella destinazione per i CSV per-file')
    args = ap.parse_args()

    # Carica timeline prezzi
    df_prices = build_price_timeline(args.input_dir)

    out_files = sorted(glob.glob(os.path.join(args.output_dir, 'output_*.json')))
    if not out_files:
        raise SystemExit('Nessun file output_*.json trovato')

    if args.per_file:
        if not args.out_dir:
            raise SystemExit('Con --per-file specifica --out-dir')
        os.makedirs(args.out_dir, exist_ok=True)
        for fp in out_files:
            acts = extract_actions_from_output(fp)
            if not acts:
                print(f'[skip] {os.path.basename(fp)}: nessuna azione')
                continue
            dfA = pd.DataFrame(acts)
            joined = join_with_future(df_prices, dfA, args.horizon_hours)
            if joined.empty:
                print(f'[skip] {os.path.basename(fp)}: nessuna merge')
                continue
            joined = mark_outcomes(joined, args.fee_bps, args.slip_bps)
            base = os.path.splitext(os.path.basename(fp))[0]
            out_csv = os.path.join(args.out_dir, f'policy_{base}_H{args.horizon_hours}.csv')
            joined.to_csv(out_csv, index=False)
            print(f'Salvato {out_csv}  | righe: {len(joined)}  | pair: {joined["pair"].nunique()}')
    else:
        if not args.out:
            raise SystemExit('Specificare --out per CSV unico')
        all_acts: List[Dict[str, Any]] = []
        for fp in out_files:
            all_acts.extend(extract_actions_from_output(fp))
        if not all_acts:
            raise SystemExit('Nessuna azione trovata negli output')
        dfA = pd.DataFrame(all_acts)
        joined = join_with_future(df_prices, dfA, args.horizon_hours)
        if joined.empty:
            raise SystemExit('Merge vuoto: controlla pair/naming/timestamps')
        joined = mark_outcomes(joined, args.fee_bps, args.slip_bps)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joined.to_csv(args.out, index=False)
        print(f'Salvato {args.out}  | righe: {len(joined)}  | pair: {joined["pair"].nunique()}')


if __name__ == '__main__':
    main()





# TO RUN
# python .\NewTrainingModels.py --input-dir ..\storico_input --output-dir ..\storico_output --out ..\aiConfig\policy_outcome_H12.csv --horizon-hours 12 --fee-bps 15 --slip-bps 5





# Yes! Ho aggiunto in canvas lo script make_policy_outcome_from_logs.py che unisce i tuoi storico_output (azioni) con i storico_input (prezzi) e ti produce un dataset per:

# calibrare la conf → p(win),

# addestrare una policy (enter/hold/size),

# stimare slippage/feasibility.

# Come usarlo
# 1) CSV unico (tutte le azioni)
# python .\NewTrainingModels.py `
#   --input-dir ..\storico_input `
#   --output-dir ..\storico_output `
#   --out ..\aiConfig\policy_outcome_H12.csv `
#   --horizon-hours 12 `
#   --fee-bps 15 `
#   --slip-bps 5



# --horizon-hours: orizzonte per misurare l’esito (es. 12h).

# --fee-bps e --slip-bps: costi totali in basis points (0.01%). Esempio 15 bps = 0.15%.

# Output colonne principali:
# pair, ts_action, side, price_action, sl, tp, conf, blend, fea_ok, fea_why, sig_*, w_*, price_at_ts, price_future, ret_h, outcome_win, tp_hit_possible, sl_hit_possible

# 2) Un CSV per ogni file di output
# python make_policy_outcome_from_logs.py \
#   --input-dir ./storico_input \
#   --output-dir ./storico_output \
#   --per-file \
#   --out-dir ./aiConfig/policy_outcome_files \
#   --horizon-hours 12


# Output tipo: ./aiConfig/policy_outcome_files/policy_output_20251002_210643_H12.csv.

# Cosa fa sotto il cofano

# Costruisce una timeline prezzi per pair dai tuoi input_*.json.

# Legge gli output_*.json e per ogni azione estrae: pair/ts/side/price/sl/tp/conf/blend, flags feasibility, e appiattisce signals/weights.

# Con un merge_asof, prende il primo prezzo ≥ ts_action + H per calcolare price_future.

# Calcola ret_h (con segno coerente con BUY/SELL) e outcome_win = ret_h > (fee+slip).

# Aggiunge stime tp_hit_possible e sl_hit_possible (approssimative, senza path intra-H).

# Subito dopo: allenamenti utili

# Calibratore conf→p(win):

# import pandas as pd
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.linear_model import LogisticRegression
# df = pd.read_csv("aiConfig/policy_outcome_H12.csv")
# X = df[["conf","blend"]].fillna(0)
# y = df["outcome_win"].astype(int)
# XY = X.copy()
# XY["outcome_win"] = y
# XY = df[["conf","blend","outcome_win"]].copy()

# model = LogisticRegression(max_iter=1000).fit(X, y)
# p_win = model.predict_proba(X_new)[:,1]





# train_conf_calibrator.py
# import os, glob, joblib
# import pandas as pd
# from sklearn.linear_model import LogisticRegression

# CSV = os.path.join("aiConfig", "policy_outcome_H12.csv")
# MODEL_OUT = os.path.join("aiConfig", "conf_calibrator.pkl")

# df = pd.read_csv(CSV)

# # features minime (stesse che avevamo nel blocco di esempio)
# X = df[["conf","blend"]].fillna(0)
# y = df["outcome_win"].astype(int)

# # allena
# model = LogisticRegression(max_iter=1000)
# model.fit(X, y)

# os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
# joblib.dump(model, MODEL_OUT)
# print(f"Salvato modello: {MODEL_OUT} | n={len(df)}")




# Slippage/feasibility model: usa feas_ok/feas_why come label, spread/slip dagli input come feature (puoi unirli dal CSV LGBM).

# Se vuoi, ti preparo anche uno script “concat” che unisce tutti i CSV per-file in uno solo con rimozione duplicati per (pair, ts_action).
