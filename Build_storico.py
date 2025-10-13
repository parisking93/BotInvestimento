import os, re, json, glob
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

TS_RE = re.compile(r"input_(\d{8})_(\d{6})\.json$", re.IGNORECASE)

# etichette richieste
RANGES = ["6H", "24H", "48H", "7D", "1M", "6M", "1Y"]
# finestre in ore (approx per mesi/anno)
RANGE_HOURS = {
    "6H": 6,
    "24H": 24,
    "48H": 48,
    "7D": 24 * 7,
    "1M": 24 * 30,
    "6M": 24 * 30 * 6,
    "1Y": 24 * 365,
}

TF_ALL = ["NOW", "5M", "15M", "30M", "1H", "6H", "24H", "48H", "7D", "30D", "1Y"]


# ---------------- utils ----------------
def _basename_no_ext(path: str) -> str:
    b = os.path.basename(path)
    return os.path.splitext(b)[0]


def _parse_dt_from_name(path: str) -> Optional[datetime]:
    m = TS_RE.search(os.path.basename(path))
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    except Exception:
        return None


def _split_pair(pair: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not pair:
        return None, None
    if "/" in pair:
        b, q = pair.split("/", 1)
        return (b or None), (q or None)
    return pair, None


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------- core ----------------
class ResistenzeBuilder:
    """
    Stato salvato (pubblico):
    {
      "file_Letti": [...],
      "per_currency": {
        "BTC/EUR": [
          {"ts":"24H","base":"BTC","quote":"EUR","max_resistance":..., "min_support":...},
          {"ts":"48H", ...},
          {"ts":"7D",  ...},
          {"ts":"1M",  ...},
          {"ts":"6M",  ...},
          {"ts":"1Y",  ...}
        ],
        ...
      }
    }

    Stato interno (privato, per ricalcoli incrementali):
    "_obs": {
      "BTC/EUR": [ {"t": 1734028800, "hi": 0.00329, "lo": 0.00327}, ... ],
      ...
    }
    """

    def __init__(self, state_filename: str = "ai_features_state.json"):
        self.state_filename = state_filename
        self.state: Dict[str, Any] = {"file_Letti": [], "per_currency": {}, "_obs": {}}

    # ---------- API ----------
    def run(self, input_dir: str) -> Dict[str, Any]:
        self._load_state()

        # trova file nuovi (per basename senza estensione)
        already = set(self.state.get("file_Letti", []))
        paths = sorted(
            p for p in glob.glob(os.path.join(input_dir, "input_*.json"))
            if TS_RE.search(os.path.basename(p))
        )
        new_files = [p for p in paths if _basename_no_ext(p) not in already]
        if not new_files:
            # nessun file nuovo → ricalcola output (per sicurezza) dall'osservato esistente
            self._rebuild_output_from_obs()
            self._save_state()
            return self._public_state()

        # elabora in ordine cronologico
        for path in new_files:
            dt = _parse_dt_from_name(path)
            if not dt:
                continue
            ts_epoch = int(dt.timestamp())

            try:
                payload = _load_json(path)
            except Exception:
                continue
            if not isinstance(payload, list):
                continue

            # per ogni record del file, ricava hi/lo aggregando TUTTI i TF disponibili (NOW, 5M, ..., 1Y)
            for rec in payload:
                pair = rec.get("pair")
                base = rec.get("base")
                quote = rec.get("quote") or 'EUR'
                if not base or not quote:
                    b2, q2 = _split_pair(pair)
                    base = base or b2
                    quote = quote or q2

                # normalizza pair/base/quote
                if base and quote:
                    pair = f"{base}/{quote}"
                elif base and not quote:
                    pair = f"{base}/None"
                else:
                    continue  # record malformato

                info = rec.get("info") or {}
                # raccogli high/low da tutti i timeframe del record (ignorando None)
                highs, lows = [], []
                for tf in TF_ALL:
                    tfdata = info.get(tf)
                    if not isinstance(tfdata, dict):
                        continue
                    hi = tfdata.get("or_high")
                    lo = tfdata.get("or_low")
                    if hi is None: hi = tfdata.get("high")
                    if lo is None: lo = tfdata.get("low")
                    if isinstance(hi, (int, float)): highs.append(hi)
                    if isinstance(lo, (int, float)): lows.append(lo)

                if not highs and not lows:
                    # tutto null in questo record: ignora
                    continue

                hi_val = max(highs) if highs else None
                lo_val = min(lows) if lows else None

                # append osservazione
                obs = self.state.setdefault("_obs", {})
                obs.setdefault(pair, []).append({"t": ts_epoch, "hi": hi_val, "lo": lo_val})

            # segna file letto
            self.state["file_Letti"].append(_basename_no_ext(path))

        # ricostruisci output pubblico dai dati osservati
        self._rebuild_output_from_obs()
        self._save_state()
        return self._public_state()

    # ---------- internals ----------
    def _load_state(self):
        path = os.path.join(os.getcwd(), self.state_filename)
        if os.path.exists(path):
            try:
                data = _load_json(path)
                if isinstance(data, dict):
                    self.state["file_Letti"] = list(data.get("file_Letti", []))
                    self.state["per_currency"] = dict(data.get("per_currency", {}))
                    # area privata osservazioni (se manca, la creo vuota)
                    self.state["_obs"] = dict(data.get("_obs", {}))
            except Exception:
                self.state = {"file_Letti": [], "per_currency": {}, "_obs": {}}

    def _save_state(self):
        path = os.path.join(os.getcwd(), self.state_filename)
        _save_json(path, self.state)

    def _public_state(self) -> Dict[str, Any]:
        # restituisci solo le chiavi pubbliche
        return {k: self.state[k] for k in ("file_Letti", "per_currency")}

    def _rebuild_output_from_obs(self):
        """
        Costruisce per_currency aggregando sugli intervalli richiesti
        rispetto all'ULTIMO timestamp osservato per ciascuna currency.
        """
        obs = self.state.get("_obs", {})
        out: Dict[str, List[Dict[str, Any]]] = {}

        for pair, series in obs.items():
            if not isinstance(series, list) or not series:
                continue
            # ultimo timestamp visto per questa currency
            last_t = max((s.get("t") or 0) for s in series)
            if not last_t:
                continue

            # base/quote dal pair
            base, quote = _split_pair(pair)
            # prepara array con una entry per ogni range richiesto
            arr: List[Dict[str, Any]] = []

            for label in RANGES:
                hours = RANGE_HOURS[label]
                t_min = last_t - int(timedelta(hours=hours).total_seconds())

                # filtra le osservazioni nel range
                window = [s for s in series if isinstance(s, dict) and (s.get("t") or 0) >= t_min]
                if not window:
                    item = {"ts": label, "pair": pair, "base": base, "quote": quote,
                            "max_resistance": None, "min_support": None}
                    arr.append(item)
                    continue

                # prendi max(hi) e min(lo), ignorando None
                hi_vals = [s.get("hi") for s in window if isinstance(s.get("hi"), (int, float))]
                lo_vals = [s.get("lo") for s in window if isinstance(s.get("lo"), (int, float))]

                max_res = max(hi_vals) if hi_vals else None
                min_sup = min(lo_vals) if lo_vals else None

                arr.append({
                    "ts": label,
                    "pair": pair,
                    "base": base,
                    "quote": quote,
                    "max_resistance": max_res,
                    "min_support": min_sup,
                })

            out[pair] = arr

        self.state["per_currency"] = out


# ====== esempio d’uso ======
# if __name__ == "__main__":
#     rb = ResistenzeBuilder(state_filename="ai_features_state.json")
#     stato = rb.run("./storico_input")
#     print(json.dumps(stato, indent=2, ensure_ascii=False))
