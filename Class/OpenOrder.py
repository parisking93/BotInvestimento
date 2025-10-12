from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, TypeVar
import os, json

T = TypeVar("T")

# ---------- helpers ----------
def _to_float(v) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v))
    except Exception: return None

def _to_int(v) -> Optional[int]:
    if v is None: return None
    if isinstance(v, bool): return int(v)
    try: return int(float(str(v)))
    except Exception: return None

def _to_bool(v) -> Optional[bool]:
    if v is None: return None
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}: return True
    if s in {"false", "0", "no", "n"}: return False
    return None

# ---------- models ----------
@dataclass
class OpenOrder:
    kr_pair: Optional[str] = None
    pair: Optional[str] = None
    type: Optional[str] = None
    ordertype: Optional[str] = None
    price: Optional[float] = None
    price2: Optional[float] = None
    vol_rem: Optional[float] = None
    base: Optional[str] = None
    quote: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpenOrder":
        return OpenOrder(
            kr_pair=d.get("kr_pair"), pair=d.get("pair"),
            type=d.get("type"), ordertype=d.get("ordertype"),
            price=_to_float(d.get("price")), price2=_to_float(d.get("price2")),
            vol_rem=_to_float(d.get("vol_rem")), base=d.get("base"),
            quote=d.get("quote"),
        )

@dataclass
class Trade:
    ordertxid: Optional[str] = None
    postxid: Optional[str] = None
    pair: Optional[str] = None
    aclass: Optional[str] = None
    time: Optional[float] = None
    type: Optional[str] = None
    ordertype: Optional[str] = None
    price: Optional[float] = None
    cost: Optional[float] = None
    fee: Optional[float] = None
    vol: Optional[float] = None
    margin: Optional[float] = None
    leverage: Optional[float] = None
    misc: Optional[str] = None
    trade_id: Optional[int] = None
    maker: Optional[bool] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trade":
        return Trade(
            ordertxid=d.get("ordertxid"), postxid=d.get("postxid"),
            pair=d.get("pair"), aclass=d.get("aclass"),
            time=_to_float(d.get("time")), type=d.get("type"),
            ordertype=d.get("ordertype"), price=_to_float(d.get("price")),
            cost=_to_float(d.get("cost")), fee=_to_float(d.get("fee")),
            vol=_to_float(d.get("vol")), margin=_to_float(d.get("margin")),
            leverage=_to_float(d.get("leverage")), misc=d.get("misc"),
            trade_id=_to_int(d.get("trade_id")), maker=_to_bool(d.get("maker")),
        )

@dataclass
class Ledger:
    aclass: Optional[str] = None
    amount: Optional[float] = None
    asset: Optional[str] = None
    balance: Optional[float] = None
    fee: Optional[float] = None
    refid: Optional[str] = None
    time: Optional[float] = None
    type: Optional[str] = None
    subtype: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Ledger":
        return Ledger(
            aclass=d.get("aclass"), amount=_to_float(d.get("amount")),
            asset=d.get("asset"), balance=_to_float(d.get("balance")),
            fee=_to_float(d.get("fee")), refid=d.get("refid"),
            time=_to_float(d.get("time")), type=d.get("type"),
            subtype=d.get("subtype"),
        )

@dataclass
class PortfolioRow:
    code: Optional[str] = None
    asset: Optional[str] = None
    qty: Optional[float] = None
    px_EUR: Optional[float] = None
    val_EUR: Optional[float] = None
    avg_buy_EUR: Optional[float] = None
    pnl_pct: Optional[float] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Optional["PortfolioRow"]:
        if not d:
            return None
        return PortfolioRow(
            code=d.get("code"), asset=d.get("asset"),
            qty=_to_float(d.get("qty")), px_EUR=_to_float(d.get("px_EUR")),
            val_EUR=_to_float(d.get("val_EUR")),
            avg_buy_EUR=_to_float(d.get("avg_buy_EUR")),
            pnl_pct=_to_float(d.get("pnl_pct")),
        )

@dataclass
class Portfolio:
    row: Optional[PortfolioRow] = None
    trades: List[Trade] = field(default_factory=list)
    ledgers: List[Ledger] = field(default_factory=list)
    available: Dict[str, Optional[float]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Portfolio":
        d = d or {}
        return Portfolio(
            row=PortfolioRow.from_dict(d.get("row") or {}),
            trades=[Trade.from_dict(t) for t in (d.get("trades") or [])],
            ledgers=[Ledger.from_dict(l) for l in (d.get("ledgers") or [])],
            available={
                "base": _to_float((d.get("available") or {}).get("base")),
                "quote": _to_float((d.get("available") or {}).get("quote")),
            } if d.get("available") is not None else {},
        )

@dataclass
class Currency:
    base: Optional[str] = None
    quote: Optional[str] = None
    pair: Optional[str] = None
    kr_pair: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    pair_limits: Optional[Dict[str, Any]] = None
    open_orders: List[OpenOrder] = field(default_factory=list)
    portfolio: Portfolio = field(default_factory=Portfolio)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Currency":
        return Currency(
            base=d.get("base"), quote=d.get("quote"), pair=d.get("pair"),
            kr_pair=d.get("kr_pair"), info=d.get("info") or {},
            pair_limits=d.get("pair_limits"),
            open_orders=[OpenOrder.from_dict(o) for o in (d.get("open_orders") or [])],
            portfolio=Portfolio.from_dict(d.get("portfolio") or {}),
        )

# ---------- IO GENERICO ----------
class JsonObjectIO:
    """
    Loader/Writer generico di liste di oggetti da/verso JSON.
    Il path è risolto partendo dal file chiamante (passato come stringa).
    """

    @staticmethod
    def _resolve_path(file_name: str, rel_folder_from_caller: str, caller_file_path: str) -> str:
        base_dir = os.path.dirname(os.path.abspath(caller_file_path))
        folder = os.path.join(base_dir, rel_folder_from_caller) if rel_folder_from_caller else base_dir
        return os.path.join(folder, file_name)

    @staticmethod
    def load_list_from_json(
        file_name: str,
        rel_folder_from_caller: str,
        caller_file_path: str,
        factory: Callable[[Dict[str, Any]], T],
        *,
        top_field: Optional[str] = None,   # es. "results" se il JSON è {"results":[...]}
    ) -> List[T]:
        """
        Legge il JSON <file_name> dentro <rel_folder_from_caller> relativo a <caller_file_path>
        e ritorna List[T] usando la factory.
        """
        path = JsonObjectIO._resolve_path(file_name, rel_folder_from_caller, caller_file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if top_field and top_field in data and isinstance(data[top_field], list):
                items = data[top_field]
            else:
                # se è un singolo oggetto, avvolgilo in lista
                items = [data]
        else:
            raise TypeError(f"Formato JSON non supportato: {type(data)}")

        return [factory(obj or {}) for obj in items]

    @staticmethod
    def save_list_to_json(
        file_name: str,
        rel_folder_from_caller: str,
        caller_file_path: str,
        objects: List[Any],
        *,
        indent: int = 2
    ) -> str:
        """
        Salva una lista di dataclass/dict in JSON nel percorso relativo al chiamante.
        Ritorna il percorso completo del file scritto.
        """
        path = JsonObjectIO._resolve_path(file_name, rel_folder_from_caller, caller_file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def _as_dict(x):
            try:
                return asdict(x)  # dataclass
            except Exception:
                return x          # dict già pronto

        with open(path, "w", encoding="utf-8") as f:
            json.dump([_as_dict(x) for x in objects], f, ensure_ascii=False, indent=indent)
        return path

    # wrapper comodo per Currency
    @staticmethod
    def load_currencies(
        file_name: str,
        rel_folder_from_caller: str,
        caller_file_path: str,
        *,
        top_field: Optional[str] = None
    ) -> List[Currency]:
        return JsonObjectIO.load_list_from_json(
            file_name=file_name,
            rel_folder_from_caller=rel_folder_from_caller,
            caller_file_path=caller_file_path,
            factory=Currency.from_dict,
            top_field=top_field,
        )
