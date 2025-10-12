class Action:
    def __init__(
        self,
        pair: str | None,
        tipo: str,                # 'sell' | 'buy' | 'order'
        quando: str,              # 'adesso' | 'al_limite' | 'al_break' | 'imposta_stop'
        lato: str | None = None,  # 'long' | 'short' | None
        prezzo: float | None = None,
        limite: float | None = None,
        break_price: float | None = None,  # soglia di rottura
        stop_loss: float | None = None,
        take_profit: float | None = None,
        quantita: float | None = None,
        timeframe: str | None = None,
        motivo: str | None = None,
        quantita_eur: float | None = None,
        leverage: str | None = None,   # <-- aggiunto (es. "2:1")
    ):
        self.pair = pair
        self.tipo = tipo
        self.quando = quando
        self.lato = lato
        self.prezzo = prezzo
        self.limite = limite
        self.break_price = break_price      # campo usato dall'esecutore ordini
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.quantita = quantita
        self.timeframe = timeframe
        self.motivo = motivo
        self.leverage = leverage
        self.quantita_eur = quantita_eur

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "tipo": self.tipo,
            "quando": self.quando,
            "lato": self.lato,
            "prezzo": self.prezzo,
            "limite": self.limite,
            "break_price": self.break_price ,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantita": self.quantita,
            "timeframe": self.timeframe,
            "motivo": self.motivo,
            "leverage": self.leverage,  # <-- esportato
        }


class Actions:
    def __init__(self):
        self.items: list[Action] = []

    def add(self, action: Action) -> None:
        self.items.append(action)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def to_list(self) -> list[dict]:
        return [a.to_dict() for a in self.items]
