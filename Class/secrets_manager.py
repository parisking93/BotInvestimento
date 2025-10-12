# Class/secrets_manager.py
from __future__ import annotations
import keyring
import getpass

SERVICE = "BotInvestimento"

def set_secret(key: str, value: str) -> None:
    keyring.set_password(SERVICE, key, value)

def get_secret(key: str) -> str | None:
    return keyring.get_password(SERVICE, key)

def interactive_setup() -> None:
    print("== Primo setup segreti (Credential Manager) ==")
    for k in ("OPENAI_API_KEY", "KRAKEN_KEY", "KRAKEN_SECRET"):
        cur = get_secret(k)
        if cur:
            print(f"{k}: già presente (tenuto).")
            continue
        val = getpass.getpass(f"Inserisci {k}: ")
        if val:
            set_secret(k, val)
            print(f"{k}: salvato.")
    print("Fatto!")

if __name__ == "__main__":
    interactive_setup()
