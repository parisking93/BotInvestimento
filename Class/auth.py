"""
Kraken Authorization module
- Defines two variables for credentials (API_KEY, API_SECRET)
- Exposes `authorize()` which returns an AuthResult object containing a WebSocket token
  and other handy data for subsequent API calls.
- Prints individual attributes if run as a script.

Works with: requests (standard), no third‑party deps.
It can also read credentials from environment or .env if you prefer.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    # Optional: load .env automatically if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import requests

# === 1) Two variables (replace with your values, or set env KRAKEN_KEY/KRAKEN_SECRET) ===
API_KEY: str = os.getenv("KRAKEN_KEY", "")
API_SECRET: str = os.getenv("KRAKEN_SECRET", "")  # base64 string from Kraken UI

API_BASE = "https://api.kraken.com"


@dataclass
class AuthResult:
    ok: bool
    ws_token: Optional[str]
    ws_expires_in: Optional[int]
    server_unixtime: Optional[int]
    balances: Dict[str, str]
    key_last4: str
    error: Optional[str] = None


def _sign(path: str, data: Dict[str, Any], secret_b64: str) -> str:
    """Create API-Sign header for Kraken private endpoints."""
    # Kraken requires: HMAC-SHA512(secret, path + SHA256(nonce + postdata))
    postdata = requests.models.RequestEncodingMixin._encode_params(data)
    # nonce must be included in hash
    m = hashlib.sha256()
    m.update((str(data["nonce"]) + postdata).encode("utf-8"))
    sha256 = m.digest()
    mac = hmac.new(base64.b64decode(secret_b64), (path.encode("utf-8") + sha256), hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def _private(endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing API_KEY/API_SECRET. Set variables or env KRAKEN_KEY/KRAKEN_SECRET.")
    url_path = f"/0/private/{endpoint}"
    url = API_BASE + url_path
    data: Dict[str, Any] = payload.copy() if payload else {}
    data["nonce"] = int(time.time() * 1000)
    headers = {
        "API-Key": API_KEY,
        "API-Sign": _sign(url_path, data, API_SECRET),
    }
    resp = requests.post(url, data=data, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _public(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE}/0/public/{endpoint}"
    resp = requests.get(url, params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def authorize() -> AuthResult:
    """Fetch WebSockets token + handy context for later use."""
    server_time = None
    try:
        t = _public("Time")
        if not t.get("error"):
            server_time = int(t["result"]["unixtime"])  # type: ignore
    except Exception:
        server_time = None

    ws_token = None
    ws_expires = None
    balances: Dict[str, str] = {}
    err: Optional[str] = None

    try:
        # 1) Get WebSockets private token
        r = _private("GetWebSocketsToken")
        if r.get("error"):
            err = "; ".join(r["error"])  # type: ignore
        else:
            res = r.get("result", {})
            ws_token = res.get("token")
            ws_expires = res.get("expires")  # seconds (if provided by API)
    except Exception as e:
        err = f"GetWebSocketsToken failed: {e}"

    try:
        # 2) Read balances (useful to confirm perms & funds)
        r = _private("Balance")
        if not r.get("error"):
            balances = r.get("result", {})
    except Exception:
        pass

    key_last4 = (API_KEY[-4:] if API_KEY else "")
    ok = bool(ws_token)
    return AuthResult(
        ok=ok,
        ws_token=ws_token,
        ws_expires_in=ws_expires,
        server_unixtime=server_time,
        balances=balances,
        key_last4=key_last4,
        error=err,
    )


if __name__ == "__main__":
    auth = authorize()
    print("OK:", auth.ok)
    print("Key last4:", auth.key_last4)
    print("Server time:", auth.server_unixtime)
    print("WS token:", (auth.ws_token[:8] + "…" if auth.ws_token else None))
    print("WS expires in:", auth.ws_expires_in)
    print("Balances:", auth.balances)
    if auth.error:
        print("Error:", auth.error)
