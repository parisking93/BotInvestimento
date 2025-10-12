#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kraken quick tester: REST + WS v1 (public/private) + WS v2 (ping)
- Nessun replace/alias: usa i nomi ufficiali come forniti da AssetPairs/WS.
- Logga ogni request (>>) e ogni response (<<), abbreviando output troppo lunghi.
Run:
  pip install websockets aiohttp
  KRAKEN_KEY=... KRAKEN_SECRET=...  python test.py
"""

import os, json, time, hmac, hashlib, base64, urllib.parse, asyncio
from typing import Dict, Any, List, Tuple
import aiohttp
import websockets

API_URL = "https://api.kraken.com"
WS_V1_URL = "wss://ws.kraken.com"
WS_V2_URL = "wss://ws.kraken.com/v2"

LOG_TRIM = 220  # tronca le righe lunghissime per il log

# ---- logging helpers ---------------------------------------------------------
def jdump(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def log_send(obj: Any, tag: str = "") -> None:
    s = jdump(obj)
    if len(s) > LOG_TRIM: s = s[:LOG_TRIM] + "…"
    print(f">> {tag}{s}")

def log_recv(msg: Any, tag: str = "") -> None:
    if isinstance(msg, (bytes, bytearray)):
        msg = msg.decode("utf-8", "replace")
    s = msg
    if len(s) > LOG_TRIM: s = s[:LOG_TRIM] + "…"
    print(f"<< {tag}{s}")

# ---- REST helpers ------------------------------------------------------------
async def rest_public(session: aiohttp.ClientSession, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = API_URL + path
    async with session.get(url, params=params, timeout=20) as resp:
        resp.raise_for_status()
        data = await resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken REST error {path}: {data['error']}")
        return data

async def rest_private(
    session: aiohttp.ClientSession, path: str, data: Dict[str, Any] | None, key: str, secret: str
) -> Dict[str, Any]:
    url = API_URL + path
    data = dict(data or {})
    nonce = str(int(time.time() * 1000))
    data["nonce"] = nonce
    postdata = urllib.parse.urlencode(data)
    # API-Sign = base64( HMAC-SHA512( urlpath + SHA256(nonce+postdata), base64_decode(secret) ) )
    message = (nonce + postdata).encode()
    sha256 = hashlib.sha256(message).digest()
    mac = hmac.new(base64.b64decode(secret), path.encode() + sha256, hashlib.sha512)
    sig = base64.b64encode(mac.digest()).decode()

    headers = {"API-Key": key, "API-Sign": sig}
    async with session.post(url, data=data, headers=headers, timeout=20) as resp:
        resp.raise_for_status()
        data = await resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken REST error {path}: {data['error']}")
        return data

async def get_ws_token(session: aiohttp.ClientSession, key: str, secret: str) -> str:
    out = await rest_private(session, "/0/private/GetWebSocketsToken", {}, key, secret)
    return out["result"]["token"]

async def dump_assetpairs(session: aiohttp.ClientSession, quote_filter: str = "EUR", sample: int = 8) -> Tuple[List[str], List[str]]:
    """Ritorna (ws_pairs, alt_pairs) solo per la quote richiesta; non normalizza nulla."""
    data = await rest_public(session, "/0/public/AssetPairs")
    result = data["result"]  # dict: { pair_code: meta }

    rows = []
    for code, meta in result.items():
        base = meta.get("base")
        quote = meta.get("quote")
        wsname = meta.get("wsname")
        alt = meta.get("altname")
        if quote_filter:
            q = (quote or "").upper()
            if q not in (quote_filter.upper(), "Z" + quote_filter.upper()):
                continue
        rows.append(
            {
                "kr_code": code,
                "wsname": wsname,
                "altname": alt,
                "base": base,
                "quote": quote,
                "pair_decimals": meta.get("pair_decimals"),
                "lot_decimals": meta.get("lot_decimals"),
                "ordermin": meta.get("ordermin"),
            }
        )
    print(f"[REST] AssetPairs (quote={quote_filter}) total={len(rows)} — sample:")
    for r in rows[:sample]:
        print("   ", r)

    # pick 1–2 pairs for demos
    ws_pairs = [r["wsname"] for r in rows if r["wsname"]][:2] or ["XBT/EUR"]
    alt_pairs = [r["altname"] for r in rows if r["altname"]][:2] or ["XBTEUR"]
    return ws_pairs, alt_pairs

# ---- WS v1 demos -------------------------------------------------------------
async def ws_v1_public_demo(pairs_wsname: List[str], seconds: int = 5) -> None:
    """Sub a 'ticker' (public). Usa wsname: es. 'XBT/EUR'."""
    async with websockets.connect(WS_V1_URL, ping_interval=30) as ws:
        sub = {"event": "subscribe", "pair": pairs_wsname, "subscription": {"name": "ticker"}}
        log_send(sub, "ws1 ")
        await ws.send(jdump(sub))

        start = time.time()
        while time.time() - start < seconds:
            msg = await ws.recv()
            log_recv(msg, "ws1 ")
        # unsubscribe (best practice)
        unsub = {"event": "unsubscribe", "pair": pairs_wsname, "subscription": {"name": "ticker"}}
        log_send(unsub, "ws1 ")
        await ws.send(jdump(unsub))

async def ws_v1_private_demo(token: str, seconds: int = 5) -> None:
    """Sub a 'openOrders' (private) usando il token di GetWebSocketsToken."""
    async with websockets.connect(WS_V1_URL, ping_interval=30) as ws:
        sub = {"event": "subscribe", "subscription": {"name": "openOrders", "token": token}}
        log_send(sub, "ws1 ")
        await ws.send(jdump(sub))

        start = time.time()
        while time.time() - start < seconds:
            msg = await ws.recv()
            log_recv(msg, "ws1 ")
        # unsubscribe
        unsub = {"event": "unsubscribe", "subscription": {"name": "openOrders", "token": token}}
        log_send(unsub, "ws1 ")
        await ws.send(jdump(unsub))

# ---- WS v2 demo --------------------------------------------------------------
# Lista di messaggi WS v2 da inviare (puoi aggiungere altri metodi trading qui)
REQUESTS_V2: List[Dict[str, Any]] = [
    {"method": "ping", "params": {}},
    # Esempi che puoi attivare dopo aver verificato sulla tua doc/versione:
    # {"method": "get_open_orders", "params": {"symbol": "BTC/EUR"}},
    # {"method": "add_order", "params": {"order_type":"limit","side":"buy","symbol":"BTC/EUR","order_qty":0.001,"limit_price":25000}},
]

async def ws_v2_demo(messages: List[Dict[str, Any]], seconds: int = 5) -> None:
    """WS v2 generic: invia i messaggi in REQUESTS_V2 e logga le risposte."""
    async with websockets.connect(WS_V2_URL, ping_interval=30) as ws:
        for req in messages:
            log_send(req, "ws2 ")
            await ws.send(jdump(req))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=seconds)
                log_recv(msg, "ws2 ")
            except asyncio.TimeoutError:
                print("<< ws2 timeout (nessuna risposta)")

# ---- main --------------------------------------------------------------------
async def main() -> None:
    key = os.getenv("KRAKEN_KEY")
    secret = os.getenv("KRAKEN_SECRET")

    async with aiohttp.ClientSession() as session:
        # 1) REST: elenco pair
        ws_pairs, alt_pairs = await dump_assetpairs(session, quote_filter="EUR", sample=6)

        # 2) WS v1 public
        print("\n[DEMO] WS v1 PUBLIC ticker on:", ws_pairs)
        await ws_v1_public_demo(ws_pairs, seconds=6)

        # 3) WS v1 private (solo se key/secret presenti)
        if key and secret:
            print("\n[DEMO] WS v1 PRIVATE openOrders")
            token = await get_ws_token(session, key, secret)
            await ws_v1_private_demo(token, seconds=6)
        else:
            print("\n[SKIP] WS v1 PRIVATE: set KRAKEN_KEY and KRAKEN_SECRET to run this step.")

        # 4) WS v2 (ping + eventuali altri messaggi in REQUESTS_V2)
        print("\n[DEMO] WS v2 messages")
        await ws_v2_demo(REQUESTS_V2, seconds=6)

if __name__ == "__main__":
    asyncio.run(main())
