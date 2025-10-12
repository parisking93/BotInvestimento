import os, krakenex
k = krakenex.API(key=os.environ["KRAKEN_KEY"], secret=os.environ["KRAKEN_SECRET"])
tr = k.query_private('TradesHistory')
print(tr)
led = k.query_private('Ledgers')
print(list(led['result']['ledger'].items())[:5])  # prime 5 righe
