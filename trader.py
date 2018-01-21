from binance.client import Client
import numpy as np

api_key = "sGfyIE0zYoKkZ3M0CtnuZpf070GGkUmXrDimdyavp3FMShQkh7unnrzue9pwGEay"
api_secret = "lz9NXEITdTH6sF8UZ1IPFaWs3MjjScHhjuMvyznS27GMFQviTWqlF1RvFP0D7snl"

client = Client(api_key, api_secret)

info = client.get_account()
print(info)
print(client.get_all_tickers())

names = ["TRXETH",
         "OMGETH",
         "NEOETH",
         "LRCETH",
         "AMBETH"]
for name in names:
    print(name + ":")
    print(client.get_klines(symbol=name, interval=client.KLINE_INTERVAL_30MINUTE))
