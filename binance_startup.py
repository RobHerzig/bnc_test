from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceWithdrawException
import numpy as np


api_key = "sGfyIE0zYoKkZ3M0CtnuZpf070GGkUmXrDimdyavp3FMShQkh7unnrzue9pwGEay"
api_secret = "lz9NXEITdTH6sF8UZ1IPFaWs3MjjScHhjuMvyznS27GMFQviTWqlF1RvFP0D7snl"

client = Client(api_key, api_secret)

info = client.get_account()
# print(info)


def get_ratios_1min_24h(currency):
    # fetch 1 minute klines for the last day up until now
    klines = client.get_historical_klines(currency, Client.KLINE_INTERVAL_3MINUTE, "30 days ago UTC")
    print("KLINES: " + str(klines))

    close_data = np.array(klines)[:, 4]
    # print("CLOSE DATA" + str(close_data))
    result = np.zeros(len(close_data))
    for h in range(0, (len(close_data))):
        # ratio = float(close_data[h]) / float(close_data[h - 10])
        ratio = float(close_data[h])
        # print("DIVIDE : " + close_data[h] + " / " + close_data[h-10])
        # print(ratio)
        result[h] = ratio
    return result

def get_last_30min(currency):
    # fetch 1 minute klines for the last day up until now
    klines = client.get_historical_klines(currency, Client.KLINE_INTERVAL_3MINUTE, "2 days ago UTC")
    # print("KLINES: " + str(klines))

    close_data = np.array(klines)[:, 3]
    # print("CLOSE DATA" + str(close_data))
    result = np.zeros(len(close_data))
    for h in range(0, (len(close_data))):
        # ratio = float(close_data[h]) / float(close_data[h - 10])
        ratio = float(close_data[h])
        # print("DIVIDE : " + close_data[h] + " / " + close_data[h-10])
        # print(ratio)
        result[h] = ratio
    return result

