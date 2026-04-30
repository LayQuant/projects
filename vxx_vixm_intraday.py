"""

Based on https://substack.com/@quantgalore/note/c-249177010

A short/long VXX/VIXM Intraday trade seems to work?
But only when you Exclude Thursdays...

"""


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


long = "VIXM"
short = "VXX"

ticker_list = [long, short, "^VIX", "^VVIX", "SPY", "^VIX3M", "QQQ"]
data_list = []
for ticker in ticker_list:
    temp = yf.download(ticker, start="2018-01-01", end="2026-04-27")
    temp.columns = temp.columns.droplevel(1)
    temp["O2C"] = (temp["Close"] - temp["Open"]) / temp["Open"]
    temp["C2O"] = (temp["Open"] - temp["Close"].shift(1)) / temp["Open"].shift(1)
    temp["C2C"] = (temp["Close"] - temp["Close"].shift(1)) / temp["Open"].shift(1)
    temp["IBS"] = (temp["Close"] - temp["Low"]) / (temp["High"] - temp["Low"])
    temp["ema"] = temp["Close"].shift(1).rolling(10).mean()
    temp = temp.add_suffix(f"_{ticker}")
    data_list.append(temp)
data = pd.concat(data_list, axis=1)
data["date"] = pd.to_datetime(data.index)
data["day_name"] = data["date"].dt.day_name()
data["L/S"] = data[f"O2C_{long}"] - data[f"O2C_{short}"]
data = data.fillna(0)
#print(data.head().to_string())
#print(data.tail().to_string())

# sorta works:
data["L/S_Thursday"] = np.where(data["day_name"].isin(["Thursday"]), data["L/S"], 0)
data["L/S"] = np.where(~data["day_name"].isin(["Thursday"]), data["L/S"], 0)

plt.title("VIXM/VXX Intraday Trade")
plt.plot(data["date"], (1+data["L/S"]).cumprod(), label="L/S_no_Thursday")
plt.plot(data["date"], (1+data["L/S_Thursday"]).cumprod(), label="L/S_Thursday_only")
plt.plot(data["date"], (1+data["O2C_VIXM"]).cumprod(), label="VIXM")
plt.plot(data["date"], (1+data["O2C_VXX"]).cumprod(), label="VXX")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.yscale("linear")
plt.show()

# The Intraday Sharpe is ~0.5, which is decent, but the avg return is 0.09 and that might have issues with spreads...
print("Statistics for VXX/VIXM Intraday Trade")
print("Avg return:", round(data["L/S"].mean()*100,2), "%")
print("Std Deviation Return:", round(data["L/S"].std()*100,2), "%")
print("Sharpe: ",round((data["L/S"].mean() - 0.03/252) / data["L/S"].std() * np.sqrt(252),2))
print("SPY Sharpe: ", round((data["O2C_SPY"].mean() - 0.03/252) / data["O2C_SPY"].std() * np.sqrt(252),2))