"""
I came across this by accident, but it's still interesting.


"""


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ibs_check_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
risk_free_rate = 0.03/252
ticker_list = ["SPY", "SVXY", "VIXY", "SVIX", "^VIX", "^VIX3M", "^VIX1D"]
data_list =[]
for ticker in ticker_list:
    temp = yf.download(ticker, start="2011-01-01", end="2030-01-01")[["Open", "High", "Low", "Close"]]
    temp.columns = temp.columns.droplevel(1)
    temp["Return"] = (temp["Close"] - temp["Close"].shift(1)) / temp["Close"].shift(1)
    temp["nextReturn"] = temp["Return"].shift(-1)
    temp["IBS"] = round((temp["Close"] - temp["Low"]+0.000001)/ (temp["High"] - temp["Low"]+0.00000000001), 4)
    temp["cum_return"] = (1 + temp["nextReturn"]).cumprod()
    temp = temp.add_suffix(f"_{ticker}")
    data_list.append(temp)
data = pd.concat(data_list, axis=1)
# Make sure we don't trade VIXY on monday?
data["weekday"] = pd.to_datetime(data.index).day_name()


# Now make returns lists:
for i in ibs_check_list:
    data[f"return_strat_{i}"] = 0
    data[f"return_strat_{i}"][data["IBS_SPY"] <= i] = data["nextReturn_SVXY"]
    data[f"return_strat_{i}"][data["IBS_SPY"] > i] = data["nextReturn_VIXY"]
    # Avoid holding VIXY on mondays? (We use those days for SVXY Monday trades anyway...)
    #data[f"return_strat_{i}"][(data["weekday"]=="Friday") & (data["IBS_SPY"] > i)] = 0
    # Avoid holding when VIX in backwardation
    data[f"return_strat_{i}"][(data["Close_^VIX"] >= data["Close_^VIX3M"]) & (data["IBS_SPY"] < i)] = 0
    data[f"cum_return_{i}"] = (1+data[f"return_strat_{i}"]).cumprod()

# now let's compare normal Volatilty Reversion
# "VRC" Volatility Reversion Classic
data["nextReturn_VRC"] = 0
data["nextReturn_VRC"][(data["Close_^VIX1D"] >= 15) & (data["Close_^VIX"] < data["Close_^VIX3M"])] = data["nextReturn_SVIX"]
data["nextReturn_VRC"][(data["Close_^VIX1D"] <= 10)] = data["nextReturn_VIXY"]
data[f"cum_return_VRC"] = (1+data[f"nextReturn_VRC"]).cumprod()

print(data.head(5).to_string())
print(data.tail(5).to_string())


for i in  ibs_check_list:
    temp = data[[f"return_strat_{i}",f"cum_return_{i}"]]
    temp = temp.dropna()
    print(f"Return Stats for: {i} IBS threshold")
    print(f"Avg Daily Return: {round(temp[f"return_strat_{i}"].mean()*100,4)}%")
    print(f"Avg Std Dev Return: {round(temp[f"return_strat_{i}"].std()*100, 4)}%")
    print(f"Sharpe: {round((temp[f"return_strat_{i}"].mean() - risk_free_rate)/temp[f"return_strat_{i}"].std() *np.sqrt(252), 2)}")
    print(f"Sortino: {round((temp[f"return_strat_{i}"].mean() - risk_free_rate)/
            np.where(temp[f"return_strat_{i}"]  > 0,temp[f"return_strat_{i}"],  0).std() * np.sqrt(252),2)}")
    print(f"Profit Factor: {round(np.where(temp[f"return_strat_{i}"] > 0, temp[f"return_strat_{i}"], 0).sum()/np.where(temp[f"return_strat_{i}"] < 0, -temp[f"return_strat_{i}"], 0).sum(), 2)}")
    print("\n")

print("Now for SVXY and SPY Buy+Hold, the Volatility Reversion Classic")
for i in ["SVXY", "SPY", "VRC"]:
    print(f"Return Stats for: {i}")
    temp = data[[f"nextReturn_{i}",f"cum_return_{i}"]]
    temp = temp.dropna()
    # we need to cut off VRC data before 2023-04, since VIX1D didn't exist before then.
    if i == "VRC": temp = temp[temp.index >= "2023-04-01"]
    print(f"Days Traded: {len(temp)}" )
    print(f"Avg Daily Return: {round(temp[f"nextReturn_{i}"].mean() * 100, 4)}%")
    print(f"Avg Std Dev Return: {round(temp[f"nextReturn_{i}"].std() * 100, 4)}%")
    print(
        f"Sharpe: {round((temp[f"nextReturn_{i}"].mean() - risk_free_rate) / temp[f"nextReturn_{i}"].std() * np.sqrt(252), 2)}")
    print(f"Sortino: {round((temp[f"nextReturn_{i}"].mean() - risk_free_rate) /
                            np.where(temp[f"nextReturn_{i}"] > 0, temp[f"nextReturn_{i}"], 0).std() * np.sqrt(252), 2)}")
    print(
        f"Profit Factor: {round(np.where(temp[f"nextReturn_{i}"] > 0, temp[f"nextReturn_{i}"], 0).sum() / np.where(temp[f"nextReturn_{i}"] < 0, -temp[f"nextReturn_{i}"], 0).sum(), 2)}")
    print("\n")

plt.plot(figsize=(12,10))
plt.title("IBS-focused Volatility Reversion Strategy")
legend_list = []
for i in ibs_check_list:
    temp = data[[f"return_strat_{i}",f"cum_return_{i}"]]
    temp = temp.dropna()
    label = f"Trade SPY IBS threshold {i}"
    plt.plot(temp.index, temp[f"cum_return_{i}"], label=label)
    legend_list.append(label)
for i in ["SVXY", "SPY", "VRC"]:
    temp = data[[f"nextReturn_{i}",f"cum_return_{i}"]]
    # we need to cut off VRC data before 2023-04, since VIX1D didn't exist before then.
    if i == "VRC": temp = temp[temp.index >= "2023-04-01"]
    temp = temp.dropna()
    plt.plot(temp.index, temp[f"cum_return_{i}"], label=f"{i} return")
legend_list.append("B+H SVXY Return")
legend_list.append("B+H SPY Return")
legend_list.append("Vol Reversion Classic Return")
plt.legend(legend_list)
plt.xlabel("Time")
plt.ylabel("Return of $1")
plt.yscale("log")
plt.show()