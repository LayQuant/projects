import yfinance as yf
import warnings
import pandas as pd
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


# FUNCTIONS:
def rsi(df, periods=20, ema=False, ref_point="close"):

    close_delta = df[ref_point].diff()
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    if ema == True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods).mean()
        ma_down = down.rolling(window=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


# We stop our data collection for that at 2022-10-01, so that when we can test the strategy without look-ahead bias.
start = "2000-01-01"
end = "2022-10-01"
# Let's look at VIX as well.
price_list = ["Open", "High", "Low", "Close"]


asset_list = [
              "^VIX", "^VVIX", "^VIX3M",
              ]


total_data = yf.download("SPY", start=start, end=end)[["Volume"]].add_suffix("placeholder")
total_data["Date-str"] = total_data.index.astype(str)
total_data["weekday_name"] = pd.to_datetime(total_data.index).day_name()
total_data["weekday"] = pd.to_datetime(total_data.index).dayofweek
total_data["year"] = pd.to_datetime(total_data.index).year
total_data["month"] = pd.to_datetime(total_data.index).month

# Get Trading Day of month Count:
total_data['day_num'] = total_data.groupby(["year", "month"])['Date-str'].rank(method='first').astype(int)

# GLD, HYG, TLT, /ZN=F, /CL=F, ^VIX, SPY
for ticker in asset_list:
    temp = yf.download(ticker, start=start, end=end, auto_adjust=True)
    temp["Gap"] = temp["Open"] - temp["Close"].shift(1)
    temp["Gap_pct"] = temp["Gap"] / temp["Close"].shift(1)
    temp["Open2Close"] = temp["Close"] - temp["Open"]
    temp["Open2Close_pct"] = temp["Open2Close"] / temp["Open"].shift(1)
    temp["Close2Close"] = temp["Close"] - temp["Close"].shift(1)
    temp["Close2Close_pct"] = temp["Close2Close"] / temp["Close"].shift(1)
    temp["Vol_chg"] = temp["Volume"] - temp["Volume"].shift(1)
    temp["Vol_chg_pct"] = temp["Vol_chg"] / temp["Volume"].shift(1)
    for price in ["Open", "High", "Low", "Close", "Volume"]:
        temp[f"{price}_prior"] = temp[price].shift(1)
        for num in [5, 20]:
            temp[f"{price}_RSI_{num}"] = rsi(temp, periods=num, ref_point=f"{price}")
            temp[f"{price}_EMA_{num}"] = temp[f"{price}_prior"].ewm(span=num).mean()
            temp[f"{price}_Regime_{num}"] = np.where(temp[f"{price}_prior"] > temp[f"{price}_EMA_{num}"], 1, 0)

    for i in [1, 3, 5]:
        temp[f"Close2Close{i}"] = temp["Close"].shift(-i) - temp["Close"]
        temp[f"Close2Close{i}_pct"] = temp[f"Close2Close{i}"] / temp["Close"]

    total_data = pd.merge(total_data, temp.add_suffix(f"_{ticker}"), how="left", left_index= True, right_index= True)


# Comp list
comp_list = []
for i in range(0, len(asset_list)):
    for j in range(0, len(asset_list)):
        if i != j:
            total_data[f"{asset_list[i]}-{asset_list[j]}"] = \
                total_data[f"Close2Close_pct_{asset_list[i]}"] - total_data[f"Close2Close_pct_{asset_list[j]}"]
            total_data[f"{asset_list[i]}r{asset_list[j]}"] = \
                total_data[f"Close_{asset_list[i]}"] / total_data[f"Close_{asset_list[j]}"]
            # Maybe make a 5 or 20-day moving average for the ratio???
            for num in [1, 3, 5]:
                total_data[f"{asset_list[i]}-{asset_list[j]}{num}"] = \
                    total_data[f"Close2Close{num}_pct_{asset_list[i]}"] - total_data[f"Close2Close{num}_pct_{asset_list[j]}"]
                total_data[f"{asset_list[i]}r{asset_list[j]}{num}"] = \
                    total_data[f"Close2Close{num}_{asset_list[i]}"] / total_data[
                        f"Close2Close{num}_{asset_list[j]}"]
            total_data[f"{asset_list[i]}-{asset_list[j]}_vol"] = \
                total_data[f"Vol_chg_pct_{asset_list[i]}"] - total_data[f"Vol_chg_pct_{asset_list[j]}"]
            comp_list.append(f"{asset_list[i]}-{asset_list[j]}")



# Let's make test and train data.
test_data = total_data[total_data.index >= "2020-01-01"]
total_data = total_data[(total_data.index >= "2014-01-01") & (total_data.index < "2020-01-01")]


# For "^VVIXr^VIX", Key values are 4 and 6, creating 3 regions, <4, 4-6, 6+
# For ^VIXr^VIX3M, and ^VIX3Mr^VIX6M Key Values are 1 (difference between contango and backwardation?)
# Trading VIX itself (Using Options):
# I'm plotting "^VVIXr^VIX" (VVIX-ratio-over-VIX) to ^VIXr^VIX3M (The ratio between VIX and VIX3Month)
# Using EoD Close Prices, 3-5 days.
# For 3-5 days out:VVIXrVIX < 6 and VIXrVIX3M between (0.9 and 1.5) ~70% of the time the VIX is negative in next 5 days.
# For 5 days out:VVIXrVIX > 6 and VIXrVIX3M between (0.7 and 1) ~60% of the time the VIX is positive in next 5 days.
# For 5 days out:VVIXrVIX between (4, 6) and VIXrVIX3M < 0.9, VIX is 50/50 up or down, but the %Vol is large.
    # Could be a good idea to run a long straddle here? Maybe see what a backtest shows.


xvar = "^VVIXr^VIX"
yvar = "^VIXr^VIX3M"
ticker_check = "^VIX"
days_out = 5 # How many days in the future are we trying to predict?
metric = "Close2Close" # what are we trying to see.
check = f"{metric}{days_out}_pct_{ticker_check}"

total_data[f"{check}_up"] = np.where(total_data[check] > 0, 1, 0)
test_data[f"{check}_up"] = np.where(test_data[check] > 0, 1, 0)
for data in [test_data, total_data]:
    data["checking"] = np.where(
        (data[xvar].between(0, 6))
        &
        (data[yvar].between(0.9, 1.5))
        , 1, 0)


sns.scatterplot(data=total_data, x=xvar, y=yvar, hue=check)
plt.show()
sns.scatterplot(data=test_data, x=xvar, y=yvar, hue=check)
plt.show()


i = 0
data_type_list = ["Overall early data", "Data fitting initial screen", "test data"]
for data in [total_data, total_data[total_data["checking"]==1], test_data[test_data["checking"]==1]]:
    print(data_type_list[i])
    i+=1
    print(f'Trading Day Count: {len(data)}')
    print(f'Percent of VIX periods up {days_out} days after: {data[f"{check}_up"].mean()*100}')
    print(f'Average change in VIX {days_out} days out: {data[check].mean() * 100}')
    print(f'Std Dev of change in VIX {days_out} days out: {data[check].mean() * 100}')
    print(f'Distribution of up vs down Vix periods: {data[f"{check}_up"].value_counts()}')
    print("\n")

