"""
Commodity Calendar Trade:

Okay, so we find a list of Commodities and we need to see how they do each month to see if there are patterns.

If you're doing a simple strategy:
Find Commodities (or at least ETFs of commodities) with 1+ Sharpe on a certain month. Buy+Hold the unlevered ETF for the month.
Can get a 1 Sharpe Pretty easily

More Complicated Strategy:
Can get a 2 Sharpe, and just have to worry about End-of-day trades. Not too bad?
Long or Short the Leveraged ETF based on what SPX or VIX are doing at end of day.


"Commodities" (Anything With Futures)
(Underlying Futures, 1x ETF, 2-3/-2-3X ETF if possible)
# INDICES:
S&P500: ES=F, SPY, SPXL, SPXS, # does well in NOVEMBER?
NASDAQ: NQ=F, QQQ, TQQQ, SQQQ,
RUSSELL 2000: RTY=F, ^RUT, IWM,
DJIA: YM=F, DJI,
VIX: ^VIX, SVXY, UVIX, VIXY, SVIX, VXX
NIKKEI 225: NKD=F, ^NKD, JPXN(sorta)

# ENERGY:
# Crude Oil WTI: CL=F, USO, SCO, UCO, USL,
# Crude Oil Brent: BZ=F, BNO
# Heating Oil: HO=F
# Gasoline RBOB: RB=F, UGA
# Natural Gas: NG=F, UNG, UNL, BOIL, KOLD
# Uranium: UX

# BONDS:
# 30-Year ZB=F, TLT
# 10-Year ZN=F
# 5-Year ZF=F
# 2-Year ZT=F

# SOFTS:
# Cocoa CC=F,
# Cotton CT=F,
# Orange Juice OJ=F,
# Coffee KC=F,
# Lumber LBS=F,
# Sugar SB=F, CANE


# METALS:
# Gold: GC=F, GLD, UGL, SGL
# Silver: SI=F, SLV
# Platinum: PL=F, PPLT, PLTM,
# Copper: HG=F, CPER, CPXR,
# Palladium: PA=F, PALL

# GRAINS:
# Soybeans: ZS=F, SOYB
# Soybean Meal:
# Soybean Oil: ZL=F,
# Corn: ZC=F, CORN, CXRN, CORX,
# Wheat: KE=F, WEAT, WXET,
# Rough Rice:
# Oats: ZO=F,
# Canola:

# MEAT:
# Live Cattle: LE=F
# Lean Hogs: HE=F
# Feeder Cattle: GF=F

# SPDR Select Funds: (I think they have futures)
# XLRE, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY


# Currencies: (don't really work on a month basis)
# USD: DX=F, UUP, UDN
# EUR: FXE, ULE, EUO,
# JPY: YCL, YCS, FXY,
# GBP: FXB,
# CAD: FXC,
# CHF: FXF,
# AUD: FXA
# BTC: BTC=F, GBTC,
# ETH:

Freight:
# BDRY # Shipping? Seems to be good in June, assuming VIX goes up? Don't know enough about freight.
# BWET

"""

import time
import sys
from datetime import timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


import pandas_market_calendars as  mcal
from urllib.request import urlopen
import ssl

measure_list = [#"^VVIX", # Vol-of-Vol
                "^VIX", # Vol
                #"^SPX", # S&P 500
                #"TLT", # Long Term Bonds
                ]
ticker_list = ["^SPX"]
month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
start_date = "2015-01-01"
end_date = "2030-01-01"
risk_free_rate = 0.03/252


data_list = []
for ticker in ticker_list+measure_list:
    temp = yf.download(ticker, start=start_date, end=end_date, interval="1d")[["Close", "Open", "High", "Low"]]
    temp.columns = temp.columns.droplevel(1)
    temp["C2C"] = (temp["Close"] - temp["Close"].shift(1)) /  temp["Close"].shift(1) * 100
    temp = temp.add_suffix(f"_{ticker}")
    data_list.append(temp)

data = pd.concat(data_list, axis=1)
data["Date-str"] = data.index.astype(str)
data["Date-date"] = data.index
data["year"] = pd.to_datetime(data.index).year
data["month"] = pd.to_datetime(data.index).month
data["day"] = pd.to_datetime(data.index).day
data['day_num'] = data.groupby(["year", "month"])['Date-str'].rank(method='first').astype(int)
data['day_numr'] = data.groupby(["year", "month"])['Date-str'].rank(method='first', ascending=False).astype(int)
data["day_name"] = pd.to_datetime(data.index).day_name()
data["days_skip"] = (data["Date-date"] - data["Date-date"].shift(1)).dt.days - 1


# Let's try some things:
data_dict = dict()
data_dict["Normal Data"] = data
for ticker in measure_list:
    data_dict[f"{ticker} up"] = data[data[f"C2C_{ticker}"].shift(1)>=0]
    data_dict[f"{ticker} down"] = data[data[f"C2C_{ticker}"].shift(1)<0]

#print(data.head().to_string())
#print(data.tail().to_string())

# Now, let's make line plots of the daily returns by month:
i = 0
for key, value in data_dict.items():
    print(key)
    for ticker in ticker_list:
        temp = value.dropna(subset=[f'C2C_{ticker}'], inplace=False)
        print(f"\t Return Data for {ticker}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
        ax1 = sns.barplot(x="month", y=f'C2C_{ticker}', data=temp, ax=ax1)
        ax1.set_title(f"Monthly Bar+Whisker Plots for {ticker} {key} by month")
        ax1.set_yscale("linear")
        # Add the average value labels on top of the bars
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f')  # Formats the label to 2 decimal places
        # By Month Line chart:
        #legend_list = [f"{ticker} total"]
        temp[f"ret_{ticker}"] =  (1+temp[f"C2C_{ticker}"]/100.0).cumprod()
        ax2 = sns.lineplot(data=temp, x="Date-date", y=f"ret_{ticker}", label=f"{ticker} Return", ax=ax2)
        ax2.set_title(f"Cumulative Return Plots for {ticker} {key} by month")
        ax2.set_yscale("log")
        for j in range(1,13):
            temp[f"ret_{j}"] = np.where(temp["month"] == j, temp[f"C2C_{ticker}"]/100.0,0)
            temp[f"ret_{j}_cum"] = (1+temp[f"ret_{j}"]).cumprod()
            #legend_list.append(f"{ticker} month {j} Return")
            sns.lineplot(data=temp, x="Date-date",y=f"ret_{j}_cum", label=f"{ticker} {month_list[j-1]} Return")

            # Print the returns:
            print(f"\t \t {month_list[j-1]} "
                  f"Days Traded: {np.where((temp["month"] == j) & (temp[f"ret_{j}"]!=0), 1, 0).sum()}, "
                  f"Avg Daily Return: {round(temp[temp["month"] == j][f"ret_{j}"].mean()*100,2)}%, ",
                  f"Pct Days Profitable: {round(np.where(temp[temp["month"] == j][f"ret_{j}"] > 0, 1,0).mean()*100, 2)}%, ",
                  f"Est. Sharpe: {round((temp[temp["month"] == j][f"ret_{j}"].mean() - risk_free_rate) / temp[temp["month"] == j][f"ret_{j}"].std() * np.sqrt(252), 2)}, "
                  f"Total (Geo) Return: {round((temp[f"ret_{j}_cum"].iloc[-1] -1)*100, 2)}%"
                  )

        plt.legend()
        plt.tight_layout()
        plt.show()
    i+=1



# Trade system backtest.
trade_dict = {}
# method follows
# ETFs:
trade_dict[1] = {"ticker": "GLD", "L/S": "Long", "measure": "^SPX", "direction": "Up"}
trade_dict[2] = {"ticker": "UGA", "L/S": "Long", "measure": "^VIX", "direction":"Down"}
trade_dict[3] = {"ticker": "XLU", "L/S": "Long", "measure": "^VIX", "direction":"Up"} # Utilities are a commodity, right???
trade_dict[4] = {"ticker": "BOIL", "L/S": "Long", "measure": "^VIX", "direction":"Down"}
trade_dict[5] = {"ticker": "UGA", "L/S": "Long", "measure": "^VIX", "direction":"Down"} # or SVXY
trade_dict[6] = {"ticker": "UCO", "L/S": "Long", "measure": "^VIX", "direction": "Up"} # not certain yet
trade_dict[7] = {"ticker": "PALL", "L/S": "Long", "measure": "^SPX", "direction":"Up"} # or SVXY, or GLD
trade_dict[8] = {"ticker": "BOIL", "L/S": "Long", "measure": "^VIX", "direction": "Up"} #  TLT does okay?
trade_dict[9] = {"ticker": "CANE", "L/S": "Long", "measure": "^VIX", "direction":"Up"} # Real Estate's a Commodity, right??? Uranium isn't... difficult, right?
trade_dict[10] = {"ticker": "CORN", "L/S": "Long", "measure": "^VIX", "direction":"Up"} # or XLF??
trade_dict[11] = {"ticker": "SCO", "L/S": "Long", "measure": "^VIX", "direction":"Up"}
trade_dict[12] = {"ticker": "KOLD", "L/S": "Long", "measure": "^VIX", "direction":"Down"}




data_list = []
data_list_easy = []
for key, value in trade_dict.items():
    print(key, value)
    if value["ticker"] is not None:
        temp = yf.download(value["ticker"], start=start_date, end=end_date)[["Close"]]
        temp.columns = temp.columns.droplevel(1)
        temp["C2C"] = (temp["Close"] - temp["Close"].shift(1)) / temp["Close"].shift(1)
        if value["L/S"] == "Short":
            # might want to add a 1-2bp cost to short per day???
            temp["C2C"] *= -1
        temp["month"] = pd.to_datetime(temp.index).month
        temp = temp.add_suffix(f"_{value["ticker"]}")

        # Add the easy returns to the "easy" one
        temp_easy = temp[temp[f"month_{value["ticker"]}"] == key]
        temp_easy = temp_easy[[f"C2C_{value["ticker"]}"]]
        temp_easy = temp_easy.rename(columns={f"C2C_{value["ticker"]}": "position"})
        temp_easy["ticker"] = value["ticker"]
        data_list_easy.append(temp_easy)

        # Add the more complicated stuff
        if value["measure"] is not None:
            temp2 = yf.download(value["measure"], start=start_date, end=end_date)[["Close"]]
            temp2.columns = temp2.columns.droplevel(1)
            temp2["C2C"] = (temp2["Close"] - temp2["Close"].shift(1)) / temp2["Close"].shift(1)
            temp2 = temp2.add_suffix(f"_{value["measure"]}")
            temp = pd.merge(temp, temp2, how="left", left_index=True, right_index=True)
            if value["direction"] == "Up":
                temp = temp[temp[f"C2C_{value["measure"]}"].shift(1) > 0]
            elif value["direction"] == "Down":
                temp = temp[temp[f"C2C_{value["measure"]}"].shift(1) < 0]
        # now separate out the one-month's data
        temp = temp[temp[f"month_{value["ticker"]}"] == key]
        temp = temp[[f"C2C_{value["ticker"]}"]]
        temp = temp.rename(columns={f"C2C_{value["ticker"]}": "position"})
        temp["ticker"] = value["ticker"]
        # add to the data list.
        data_list.append(temp)


data_trade = pd.concat(data_list, axis=0).sort_index()
data_trade_easy = pd.concat(data_list_easy, axis=0).sort_index()
data_trade["cum_return"] = (1 + data_trade["position"]).cumprod()
data_trade["arith_return"] = 1+ (data_trade["position"]).cumsum()
data_trade_easy["cum_return"] = (1 + data_trade_easy["position"]).cumprod()
data_trade_easy["arith_return"] = 1 + (data_trade_easy["position"]).cumsum()


print(data_trade.head().to_string())
print(data_trade.tail().to_string())

# Now let's get the returns:
data_names = ["Filtered Portfolio", "Basic Buy+Hold 1-month"]
i = 0
for _data in [data_trade, data_trade_easy]:
    print("\n")
    print(data_names[i])
    i+=1
    print(f"Days invested:", len(_data), "From:", str(_data.index[0])[:10], "To: ", str(_data.index[-1])[:10])
    print(f"Pct Profitable", round(np.where(_data["position"]>0, 1, 0).mean()*100,2),"%")
    print(f"Avg Daily Return When Invested:", round(_data["position"].mean()*100, 2),"%")
    print(f"Estimated Sharpe:", round((_data["position"].mean() - risk_free_rate)/_data["position"].std()*np.sqrt(252),2))
    print(f"Total Return Geometric:", round((_data["cum_return"].iloc[-1]-1)*100, 2),"%", f"Total Return Arithmatic:", round((_data["arith_return"].iloc[-1]-1)*100, 2),"%",)


plt.figure(dpi=150)
plt.title("Geometric returns across time")
plt.plot(data_trade.index, data_trade["cum_return"], label="Position")
plt.plot(data_trade_easy.index, data_trade_easy["cum_return"], label="Easy Position")
plt.yscale("log")
plt.legend(["VIX-Aware Commodity Calendar", "Simple Commodity Calendar"])
plt.show()

plt.figure(dpi=150)
plt.title("Arithmatic returns across time")
plt.plot(data_trade.index, data_trade["arith_return"], label="Position")
plt.plot(data_trade_easy.index, data_trade_easy["arith_return"], label="Easy Position")
plt.yscale("linear")
plt.legend(["VIX-Aware Commodity Calendar", "Simple Commodity Calendar"])
plt.show()

# Percent of total returns from each ETF:

#print(data_trade.head().to_string())
#print(data_trade.tail().to_string())

data_trade2 = data_trade.groupby("ticker")["position"].sum()
data_trade_easy2 = data_trade_easy.groupby("ticker")["position"].sum()
print(data_trade2.to_string())
print(data_trade_easy2.to_string())

# We can see that the real workhorses of the trade are the Natural Gas, Oil, and Gasoline trades,
# but the others seem to help.