import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
from fredapi import Fred
warnings.filterwarnings("ignore")


# FUNCTIONS:
def rsi(df, periods=20, ema=False, ref_point="close"):
    """
    Returns a pd.Series with the relative strength index.
    """
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



# Backtest:
start = "2009-04-17" # when TMF and TMV shares started trading.
end = "2030-01-01"
interval = "1d"
# Don't actually care about 'norm' ticker it's just a placeholder
norm_ticker = "QQQ"
starting_capital = 1000
# Get initial data that we're trading: TMF
full_data = yf.download(norm_ticker, start = start, end = end, interval = interval).add_suffix(f"_{norm_ticker}")
ticker_list = ["TLT",
               "TMF",
               "TMV",
               "/ZN=F",
               "^MOVE", # Not really necessary for this strategy, but interesting to tinker with.
               ]
for data in ticker_list:
    temp_ticker = data
    data = yf.download(temp_ticker, start = start, end = end , interval = interval, auto_adjust=True)
    # Get Metrics:
    # Volume Change t-1 to t
    data["VolChange"] = data["Volume"] - data["Volume"].shift(1)
    data["VolChange_pct"] = data["VolChange"] / data["Volume"].shift(1)

    # Open t 2 Close t
    data["Open2Close"] = data["Close"] - data["Open"]
    data["Open2Close_pct"] = data["Open2Close"] / data["Open"]

    # Open t+1 2 Close t+1
    data["Open2Close1"] = data["Close"].shift(-1) - data["Open"].shift(-1)
    data["Open2Close1_pct"] = data["Open2Close1"] / data["Open"].shift(-1)

    # Close t-1 to Open t
    data["Close2Open"] = data["Open"].shift(-1) - data["Close"]
    data["Close2Open_pct"] = data["Close2Open"] / data["Close"]

    # Close t to Close t+1
    data["Close2Close"] = data["Close"].shift(-1) - data["Close"]
    data["Close2Close_pct"] = data["Close2Close"] / data["Close"]

    for price in ["Open", "High", "Low", "Close", "Volume"]:
        for i in [5, 20]:
            #data[f"{price}_EMA_{i}"] = data[price].ewm(span=i).mean()
            data[f"{price}_RSI_{i}"] = rsi(data, periods=i, ref_point=price)
    # Merge to our original data
    full_data = pd.merge(full_data, data.add_suffix(f"_{temp_ticker}"), how="left", left_index=True, right_index=True)


#print(full_data.tail().to_string())
full_data["Date-date"] = full_data.index
full_data["Date-str"] = full_data.index.astype(str)
full_data["year"] = pd.to_datetime(full_data.index).year
full_data["month"] = pd.to_datetime(full_data.index).month
full_data["weekday"] = pd.to_datetime(full_data.index).day_name()
# Get Trading Day of month Count:
full_data['day_num'] = full_data.groupby(["year", "month"])['Date-str'].rank(method='first').astype(int)


# If trading on ^MOVE, we need to drop the few days where there isn't ^MOVE data (Will need to get that raw data later and put it back in)
#full_data = full_data.dropna(subset=["Close_^MOVE"])
#full_data = full_data[full_data.index >= "2009-01-01"]


# Make the Buy+Sell Indicator:
full_data["B/S"] = 0

"""
# This one is proven to work, save it:
full_data.loc[((full_data["Close_RSI_20_^MOVE"] < 80) & (full_data["day_num"] >= 6)) | ((full_data["Close_RSI_20_^MOVE"] < 20) & (full_data["day_num"] <= 5)),"B/S"] = 1
full_data.loc[((full_data["Close_RSI_20_^MOVE"] > 20) & (full_data["day_num"] <= 5)) | ((full_data["Close_RSI_20_^MOVE"] > 80) & (full_data["day_num"] >= 6)),"B/S"] = -1

# This one is proven to work even better, save it:
full_data.loc[((full_data["Close_RSI_20_/ZN=F"] < 80) & (full_data["day_num"] >= 6)) | ((full_data["Close_RSI_20_/ZN=F"] < 20) & (full_data["day_num"] <= 5)),"B/S"] = 1
full_data.loc[((full_data["Close_RSI_20_/ZN=F"] > 20) & (full_data["day_num"] <= 5)) | ((full_data["Close_RSI_20_/ZN=F"] > 80) & (full_data["day_num"] >= 6)),"B/S"] = -1

# Strongest base for trades I could find.
full_data.loc[(full_data["day_num"] >= 6),"B/S"] = 1
full_data.loc[(full_data["day_num"] <= 5),"B/S"] = -1



"""

# This one is working even better:
full_data.loc[((full_data["Close_RSI_20_/ZN=F"] < 80) & (full_data["day_num"] >= 6)) | ((full_data["Close_RSI_20_/ZN=F"] < 20) & (full_data["day_num"] <= 5)),"B/S"] = 1
full_data.loc[((full_data["Close_RSI_20_/ZN=F"] > 20) & (full_data["day_num"] <= 5)) | ((full_data["Close_RSI_20_/ZN=F"] > 80) & (full_data["day_num"] >= 6)),"B/S"] = -1


print("Trade Count:", sum(full_data["B/S"].abs()))
for i in ["TLT", "TMF"]:
    print(f"{i} Total Possible Gains per Share (Abs): $", round(full_data[f"Close2Close_{i}"].abs().sum(skipna=True), 2))
    print(f"{i} Per Share Gains with Strategy: $", round((full_data[f"Close2Close_{i}"]*full_data["B/S"]).sum(skipna=True), 2))
    print(f"{i} Per Share Gains with Buy and Hold: $", round(full_data[f"Close2Close_{i}"].sum(skipna=True), 2))
    print("\n")


# Several Strategies:
# Buy+Hold (TMF, TMV, TLT)
# Strategy TLT, TMF (Long), TMF (Long) TMV (Long), TMF+TMV Long+Short.
for timeframe in ["Open2Close1", "Close2Open", "Close2Close"]:
    for ticker in ["TLT", "TMV", "TMF"]:
        full_data[f"B+H {ticker}"] = starting_capital * ((1 + full_data[f"{timeframe}_pct_{ticker}"]).cumprod())
        full_data[f"Gap and Intraday Flip {ticker}"] = starting_capital * \
            (((1 + full_data["B/S"]*full_data[f"Open2Close1_pct_{ticker}"]) *
             (1 - full_data["B/S"]*full_data[f"Close2Open_pct_{ticker}"])) .cumprod())

    # Now, strategies. First we'll do TLT only: (Long+short)
    full_data[f"{timeframe} Strategy TLT"] = starting_capital * ((1 + (full_data["B/S"] * full_data[f"{timeframe}_pct_TLT"])).cumprod())
    # Now, strategies. First we'll do TMF only: (Long+short)
    full_data[f"{timeframe} Strategy TMF"] = starting_capital * ((1 + (full_data["B/S"] * full_data[f"{timeframe}_pct_TMF"])).cumprod())
    # Now, we'll do TMF Long + TMV short. very, very leveraged. (6x on zero capital outlay)
    full_data[f"{timeframe} Strategy TMF+TMV LS"] = starting_capital * ((1 +
        (full_data["B/S"] * full_data[f"{timeframe}_pct_TMF"]) - (full_data["B/S"] * full_data[f"{timeframe}_pct_TMV"])).cumprod())
    # Now, let's do TMF long and TMV long, switching off as needed when strategy calls for long or short.
    full_data[f"{timeframe} Long_both_returns"] = 0
    full_data[f"{timeframe} Long_both_returns"].loc[full_data["B/S"] == 1] = full_data[f"{timeframe}_pct_TMF"]
    full_data[f"{timeframe} Long_both_returns"].loc[full_data["B/S"] == -1] = full_data[f"{timeframe}_pct_TMV"]
    # So for here, we switch between Long and
    full_data[f"{timeframe} Strategy TMF+TMV LO"] = starting_capital  * ((1 + full_data[f"{timeframe} Long_both_returns"]).cumprod())


# Now let's do some plotting?
# Plot the Buy and sells: (needs to be zoomed in to see trends.)
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TLT Buy/Sell Decision Chart at {timeframe} levels")
plt.title(f"1 is Buy, -1 Sell, 0 is No Position") # Get the dates from X to Y as well.
legend_list = []
plt.plot(full_data.index, full_data[f"B/S"], label="B/S Decisions")
plt.legend()
plt.show()




# Buy + Hold Results:
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TLT Buy And Hold Results for ${starting_capital} at {timeframe} levels")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for trade_frame in ["Close2Close"]:
    for ticker in ["TLT", "TMF", "TMV"]:
        temp = f"{ticker} Profits on {trade_frame}"
        plt.plot(full_data.index, full_data[f"B+H {ticker}"], label=temp)
        legend_list.append(temp)
plt.legend(legend_list)
plt.show()


# Flipping on the recommendation on the open for Gap and Intraday
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TLT, TMF, TMV Strategy Flipping Results for ${starting_capital} at {timeframe} levels")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for trade_frame in [f"Gap and Intraday Flip"]:
    for ticker in ["TLT", "TMF", "TMV"]:
        temp = f"{ticker} Profits on {trade_frame}"
        plt.plot(full_data.index, full_data[f"{trade_frame} {ticker}"], label=temp)
        legend_list.append(temp)
plt.legend(legend_list)
plt.show()


# Now, strategy results for TLT and TMF
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TLT Strategy Results for ${starting_capital} at {timeframe} levels")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for trade_frame in ["Open2Close1", "Close2Open", "Close2Close"]:
    for ticker in ["TLT"]:
        temp = f"{ticker} Profits on {trade_frame}"
        plt.plot(full_data.index, full_data[f"{trade_frame} Strategy {ticker}"], label=temp)
        legend_list.append(temp)
plt.legend(legend_list)
plt.show()


plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TMF Strategy Results for ${starting_capital} at {timeframe} levels")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for trade_frame in ["Open2Close1", "Close2Open", "Close2Close"]:
    for ticker in ["TMF"]:
        temp = f"{ticker} Profits on {trade_frame}"
        plt.plot(full_data.index, full_data[f"{trade_frame} Strategy {ticker}"], label=temp)
        legend_list.append(temp)
plt.legend(legend_list)
plt.show()


# Now for switching between TMF and TMV Longs (TMV long simulates a short position)
# It's similar enough to the previous graph.
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TMF+TMV Long Strategy Results for ${starting_capital} at {timeframe} levels")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for trade_frame in ["Open2Close1", "Close2Open", "Close2Close"]:
    for ticker in ["TMF+TMV LO"]:
        temp = f"{ticker} Profits on {trade_frame}"
        plt.plot(full_data.index, full_data[f"{trade_frame} Strategy {ticker}"], label=temp)
        legend_list.append(temp)
plt.legend(legend_list)
plt.show()


# Now for pure silliness, let's do Long+Short TMF+TMV as needed.
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"TMF+TMV Long/Short Strategy Results for ${starting_capital} at {timeframe} levels")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for trade_frame in ["Open2Close1", "Close2Open", "Close2Close"]:
    for ticker in ["TMF+TMV LS"]:
        temp = f"{ticker} Profits on {trade_frame}"
        plt.plot(full_data.index, full_data[f"{trade_frame} Strategy {ticker}"], label=temp)
        legend_list.append(temp)
plt.legend(legend_list)
plt.show()


