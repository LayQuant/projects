import time
import requests
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime as datetime
import numpy as np
from urllib.request import urlopen
import certifi
import json
import warnings

warnings.filterwarnings("ignore")
polygon_api_key = "<<API KEY>>"
fmp_api_key = "<<API KEY>>"
folder = "<<Folder for Data>>"

# Get JSON from FMP and organize it into a dataframe.
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


#####
# Get a list of all trading dates in our study
#####
period_start = "2022-01-01"
today = datetime.date.today()
timeframe = "1min"
ticker = "SPY"
trading_days = list(yf.download(ticker, period_start, today).index.astype(str))
n = 2
trading_day_list = [trading_days[i:i+n] for i in range(0, len(trading_days), n)]

#####
# Pull data from FMP and yFinance
#####

# This pulls and saves our data.
def pull_fmp_data(ticker=ticker,timeframe=timeframe, day_list = trading_day_list):
    full = pd.DataFrame(columns=["date", "open", "low", "high", "close", "volume"])
    base_url = f"https://financialmodelingprep.com/api/v3/"
    endpoint = f"historical-chart"
    for date_list in day_list:
        time.sleep(0.1)
        start_date = date_list[0]
        end_date = date_list[-1]
        combined_url = base_url + endpoint + f"/{timeframe}/" + ticker + f"?from={start_date}" f"&to={end_date}" + f"&apikey={fmp_api_key}"
        X = get_jsonparsed_data(combined_url)
        X = pd.DataFrame.from_dict(X)
        full = pd.concat([full, X])
        print(f"{ticker} {date_list}")
    full.to_csv(path_or_buf=f"{folder}/{ticker}_{timeframe}_intraday_data.csv")
    print(f"{ticker} done")
    return


#####
# Pull 1-minute Data from FMP
#####

#for i in ["SPY", "^SPX", "^VIX", "^VVIX"]:
#    pull_fmp_data(ticker=i, timeframe="1min", day_list=trading_day_list)

spy_1min = pd.read_csv(f"{folder}/SPY_1min_intraday_data.csv").sort_values(by="date", ascending=True)
vix_1min = pd.read_csv(f"{folder}/^VIX_1min_intraday_data.csv").sort_values(by="date", ascending=True)
vvix_1min = pd.read_csv(f"{folder}/^VVIX_1min_intraday_data.csv").sort_values(by="date", ascending=True)
spy_1min["ticker"] = "SPY"
vix_1min["ticker"] = "VIX"
vvix_1min["ticker"] = "VVIX"


# How long are the Exponential Moving Averages we'll use? (between 3 and 30?)
regime = 30
for data in [spy_1min, vix_1min, vvix_1min]:
    data["date-date"] = pd.to_datetime(data["date"]).dt.date
    data["date-datestr"] = data["date-date"].astype(str)
    data["date-year"] = pd.to_datetime(data["date"]).dt.year
    data["date-month"] = pd.to_datetime(data["date"]).dt.month
    data["date-day"] = pd.to_datetime(data["date"]).dt.day
    data["date-hour"] = pd.to_datetime(data["date"]).dt.hour
    data["date-minute"] = pd.to_datetime(data["date"]).dt.minute
    data["date-weekday"] = pd.to_datetime(data["date"]).dt.day_name()
    data["open2close"] = (data["close"] - data["open"]) / data["open"] * 100
    data["high2low"] = (data["high"] - data["low"]) / data["open"] * 100
    data["open2close_ud"] = np.where(data["open2close"] > 0, 1, 0)
    data["gap"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1) * 100
    data["gap_ud"] = np.where(data["gap"] > 0, 1, 0)
    for price in ["open", "close", "high", "low", "volume"]:
        data[f"{price} EMA-{regime}"] = data[price].ewm(span=regime).mean()
        data[f"{price} Std Dev"] = data[price].std() / data[price]
        data[f"{price} Regime"] = np.where(data[f"{price} EMA-{regime}"] < data[price], 1, 0)

# Combine SPY, VIX, and VVIX data
spy_1min = pd.merge(spy_1min, vix_1min.add_suffix("_vix"), how="left", left_on="date", right_on="date_vix")
spy_1min = pd.merge(spy_1min, vvix_1min.add_suffix("_vvix"), how="left", left_on="date", right_on="date_vvix")


# Get SPY, VIX, and VVIX data
spy_day = yf.download("SPY", start="2014-08-01", end=today)
vix_day = yf.download("^VIX", start="2014-08-01", end=today)
vvix_day = yf.download("^VVIX", start="2014-08-01", end=today)
for data in [spy_day, vix_day, vvix_day]:
    data["Date"] = data.index
    data["C2C Return"] = (data["Adj Close"] - data["Adj Close"].shift(1)) / data["Adj Close"].shift(1) * 100

#####
# Charts
#####

# Plot VIX vs VVIX
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.title(f"Daily Close prices of VIX and VVIX Indices, 10 Years")
plt.plot(vix_day["Date"], vix_day["Close"])
plt.plot(vvix_day["Date"], vvix_day["Close"])
plt.tick_params(axis='x', which='both', bottom=False)
plt.legend(["VIX Close", "VVIX Close"])
plt.show()

# Plot VIX versus SPY daily returns.
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.title(f"Daily Close prices of VIX and SPY 1-Day Returns, 10 Years")
plt.plot(vix_day["Date"], vix_day["Close"])
plt.plot(spy_day["Date"], spy_day["C2C Return"])
plt.tick_params(axis='x', which='both', bottom=False)
plt.legend(["VIX Close", "SPY 1-Day Returns"])
plt.show()


# Chart the Averages of the 1-minute Open to Close, High to Low difference by %.
vol_chart = spy_1min.groupby(["date-hour", "date-minute"], as_index=False)[["open2close", "high2low", "volume"]].agg(np.mean)
vol_chart["Time"] = vol_chart["date-hour"].astype(str) + ":" + vol_chart["date-minute"].astype(str)
vol_chart["log_volume"] = np.log(vol_chart["volume"])
plt.figure(dpi=100)
plt.title(f"Averages of 1min {ticker} O2C and H2L Distances in %")
plt.plot(vol_chart["Time"], vol_chart["open2close"])
plt.plot(vol_chart["Time"], vol_chart["high2low"])
plt.tick_params(axis='x', which='both', bottom=False)
plt.legend(["Open-to-Close", "High-to-Low"])
plt.show()

# Chart the Averages of the 1-minute Volumes, converted to Log
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.title(f"Averages of 1min {ticker} Log-Volume")
plt.plot(vol_chart["Time"], vol_chart["log_volume"])
plt.tick_params(axis='x', which='both', bottom=False)
plt.legend(["Volume"])
plt.show()



data = spy_1min
ticker = "SPY"
options_ticker = "SPY"
# how much are the strikes moved up or down?
offset = 0
wing_size = 3
trade_list0 = []
times = []

trade_time0 = "9:59"
close_time = "15:00"
from pandas_market_calendars import get_calendar
calendar = get_calendar("NYSE")
trading_dates = pd.to_datetime(
    data[
        (data["date-hour"] == 9) & (data["date-minute"] == 59) & (data["date-datestr"] >= "2022-08-01")
            ]["date-date"].unique()).strftime("%Y-%m-%d").values


for date in trading_dates:
    try:
        start_time = datetime.datetime.now()
        trade_date = str(date)[:10]

        curr_data = data[(data["date-datestr"] == trade_date) & (data["date-hour"] == 9) & (data["date-minute"] == 59)]
        price0 = curr_data["close"].values[0]
        priceend = data[(data["date-datestr"] == trade_date) & (data["date-hour"] == 15) & (data["date-minute"] == 0)]["close"].values[0]

        #Get the data for the curent moment:
        vvix_regime = curr_data["close Regime_vvix"].values[0]
        vix_regime = curr_data["close Regime_vix"].values[0]
        reg_regime = curr_data["close Regime"].values[0]
        print(price0, priceend, vix_regime, vvix_regime, reg_regime, trade_date)
        calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=call&as_of={trade_date}&expiration_date={trade_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=put&as_of={trade_date}&expiration_date={trade_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        calls["distance_from_price0"] = abs(calls["strike_price"] - price0)
        puts["distance_from_price0"] = abs(puts["strike_price"] - price0)

        # Let's just get the closest strike to the price that's whole number.
        atm_strike0 = int(calls.nsmallest(1, "distance_from_price0")["strike_price"].iloc[0])
        trade_type = "None"
        fees = 0
        # This is for the long Calls: (Done)
        if (vvix_regime == 0) & (vix_regime == 0) & (reg_regime == 1):
            trade_type = "Long Call"
            fees = 0.02
            long_call0 = calls[calls["strike_price"] == atm_strike0 + offset]
            short_call0 = calls[calls["strike_price"] == atm_strike0 + wing_size +offset]
            # unneeded parts
            long_put0 = puts[puts["strike_price"] == atm_strike0 + offset]
            short_put0 = puts[puts["strike_price"] == atm_strike0 - wing_size + offset]

        # This is for the Call Spread: (Done)
        elif (vvix_regime == 1) & (vix_regime == 1) & (reg_regime == 1):
            trade_type = "Call Debit Spread"
            fees = 0.03
            long_call0 = calls[calls["strike_price"] == atm_strike0 + offset]
            short_call0 = calls[calls["strike_price"] == atm_strike0 + wing_size + offset]
            # unneeded parts
            long_put0 = puts[puts["strike_price"] == atm_strike0 + offset]
            short_put0 = puts[puts["strike_price"] == atm_strike0 - wing_size + offset]

        # This is for the Short Put Credit Spread: (Done)
        elif (vvix_regime == 0) & (vix_regime == 0) & (reg_regime == 0):
            trade_type = "Short Put Credit Spread"
            fees = 0.03
            short_call0 = calls[calls["strike_price"] == atm_strike0 + offset]
            long_call0 = calls[calls["strike_price"] == atm_strike0 + wing_size + offset]
            # unneeded parts
            short_put0 = puts[puts["strike_price"] == atm_strike0 + offset]
            long_put0 = puts[puts["strike_price"] == atm_strike0 - wing_size + offset]

        elif  (vvix_regime == 0) & (vix_regime == 1) & (reg_regime >=0):
            trade_type = "Iron Butterfly"
            fees = 0.05
            # Try an Iron Butterfly?
            short_put0 = puts[puts["strike_price"] == atm_strike0 + offset]
            long_put0 = puts[puts["strike_price"] == atm_strike0 - wing_size + offset]
            short_call0 = calls[calls["strike_price"] == atm_strike0 + offset]
            long_call0 = calls[calls["strike_price"] ==  atm_strike0 + wing_size + offset]

        else:
            print("No trades")

        # Get OHLCV of the short/long call/put options that we've found
        sc_ohlcv0 = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{short_call0['ticker'].iloc[0]}/range/1/minute/{trade_date}/{trade_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        lc_ohlcv0 = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{long_call0['ticker'].iloc[0]}/range/1/minute/{trade_date}/{trade_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        sp_ohlcv0 = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{short_put0['ticker'].iloc[0]}/range/1/minute/{trade_date}/{trade_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        lp_ohlcv0 = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{long_put0['ticker'].iloc[0]}/range/1/minute/{trade_date}/{trade_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")

        # Set the index
        sc_ohlcv0.index = pd.to_datetime(sc_ohlcv0.index, unit="ms", utc=True).tz_convert("America/New_York")
        lc_ohlcv0.index = pd.to_datetime(lc_ohlcv0.index, unit="ms", utc=True).tz_convert("America/New_York")
        sp_ohlcv0.index = pd.to_datetime(sp_ohlcv0.index, unit="ms", utc=True).tz_convert("America/New_York")
        lp_ohlcv0.index = pd.to_datetime(lp_ohlcv0.index, unit="ms", utc=True).tz_convert("America/New_York")

        # combine the option data and keep the data that's between our open and close period for the trades.
        _data = pd.concat([lc_ohlcv0.add_prefix("lc_"),
                        lp_ohlcv0.add_prefix("lp_"),
                        sc_ohlcv0.add_prefix("sc_"),
                        sp_ohlcv0.add_prefix("sp_"),], axis=1).dropna()
        _data = _data[(_data.index.time >= pd.Timestamp(trade_time0).time())
                      & (_data.index.time <= pd.Timestamp(close_time).time())].copy()

        # This is for Call Debit Spread:
        if (vvix_regime == 1) & (vix_regime == 1) & (reg_regime == 1):
            _data["position_value"] = -_data["sc_c"] + _data["lc_c"]

        # This is for Short Put Credit Spread:
        elif (vvix_regime == 0) & (vix_regime == 0) & (reg_regime == 0):
            _data["position_value"] = -_data["sp_c"] + _data["lp_c"]

        # This is for Long Call:
        elif (vvix_regime == 0) & (vix_regime == 0) & (reg_regime == 1):
            _data["position_value"] = _data["lc_c"]

        # For Iron Butterfly:
        elif (reg_regime >= 0) & (vvix_regime == 0) & (vix_regime == 1):
            _data["position_value"] = _data["lc_c"] + _data["lp_c"] - _data["sc_c"] - _data["sp_c"]

        else:
            print("No Trades today")

        cost0 = _data["spread_value"].iloc[1]

        # Get the starting and finishing Option Values:
        long_call0 = _data["lc_c"].iloc[1]
        long_callend = _data["lc_c"].iloc[-1]
        long_put0 = _data["lp_c"].iloc[1]
        long_putend = _data["lp_c"].iloc[-1]

        short_call0 = _data["sc_c"].iloc[1]
        short_callend = _data["sc_c"].iloc[-1]
        short_put0 = _data["sp_c"].iloc[1]
        short_putend = _data["sp_c"].iloc[-1]

        _data["position_pnl"] = _data["position_value"] - cost0
        _data["position_pnl_percent"] = round(((_data["position_pnl"] / cost0) * 100), 2)
        max_loss0 = cost0
        final_value0 = _data["position_value"].iloc[-1]
        gross_pnl0 = (final_value0 - cost0)
        gross_pnl_percent0 = round((gross_pnl0 / cost0) * 100, 2)

        trade_data = pd.DataFrame([{"date": trade_date, "trade type:": trade_type,
                                    "cost": cost0, "final_price": final_value0, "gross_pnl": gross_pnl0,
                                    "gross_pnl_percent": gross_pnl_percent0,
                                    "max_loss": max_loss0, "ticker": ticker, "exit_time": close_time,
                                    "price_at_start": price0, "price_at_end": priceend,
                                    "long_call0": long_call0, "short_call0": short_call0,
                                    "long_put0": long_put0, "short_put0": short_put0,
                                    "long_callend": long_callend, "short_callend": short_callend,
                                    "long_putend": long_putend, "short_putend": short_putend,
                                    "fees": fees,
                                    }])

        trade_list0.append(trade_data)

    except Exception as data_error:
        print(data_error)
        continue

#############################################

starting_capital = 2000

#####
# Trade from 10AM to 3PM:
####
print(f"10am to 3pm {ticker} trade:")
all_trades = pd.concat(trade_list0).drop_duplicates("date").set_index("date")
all_trades.index = pd.to_datetime(all_trades.index).tz_localize("America/New_York")

all_trades["contracts"] = 2
all_trades["max_loss"] = (all_trades["max_loss"]) * all_trades["contracts"]
all_trades["fees"] *= all_trades["contracts"]
all_trades["net_pnl"] = (all_trades["gross_pnl"] * all_trades["contracts"]) - all_trades["fees"]
all_trades["stock_return"] = (all_trades["price_at_end"] - all_trades["price_at_start"]) / all_trades["price_at_start"]

all_trades["net_capital"] = starting_capital + (all_trades["net_pnl"] * 100).cumsum()
all_trades["net_capital_stock"] = starting_capital * (1 + all_trades["stock_return"]).cumprod()

# Let's take a look at our trades and the growth of the account.
print(all_trades.to_string())

plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Trading {ticker} Options Intraday")
plt.title(f"10:00am - 3:00pm")
plt.plot(all_trades.index, all_trades["net_capital"])
plt.plot(all_trades.index, all_trades["net_capital_stock"])
plt.legend(["Net Profits", "Long Stock Strategy"])
plt.show()

# Histogram of the net PnL's
plt.hist(all_trades["net_pnl"], bins=20)
plt.title(f"Net PNL Distribution {ticker} at {trade_time0}")
plt.show()

####

wins = all_trades[all_trades["net_pnl"] > 0]
losses = all_trades[all_trades["net_pnl"] < 0]
avg_win = wins["net_pnl"].mean()
avg_loss = losses["net_pnl"].mean()
win_rate = round(len(wins) / len(all_trades), 2)
expected_value = round((win_rate * avg_win) + ((1 - win_rate) * avg_loss), 2)

print(f"Number of trades: {len(all_trades)}")
print(f"Percent of Trading Days Active: {round(len(all_trades)/len(trading_days)*100, 2)}%")
print(f"Expected Value per trade: ${round(expected_value * 100,2)}")
print(f"Win Rate: {win_rate * 100}%")
print(f"Avg Profit: ${round(avg_win * 100, 2)}")
print(f"Avg Loss: ${round(avg_loss * 100, 2)}")
print(f"Total Profit: ${round(all_trades['net_pnl'].sum() * 100,2)}")
print(f"Percent Return: {round(all_trades['net_pnl'].sum() * 100 / starting_capital * 100,2)}%")
print(f"Percent Return from Long Stock {ticker}, 100% allocation:"
      f"{round((all_trades['net_capital_stock'][-1] - all_trades['net_capital_stock'][0]) / all_trades['net_capital_stock'][0] * 100,2)}")
