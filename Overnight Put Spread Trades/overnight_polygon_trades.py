import time
import requests
import matplotlib.pyplot as plt
import yfinance
import yfinance as yf
import pandas as pd
import datetime as datetime
import numpy as np
from urllib.request import urlopen
import certifi
import json
import warnings
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

warnings.filterwarnings("ignore")
polygon_api_key = "API_KEY HERE"
fmp_api_key = "FMP KEY HERE"
folder = f"PATH TO SAVE SPOT FOR DATA"

# Get JSON from FMP and organize it into a dataframe.
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

#####
# Get a list of all trading dates in our study
#####
period_start = "2022-08-01"
today = datetime.today()
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
    full.to_csv(path_or_buf=f"{folder}/{ticker}_{timeframe}_overnight_data.csv")
    print(f"{ticker} done")
    return


#####
# Pull 1-minute Data from FMP
#####

# You can probably get similar data from yFinance or EODHD for cheaper or free.
#for i in ["SPY", "^SPX", "^VIX", "^VVIX", "^VIX1D"]:
#    pull_fmp_data(ticker=i, timeframe="1min", day_list=trading_day_list)

# pull yfinance data (daily):
pull_list = ["SPY", "^VIX", "^VVIX", "^VIX1D"]
daily_data = yfinance.download(pull_list, start=period_start, end="2030-01-01")["Close"]
daily_data["Date"] = daily_data.index
daily_data["Date-datestr"] = daily_data["Date"].astype(str)
daily_data["Date-year"] = pd.to_datetime(daily_data["Date"]).dt.year
daily_data["Date-month"] = pd.to_datetime(daily_data["Date"]).dt.month
daily_data["Date-day"] = pd.to_datetime(daily_data["Date"]).dt.day
regime_day = 5
for ticker in pull_list:
    daily_data[f"{ticker} EMA-{regime_day}"] = daily_data[ticker].ewm(span=regime_day).mean()
    daily_data[f"{ticker} Regime"] = np.where(daily_data[f"{ticker} EMA-{regime_day}"] < daily_data[ticker], 1, 0)
print(daily_data.head(10).to_string())
# Pull 1-minute data.
spy_1min = pd.read_csv(f"{folder}/SPY_1min_overnight_data.csv").sort_values(by="date", ascending=True)
tlt_1min = pd.read_csv(f"{folder}/TLT_1min_overnight_data.csv").sort_values(by="date", ascending=True)
vix_1min = pd.read_csv(f"{folder}/^VIX_1min_overnight_data.csv").sort_values(by="date", ascending=True)
vix1d_1min = pd.read_csv(f"{folder}/^VIX1D_1min_overnight_data.csv").sort_values(by="date", ascending=True)
vvix_1min = pd.read_csv(f"{folder}/^VVIX_1min_overnight_data.csv").sort_values(by="date", ascending=True)
regime = 5
for data in [spy_1min, tlt_1min, vix_1min, vix1d_1min, vvix_1min]:
    # Date Data:
    data["Date"] = pd.to_datetime(data["date"]).dt.date
    data["Date-datestr"] = data["date"].astype(str)
    data["Date-year"] = pd.to_datetime(data["date"]).dt.year
    data["Date-month"] = pd.to_datetime(data["date"]).dt.month
    data["Date-day"] = pd.to_datetime(data["date"]).dt.day
    data["Date-hour"] = pd.to_datetime(data["date"]).dt.hour
    data["Date-minute"] = pd.to_datetime(data["date"]).dt.minute
    # Regimes and EMA's
    for price in ["close", "open", "high", "low", "volume"]:
        data[f"{price} EMA-{regime}"] = data[price].ewm(span=regime).mean()
        data[f"{price} Regime"] = np.where(data[f"{price} EMA-{regime}"] < data[price], 1, 0)

#print(vvix_1min.head(10).to_string())
#print(vix_1min.head(10).to_string())
#print(tlt_1min.head(10).to_string())
#print(vix1d_1min.head(10).to_string())
#print(spy_1min.head(10).to_string())

# Smaller regimes seem to work.
# Let's see if closing right at 9:30 (first minute) actually works?
ticker = "SPY"
open_trade_time = "15:55"
open_trade_time_hour = 15
open_trade_time_minute = 55
# Let's try a non-9:30 time at close?
# I'm picking 3pm, because the broker might throw out the data anyway
close_trade_time = "9:30"
close_trade_time_hour = 9
close_trade_time_minute = 30


# 3, 4, 5? How far should it go?
#spread_width = 3 # We're using VIX instead to determine size
# What if we move our strikes up 1?
adjustment = 0

atm_diff = 0 # increase the call strike, decrease the put strike by this amount, if needed



trade_list = []
times = []
calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date="2022-09-01", end_date = (datetime.today()-timedelta(days = 1))).index.strftime("%Y-%m-%d").values

for date in trading_dates[1:-2]:

    try:

        start_time = datetime.now()
        prior_day = trading_dates[np.where(trading_dates==date)[0][0]-1]
        next_day = trading_dates[np.where(trading_dates==date)[0][0]+1]
        exp_date = next_day#date

        # We can filter out weekends and skipped days here. So like, what happens if we do 1DTE, 2, 3+? etc.
        dte = (pd.to_datetime(next_day) - pd.to_datetime(date)).days

        # Get Daily Data:
        daily_data_day = daily_data[daily_data["Date"] == date]
        #print(daily_data_day.to_string())
        SPY_day_Regime = daily_data_day["SPY Regime"][0]
        #TLT_day_Regime = daily_data_day["TLT Regime"][0]
        VIX_day_Regime = daily_data_day["^VIX Regime"][0]
        #VIX1D_day_Regime = daily_data_day["^VIX1D Regime"][0]
        VVIX_day_Regime = daily_data_day["^VVIX Regime"][0]
        VIX_level = daily_data_day["^VIX"][0]

        # Get the Price data for SPY, TLT VIX, VVIX, VIX1D?
        spy_price = spy_1min[(spy_1min["Date"].astype(str) == date)
                            & (spy_1min["Date-hour"] == open_trade_time_hour)
                            & (spy_1min["Date-minute"] == open_trade_time_minute)]

        vix_price = vix_1min[(vix_1min["Date"].astype(str) == date)
                             & (vix_1min["Date-hour"] == open_trade_time_hour)
                             & (vix_1min["Date-minute"] == open_trade_time_minute)]

        vvix_price = vvix_1min[(vvix_1min["Date"].astype(str) == date)
                             & (vvix_1min["Date-hour"] == open_trade_time_hour)
                             & (vvix_1min["Date-minute"] == open_trade_time_minute)]

       # Can get prices and Regimes from here.
        spy_price = spy_price["close"].values[0]
        vvix_price = vvix_price["close"].values[0]
        spy_strike = round(spy_price) + adjustment
        #print(spy_price, round(spy_price))#,vix1d_price,vix_price,vvix_price,tlt_price)
        # Get the expected 1-day move:
        EM_pct = VIX_level/16*0.01
        # Let's say we want to make the Spread adjust to VIX's size.
        # Here we're doing 0.75-EM:
        spread_width = round(spy_price*EM_pct * 0.75)
        print(EM_pct, spread_width)
        # Get SPY prices
        valid_calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type=call&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        valid_calls = valid_calls[valid_calls["ticker"].str.contains(ticker)].copy()
        valid_calls = valid_calls[valid_calls["expiration_date"] == next_day]
        valid_calls["distance_from_price"] = abs(valid_calls["strike_price"] - spy_price)

        valid_puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type=put&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        valid_puts = valid_puts[valid_puts["ticker"].str.contains(ticker)].copy()
        valid_puts = valid_puts[valid_puts["expiration_date"] == next_day]
        valid_puts["distance_from_price"] = abs(spy_price - valid_puts["strike_price"])

        # Get the ticker for the strikes we want.
        atm_call = valid_calls[valid_calls["strike_price"] == spy_strike + atm_diff]
        atm_put = valid_puts[valid_puts["strike_price"] == spy_strike - atm_diff]
        otm_call = valid_calls[valid_calls["strike_price"] == spy_strike + spread_width + atm_diff]
        otm_put = valid_puts[valid_puts["strike_price"] == spy_strike - spread_width - atm_diff]

        # Get ATM and OTM Call and Put prices
        # ATM Call
        atm_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_call['ticker'].iloc[0]}/range/5/minute/{date}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        atm_call_ohlcv.index = pd.to_datetime(atm_call_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")

        # ATM Put
        atm_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_put['ticker'].iloc[0]}/range/5/minute/{date}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        atm_put_ohlcv.index = pd.to_datetime(atm_put_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")

        # OTM Call
        otm_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{otm_call['ticker'].iloc[0]}/range/5/minute/{date}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        otm_call_ohlcv.index = pd.to_datetime(otm_call_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")

        # OTM Put
        otm_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{otm_put['ticker'].iloc[0]}/range/5/minute/{date}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        otm_put_ohlcv.index = pd.to_datetime(otm_put_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")


        # Get the open and close prices.
        # ATM Call
        atm_call_open = atm_call_ohlcv[(atm_call_ohlcv.index.time == pd.Timestamp(open_trade_time).time())]['c'][0]
        atm_call_close = atm_call_ohlcv[(atm_call_ohlcv.index.time == pd.Timestamp(close_trade_time).time())]['c'][1]

        # ATM Put
        atm_put_open = atm_put_ohlcv[(atm_put_ohlcv.index.time == pd.Timestamp(open_trade_time).time())]['c'][0]
        atm_put_close = atm_put_ohlcv[(atm_put_ohlcv.index.time == pd.Timestamp(close_trade_time).time())]['c'][1]

        # OTM Call
        otm_call_open = otm_call_ohlcv[(otm_call_ohlcv.index.time == pd.Timestamp(open_trade_time).time())]['c'][0]
        otm_call_close = otm_call_ohlcv[(otm_call_ohlcv.index.time == pd.Timestamp(close_trade_time).time())]['c'][1]

        # OTM Put
        otm_put_open = otm_put_ohlcv[(otm_put_ohlcv.index.time == pd.Timestamp(open_trade_time).time())]['c'][0]
        otm_put_close = otm_put_ohlcv[(otm_put_ohlcv.index.time == pd.Timestamp(close_trade_time).time())]['c'][1]

        #print(atm_call_open, atm_call_close)
        #print(atm_put_open, atm_put_close)
        #print(otm_call_open, otm_call_close)
        #print(otm_put_open, otm_put_close)

        # Now make gains and losses for each day, for each strategy.
        # Let's do all 3 at once.
        # Straddle Values
        straddle_open = atm_call_open + atm_put_open
        straddle_close = atm_call_close + atm_put_close
        straddle_profit = straddle_close - straddle_open

        # Iron Condor values
        iron_condor_open = otm_call_open + otm_put_open - atm_call_open - atm_put_open
        iron_condor_close = otm_call_close + otm_put_close - atm_call_close - atm_put_close
        iron_condor_profit = iron_condor_close - iron_condor_open

        # Reverse Iron Condor Values
        reverse_ic_open = atm_call_open + atm_put_open - otm_call_open - otm_put_open
        reverse_ic_close = atm_call_close + atm_put_close - otm_call_close - otm_put_close
        reverse_ic_profit = reverse_ic_close - reverse_ic_open

        # Call Credit Spread
        call_credit_open = otm_call_open - atm_call_open
        call_credit_close = otm_call_close - atm_call_close
        call_credit_profit = call_credit_close - call_credit_open

        # Put Credit Spread
        put_credit_open = otm_put_open - atm_put_open
        put_credit_close = otm_put_close - atm_put_close
        put_credit_profit = put_credit_close - put_credit_open

        # Make a 2-step strategy. None worked so far, but it's good to keep code on it here.
        if (VIX_day_Regime == 0):
            chosen_open = put_credit_open
            chosen_profit = put_credit_profit
            fees = 0.03
        else:
            chosen_open = iron_condor_open
            chosen_profit = iron_condor_profit
            fees = 0.03

        trade_data = pd.DataFrame([{"date": date,
                                    "straddle_open": straddle_open,
                                    "iron_condor_open": iron_condor_open,
                                    "reverse_ic_open": reverse_ic_open,
                                    "chosen_open": chosen_open,
                                    "call_credit_open": call_credit_open,
                                    "put_credit_open": put_credit_open,
                                    "straddle_profit": straddle_profit,
                                    "iron_condor_profit": iron_condor_profit,
                                    "reverse_ic_profit": reverse_ic_profit,
                                    "call_credit_profit": call_credit_profit,
                                    "put_credit_profit": put_credit_profit,
                                    "chosen_profit": chosen_profit,
                                    "ticker": ticker,
                                    "exit_time": close_trade_time,
                                    "dte": dte,
                                    "fees": fees,
                                    }])

        # the ">" can be switched out for ">=", and "==" depending on what group you're looking at.
        if (dte > 1):
            trade_list.append(trade_data)
        print(date)

    except Exception as data_error:
        print(data_error)
        continue

all_trades = pd.concat(trade_list).drop_duplicates("date").set_index("date")
all_trades.index = pd.to_datetime(all_trades.index).tz_localize("America/New_York")

all_trades["contracts"] = 1
#all_trades["fees"] = 0.03

all_trades["straddle_pnl"] = (all_trades["straddle_profit"] - all_trades["fees"]) * all_trades["contracts"]
all_trades["iron_condor_pnl"] = (all_trades["iron_condor_profit"] - all_trades["fees"]) * all_trades["contracts"]
all_trades["reverse_ic_pnl"] = (all_trades["reverse_ic_profit"] - all_trades["fees"]) * all_trades["contracts"]
all_trades["call_credit_pnl"] = (all_trades["call_credit_profit"] - all_trades["fees"]) * all_trades["contracts"]
all_trades["put_credit_pnl"] = (all_trades["put_credit_profit"] - all_trades["fees"]) * all_trades["contracts"]
all_trades["chosen_pnl"] = (all_trades["chosen_profit"] - all_trades["fees"]) * all_trades["contracts"]

# For each, let's show changes in Net capital
all_trades["straddle_net_capital"] = 3000 + (all_trades["straddle_pnl"]*100).cumsum()
all_trades["iron_condor_net_capital"] = 3000 + (all_trades["iron_condor_pnl"]*100).cumsum()
all_trades["reverse_ic_net_capital"] = 3000 + (all_trades["reverse_ic_pnl"]*100).cumsum()
all_trades["call_credit_net_capital"] = 3000 + (all_trades["call_credit_pnl"]*100).cumsum()
all_trades["put_credit_net_capital"] = 3000 + (all_trades["put_credit_pnl"]*100).cumsum()
all_trades["chosen_net_capital"] = 3000 + (all_trades["chosen_pnl"]*100).cumsum()

print(all_trades.to_string())

plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Selling 1-DTE Strategies (Incl. Fees)")
plt.title(f"{open_trade_time} Trading Day T - {close_trade_time} Trading Day T+1")
plt.plot(all_trades.index, all_trades["straddle_net_capital"], label="Straddle")
plt.plot(all_trades.index, all_trades["iron_condor_net_capital"], label="Iron Condor")
plt.plot(all_trades.index, all_trades["reverse_ic_net_capital"], label="Reverse Iron Condor")
plt.plot(all_trades.index, all_trades["chosen_net_capital"], label="Chosen Strategy")
plt.plot(all_trades.index, all_trades["call_credit_net_capital"], label="Call Credit Spread")
plt.plot(all_trades.index, all_trades["put_credit_net_capital"], label="Put Credit Spread")
plt.legend(["Straddle PnL", "Iron Condor PnL", "Reverse Iron Condor PnL",
            "Chosen Strategy PnL", "Call Credit PnL", "Put Credit PnL"])
plt.show()

# Let's look at each thing:

for trade_type in ["straddle", "iron_condor", "reverse_ic", "chosen", "call_credit", "put_credit"]:

    wins = all_trades[all_trades[f"{trade_type}_pnl"] > 0]
    losses = all_trades[all_trades[f"{trade_type}_pnl"] < 0]

    avg_win = wins[f"{trade_type}_pnl"].mean()
    avg_loss = losses[f"{trade_type}_pnl"].mean()

    avg_position_size = abs(all_trades[f"{trade_type}_open"].mean())

    win_rate = round(len(wins) / len(all_trades), 2)

    expected_value = round((win_rate * avg_win) + ((1 - win_rate) * avg_loss), 2)
    print(f"{trade_type} Results")
    print(f"EV per trade: ${expected_value * 100}")
    print(f"Win Rate: {round(win_rate * 100,2)}%")
    print(f"Avg Profit: ${round(avg_win * 100, 2)}")
    print(f"Avg Loss: ${round(avg_loss * 100, 2)}")
    print(f"Avg Position Size ${round(avg_position_size * 100, 2)}")
    print(f"Total Profit: ${all_trades[f'{trade_type}_pnl'].sum() * 100}")
    print("\n")
