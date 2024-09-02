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
import py_vollib_vectorized as pvv
from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")


#####
# Get a list of all trading dates in our study
#####
period_start = "2000-01-01"
today = datetime.today()
ticker = "SPY"
trading_days = list(yf.download(ticker, period_start, today).index.astype(str))
n = 5
trading_day_list = [trading_days[i:i+n] for i in range(0, len(trading_days), n)]
fmp_api = "FMP API HERE" # Just need this for our Risk-Free-Rates.
folder = "FOLDER FOR DATA HERE"
# Pretty sure you can also get free FED interest rate data for the 1-year.

# What is the stock/ETF we want to check?
ticker = "SPY"
vol_ticker = "^VIX"
# What percent of the implied move are we pushing out the OTM for?
implied_move_pct = 1
strike_size = 1 # How many $ between strikes?
starting_capital = 3000


# Functions:
# Useful for an asset like SPX which has strikes that are in intervals of $5
def myround(x, base=5):
    return base * round(x/base)

# Get JSON from FMP and organize it into a dataframe.
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


# Make a function to get all treasury data (per year, only need the year1 data)
def get_rfr(start= "2000-01-01", end="2025-01-01"):
    full = pd.DataFrame(columns=["date", "month1", "month2", "month3", "month6", "year1", "year2", "year3",
                                 "year5", "year10", "year20", "year30"])
    for date_list in trading_day_list:
        time.sleep(0.15)
        base_url = f"https://financialmodelingprep.com/api/v4/"
        endpoint = f"treasury?"
        start = date_list[0]
        end = date_list[-1]
        full_url = f"{base_url}{endpoint}from={start}&to={end}&apikey={fmp_api}"
        temp = pd.DataFrame(get_jsonparsed_data(full_url))
        if len(temp) > 0:
            full = pd.concat([full, temp])
        print(date_list)
        full = full.sort_values(by="date")
        full.to_csv(path_or_buf =f"{folder}treasuries.csv")
    return

#get_rfr()
rfr = pd.read_csv(filepath_or_buffer=f"{folder}treasuries.csv")
rfr = rfr.drop_duplicates()
rfr = rfr.sort_values(by="date")
rfr.index =pd.to_datetime(rfr["date"])
rfr = rfr[["year1"]]
print(rfr.head().to_string())
# If you don't feel like using an FMP membership, you can maybe replace this with just 0.03 for "year1",
# since we're using very short timeframes,
# there's probably not a huge change in value +/- a few percent in the annualized interest rate for our options.

# Just pull Open and Close price data from yfinance again.
open_data = yf.download([ticker, vol_ticker], start="2001-01-01", end= "2030-01-01")["Open"]
close_data = yf.download([ticker, vol_ticker], start="2001-01-01", end= "2030-01-01")["Close"]
stock_data = pd.merge(open_data.add_suffix("_o"), close_data.add_suffix("_c"),
                      how="left", left_index=True, right_index=True)
stock_data = pd.merge(stock_data, rfr, how="left", left_index=True, right_index=True)
stock_data["open2close_r"] = (stock_data[f"{ticker}_c"] - stock_data[f"{ticker}_o"]) / stock_data[f"{ticker}_o"] * 100
stock_data["close2open_r"] = (stock_data[f"{ticker}_o"].shift(-1) - stock_data[f"{ticker}_c"]) / stock_data[f"{ticker}_c"] * 100
stock_data["close2close_r"] = (stock_data[f"{ticker}_c"].shift(-1) - stock_data[f"{ticker}_c"]) / stock_data[f"{ticker}_c"] * 100


# need a "forward day gap" Basically how many days until next trading day.
stock_data["days_to_next_trading_day"] = round(stock_data.index.to_series().diff().astype(int)/86400000000000)

# Get the implied 1-day move (volatility) from the VIX
stock_data["implied_move_c"] = stock_data[f"{vol_ticker}_c"] * np.sqrt(stock_data["days_to_next_trading_day"] / 365)
stock_data["implied_move_o"] = stock_data[f"{vol_ticker}_o"] * np.sqrt(stock_data["days_to_next_trading_day"] / 365)
stock_data["ATM_strike_o"] = myround(stock_data[f"{ticker}_o"], base=strike_size)
stock_data["ATM_strike_c"] = myround(stock_data[f"{ticker}_c"], base=strike_size)
stock_data["OTM_strike_o"] = myround(stock_data[f"{ticker}_o"] - stock_data[f"{ticker}_o"]/100*stock_data["implied_move_o"] * implied_move_pct, base=strike_size)
stock_data["OTM_strike_c"] = myround(stock_data[f"{ticker}_c"] - stock_data[f"{ticker}_c"]/100*stock_data["implied_move_c"] * implied_move_pct, base=strike_size)
# Now if we want "up" for the OTM Calls later.
stock_data["ATM_strike_o_call"] = myround(stock_data[f"{ticker}_o"], base=strike_size)
stock_data["ATM_strike_c_call"] = myround(stock_data[f"{ticker}_c"], base=strike_size)
stock_data["OTM_strike_o_call"] = myround(stock_data[f"{ticker}_o"] + stock_data[f"{ticker}_o"]/100*stock_data["implied_move_o"] * implied_move_pct, base=strike_size)
stock_data["OTM_strike_c_call"] = myround(stock_data[f"{ticker}_c"] + stock_data[f"{ticker}_c"]/100*stock_data["implied_move_c"] * implied_move_pct, base=strike_size)


# now let's get the risk-free rate per day. Not working at the moment.
# Need to get from FRED or some other source ASAP.
stock_data["rfr"] = stock_data["year1"] * 0.0100
#stock_data["rfr"] = stock_data["rfr"].fillna(0.00)
stock_data = stock_data.dropna()

# Let's get the Next trading day's Open and Close values
stock_data[f"{ticker}_o_next_day"] = stock_data[f"{ticker}_o"].shift(1)
stock_data[f"{ticker}_c_next_day"] = stock_data[f"{ticker}_c"].shift(1)
stock_data[f"{vol_ticker}_o_next_day"] = stock_data[f"{vol_ticker}_o"].shift(1)
stock_data[f"{vol_ticker}_c_next_day"] = stock_data[f"{vol_ticker}_c"].shift(1)


# Now, let's get put prices at open and close for all items.
flag = 'p'
S = [1] # underlying Asset Prices.
K = [1] # strike prices
t = [1.0] # (Annualized times-to-expiration
r = [0.03]
iv = [0.2] # implied Volatilities

for i in ["o", "c"]:
    for option in ["ATM", "OTM"]:
        stock_data[f"{option}_strike_put_{i}"] = pvv.vectorized_black_scholes(flag=flag, S=stock_data[f"{ticker}_{i}"],
                                                            K=stock_data[f"{option}_strike_{i}"],
                                                            t = stock_data["days_to_next_trading_day"]/365.0,
                                                            r = stock_data["rfr"], sigma=stock_data[f"{vol_ticker}_{i}"]*0.01,
                                                            return_as='array').round(decimals=2)
        if i == "c":
            temp_t = 0.00
        else:
            temp_t = 0.33 # not entirely sure if this is the right number, but at market open, you're about 7-8 hours before the option expires.
        stock_data[f"{option}_strike_put_{i}_next_day"] = pvv.vectorized_black_scholes(flag=flag, S=stock_data[f"{ticker}_{i}_next_day"],
                                                            K=stock_data[f"{option}_strike_{i}"],
                                                            t = temp_t/365.0,
                                                            r = stock_data["rfr"],
                                                            sigma=stock_data[f"{vol_ticker}_{i}_next_day"]*0.01,
                                                            return_as='array').round(decimals=2)

# Now let's get Spread Size (for risk amount), and profits.
# Basically how much is our Risk for total loss.
for i in ["o", "c"]:
    stock_data[f"spread_size_{i}"] = abs(stock_data[f"ATM_strike_{i}"] - stock_data[f"OTM_strike_{i}"])
    stock_data[f"intial_credit_{i}"] = abs(stock_data[f"ATM_strike_put_{i}"] - stock_data[f"OTM_strike_put_{i}"])
    stock_data[f"risk_size_{i}"] = stock_data[f"spread_size_{i}"] \
                                   - stock_data[f"ATM_strike_put_{i}"] + stock_data[f"OTM_strike_put_{i}"]

# Now get profits for our Put Spreads: (the +1 is the next trading day, not next calendar day)
# Market Open t to Market Open t
stock_data["o2c_same_day_profit"] = round(stock_data["ATM_strike_put_o"] - stock_data["ATM_strike_put_c"] - \
                                        stock_data["OTM_strike_put_o"] + stock_data["OTM_strike_put_c"], 2)
# Market Open t to Market Open t+1
stock_data["o2o_next_day_profit"] = round(stock_data["ATM_strike_put_o"] - stock_data["ATM_strike_put_o_next_day"] - \
                                        stock_data["OTM_strike_put_o"] + stock_data["OTM_strike_put_o_next_day"], 2)
# Market Close t to Market Open t+1
stock_data["c2o_next_day_profit"] = round(stock_data["ATM_strike_put_c"] - stock_data["ATM_strike_put_o_next_day"] - \
                                        stock_data["OTM_strike_put_c"] + stock_data["OTM_strike_put_o_next_day"], 2)
# Market Close t to Market Close t+1
stock_data["c2c_next_day_profit"] = round(stock_data["ATM_strike_put_c"] - stock_data["ATM_strike_put_c_next_day"] - \
                                        stock_data["OTM_strike_put_c"] + stock_data["OTM_strike_put_c_next_day"], 2)
stock_data["fees"] = 2 # round trip for an open and close on a put spread.
stock_data["contracts"] = 1 # contract multiple we're using.

# Now, we switch to the strategy we're sampling:
stock_data_use = stock_data

# Let's just worry about 2+ days away
stock_data_use = stock_data_use[stock_data_use["days_to_next_trading_day"] > 1]
# Doing it for just 2 years roughly matches the trade I had.

# Check a difficult market period like 2019-2021:
#stock_data_use = stock_data_use[(stock_data_use.index > "2008-01-01") & (stock_data_use.index < "2010-01-01")]
#stock_data_use = stock_data_use[(stock_data_use.index > "2018-01-01") & (stock_data_use.index < "2021-01-01")]
#stock_data_use = stock_data_use[(stock_data_use.index > "2008-01-01") & (stock_data_use.index < "2010-01-01")]

# The rough period we're using with Polygon's Data:
#stock_data_use = stock_data_use[(stock_data_use.index > "2022-09-01") & (stock_data_use.index < "2025-01-01")]

# Check to see how it handles the real introduction of Monday-expiration SPY options.
#stock_data_use = stock_data_use[stock_data_use.index > "2018-03-01"]

# Check to see if it handles a hypothetical run from 2001.
# stock_data_use = stock_data_use[stock_data_use.index > "2001-01-01"]

# If we limit ourselves to only entering the trade when VIX > 40, we increase our EV and lose a little volatility.
#stock_data_use = stock_data_use[stock_data_use[f"{vol_ticker}_o"] < 30]

# Get cumulative PNL assuming 1 contract used.
for test in ["o2c_same_day", "o2o_next_day", "c2o_next_day", "c2c_next_day"]:
    stock_data_use[f"{test}_pnl"] = starting_capital + (stock_data_use[f"{test}_profit"]*stock_data_use["contracts"]*100).cumsum() \
                                    - stock_data_use["fees"] * stock_data_use["contracts"]



# If we want to scale our Contracts by the size of our capital at each day, here's how we do it.
for test in ["o2c_same_day", "o2o_next_day", "c2o_next_day", "c2c_next_day"]:
    testing = f"{test}_pnl_scaling"
    stock_data_use[testing] = starting_capital
    stock_data_use[f"{test}_contracts"] = 1
    for i in range(1, len(stock_data_use)):
        stock_data_use[f"{test}_contracts"][i] = max(np.floor((stock_data_use[testing][i-1] - starting_capital) / starting_capital) + 1, 1)
        stock_data_use[testing][i] = (stock_data_use[testing][i-1] + (stock_data_use[f"{test}_profit"][i]*stock_data_use[f"{test}_contracts"][i]*100) - stock_data_use["fees"][i] * stock_data_use[f"{test}_contracts"][i])


plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Sell Hypothetical 'Overnight' ATM Put Credit Spreads, OTM is {implied_move_pct*100}% of Expected Move")
plt.title(f"Put Spread PNL's") # Get the dates from X to Y as well.
plt.plot(stock_data_use.index, stock_data_use["o2c_same_day_pnl"], label="Open to Close Same Day")
plt.plot(stock_data_use.index, stock_data_use["o2o_next_day_pnl"], label="Open to Open Next Day")
plt.plot(stock_data_use.index, stock_data_use["c2o_next_day_pnl"], label="Close to Open Next Day")
plt.plot(stock_data_use.index, stock_data_use["c2c_next_day_pnl"], label="Close to Close Next Day")
plt.legend(["O2C Same Trading Day", "O2O Next Trading Day", "C2O Next Trading Day", "C2C Next Trading Day"])
plt.show()

plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Hypothetical 'Overnight' ATM Put Credit Spreads, OTM is {implied_move_pct*100}% of Expected Move")
plt.title(f"Put Spread PNL's")
plt.plot(stock_data_use.index, stock_data_use["o2c_same_day_pnl_scaling"], label="Open to Close Same Day")
plt.plot(stock_data_use.index, stock_data_use["o2o_next_day_pnl_scaling"], label="Open to Open Next Day")
plt.plot(stock_data_use.index, stock_data_use["c2o_next_day_pnl_scaling"], label="Close to Open Next Day")
#plt.plot(stock_data.index, stock_data["c2c_next_day_pnl_scaling"], label="Close to Close Next Day")
plt.legend(["O2C Same Trading Day Scaling", "O2O Next Trading Day Scaling",
            "C2O Next Trading Day Scaling"])
plt.show()

print(stock_data_use[stock_data_use["days_to_next_trading_day"] > 1].to_string())


# Get average returns for the data:
print("Open to Close Average, STD, Percent Positive returns:")
print(stock_data_use["open2close_r"].mean(), stock_data_use["open2close_r"].std(), np.where(stock_data_use["open2close_r"] > 0, 1, 0).mean()*100)
print("Close to Open Average, STD:")
print(stock_data_use["close2open_r"].mean(), stock_data_use["close2open_r"].std(), np.where(stock_data_use["close2open_r"] > 0, 1, 0).mean()*100)
print("Close to Close Average, STD:")
print(stock_data_use["close2close_r"].mean(), stock_data_use["close2close_r"].std(), np.where(stock_data_use["close2close_r"] > 0, 1, 0).mean()*100)


# Now let's look at metrics:
for trade_type in ["o2c_same_day", "o2o_next_day", "c2o_next_day", "c2c_next_day"]:
    wins = stock_data_use[stock_data_use[f"{trade_type}_profit"] > 0]
    losses = stock_data_use[stock_data_use[f"{trade_type}_profit"] < 0]

    avg_win = wins[f"{trade_type}_profit"].mean()
    avg_loss = losses[f"{trade_type}_profit"].mean()

    #avg_position_size = abs(stock_data[f"{trade_type}_open"].mean())
    sharpe = stock_data_use[f"{trade_type}_profit"].mean() / stock_data_use[f"{trade_type}_profit"].std()

    win_rate = round(len(wins) / len(stock_data_use), 2)

    expected_value = round((win_rate * avg_win) + ((1 - win_rate) * avg_loss), 2)
    print(f"{trade_type} Results")
    print(f"EV per trade: ${expected_value * 100}")
    print(f"Win Rate: {round(win_rate * 100,2)}%")
    print(f"Avg Profit: ${round(avg_win * 100, 2)}")
    print(f"Avg Loss: ${round(avg_loss * 100, 2)}")
    print(f"'Sharpe': {round(sharpe, 2)}")
    #print(f"Avg Position Size ${round(avg_position_size * 100, 2)}")
    print(f"Total Profit: ${round(stock_data_use[f'{trade_type}_profit'].sum() * 100,2)}")
    print("\n")



# Let's see if each hypothetical strategy is statistically significant?
for trade_type in ["o2c_same_day", "o2o_next_day", "c2o_next_day", "c2c_next_day"]:
    print(f"{trade_type} Statistical Significance?")
    N = len(stock_data_use)
    T = round(stock_data_use[f"{trade_type}_profit"].mean() * 100, 2)
    # To calculate the t-statistic, we need the Sample and population data.
    # One Sample T-test.
    sample_mean = round(stock_data_use[f"{trade_type}_profit"].mean() * 100, 2)
    population_mean = round(stock_data[f"{trade_type}_profit"].mean() * 100, 2)
    sample_size = N
    sample_std_dev = round(stock_data_use[f"{trade_type}_profit"].std() * 100, 2)
    degrees_freedom = N - 2
    t = round((sample_mean - population_mean) / (sample_std_dev / np.sqrt(N)), 4)
    # If using a 1% alpha, 1-sided t-test:
    alpha = 0.01
    # We need a score to compare at the alpha = 1% level:
    test_stat = 2.326
    CI = round(t * sample_std_dev / np.sqrt(N), 2)
    if t > test_stat:
        print(f"Avg Profit: {T} is statistically significant at the {int(alpha*100)}% level, t-stat equals: {t} against {test_stat} Test Stat.")
    else:
        print("Not Significant")



