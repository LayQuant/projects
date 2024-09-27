import yfinance as yf
import pandas as pd
import numpy as np
from urllib.request import urlopen
import certifi
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# API's:
polygon_api_key = "API_KEY_HERE"

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

# Get JSON from FMP and organize it into a dataframe.
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

# For rounding.
def myround(x, base=5):
    return base * round(x/base)


#####
# DATA PROCESSING
#####

start= "2022-09-01"
end = "2024-09-25"

underlying_ticker = "VIX"
underlying_suffix = "^vix"
chain_check = "I:VIXW"
vol_index = "^VVIX"
vol_index_suffix = vol_index.lower()
starting_capital = 3000
ticker_list = [underlying_suffix, vol_index_suffix, "^vix3m", "^spx", ]
min_dte = 5 # What is the minimum number of trading days we want our option to last for?
spread_size = 2
strike_size = 1 # What is the spread between the strikes?
scale_spread_size = False

# Build out the yfinance data:
total_data = yf.download("SPY", start=start, end=end, auto_adjust=True)[["Volume"]]
for ticker in ticker_list:
    data = yf.download(ticker.upper(),  start=start, end=end, auto_adjust=True)
    # Get returns:
    # Open t, to Close t
    data["Open2Close"] = data["Close"] - data["Open"]
    data["Open2Close_pct"] = data["Open2Close"] / data["Open"] * 100
    # Close t to Open t+1
    data["Close2Open"] = data["Open"] - data["Close"].shift(1)
    data["Close2Open_pct"] = data["Close2Open"] / data["Close"].shift(1) * 100
    # CLose t to Close t+1
    data["Close2Close"] = data["Close"] - data["Close"].shift(1)
    data["Close2Close_pct"] = data["Close2Close"] / data["Close"].shift(1) * 100

    # Then Close t-1 to Close t
    data["Close2Close_prior"] = data["Close2Close"].shift(1)
    data["Close2Close_prior_pct"] =  data["Close2Close_pct"].shift(1)
    #print(data.head(10).to_string())
    # Try for an RSI too
    for price in ["Open", "High", "Low", "Close", "Volume"]:
        data[f"{price}_prior"] = data[price].shift(1)
        for num in [5, 20]:
            data[f"{price}_RSI_{num}"] = rsi(data, periods=num, ref_point=f"{price}")
            data[f"{price}_EMA_{num}"] = data[f"{price}_prior"].ewm(span=num).mean()
            data[f"{price}_Regime_{num}"] = np.where(data[f"{price}_prior"] > data[f"{price}_EMA_{num}"], 1, 0)
    total_data = pd.merge(total_data, data.add_suffix(f"_{ticker}"), how="left", left_index=True, right_index=True)

# Get Date-based Data
total_data["Date-date"] = total_data.index
total_data["Date-str"] = total_data.index.astype(str)
total_data["year"] = pd.to_datetime(total_data.index).year
total_data["month"] = pd.to_datetime(total_data.index).month
total_data["weekday"] = pd.to_datetime(total_data.index).day_name()
# Get Trading Day of month Count:
total_data['day_num'] = total_data.groupby(["year", "month"])['Date-str'].rank(method='first').astype(int)
total_data['year_num'] = total_data.groupby(["year", "month"])['Date-str'].rank(method='first').astype(int)
total_data["days_to_next_trading_day"] = abs((total_data['Date-date'] - total_data['Date-date'].shift(-1)).dt.days)
total_data["do"] = 1 # Just so our code doesn't bug out if we don't have a strategy used.


#####
# Scaling Spread size by the Volatility Index
#####
total_data[f"{underlying_ticker}_EM_pct"] = np.sqrt(min_dte) / 16 * total_data[f"Close_{vol_index_suffix}"] * 0.01
total_data[f"{underlying_ticker}_EM_val"] = np.sqrt(min_dte) / 16 * total_data[f"Close_{vol_index_suffix}"] * 0.01 * total_data[f"Close_{underlying_suffix}"]
total_data[f"{underlying_ticker}_EM_val_r"] = myround(total_data[f"{underlying_ticker}_EM_val"], strike_size) # for solid strikes.


# check our normal data:
print(total_data.head(10).to_string())
print(total_data.tail(10).to_string())

# Get "equity" returns for each ticker in our list, especially underlying
for ticker in ticker_list:
    for return_type in ["Close2Close", "Open2Close", "Close2Open"]:
        total_data[f"{return_type}_{ticker}_ret_norm"] = (1+(total_data[f"{return_type}_pct_{ticker}"]/100)).cumprod()



# This is the trigger for when we trade.
total_data["do"] = np.where((total_data["Close_^vvix"] / total_data["Close_^vix"] < 6) # Close2Close
                & (total_data["Close_^vix"] / total_data["Close_^vix3m"] >= 0.9)
                & (total_data["Close_^vix"] / total_data["Close_^vix3m"] <= 1.5)
                ,1 , 0)


temp = total_data
# Now get returns for all data.
for ticker in ticker_list:
    for return_type in ["Close2Close", "Open2Close", "Close2Open"]:
        temp[f"{return_type}_{ticker}_ret_strat"] = (1+(temp[f"{return_type}_pct_{ticker}"]/100)).cumprod()


print(f"Instance Count: {len(temp)}")
print(f"Universe Instances: {len(total_data)}")
for ticker in ticker_list:
    for i in ["Close2Open_pct", "Open2Close_pct", "Close2Close_pct"]:
        print(
            f'{ticker.upper()} {i} ',
            f'{ticker.upper()} Avg Return: {round(temp[f"{i}_{ticker}"].mean(),2)}% '
            f'{ticker.upper()} Return Std Dev: {round(temp[f"{i}_{ticker}"].std(), 2)}% '
            f'\n'
            f'{ticker.upper()} Percent Positive: {round(np.where(temp[f"{i}_{ticker}"] > 0, 1, 0).mean()*100, 2)}% '
            f'{ticker.upper()} Cumulative Return since {start}: {round((1+(temp[f"{i}_{ticker}"]/100)).cumprod()[-1]*100 - 100, 2)}% '
        )
    print("\n")






strategy_list = [
    "long_call", "long_put", # I don't have enough $$ for this strategy anyway
    "long_straddle", "short_straddle",# Don't have a large enough portfolio
    #"long_strangle", #"short_strangle",
    "credit_call_spread", "credit_put_spread",
    # "debit_call_spread", # "debit_put_spread",
    "iron_condor", "reverse_iron_condor",
    "combo",
    "long_underlying",
]


trade_data_list = []

# Let's give ourselves 3-5 trading days of breathing room.
for trade_day in total_data.index:
        try:
            row = total_data[total_data["Date-date"] == trade_day] # "current day"
            trigger = row["do"].values[0]
            # ignore days we're not trading.
            row_next = total_data[total_data["Date-date"].shift(1) == trade_day] # next day
            min_rows_ahead = total_data[total_data["Date-date"].shift(min_dte) == trade_day]
            max_rows_ahead = total_data[total_data["Date-date"].shift(min_dte+5) == trade_day]
            trade_day_str = row["Date-str"].values[0]
            next_trade_day_str = row_next["Date-str"].values[0]
            underlying_value = row[f"Close_{underlying_suffix}"].values[0]
            vol_index_value = row[f"Close_{vol_index_suffix}"].values[0]
            underlying_return = row[f"Close2Close_pct_{underlying_suffix}"].values[0]
            vol_index_return = row[f"Close2Close_pct_{vol_index_suffix}"].values[0]
            print(trade_day_str, underlying_value, underlying_return, vol_index_value, vol_index_return, )
            if scale_spread_size == True:
                spread_size = row[f"{underlying_ticker}_EM_val_r"].values[0]
            pg_url = f"https://api.polygon.io/v3/reference/options/contracts?" \
                     f"underlying_ticker={underlying_ticker}&" \
                     f"expiration_date.gt={trade_day_str}&" \
                     f"expiration_date.lte={max_rows_ahead['Date-str'].values[0]}&" \
                     f"strike_price.gte={round(underlying_value, 2) - spread_size - 1 }&" \
                     f"strike_price.lte={round(underlying_value, 2) + spread_size + 1}&" \
                     f"&expired=true&" \
                     f"limit=1000&" \
                     f"apiKey={polygon_api_key}"
            print(pg_url)
            option_contracts = pd.DataFrame.from_dict(get_jsonparsed_data(pg_url)['results'])
            print(f"options exist for {trade_day_str} to {min_rows_ahead['Date-str'].values[0]}")
            #print(option_contracts.tail().to_string())
            # Let's get the earliest contract with DTE >= 3
            option_contracts = option_contracts[option_contracts["expiration_date"] >= min_rows_ahead['Date-str'].values[0]]
            option_contracts["distance_to_strike"] = (underlying_value - option_contracts["strike_price"]).abs()
            option_contracts = option_contracts[option_contracts["expiration_date"] == option_contracts["expiration_date"].min()]
            option_contracts_calls =option_contracts[option_contracts["contract_type"] == "call"]
            option_contracts_puts = option_contracts[option_contracts["contract_type"] == "put"]
            # Puts:
            ATM_row_put = option_contracts_puts[option_contracts_puts["distance_to_strike"] == option_contracts_puts["distance_to_strike"].min()]
            #print(ATM_row.to_string())
            OTM_row_put = option_contracts_puts[option_contracts_puts["strike_price"] == ATM_row_put["strike_price"].values[0] - spread_size]
            OTM_up_row_put = option_contracts_puts[option_contracts_puts["strike_price"] == ATM_row_put["strike_price"].values[0] + spread_size]
            # Calls:
            ATM_row_call = option_contracts_calls[option_contracts_calls["distance_to_strike"] == option_contracts_calls["distance_to_strike"].min()]
            # print(ATM_row.to_string())
            OTM_row_call = option_contracts_calls[option_contracts_calls["strike_price"] == ATM_row_call["strike_price"].values[0] - spread_size]
            OTM_up_row_call = option_contracts_calls[option_contracts_calls["strike_price"] == ATM_row_call["strike_price"].values[0] + spread_size]

            exp_date = ATM_row_put["expiration_date"].values[0]
            atm_put_contract = ATM_row_put["ticker"].values[0]
            otm_put_contract = OTM_row_put["ticker"].values[0]
            otm_up_put_contract = OTM_up_row_put["ticker"].values[0]

            atm_call_contract = ATM_row_call["ticker"].values[0]
            otm_call_contract = OTM_row_call["ticker"].values[0]
            otm_up_call_contract = OTM_up_row_call["ticker"].values[0]
            # might also need strikes too.
            print(exp_date, atm_put_contract, atm_call_contract, otm_put_contract, otm_call_contract)

            # Make conditionals so that if there's partial data, we don't throw out everything.

            # Puts
            # Get ATM contract data
            atm_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_put_contract}/range/1/day/{trade_day_str}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            atm_put_ohlcv.index = pd.to_datetime(atm_put_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")
            atm_put_ohlcv["date-str"] = pd.to_datetime(atm_put_ohlcv.index).date
            print("atm put pulled")
            # Get OTM Contract data
            otm_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{otm_put_contract}/range/1/day/{trade_day_str}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            otm_put_ohlcv.index = pd.to_datetime(otm_put_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")
            otm_put_ohlcv["date-str"] = pd.to_datetime(otm_put_ohlcv.index).date
            print("otm put pulled")
            # Call
            # Get ATM contract data
            atm_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_call_contract}/range/1/day/{trade_day_str}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            atm_call_ohlcv.index = pd.to_datetime(atm_call_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")
            atm_call_ohlcv["date-str"] = pd.to_datetime(atm_call_ohlcv.index).date
            print("atm call pulled")
            # Get OTM Contract data
            otm_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{otm_up_call_contract}/range/1/day/{trade_day_str}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            otm_call_ohlcv.index = pd.to_datetime(otm_call_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")
            otm_call_ohlcv["date-str"] = pd.to_datetime(otm_call_ohlcv.index).date
            print("otm call pulled")

            print(f"https://api.polygon.io/v2/aggs/ticker/{atm_put_contract}/range/1/day/{trade_day_str}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}")
            print(f"https://api.polygon.io/v2/aggs/ticker/{otm_put_contract}/range/1/day/{trade_day_str}/{exp_date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}")
            #print(atm_call_ohlcv.to_string())
            #print(otm_call_ohlcv.to_string())

            # Just make profits and losses for each
            # Long Call:
            atm_long_call_credit = atm_call_ohlcv['c'][0]
            otm_long_call_credit = otm_call_ohlcv['c'][0]
            atm_long_call_profit = -atm_long_call_credit + atm_call_ohlcv['o'][-1]
            otm_long_call_profit = -otm_long_call_credit + otm_call_ohlcv['o'][-1]

            # Long Put:
            atm_long_put_credit = atm_put_ohlcv['c'][0]
            otm_long_put_credit = otm_put_ohlcv['c'][0]
            atm_long_put_profit = -atm_long_put_credit + atm_put_ohlcv['o'][-1]
            otm_long_put_profit = -otm_long_put_credit + otm_put_ohlcv['o'][-1]

            # Short Call:
            atm_short_call_credit = atm_call_ohlcv['c'][0]
            otm_short_call_credit = otm_call_ohlcv['c'][0]
            atm_short_call_profit = atm_short_call_credit - atm_call_ohlcv['o'][-1]
            otm_short_call_profit = otm_short_call_credit - otm_call_ohlcv['o'][-1]

            # Short Put:
            atm_short_put_credit = atm_put_ohlcv['c'][0]
            otm_short_put_credit = otm_put_ohlcv['c'][0]
            atm_short_put_profit = atm_short_put_credit - atm_put_ohlcv['o'][-1]
            otm_short_put_profit = otm_short_put_credit - otm_put_ohlcv['o'][-1]

            # Trade Types:
            # Long Call
            long_call_risk = abs(atm_long_call_credit)
            long_call_profit = atm_long_call_profit
            long_call_fee = 0.02

            # Long Put
            long_put_risk = abs(atm_long_put_credit)
            long_put_profit = atm_long_put_profit
            long_put_fee = 0.02

            # Short Straddle (sell ATM Call and Put)
            short_straddle_risk = abs(atm_short_call_credit) + abs(atm_short_put_credit)
            short_straddle_profit = atm_short_call_profit + atm_short_put_profit
            short_straddle_fee = 0.03

            # Short Strangle (sell OTM Call and Put)
            short_strangle_risk = abs(otm_short_call_credit) + abs(otm_short_put_credit)
            short_strangle_profit = otm_short_call_profit + otm_short_put_profit
            short_strangle_fee = 0.03

            # Long Straddle
            long_straddle_risk = long_call_risk + long_put_risk
            long_straddle_profit = long_call_profit + long_put_profit
            long_straddle_fee = 0.03

            # Long Strangle (buy OTM call and Put)
            long_strangle_risk = abs(otm_long_call_credit) + abs(otm_long_put_credit)
            long_strangle_profit = otm_long_call_profit + otm_long_put_profit
            long_strangle_fee = 0.03

            # Credit Call Spread
            credit_call_spread_risk = spread_size - atm_short_call_credit + otm_long_call_credit
            credit_call_spread_profit = atm_short_call_profit + otm_long_call_profit
            credit_call_spread_fee = 0.03

            # Debit Call Spread
            debit_call_spread_risk = spread_size - atm_long_call_credit + otm_short_call_credit
            debit_call_spread_profit = atm_long_call_profit + otm_short_call_profit
            debit_call_spread_fee = 0.03

            # Credit Put Spread
            credit_put_spread_risk = spread_size - atm_short_put_credit + otm_long_put_credit
            credit_put_spread_profit = atm_short_put_profit + otm_long_put_profit
            credit_put_spread_fee = 0.03

            # Debit Put Spread
            debit_put_spread_risk = spread_size - atm_long_put_credit + otm_short_put_credit
            debit_put_spread_profit = atm_long_put_profit + otm_short_put_profit
            debit_put_spread_fee = 0.03

            # Iron Condor
            iron_condor_risk = spread_size - atm_short_call_credit - atm_short_call_credit + otm_long_call_credit + otm_long_put_credit
            iron_condor_profit = atm_short_call_profit + atm_short_put_profit + otm_long_call_profit + otm_long_put_profit
            iron_condor_fee = 0.04

            # Reverse Iron Condor
            reverse_iron_condor_risk = spread_size + atm_short_call_credit + atm_short_call_credit - otm_long_call_credit - otm_long_put_credit
            reverse_iron_condor_profit = atm_long_call_profit + atm_long_put_profit + otm_short_call_profit + otm_short_put_profit
            reverse_iron_condor_fee = 0.04

            # Then here, depending on what we code, we'll get strategy risk, profit, and fees.
            # Let's make a Strategy based on Credit Put and Call Spreads? And flip them.
            if trigger == 0:
                combo_risk = credit_call_spread_risk
                combo_profit = credit_call_spread_profit
                combo_fees = credit_call_spread_fee
            else:
                combo_risk = credit_put_spread_risk
                combo_profit = credit_put_spread_profit
                combo_fees = credit_put_spread_fee

            trade_data = pd.DataFrame([{"date": trade_day, "ticker": underlying_ticker,
                "long_call_profit": long_call_profit, "long_call_risk": long_call_risk, "long_call_fee": long_call_fee,
                "long_put_profit": long_put_profit, "long_put_risk": long_put_risk, "long_put_fee": long_put_fee,
                "long_straddle_profit": long_straddle_profit, "long_straddle_risk": long_straddle_risk, "long_straddle_fee": long_straddle_fee,
                "long_strangle_profit": long_strangle_profit, "long_strangle_risk": long_strangle_risk, "long_strangle_fee": long_strangle_fee,
                "short_straddle_profit": short_straddle_profit, "short_straddle_risk": short_straddle_risk, "short_straddle_fee": short_straddle_fee,
                "short_strangle_profit": short_strangle_profit, "short_strangle_risk": short_strangle_risk, "short_strangle_fee": short_strangle_fee,
                "credit_call_spread_profit": credit_call_spread_profit, "credit_call_spread_risk": credit_call_spread_risk, "credit_call_spread_fee": credit_call_spread_fee,
                "debit_call_spread_profit": debit_call_spread_profit, "debit_call_spread_risk": debit_call_spread_risk, "debit_call_spread_fee": debit_call_spread_fee,
                "credit_put_spread_profit": credit_put_spread_profit, "credit_put_spread_risk": credit_put_spread_risk, "credit_put_spread_fee": credit_put_spread_fee,
                "debit_put_spread_profit": debit_put_spread_profit, "debit_put_spread_risk": debit_put_spread_risk, "debit_put_spread_fee": debit_put_spread_fee,
                "iron_condor_profit": iron_condor_profit, "iron_condor_risk": iron_condor_risk, "iron_condor_fee": iron_condor_fee,
                "reverse_iron_condor_profit": reverse_iron_condor_profit, "reverse_iron_condor_risk": reverse_iron_condor_risk, "reverse_iron_condor_fee": reverse_iron_condor_fee,
                "combo_profit": combo_profit, "combo_risk": combo_risk, "combo_fee": combo_fees,
                "long_underlying_profit": underlying_return, "long_underlying_risk": starting_capital, "long_underlying_fee": 0.01,
                "do": trigger,
                }])

            trade_data_list.append(trade_data)

        except Exception as error:
            print(error)
            continue
    # Get the Contract name for

# Make a "strategy" thingy and it'll be separate.
all_trades = pd.concat(trade_data_list).drop_duplicates("date").set_index("date")
strat = all_trades[all_trades["do"] == 1] # this is where we do the "trigger" trades


for strategy in strategy_list:
    all_trades[f"{strategy}_pct_profit"] = (all_trades[f"{strategy}_profit"] - all_trades[f"{strategy}_fee"]) / abs(all_trades[f"{strategy}_risk"])
    all_trades[f"{strategy}_pnl"] = starting_capital + ((all_trades[f"{strategy}_profit"] - all_trades[f"{strategy}_fee"])* 100).cumsum()
    strat[f"{strategy}_pct_profit"] = (strat[f"{strategy}_profit"] - strat[f"{strategy}_fee"]) / abs(strat[f"{strategy}_risk"])
    strat[f"{strategy}_pnl"] = starting_capital + ((strat[f"{strategy}_profit"] - strat[f"{strategy}_fee"])* 100).cumsum()


# What happens if we just buy shares in the underlying
all_trades["long_underlying_pct_profit"] = all_trades["long_underlying_profit"]
all_trades["long_underlying_pnl"] = starting_capital * (1 + all_trades["long_underlying_pct_profit"]/100).cumprod()
strat["long_underlying_pct_profit"] = strat["long_underlying_profit"]
strat["long_underlying_pnl"] = starting_capital * (1 + strat["long_underlying_pct_profit"]/100).cumprod()


#print(all_trades.to_string())

# Let's get some underlying Figures:
for trade_type in strategy_list:
    i = 0
    for data in [all_trades, strat]:
        if i == 0: print(f"Do {trade_type} every day Results:")
        else: print(f"Do {trade_type} strategy Results:")
        wins = data[data[f"{trade_type}_profit"] >= 0]
        losses = data[data[f"{trade_type}_profit"] < 0]

        avg_win = wins[f"{trade_type}_profit"].mean()
        avg_loss = losses[f"{trade_type}_profit"].mean()

        avg_position_size = data[f"{trade_type}_risk"].abs()
        sharpe = data[f"{trade_type}_profit"].mean() / data[f"{trade_type}_profit"].std()

        win_rate = round(len(wins) / (len(wins) + len(losses)), 2)

        expected_value = round((win_rate * avg_win) + ((1 - win_rate) * avg_loss), 2)
        print(f"EV per trade: ${expected_value * 100}", f"Win Rate: {round(win_rate * 100,2)}%")
        print(f"Avg Profit: ${round(avg_win * 100, 2)}", f"Avg Loss: ${round(avg_loss * 100, 2)}")
        print(f"Trade Count: {len(data)}")
        print(f"Trades: {len(wins) + len(losses)}", f"Wins: {len(wins)}", f"Losses: {len(losses)}")
        print(f"'Sharpe': {round(sharpe, 2)}")
        print(f"Spread Size: {spread_size}", f"Avg Position Size ${round(avg_position_size.mean() * 100, 2)}")
        print(f"Total Profit: ${round(data[f'{trade_type}_profit'].sum() * 100,2)}")
        print("\n")
        i+=1


plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Returns for {underlying_ticker} Option Strategies")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for strategy in strategy_list:
    temp2 = f"{underlying_ticker} {strategy} Strategy"
    plt.plot(all_trades.index, all_trades[f"{strategy}_pnl"], label=temp2)
    legend_list.append(temp2)
plt.legend(legend_list)
plt.show()


plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Strategy Returns for {underlying_ticker} Option Strategies")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for strategy in strategy_list:
    temp2 = f"{underlying_ticker} {strategy} Strategy on Strat day"
    plt.plot(strat.index, strat[f"{strategy}_pnl"], label=temp2)
    legend_list.append(temp2)
plt.legend(legend_list)
plt.show()

