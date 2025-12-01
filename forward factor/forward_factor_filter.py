"""
We take the forward factor and try and do a quick run through all the data?

Weekly? (if FF works for 7-14 ATM Calls, it should be okay to speed through this.)
(or we can just do 30-60-90 like everyone else?)

    between two expiries from their DTEs and IVs.

    Forward variance identity:
        sigma_fwd = sqrt( (sigma2^2 * T2 - sigma1^2 * T1) / (T2 - T1) )

    with T = DTE / 365 and sigma = IV / 100 (annualized).

    Forward Factor:
        FF = (FrontMonthIV − ForwardIV(1→2)) / ForwardIV(1→2)
           = (σ1 − σ_fwd) / σ_fwd

want FF >= 0.20. That's the magic number.

How do I figure out average daily option volume?
VV says it needs to be a 20-day average of >10,000 contracts traded (volume)?.
But that's not going to help me if all the options are trading in other expirations than what I need.
Probably better to just make sure there's at least 100-contract volume that day in the strike and expirations I need.

"""


import tqdm
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
import time
import ssl
import json
from urllib.request import urlopen
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
import warnings
warnings.filterwarnings("ignore")


today = datetime.today().strftime('%Y-%m-%d')
today = str(today)

###
# Functions for yfinance screener, and a Polygon screener
# Both screeners will reference a "available_weekly_options" csv file with a list of stocks with weekly options.
###

def get_jsonparsed_data(url):
    response = urlopen(url, #cafile=certifi.where()
                       context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)


def run_yf_calculator():

    today = datetime.today().strftime('%Y-%m-%d')
    today = str(today)
    # pull tickers
    ticker_list = pd.read_csv("./available_weekly_options.csv", sep="\t", header=None).get(0)
    ticker = ""
    possible_list = []
    for ticker in tqdm.tqdm(ticker_list, desc=f"Checking Ticker: {ticker}"):
        #print(ticker, end="\r")
        temp = yf.Ticker(ticker)
        # Upcoming Earnings
        current_price = temp.history(period='1d')["Close"][0]
        if (current_price < 5):
            continue
        #print("Current Price: ", round(current_price,2))
        nearest_earnings = temp.get_calendar().get("Earnings Date")
        if len(nearest_earnings) > 0:
            nearest_earnings = str(nearest_earnings[0])
        else:
            nearest_earnings = "2000-01-01"
        #print("Nearest Earnings: ", nearest_earnings)
        # Dates of options available:
        available_expirations = temp.options
        #print("Available Expirations: ",available_expirations)
        exp_list = []
        for _date in available_expirations[:8]:
            temp_data = temp.option_chain(_date).calls
            #print(temp_data.to_string())
            temp_data["exp_date"] = _date
            temp_data["DTE"] = (pd.to_datetime(temp_data["exp_date"]) - pd.to_datetime("today")).dt.days
            temp_data["distance"] = abs(current_price - temp_data["strike"]) / current_price
            temp_data = temp_data[temp_data["distance"] == temp_data["distance"].min()]
            exp_list.append(temp_data)
        exp_data = pd.concat(exp_list, axis=0)
        exp_data = exp_data[exp_data["DTE"].between(4, 35)]
        #print(exp_data.head().to_string())
        for index, row in exp_data.iterrows():
            for index2, row2 in exp_data.iterrows():
                if (row["DTE"]<row2["DTE"]) & (row2["DTE"]-row["DTE"]<=35) & (row["volume"]>100) & (row2["volume"]>100):
                    # What if we just insist that each Call option has volume of at least 100?
                    dte1, dte2 = row["DTE"], row2["DTE"]
                    volume1, volume2 = row["volume"], row2["volume"]
                    iv1, iv2 = round(row["impliedVolatility"],4), round(row2["impliedVolatility"],4)
                    T1, T2 = dte1/365, dte2/365
                    exp1, exp2 = row["exp_date"], row2["exp_date"]
                    bid1, bid2 = row["bid"], row2["bid"]
                    ask1, ask2 = row["ask"], row2["ask"]
                    # σ_fwd = sqrt((σ₂²·T₂ − σ₁²·T₁) / (T₂ − T₁))
                    sigma_fwd = np.sqrt((iv2*iv2*T2 - iv1*iv1*T1)/(T2-T1))
                    FF = round((iv1 - sigma_fwd) / sigma_fwd, 4)
                    if (FF >= 0.2) & ((today > nearest_earnings) | (exp1 < nearest_earnings)):
                        # Print a Good one ASAP, you don't have to wait.
                        print("\n",
                              ticker, ": Current Price: $", round(current_price,2), "FF:", FF,
                              "nearest_earnings:", nearest_earnings,
                              "Exp1:", exp1, "Exp2:", exp2, "DTE1:", dte1, "DTE1:", dte2,"IV1:", iv1, "IV2:", iv2,
                              "bid1-ask1:", bid1, ask1, "bid2-ask2:", bid2, ask2)
                        possible_list.append(
                            pd.DataFrame(
                                {"ticker":ticker,
                                 "stock_price": round(current_price,2), "FF": FF,
                                 "nearest_earnings": nearest_earnings,
                                 "exp1": exp1,"exp2": exp2,
                                 "dte1": dte1,"dte2": dte2,
                                 "iv1": iv1,"iv2": iv2,
                                 "short_option_volume": volume1,
                                 "long_option_volume": volume2,
                                 "bid1":bid1,
                                 "ask1":ask1,
                                 "bid2": bid2,
                                 "ask2": ask2,
                                 }, index = [0]
                            )
                        )
    print("Finished", end="\n")
    possible_data = pd.concat(possible_list,axis=0)
    possible_data = possible_data.sort_values(by='FF', ascending=False)
    print(possible_data.to_string())


    return

# The polygon script is just as slow, but it's using real time data

polygon_api_key = "POLYGON_KEY_HERE" # optional. Can just use yfinance's instead.
#ticker = "AAPL"
#date = "2025-10-24" # date for our option.
today = datetime.today().strftime('%Y-%m-%d')
today = str(today)

# Get last quote
def get_recent_price(ticker="AAPL"):
    price_url = f"https://api.polygon.io/v2/last/nbbo/{ticker}?apiKey={polygon_api_key}"
    #print(price_url)
    price_temp = get_jsonparsed_data(price_url).get("results").get("P")
    return price_temp

def get_option_data(stock_price=100, ticker=None, date=None):
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?expiration_date.gte={date}&contract_type=call&order=asc&limit=250&sort=strike_price&apiKey={polygon_api_key}"
    #print(url)
    temp = get_jsonparsed_data(url).get("results")
    if len(temp) < 3:
        return pd.DataFrame()
    temp_list, option_price_list, iv_list, day_list = [], [], [], []
    for i in range(0, len(temp)):
        #print(temp[i])
        temp_list.append(temp[i]["details"])
        option_price_list.append(temp[i]["last_quote"])
        iv = temp[i].get("implied_volatility")
        day = temp[i].get("day")
        if iv is not None: iv_list.append(iv)
        else: iv_list.append(np.nan)
        if day is not None: day_list.append(day)
        else: day_list.append(np.nan)

    option_price_df = pd.DataFrame(option_price_list)
    iv_df = pd.DataFrame(iv_list, columns=["implied_volatility"])
    day_df = pd.DataFrame(day_list)
    temp = pd.DataFrame(temp_list)
    temp = pd.concat([temp, option_price_df, iv_df, day_df], axis=1)
    temp["stock_price"] = stock_price
    temp["dist_to_stock_price"] = abs(temp["stock_price"] - temp["strike_price"])
    # Get the 3 nearest strikes.
    temp = temp.nsmallest(3, 'dist_to_stock_price')
    temp["DTE"] = (pd.to_datetime(date) - pd.to_datetime(today)).days
    #print(temp.head().to_string())
    return temp


# now let's figure out how to get expiration dates.

def pull_all_options(ticker="AAPL", DTE_limit = 90):
    temp = yf.Ticker(ticker)
    try:
        nearest_earnings = temp.get_calendar().get("Earnings Date")
    except:
        nearest_earnings = "2000-01-01"
    if len(nearest_earnings) > 0:
        nearest_earnings = str(nearest_earnings[0])
    else:
        nearest_earnings = "2000-01-01"
    stock_price = get_recent_price(ticker=ticker)
    if stock_price <= 2.5:
        print("Not getting:", ticker, " because stock price too low")
        return pd.DataFrame()
    print("Getting:", ticker,"Nearest Earnings", nearest_earnings,"Option Strikes", temp.options)
    temp_list = []
    for date in temp.options:
        if (pd.to_datetime(date) - pd.to_datetime(today)).days <= DTE_limit:
            x = get_option_data(stock_price=stock_price, ticker=ticker, date=date)
            # if the volume, or IV row doesn't exist, don't bother:
            if "volume" in x.columns:
                temp_list.append(x)
    temp = pd.concat(temp_list)
    temp["nearest_earnings"] = nearest_earnings
    temp["today"] = today
    temp["ticker"] = ticker
    return temp


def get_ff_combos(option_dataframe = pd.DataFrame()):
    ff_combo_list = []
    for index, row in option_dataframe.iterrows():
        for index2, row2 in option_dataframe.iterrows():
            if (row["expiration_date"] < row["nearest_earnings"]) | (row["nearest_earnings"] < row["today"]):
                if (row["DTE"] < row2["DTE"]) & (row["strike_price"] == row2["strike_price"]):
                    if (row["volume"] > 100) & (row2["volume"] > 100):
                            #print(row["DTE"], row2["DTE"])
                            T1, T2 = row["DTE"]/365, row2["DTE"]/365
                            iv1, iv2 = row["implied_volatility"], row2["implied_volatility"]
                            sigma_fwd = np.sqrt((iv2 * iv2 * T2 - iv1 * iv1 * T1) / (T2 - T1))
                            FF = round((iv1 - sigma_fwd) / sigma_fwd, 4)
                            if (FF > 0.2):
                                temp_row = pd.DataFrame(
                                {"ticker":row["ticker"],
                                 "stock_price": round(row["stock_price"],2), "FF": FF,
                                 "nearest_earnings": row["nearest_earnings"],
                                 "exp1": row["expiration_date"],"exp2": row2["expiration_date"],
                                 "dte1": row["DTE"],"dte2": row2["DTE"],
                                 "iv1": iv1,"iv2": iv2,
                                 "short_option_volume": row["volume"], "long_option_volume": row2["volume"],
                                 "bid1":row["bid"],"ask":row["ask"], "bid2": row2["bid"], "ask2": row2["ask"],
                                 }, index = [0]
                                )
                                print("\n",temp_row.to_string())
                                ff_combo_list.append(temp_row)
    if len(ff_combo_list) > 0:
        ff_combo = pd.concat(ff_combo_list)
    else:
        ff_combo = pd.DataFrame()
    return ff_combo


def large_run():
    ticker_list = pd.read_csv("./available_weekly_options.csv", sep="\t", header=None).get(0)
    full_list = []
    for ticker in ticker_list:
        test = pull_all_options(ticker=ticker, DTE_limit=90)
        if len(test) > 0:
            test2 = get_ff_combos(test)
            if len(test2) > 0:
                full_list.append(test2)
    full_data = pd.concat(full_list)
    full_data.sort_values(by="FF", ascending=False,inplace=True,)
    print(full_data.to_string())
    return full_data


###
# Let's run the code now.
###

start_time = time.perf_counter()
#large_run() # when using Polygon's Option Data
run_yf_calculator() # when using yfinance's (slower) Option Data
# Record the end time
end_time = time.perf_counter()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Wall-clock time: {elapsed_time:.6f} seconds")
