"""
For this trading strategy, we're assuming that extreme up/down movement on Monday,
specifically Monday afternoon, will be reversed during the rest of the week.

"""

import time
import sys
from datetime import timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas_market_calendars as  mcal
from urllib.request import urlopen
import ssl


context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
import warnings
warnings.filterwarnings("ignore")

yscale = "log"

# FUNCTIONS:
# pulling data
def get_jsonparsed_data(url):
    response = urlopen(url, #cafile=certifi.where()
                       context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

fmp_api_key = "FMP_KEY_HERE"
risk_free_rate = 0.03/52
risk_free_rate252 = 0.03/252
start_date = "2000-01-01" # change to 1920-01-01 when looking for older Index data like for ^SPX
end_date = "2025-11-01"
ticker= "SPY"
# What percent large move threshold can we use for  examining a "larger" drift effect
big_move = 0.005 # Could be 1%, 0.005%, etc.

# what type of measurement do we use for Mondays?
#return_type = "monday_intra_return" #Open to Close
return_type = "fri_mon_day_return" # Friday Close to Monday CLose
#return_type = "fri_mon_overnight_return" # Friday Close to Monday Open
#return_type = "last3hours" # From 1PM Monday to 4PM Monday. (will take longer, requires an FMP Key)

# Get the start of trading each week. Usually monday, sometimes tuesaday
# Get the end of trading each week. Usually friday, sometimes thursday
nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule.index
first_trading_days_of_week = trading_days.to_series().groupby(pd.Grouper(freq="W")).first().dropna()
first_trading_days_of_week = first_trading_days_of_week.to_frame(name="week_start_date")
first_trading_days_of_week["start_weekday"] = pd.to_datetime(first_trading_days_of_week["week_start_date"]).dt.day_name()

last_trading_days_of_week = trading_days.to_series().groupby(pd.Grouper(freq="W")).last().dropna()
last_trading_days_of_week = last_trading_days_of_week.to_frame(name="week_end_date")
last_trading_days_of_week["end_weekday"] = pd.to_datetime(last_trading_days_of_week["week_end_date"]).dt.day_name()



# Let's get 30-minute price data for the following ETFs, as much as possible:
# Start with Daily, and Analyze that.
# US Indices: ^SPX, ^DJI, ^COMP, ^NDX, ^RUA, ^N225, ^HSI, ^STOXX50E,
# Futures: ES=F, NQ=F,
# US Equity ETFs: SPY, VTI, QQQ, VTV, IWM, EEM
# Bond ETFs: TLT, HYG, EMB, MUB,
# Europe: VGK, EZU, FEZ,
# Germany: EWG
# Japan: EWJ
# UK: EWU
# China: FXI
# Canada: EWC
# Mexico: EWW


# Let's take a look at ETF returns
def pull_yf_data(ticker=ticker):
    data = yf.download(ticker, start = start_date, end=end_date)
    data["Date"] = data.index.astype(str)
    data["weekday"] = data.index.day_name()
    data.columns = data.columns.droplevel(1)
    data["1d_return"] = (data["Close"] - data["Close"].shift(1)) / data["Close"].shift(1)
    data["overnight_return"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)
    return data
# Exit before we do stuff.



data = pull_yf_data(ticker=ticker)
reg = data
# cumulative daily return
reg["cum_return"] = (1+reg["1d_return"]).cumprod()

def make_monday_df(data=None, ticker=None, return_type=None):
    # Only do the pulling of last3hours if we're using it:
    if return_type == "last3hours":
        # Let's get 30-minute data for as far back as possible.
        list_30m = []
        for i in range(2003, 2026): # FMP doesn't have 30-minute data before 2003.
            year, year2 = i, i
            for month in range(1, 13):
                time.sleep(0.1)
                _month = month
                if _month <= 11:
                    _month2 = str(month + 1).zfill(2)
                else:
                    _month2 = "01"
                    year2 = i + 1
                print(_month, year, _month2, year2)
                temp_url = (f"https://financialmodelingprep.com/stable/historical-chart/30min?symbol={ticker}"
                            f"&from={year}-{_month}-01&to={year2}-{_month2}-01&apikey={fmp_api_key}")
                print(temp_url)
                temp_df = pd.DataFrame(get_jsonparsed_data(temp_url))
                if len(temp_df) > 0:
                    temp_df["Date"] = pd.to_datetime(temp_df["date"]).dt.date
                    temp_df["hour"] = pd.to_datetime(temp_df["date"]).dt.hour
                    temp_df["minute"] = pd.to_datetime(temp_df["date"]).dt.minute
                    temp_df = temp_df[(temp_df["hour"] >= 13)]
                    if len(temp_df) > 0:
                        temp_df = temp_df.groupby('Date').apply(lambda x: x.iloc[[0, -1]])
                        temp_df["last3hours"] = (temp_df["close"] - temp_df["open"].shift(-1)) / temp_df["open"].shift(-1)
                        temp_df = temp_df[temp_df["minute"] == 30]
                        temp_df = temp_df[["Date", "last3hours"]]
                        temp_df = temp_df.reset_index(drop=True)
                        if len(temp_df) > 0:
                            list_30m.append(temp_df)

        df_30m = pd.concat(list_30m, axis=0)
        df_30m = df_30m.drop_duplicates()
        df_30m["Date"] = df_30m["Date"].astype(str)
        data = data.reset_index(drop=True)
        data = pd.merge(data, df_30m, how="left", left_on="Date", right_on="Date")


    return_groups = []
    for date in first_trading_days_of_week["week_start_date"]:
        try:
            _date = pd.to_datetime(date).date()
            _date_week_end =  pd.to_datetime(date).date() + timedelta(days=4)
            temp_data = data[(data["Date"]>=str(_date)) & (data["Date"]<=str(_date_week_end))]
            if len(temp_data) > 0:
                first_day_open = temp_data["Open"].iloc[0]
                first_day_close = temp_data["Close"].iloc[0]
                last_day_close = temp_data["Close"].iloc[-1]
                if return_type == "last3hours":
                    last3hours = temp_data["last3hours"].iloc[0]
                else:
                    last3hours = None
                temp = pd.DataFrame(
                    {
                        "Date": temp_data["Date"].iloc[0],
                        "last3hours": last3hours,
                        "fri_mon_overnight_return": temp_data["overnight_return"].iloc[0],
                        "fri_mon_day_return": temp_data["1d_return"].iloc[0],
                        "monday_intra_return": (first_day_close - first_day_open) / first_day_open,
                        "rest_of_week_return":(last_day_close-first_day_close)/first_day_close,
                    }, index=[0]
                )
                return_groups.append(temp)
        except:
            print("no data for week of", date, end="\r", flush=True)
    print(f"\n {ticker} Data pull done")
    overall_data = pd.concat(return_groups, axis=0)
    return overall_data


overall_data = make_monday_df(data=data, ticker=ticker, return_type=return_type)


def make_return_dfs(overall_data=None):
    # Regular Tues-Friday returns
    reg_mon = overall_data
    reg_mon["cum_return"]= (1+reg_mon["rest_of_week_return"]).cumprod()

    # Tues-Friday returns with a Positive Monday
    _pos_mon = overall_data[overall_data[return_type]>0]
    _pos_mon["cum_return"] = (1+_pos_mon["rest_of_week_return"]).cumprod()

    # Tues-Friday returns with a Negative Monday
    _neg_mon = overall_data[overall_data[return_type]<0]
    _neg_mon["cum_return"] = (1+_neg_mon["rest_of_week_return"]).cumprod()

    # Let's try with 1% moves?
    _pos_mon1 = overall_data[overall_data[return_type]>big_move]
    _pos_mon1["cum_return"] = (1+_pos_mon1["rest_of_week_return"]).cumprod()
    _neg_mon1 = overall_data[overall_data[return_type]<-big_move]
    _neg_mon1["cum_return"] = (1+_neg_mon1["rest_of_week_return"]).cumprod()
    return reg_mon, _pos_mon, _neg_mon, _pos_mon1, _neg_mon1


reg_mon, _pos_mon, _neg_mon, _pos_mon1, _neg_mon1 = make_return_dfs(overall_data=overall_data)

plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Returns for {ticker} Rest-of-Week Reversals")
plt.title(f"{ticker} Profit over time period from $1")  # Get the dates from X to Y as well.
legend_list = []
strategy_list = [f"B+H {ticker}", "Regular Monday Return", "Pos Monday Return", "Neg Monday Return", f"{big_move*100}%+ Monday Return", f"-{big_move*100}%+ Monday Return",
                 ]
strategy_list2 = [ "Regular Monday Return", "Pos Monday Return", "Neg Monday Return", f"{big_move*100}%+ Monday Return", f"-{big_move*100}%+ Monday Return",
                 ]
data_list = [reg,
             reg_mon,
              _pos_mon, _neg_mon,
              _pos_mon1, _neg_mon1,
              ]
data_list2 = [reg_mon, _pos_mon, _neg_mon, _pos_mon1, _neg_mon1,]
i = 0
for _data in data_list:
    temp2 = f"{strategy_list[i]}_return"
    plt.plot(pd.to_datetime(_data["Date"]), _data["cum_return"], label=temp2)
    legend_list.append(temp2)
    i+=1
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale("log")
plt.show()


print(data.head().to_string())


# Let's look at daily overall underlying:
print(f"B+H {ticker} Stats: ", "\n",
    f"Days Applicable: ", len(data), "(as %)",round(len(data)/len(reg)*100,2), "\n",
    f"Sharpe Ratio: ", round((data["1d_return"].mean() - risk_free_rate252) / data["1d_return"].std() * np.sqrt(252),2), "\n",
    f"Sortino Ratio: ", round((data["1d_return"].mean() - risk_free_rate252) / data[data["1d_return"] < 0]["1d_return"].std() * np.sqrt(252),2), "\n",
    f"Pct Days Profitable: ", round(np.where(data["1d_return"]>0, 1,0).mean()*100,2), "%", "\n",
    f"Profit Factor: ", round(np.where(data["1d_return"]>0, data["1d_return"], 0).sum()/np.where(data["1d_return"]<0, -data["1d_return"], 0).sum(),2), "\n",
    f"Total Return: ", round((data["cum_return"].iloc[-1] - 1)*100, 2), "%", "\n",
    f"Avg Period Return: ", round(data["1d_return"].mean()*100, 2),"%", "\n",
    f"Std Dev Returns: ", round(data["1d_return"].std()*100, 2), "\n",
    )

# Print out stats for how this works:
# Checking Buy + Hold for the underlying Asset.
i = 0
dec_list = strategy_list2
for data_temp in data_list2:
    print(f"{dec_list[i]} Stats: ", "\n",
    f"Weeks Applicable: ", len(data_temp), "(as %)",round(len(data_temp)/len(reg_mon)*100,2), "\n",
    f"Sharpe Ratio: ", round((data_temp["rest_of_week_return"].mean() - risk_free_rate) / data_temp["rest_of_week_return"].std() * np.sqrt(52),2), "\n",
    f"Sortino Ratio: ", round((data_temp["rest_of_week_return"].mean() - risk_free_rate) / data_temp[data_temp["rest_of_week_return"] < 0]["rest_of_week_return"].std() * np.sqrt(52),2), "\n",
    f"Pct Weeks Profitable: ", round(np.where(data_temp["rest_of_week_return"]>0, 1,0).mean()*100,2), "%", "\n",
    f"Profit Factor: ", round(np.where(data_temp["rest_of_week_return"]>0, data_temp["rest_of_week_return"], 0).sum()/np.where(data_temp["rest_of_week_return"]<0, -data_temp["rest_of_week_return"], 0).sum(),2), "\n",
    f"Total Return: ", round((data_temp["cum_return"].iloc[-1] - 1) * 100, 2), "%", "\n",
    f"Avg Period Return: ", round(data_temp["rest_of_week_return"].mean()*100, 2),"%", "\n",
    f"Std Dev Returns: ", round(data_temp["rest_of_week_return"].std()*100, 2), "\n",
    )
    i+=1

if return_type == "last3hours":
    sys.exit()

# working with multiple ETFs at once?
# Try Ranking, then pick the best one.
etf_list = ["SPY", "EWG", "EWW", "EWC", "EWU", "FXI", "TLT"]
etf_data_list = []
etf_data_side_by_side_list = []
for symbol in etf_list:
    print(symbol)
    data = pull_yf_data(ticker=symbol)
    overall_data = make_monday_df(data=data, ticker=symbol, return_type=return_type)
    etf_data_list.append(overall_data)

    # Set index for Overall Data:
    overall_data = overall_data.reset_index(drop=True)
    etf_data_side_by_side_list.append(overall_data.add_suffix(f"_{symbol}"))

etf_data = pd.concat(etf_data_list, axis=0)
etf_data_side_by_side = pd.concat(etf_data_side_by_side_list, axis=1)


# only have weeks with pos/negative Mondays?
#etf_data = etf_data[etf_data[return_type]<0] # most effective filter.

# Now, for each date, we get the worst/best-performing ETF.
# then, we plot what happens
etf_data["monday_rank"] = etf_data.groupby(["Date"])[return_type].rank(method="first", ascending=True, na_option="keep")
etf_data["monday_rankr"] = etf_data.groupby(["Date"])[return_type].rank(method="first", ascending=False,  na_option="keep")

etf_data_high = etf_data[etf_data["monday_rank"] == 1]
etf_data_low = etf_data[etf_data["monday_rankr"] == 1]
etf_data_high = etf_data_high.sort_values(by='Date')
etf_data_low = etf_data_low.sort_values(by='Date')
etf_data_high["cum_return"] = (1+etf_data_high["rest_of_week_return"]).cumprod()
etf_data_low["cum_return"] = (1+etf_data_low["rest_of_week_return"]).cumprod()

plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Returns for {etf_list} Rest-of-Week Reversals")
plt.title(f"{etf_list} Profit over time period from $1")  # Get the dates from X to Y as well.
legend_list = []
strategy_list = ["SPY B+H", "ETF High Monday Return", "ETF Low Monday Return", ]
data_list = [reg, etf_data_high, etf_data_low]
i = 0
for _data in data_list:
    temp2 = f"{strategy_list[i]}_return"
    plt.plot(pd.to_datetime(_data["Date"]), _data["cum_return"], label=temp2)
    legend_list.append(temp2)
    i+=1
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale("log")
plt.show()


strategy_list2 = ["ETF High Monday Return", "ETF Low Monday Return", ]
data_list2 = [etf_data_high, etf_data_low]
i = 0
for data_temp in data_list2:
    print(f"{strategy_list2[i]} Stats: ", "\n",
    f"Weeks Applicable: ", len(data_temp), "(as %)",round(len(data_temp)/len(reg_mon)*100,2), "\n",
    f"Sharpe Ratio: ", round((data_temp["rest_of_week_return"].mean() - risk_free_rate) / data_temp["rest_of_week_return"].std() * np.sqrt(52),2), "\n",
    f"Sortino Ratio: ", round((data_temp["rest_of_week_return"].mean() - risk_free_rate) / data_temp[data_temp["rest_of_week_return"] < 0]["rest_of_week_return"].std() * np.sqrt(52),2), "\n",
    f"Pct Weeks Profitable: ", round(np.where(data_temp["rest_of_week_return"]>0, 1,0).mean()*100,2), "%", "\n",
    f"Profit Factor: ", round(np.where(data_temp["rest_of_week_return"]>0, data_temp["rest_of_week_return"], 0).sum()/np.where(data_temp["rest_of_week_return"]<0, -data_temp["rest_of_week_return"], 0).sum(),2), "\n",
    f"Total Return: ", round((data_temp["cum_return"].iloc[-1] - 1) * 100, 2), "%", "\n",
    f"Avg Period Return: ", round(data_temp["rest_of_week_return"].mean()*100, 2),"%", "\n",
    f"Std Dev Returns: ", round(data_temp["rest_of_week_return"].std()*100, 2), "\n",
    )
    i+=1


# What happens if we just go down the line:
# If SPY doesn't have a negative Monday, check EWG, and so on down the line.
# for this one, the order of the list matters.
etf_data_side_by_side["chosen_return"] = 0
etf_data_side_by_side["chosen_etf"] = "None"
etf_data_side_by_side["chosen_return_big_move"] = 0
etf_data_side_by_side["chosen_etf_big_move"] = "None"

for ticker in etf_list:
    print("try", ticker)
    # If an ETF isn't already chosen for that week, and the ETF's monday return is negative, select it.
    etf_data_side_by_side["chosen_etf"] = np.where((etf_data_side_by_side["chosen_etf"] == "None") &
                                                   (etf_data_side_by_side[f"{return_type}_{ticker}"] < 0),
                                                ticker,
                                                etf_data_side_by_side["chosen_etf"]
                                                  )
    etf_data_side_by_side["chosen_return"] = np.where(etf_data_side_by_side["chosen_etf"] == ticker,
                                                      etf_data_side_by_side[f"rest_of_week_return_{ticker}"],
                                                      etf_data_side_by_side["chosen_return"])


# Now get cumulative returns:
etf_data_side_by_side["Date"] = etf_data_side_by_side["Date_SPY"]
etf_data_side_by_side["cum_return"] = (1+etf_data_side_by_side["chosen_return"]).cumprod()


traded_etf_counts = etf_data_side_by_side["chosen_etf"].value_counts()
print("Negative Move",traded_etf_counts)


plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Returns for {etf_list} Rest-of-Week Reversals, Layered")
plt.title(f"{etf_list} Profit over time period from $1")  # Get the dates from X to Y as well.
legend_list = []
strategy_list = ["SPY B+H", "ETFs Negative Monday Return",]
data_list = [reg, etf_data_side_by_side]
i = 0
for _data in data_list:
    temp2 = f"{strategy_list[i]}_return"
    plt.plot(pd.to_datetime(_data["Date"]), _data["cum_return"], label=temp2)
    legend_list.append(temp2)
    i+=1
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale("log")
plt.show()


strategy_list2 = ["ETF Layered check",  f"ETFs -{big_move} Monday Return"]
data_list2 = [etf_data_side_by_side,]
i = 0
for data_temp in data_list2:
    print(f"{strategy_list2[i]} Stats: ", "\n",
    f"Weeks Applicable: ", len(data_temp), "(as %)",round(len(data_temp)/len(reg_mon)*100,2), "\n",
    f"Sharpe Ratio: ", round((data_temp["chosen_return"].mean() - risk_free_rate) / data_temp["chosen_return"].std() * np.sqrt(52),2), "\n",
    f"Sortino Ratio: ", round((data_temp["chosen_return"].mean() - risk_free_rate) / data_temp[data_temp["chosen_return"] < 0]["chosen_return"].std() * np.sqrt(52),2), "\n",
    f"Pct Weeks Profitable: ", round(np.where(data_temp["chosen_return"]>0, 1,0).mean()*100,2), "%", "\n",
    f"Profit Factor: ", round(np.where(data_temp["chosen_return"]>0, data_temp["chosen_return"], 0).sum()/np.where(data_temp["chosen_return"]<0, -data_temp["chosen_return"], 0).sum(),2), "\n",
    f"Total Return: ", round((data_temp["cum_return"].iloc[-1] - 1) * 100, 2), "%", "\n",
    f"Avg Period Return: ", round(data_temp["chosen_return"].mean()*100, 2),"%", "\n",
    f"Std Dev Returns: ", round(data_temp["chosen_return"].std()*100, 2), "\n",
    )
    i+=1

