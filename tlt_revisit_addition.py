"""
Let's focus on bonds again:

TMF/TMV:
It appears that a VIX1D end-of-day up/down indicator works great for TMF/TMV
VIX1D t > VIX1D t-1: buy TMV
VIX1D t < VIX1D t-1: buy TMF

Let's pull some old code from tlt_revisit.py, at https://github.com/LayQuant/projects/blob/main/tlt_revisit.py
In terms of Sharpe and Profit Factor, just focusing on the mid-months is probably better.


"""


import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


start_date = "2023-01-01"
end_date = "2030-01-01"
# Get Close2Close returns for TLT, TMF, TMV, SPY.
yf_ticker_list = ["TLT",  "^VIX", "TMF", "TMV", "SPY", "^VIX1D"]
investable = ["TLT", "TMF", "TMV", "SPY"]
portfolio = 1
risk_free_rate = 0#.03/252
month_ret_ticker = "SPY"
return_threshold = 0 # Threshold for choosing to invest in the pos/neg strategy.
chart_scale = "log" # "linear"
daily_data = yf.download(yf_ticker_list, start=start_date, end=end_date)["Close"]
daily_data["date"] = pd.to_datetime(daily_data.index)
daily_data["year"] = daily_data["date"].dt.year
daily_data["month"] = daily_data["date"].dt.month
daily_data["day"] = daily_data["date"].dt.day
daily_data["day_name"] = daily_data["date"].dt.day_name()
daily_data["year-month"] = daily_data["date"].astype(str).str[:7]
daily_data['day_num'] = daily_data.groupby(["year", "month"])['date'].rank(method='first').astype(int)
daily_data['day_numr'] = daily_data.groupby(["year", "month"])['date'].rank(method='first', ascending=False).astype(int)
for ticker in yf_ticker_list:
    daily_data[f"{ticker}_ret_1d"] = (daily_data[ticker] - daily_data[ticker].shift(1)) / daily_data[ticker].shift(1)


# make an end of month data.
end_of_month = daily_data[daily_data["day_numr"] == 1]
end_of_month[f"{month_ret_ticker}_1month_ret"] = (end_of_month[month_ret_ticker] - end_of_month[month_ret_ticker].shift(1)) / end_of_month[month_ret_ticker].shift(1)
end_of_month[f"{month_ret_ticker}_prior_month_ret"] = end_of_month[f"{month_ret_ticker}_1month_ret"].shift(1)
end_of_month = end_of_month[[f"{month_ret_ticker}_1month_ret", f"{month_ret_ticker}_prior_month_ret", "year-month"]]

# merge end of month data.
daily_data = pd.merge(daily_data, end_of_month, how="left", left_on="year-month", right_on="year-month")

# the month-to-date value
fifth_last_day_of_month = daily_data[(daily_data["day_numr"] == 6) | (daily_data["day_num"] == 1)]
fifth_last_day_of_month[f"{month_ret_ticker}_mtd_return"] = (fifth_last_day_of_month[month_ret_ticker] - fifth_last_day_of_month[month_ret_ticker].shift(1)) / fifth_last_day_of_month[month_ret_ticker].shift(1)
fifth_last_day_of_month = fifth_last_day_of_month[fifth_last_day_of_month["day_numr"] == 6]
fifth_last_day_of_month = fifth_last_day_of_month[[f"{month_ret_ticker}_mtd_return", "year-month"]]
daily_data = pd.merge(daily_data, fifth_last_day_of_month, how="left", left_on="year-month", right_on='year-month')



print(daily_data.head().to_string())
print(daily_data.tail().to_string())

# List of strategies we use:
strat_list = [f"tmf_tmv_strat_basic", # Long TMF last 5 days, Long TMV first five days, each month.
              f"tlt_strat_basic", # Long TLT last 5 days, short TLT first five days, of each month.
              f"vix1d_strat_basic", # Basic VIX1D strat when vix1d return+, buy TMV, when vix1d return-, buy TMF
              f"tmf_tmv_strat_{month_ret_ticker}_pos", # Basic, but only invest when checked value is positive
              f"tmf_tmv_strat_{month_ret_ticker}_neg", # Basic, but only invest when checked value is negative
              f"vix1d_strat_mid_month", # vix1d strat, but not traded on first or last 5 days of each month.
              f"vix1d_strat_available", # vix1D strat, but used when tmf_tmv_SPY_pos is NOT used.
              ]

# using TMF+TMV
daily_data["tmf_tmv_strat_basic"] = risk_free_rate
daily_data["tmf_tmv_strat_basic"][daily_data["day_num"] <= 5] = daily_data["TMV_ret_1d"]
daily_data["tmf_tmv_strat_basic"][daily_data["day_numr"] <= 5] = daily_data["TMF_ret_1d"]


# Using Basic TLT
daily_data["tlt_strat_basic"] = risk_free_rate
daily_data["tlt_strat_basic"][daily_data["day_num"] <= 5] = -daily_data["TLT_ret_1d"]
daily_data["tlt_strat_basic"][daily_data["day_numr"] <= 5] = daily_data["TLT_ret_1d"]


# Using Basic VIX1D Filter:
daily_data["vix1d_strat_basic"] = risk_free_rate
daily_data["vix1d_strat_basic"][(daily_data["^VIX1D_ret_1d"].shift(1) > 0)] = daily_data["TMV_ret_1d"]
daily_data["vix1d_strat_basic"][(daily_data["^VIX1D_ret_1d"].shift(1) < 0)] = daily_data["TMF_ret_1d"]


# TMF/TMV When MTD or prior month SPY returns are positive
daily_data[f"tmf_tmv_strat_{month_ret_ticker}_pos"] = risk_free_rate
daily_data[f"tmf_tmv_strat_{month_ret_ticker}_pos"][(daily_data["day_num"] <= 5) & (daily_data[f"{month_ret_ticker}_prior_month_ret"] >= return_threshold)] = daily_data["TMV_ret_1d"]
daily_data[f"tmf_tmv_strat_{month_ret_ticker}_pos"][(daily_data["day_numr"] <= 5) & (daily_data[f"{month_ret_ticker}_mtd_return"] >= return_threshold)] = daily_data["TMF_ret_1d"]


# TMF/TMV When MTD or prior month SPY returns are negative
daily_data[f"tmf_tmv_strat_{month_ret_ticker}_neg"] = risk_free_rate
daily_data[f"tmf_tmv_strat_{month_ret_ticker}_neg"][(daily_data["day_num"] <= 5) & (daily_data[f"{month_ret_ticker}_prior_month_ret"] < return_threshold)] = daily_data["TMV_ret_1d"]
daily_data[f"tmf_tmv_strat_{month_ret_ticker}_neg"][(daily_data["day_numr"] <= 5) & (daily_data[f"{month_ret_ticker}_mtd_return"] < return_threshold)] = daily_data["TMF_ret_1d"]


daily_data["vix1d_strat_mid_month"] = risk_free_rate
daily_data["vix1d_strat_mid_month"][(daily_data["^VIX1D_ret_1d"].shift(1) > 0) & (daily_data["day_num"] > 5) & (daily_data["day_numr"] > 5)] = daily_data["TMV_ret_1d"]
daily_data["vix1d_strat_mid_month"][(daily_data["^VIX1D_ret_1d"].shift(1) < 0) & (daily_data["day_num"] > 5) & (daily_data["day_numr"] > 5)] = daily_data["TMF_ret_1d"]


daily_data["vix1d_strat_available"] = risk_free_rate
daily_data["vix1d_strat_available"][(daily_data["^VIX1D_ret_1d"].shift(1) > 0) & (daily_data["day_num"] > 5) & (daily_data["day_numr"] > 5)] = daily_data["TMV_ret_1d"]
daily_data["vix1d_strat_available"][(daily_data["^VIX1D_ret_1d"].shift(1) < 0) & (daily_data["day_num"] > 5) & (daily_data["day_numr"] > 5)] = daily_data["TMF_ret_1d"]
daily_data["vix1d_strat_available"][(daily_data["day_num"] <= 5) & (daily_data[f"{month_ret_ticker}_prior_month_ret"] < return_threshold) & (daily_data["^VIX1D_ret_1d"].shift(1) > 0)] = daily_data["TMV_ret_1d"]
daily_data["vix1d_strat_available"][(daily_data["day_num"] <= 5) & (daily_data[f"{month_ret_ticker}_prior_month_ret"] < return_threshold) & (daily_data["^VIX1D_ret_1d"].shift(1) < 0)] = daily_data["TMF_ret_1d"]
daily_data["vix1d_strat_available"][(daily_data["day_numr"] <= 5) & (daily_data[f"{month_ret_ticker}_mtd_return"] < return_threshold) & (daily_data["^VIX1D_ret_1d"].shift(1) > 0)] = daily_data["TMV_ret_1d"]
daily_data["vix1d_strat_available"][(daily_data["day_numr"] <= 5) & (daily_data[f"{month_ret_ticker}_mtd_return"] < return_threshold) & (daily_data["^VIX1D_ret_1d"].shift(1) < 0)] = daily_data["TMF_ret_1d"]


for strat in strat_list:
    daily_data[f"{strat}_ret_1d"] = daily_data[strat]#(daily_data[ticker] - daily_data[ticker].shift(1)) / daily_data[ticker].shift(1)

# Get a list of everything, and total returns:
investable2 = investable + strat_list
for ticker in investable2:
    daily_data[f"{ticker}_return_total"] = portfolio * (1 + daily_data[f"{ticker}_ret_1d"]).cumprod()
    daily_data[f"{ticker}_return_arith_total"] = portfolio + (daily_data[f"{ticker}_ret_1d"]).cumsum()

for ticker in investable2:
    daily_data[f"{ticker}_drawdown"] = (daily_data[f"{ticker}_ret_1d"] - daily_data[f"{ticker}_ret_1d"].cummax()) / daily_data[f"{ticker}_ret_1d"].cummax()
    print(f"{ticker} Sharpe Ratio: ", round((daily_data[f"{ticker}_ret_1d"].mean() - risk_free_rate) / daily_data[f"{ticker}_ret_1d"].std() * np.sqrt(252),2),
    f" Sortino Ratio: ", round((daily_data[f"{ticker}_ret_1d"].mean() - risk_free_rate) / daily_data[daily_data[f"{ticker}_ret_1d"] < 0][f"{ticker}_ret_1d"].std() * np.sqrt(252),2),
    f" Profit Factor: ", round(daily_data[f"{ticker}_ret_1d"][daily_data[f"{ticker}_ret_1d"] > 0].sum() / abs(daily_data[f"{ticker}_ret_1d"][daily_data[f"{ticker}_ret_1d"] < 0].sum()),2),
    f" % of Trades Profitable: ", round(np.where(daily_data[f"{ticker}_ret_1d"] > 0, 1, 0).mean() / np.where(daily_data[f"{ticker}_ret_1d"] != risk_free_rate, 1, 0).mean() * 100,2),
    f" Avg Daily Return (While Invested) (%): ", round((daily_data[daily_data[f"{ticker}_ret_1d"] != risk_free_rate][f"{ticker}_ret_1d"].mean()) * 100, 2),
    f" Annual Return (%): ", round((((1 + daily_data[f"{ticker}_ret_1d"].mean()) ** 252) - 1) * 100,2),
    f" Days Invested (%): ", round( np.where(daily_data[f"{ticker}_ret_1d"] != risk_free_rate, 1, 0).mean() * 100,2),
    )

plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Regular Buy+Hold Returns for {investable}")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for ticker in investable:
    temp2 =f"{ticker} Return"
    plt.plot(daily_data["date"], daily_data[f"{ticker}_return_total"], label=temp2)
    legend_list.append(temp2)
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale(chart_scale)
plt.show()



plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Returns for TMF/TMV Strategies Based on SPY and VIX1D's prior return")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for ticker in strat_list:
    temp2 =f"{ticker} Return"
    plt.plot(daily_data["date"], daily_data[f"{ticker}_return_total"], label=temp2)
    legend_list.append(temp2)
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale(chart_scale)
plt.show()

plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Investigate Arithmetic Buy+Hold Returns for TMF/TMV Strategies")
plt.title(f"Profit over time period") # Get the dates from X to Y as well.
legend_list = []
for ticker in strat_list:
    temp2 =f"{ticker} Return"
    plt.plot(daily_data["date"], daily_data[f"{ticker}_return_arith_total"], label=temp2)
    legend_list.append(temp2)
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale(chart_scale)
plt.show()