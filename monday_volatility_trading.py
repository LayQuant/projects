"""
Okay, Monday is kinda weird. Yet again. Here's how:
# Note: when did SVXY lower leverage to -0.5x Vol from -1x?
# Like... February 5th 2018?

Anyway, this is a short, quick trade that shouldn't be too hard.

C2C Fridays also looks profitable for SVXY. But it's less return over more time... (24 hours instead of 6.5 hours)

"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ticker_list = ["SVXY", "SPY", "^VIX", "^VIX3M"]
trade_list = ["SPY",  "svxy_monday", "strat", "turnaround_tuesday_too"]
returns_list = ["O2C", "C2C", "C2O"]
weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

start_date="2011-08-01"
end_date = "2030-01-01"
risk_free_rate = 0.03

data_list = []
for ticker in ticker_list:
    temp = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    temp.columns = temp.columns.droplevel(1)
    temp["C2C"] = (temp["Close"] - temp["Close"].shift(1)) / temp["Close"].shift(1)
    temp["C2O"] = (temp["Open"] - temp["Close"].shift(1)) / temp["Close"].shift(1)
    temp["O2C"] = (temp["Close"] - temp["Open"]) / temp["Open"]
    temp = temp.add_suffix(f"_{ticker}")
    data_list.append(temp)

data = pd.concat(data_list, axis=1)
data["Date-str"] = data.index.astype(str)
data["year"] = pd.to_datetime(data.index).year
data["month"] = pd.to_datetime(data.index).month
data["day"] = pd.to_datetime(data.index).day
data['day_num'] = data.groupby(["year", "month"])['Date-str'].rank(method='first').astype(int)
data['day_numr'] = data.groupby(["year", "month"])['Date-str'].rank(method='first', ascending=False).astype(int)
data["day_name"] = pd.to_datetime(data.index).day_name()
#data = data.dropna()

# Let's only worry about Mondays so far:
print(data.head().to_string())
print(data.tail().to_string())

# For each weekday, and each return type, I want to know Average Return, St Deviation of return, and Percent of days profitable
temp_ticker = "^VIX"
for day in weekday_list:
    temp = data[data["day_name"] == day]
    print(temp_ticker, day)
    for i in returns_list:
        print(
            f"Avg Return {i}:", round(temp[f"{i}_{temp_ticker}"].mean()*100, 2),"%",
            f" StDev Return {i}:", round(temp[f"{i}_{temp_ticker}"].std()*100, 2),"%",
            f" Pct Profitable {i}", round(np.where(temp[f"{i}_{temp_ticker}"] > 0, 1, 0).mean()*100, 2), "%",
        )
    print("\n")



for i in returns_list:
    data[f"{i}_svxy_monday"] = 0
    data[f"{i}_svxy_monday"][(data["day_name"] == "Monday")] = data[f"{i}_SVXY"]
    data[f"{i}_strat"] = 0
    data[f"{i}_strat"][(data["Close_^VIX"].shift(1) < data["Close_^VIX3M"].shift(1)) & (data["day_name"] == "Monday")] = data[f"{i}_SVXY"]
    # So if SPY is down on Monday, we hold SVXY at the close, and what happens???
    data[f"{i}_turnaround_tuesday_too"] = data[f"{i}_strat"]
    data[f"{i}_turnaround_tuesday_too"][(data["Close_^VIX"].shift(1) < data["Close_^VIX3M"].shift(1)) & (data["day_name"] == "Monday")  & (data["day_name"].shift(-1) == "Tuesday")
                                        & (data[f"{i}_SPY"] < 0)] += data[f"C2C_SVXY"].shift(-1)


# For calculating
monday_data = data[(data["day_name"] == "Monday")]
monday_data = monday_data.dropna()

# Let's do a t-test to see if these returns are statistically significant
# Compare
t_stat_ind, p_value_ind = stats.ttest_ind(monday_data["O2C_SVXY"], monday_data["O2C_SPY"]) # Automatically handles sample variances
print(f"\nMonday Intraday SVXY vs. Monday Intraday SPY")
print("Mondays in Sample:", len(monday_data))
print(f"Avg Monday Intraday Return SVXY: {round(monday_data["O2C_SVXY"].mean()*100,2)}%")
print(f"Avg Monday Intraday Return SPY: {round(monday_data["O2C_SPY"].mean()*100,2)}%")
print(f"T-statistic: {t_stat_ind:.4f}, P-value: {p_value_ind:.4f}")

alpha = 0.05
if p_value_ind < alpha:
    print(f"Reject null hypothesis: Mean returns of SVXY and SPY are significantly different.")
else:
    print(f"Fail to reject null hypothesis: Mean returns of SVXY and SPY are not significantly different.")

# Are monday's SVXY returns significant from all SVXY returns?
t_stat_ind, p_value_ind = stats.ttest_ind(monday_data["O2C_SVXY"].dropna(), data[data["day_name"]!= "Monday"]["O2C_SVXY"].dropna(), equal_var=False) # Automatically handles sample variances
print(f"\nMonday Intraday SVXY vs. Non-Monday Intraday SVXY")
print("Mondays in Sample:", len(monday_data), "Days in sample", len(data))
print(f"Avg Monday Intraday Return SVXY: {round(monday_data["O2C_SVXY"].mean()*100,2)}%")
print(f"Avg Intraday Return SVXY: {round(data["O2C_SVXY"].mean()*100,2)}%")
print(f"T-statistic: {t_stat_ind:.4f}, P-value: {p_value_ind:.4f}")

alpha = 0.05
if p_value_ind < alpha:
    print(f"Reject null hypothesis: Mean returns of SVXY on Mondays are significantly different.")
else:
    print(f"Fail to reject null hypothesis: Mean returns of SVXY on Mondays are not significantly different.")



print("\nTrades stats for Monday intraday:")
for ticker in trade_list:
    # we're using 2 days for Turnaround Tuesday too, so we'll need to halve it for the Sharpe+Sortino calculations
    sharpe_mult = 252
    if ticker == "turnaround_tuesday_too":
        sharpe_mult /= 2

    print(f"{ticker} Sharpe Ratio: ", round((monday_data[f"O2C_{ticker}"].mean() - risk_free_rate/sharpe_mult) / monday_data[f"O2C_{ticker}"].std() * np.sqrt(sharpe_mult),2),
    f" Sortino Ratio: ", round((monday_data[f"O2C_{ticker}"].mean() - risk_free_rate/sharpe_mult) / monday_data[monday_data[f"O2C_{ticker}"] < 0][f"O2C_{ticker}"].std() * np.sqrt(sharpe_mult),2),
    f" Profit Factor: ", round(monday_data[f"O2C_{ticker}"][monday_data[f"O2C_{ticker}"] > 0].sum() / abs(monday_data[f"O2C_{ticker}"][monday_data[f"O2C_{ticker}"] < 0].sum()),2),
    f" % of Trades Profitable: ", round(np.where(monday_data[f"O2C_{ticker}"] > 0, 1, 0).mean() / np.where(monday_data[f"O2C_{ticker}"] != 0, 1, 0).mean() * 100,2),"%",
    f" Avg Daily Return (While Invested) (%): ", round((monday_data[monday_data[f"O2C_{ticker}"] != 0][f"O2C_{ticker}"].mean()) * 100, 2),
    f" Avg Daily Std Dev (While Invested) (%): ",round((monday_data[monday_data[f"O2C_{ticker}"] != 0][f"O2C_{ticker}"].std()) * 100, 2),
    f" Annualized Return (%): ", round((((1 + monday_data[f"O2C_{ticker}"].mean()) ** sharpe_mult) - 1) * 100,2),
    )

# PLOTS
plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"VIX Index Plot")
plt.title(f"VIX Level over time") # Get the dates from X to Y as well.
plt.plot(data.index, data["Close_^VIX"], label="^VIX")
plt.legend("^VIX")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale("linear")
plt.show()

# What happens to the VIX on each day of the year, month, week?
metric = "mean"
temp = data.groupby(['day_name']).aggregate({'C2C_^VIX': metric, "O2C_^VIX": metric, "C2O_^VIX": metric }).reindex(index=weekday_list)
plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.title(f"Average VIX Pct Change by Weekday") # Get the dates from X to Y as well.
legend_list = []
for returns in returns_list:
    plt.plot(temp.index, temp[f"{returns}_^VIX"]*100, label=returns)
    legend_list.append(returns)
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()



# Can this be used on a volatility ETF????
metric = "mean"
temp = data.groupby(['day_name']).aggregate({'C2C_SVXY': metric, "O2C_SVXY": metric, "C2O_SVXY": metric }).reindex(index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.title(f"Average SVXY Pct Change by Weekday") # Get the dates from X to Y as well.
legend_list = []
for returns in returns_list:
    plt.plot(temp.index, temp[f"{returns}_SVXY"]*100, label=returns)
    legend_list.append(returns)
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()


plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Strategy Returns for SVXY")
plt.title(f"Profit over time") # Get the dates from X to Y as well.
legend_list = []
for ticker in trade_list:
    plt.plot(monday_data.index, (1+monday_data[f"O2C_{ticker}"]).cumprod(), label=ticker)
    legend_list.append(ticker)
plt.legend(legend_list)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.yscale("log")
plt.show()
