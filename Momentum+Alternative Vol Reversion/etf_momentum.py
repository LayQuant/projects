"""

If you buy at the monthly level when momentum is up: Generally the Returns and Sharpe are better:
SPY, QQQ, IWM, GLD, PDP, HYG.
Doesn't work for TLT. I'm assuming because of the Fed Reserve influence.

"""
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")



ticker= "SPY"
data_d = yf.download(ticker, start="1900-01-01", end="2030-01-01", interval="1d")
data_w = yf.download(ticker, start="1900-01-01", end="2030-01-01", interval="1wk")
data_m = yf.download(ticker, start="1900-01-01", end="2030-01-01", interval="1mo")

for _data in [data_d, data_w, data_m]:
    _data.columns = _data.columns.droplevel(1)
    _data["C2C"] = (_data["Close"] - _data["Close"].shift(1)) / _data["Close"].shift(1)
    _data["return_1_next"] = _data["C2C"].shift(-1)
    _data["return_22_next"] = (_data["Close"].shift(-22) - _data["Close"]) / _data["Close"]
    _data["weekday"] = pd.to_datetime(_data.index).day_name()
    _data["date"] = pd.to_datetime(_data.index)
    _data["month"] = _data["date"].dt.month
    _data["year"] = _data["date"].dt.year
    _data['day_num'] = _data.groupby(["year", "month"])["date"].rank(method='first').astype(int)


# Monthly Momentum Factors:
data_m["mom-12-1"] = data_m["C2C"].rolling(12).sum() - data_m["C2C"]
data_m["mom-4-1"] = data_m["C2C"].rolling(4).sum() - data_m["C2C"]
data_m["mom_factor"] = data_m["mom-12-1"]/data_m["C2C"].rolling(12).std()
data_m["new_skew"] = (data_m["C2C"].rolling(12).mean() - data_m["C2C"].rolling(12).median())/data_m["C2C"].rolling(12).std()


# Try at daily trailing level?
data_d["mom-12-1"] = data_d["C2C"].rolling(252).sum() - data_d["C2C"].rolling(21).sum()
data_d["mom_factor"] = data_d["mom-12-1"]/data_d["C2C"].rolling(252).std()
data_d["new_skew"] = (data_m["C2C"].rolling(252).mean() - data_m["C2C"].rolling(252).median())/data_m["C2C"].rolling(252).std()

# Try at weekly trailing level?
data_w["mom-12-1"] = data_w["C2C"].rolling(52).sum() - data_w["C2C"].rolling(4).sum()
data_w["mom_factor"] = data_w["mom-12-1"]/data_w["C2C"].rolling(52).std()
data_w["new_skew"] = (data_w["C2C"].rolling(52).mean() - data_w["C2C"].rolling(52).median())/data_w["C2C"].rolling(52).std()


# Let's look at various momentum factors:
print(data_d.tail().to_string())
print(data_w.tail().to_string())
print(data_m.tail().to_string())


# Now get rid of the first ~12 months?
data_m = data_m[12:]
data_w = data_w[52:]
data_d = data_d[252:]

# Let's try different momentum factors:
data_mom = data_m[data_m["mom-12-1"] > 0]
data_mom2 = data_m[data_m["mom-12-1"] > data_m["mom-12-1"].shift(1)]
data_mom3 = data_m[data_m["mom_factor"] > data_m["mom_factor"].shift(1)] # doesn't work well.
data_mom4 = data_m[(data_m["mom-12-1"] > 0) & (data_m["Volume"] > data_m["Volume"].shift(1))]

# Try at weekly and daily level?
data_mom5 = data_w[(data_w["mom-12-1"] > 0) & (data_w["Volume"] > data_w["Volume"].shift(1))]
#data_mom6 = data_d[data_d["mom-12-1"] > 0] # Works similarly
data_mom6 = data_w[data_w["mom-12-1"] > 0]



# what's the return when Momentum is positive vs negative.?
i=0
data_name = ["Buy & Hold", "+12-1 Momentum", "Growing Momentum", "Growing Risk-Adj Momentum","+12-1 Momentum More Volume", "+12-1 Momentum Weekly More Volume", "+12-1 Momentum Weekly"]
data_list = [data_m, data_mom, data_mom2, data_mom3, data_mom4, data_mom5, data_mom6]
period_list = [12, 12, 12, 12, 12, 52, 52, 52]
return_type = "return_1_next"
for _data in data_list:
    print(data_name[i])
    print("Number of Periods Active:", len(_data))
    print("Avg Return: ", round(_data[return_type].mean()*100,4),"%", "Std Dev Return:", round(_data[return_type].std()*100,4),"%")
    print("Profit Factor: ", round(_data[_data[return_type] > 0][return_type].sum()/ (-1*_data[_data[return_type] < 0][return_type].sum()), 2))
    print("Sharpe: ", round((_data[return_type].mean() - 0.03/period_list[i])/_data[return_type].std()*np.sqrt(period_list[i]),2))
    print("Pct Periods Positive:", round( np.where(_data[return_type]>0, 1, 0).mean()*100, 2), "%")
    print("\n")
    i+=1
i = 0
plt.plot(figsize=(12,10))
for _data in data_list:
    plt.plot((1+_data[return_type]).cumprod(), label=f'{ticker} {data_name[i]}')
    i+=1
plt.title(f'Momentum Types vs B+H {ticker}')
plt.ylabel('Return of $1')
plt.grid(True)
plt.legend()
plt.yscale("log")
plt.show()


# what happens if we take a look at monthly/weekly/daily momentum starting at different 1-20 starting days of each month:
# Instead of just counting momentum at the start/end of the month.
# Maybe there's a calendar effect there?
plt.plot(figsize=(12,10))
plt.title(f"Regular 22-day {ticker} return by start date of month")
legend_list = []
for i in range(1,15):
    label = f"start day {i} of month"
    temp = data_d[(data_d["day_num"] == i)]
    plt.plot(temp.index, (1+temp["return_22_next"]).cumprod(), label=label)
    legend_list.append(label)
plt.legend(legend_list)
plt.yscale("log")
plt.plot()
plt.show()

plt.plot(figsize=(12,10))
plt.title(f"Positive Momentum 22-day {ticker} return by start date of month")
legend_list = []
for i in range(1,15):
    label = f"12-1MOM start day {i} of month"
    temp = data_d[(data_d["day_num"] == i) & (data_d["mom-12-1"] > 0)]
    plt.plot(temp.index, (1+temp["return_22_next"]).cumprod(), label=label)
    legend_list.append(label)
plt.legend(legend_list)
plt.yscale("log")
plt.plot()
plt.show()

for i in range(1,15):
    temp = data_d[(data_d["day_num"] == i)]
    temp_mom = data_d[(data_d["day_num"] == i) & (data_d["mom-12-1"] > 0)]
    print(f"Doing the {ticker} momentum trade by {i} (trading) Day of the month")
    print(f"Avg Return: {round(temp["return_22_next"].mean()*100,2)}%, Avg Positive-Momentum Return: {round(temp_mom["return_22_next"].mean()*100,2)}%")
    print(f"Number of months traded: {len(temp)}, Num Positive-Momentum Months traded: {len(temp_mom)}")
    print(f"Regular Sharpe: {round(temp["return_22_next"].mean()/temp["return_22_next"].std()*np.sqrt(12),2)}, "
          f"Positive-Momentum Sharpe:{round(temp_mom["return_22_next"].mean()/temp_mom["return_22_next"].std()*np.sqrt(12),2)}")
    print("\n")



# What if we cycle through momentum?
data_list = []
ticker_list = ["SPY", "GLD", "TLT", "EEM", "EMB"]
#ticker_list = ['SPY', 'EWH', 'EWU', 'EWN', 'EWI', 'EWP', 'EWO', 'EWA', 'EWK', 'EWD', 'EWM', 'EWJ', 'EWC', 'EWW', 'EWL', 'EWG', 'EWQ', 'EWS'] # this one doesn't work.
#ticker_list = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLRE", "XLU"] # this one doesn't work well either.
#ticker_list = ["ITA", "IYM", "IAI", "IYC", "IEDI", "IYK", "IDGT", "IYE", "IYG", "IYF", "IYH", "IHF", "ITB", "IYJ", "IAK", "IHI", "IEO", "IEZ", "IHE", "IYR", "IAT", "IYW", "IYZ", "IYT", "IDU"]

for ticker in ticker_list:
    temp = yf.download(ticker, start="2005-01-01", end="2030-01-01", interval="1mo")
    temp.columns = temp.columns.droplevel(1)
    temp["C2C"] = (temp["Close"] - temp["Close"].shift(1)) / temp["Close"].shift(1)
    temp["return_1_next"] = temp["C2C"].shift(-1)
    temp["mom-12-1"] = temp["C2C"].rolling(12).sum() - temp["C2C"]
    temp["mom_factor"] = (temp["C2C"].rolling(12).sum() - temp["C2C"])
    #temp["mom_factor"] = temp["mom-12-1"] - temp["mom-12-1"].shift(1) # Interesting
    #temp["mom_factor"] =  temp["C2C"].rolling(12).median() / temp["C2C"].rolling(12).std()
    temp["ticker"] = ticker
    data_list.append(temp)
data = pd.concat(data_list, axis=0)
data["_Date"] = data.index
# What do we rank:
rank_variable = "mom-12-1" # Pure Momentum Ranking at monthly level has the 2nd best momentum win????
#rank_variable = "mom_factor" # Growing Momentum
data["momentum_rank"] = data.groupby("_Date")[rank_variable].rank(method="first", ascending=False)
#print(data.to_string())

# Let's look at the top momentum_rank

plt.plot(figsize=(12,10))
plt.plot((1+data[data["ticker"]=="SPY"][f"return_1_next"]).cumprod(), label=f'SPY Buy+Hold')
for i in range(int(data["momentum_rank"].min()),int(data["momentum_rank"].max())+1):
    _data = data[data["momentum_rank"] == i]
    _data = _data.sort_index(ascending=True)
    print("Month with Rank ", i)
    print("Number of Periods Active:", len(_data))
    print("Avg Return: ", round(_data["return_1_next"].mean(), 4), "%", "Std Dev Return:",
          round(_data["return_1_next"].std(), 4), "%")
    print("Profit Factor: ", round(_data[_data["return_1_next"] > 0]["return_1_next"].sum() / (
                -1 * _data[_data["return_1_next"] < 0]["return_1_next"].sum()), 2))
    print("Sharpe: ", round(
        (_data["return_1_next"].mean() - 0.03/12) / _data["return_1_next"].std() * np.sqrt(
            12), 2))
    print("Pct Periods Positive:", round(np.where(_data["return_1_next"] > 0, 1, 0).mean() * 100, 2), "%")
    print("\n")
    plt.plot((1+_data[f"return_1_next"]).cumprod(), label=f'Momentum Rank {i}')
    i+=1
plt.title(f'Momentum Ranks for {ticker_list}')
plt.ylabel('Return of $1')
plt.grid(True)
plt.legend()
plt.yscale("log")
plt.show()
