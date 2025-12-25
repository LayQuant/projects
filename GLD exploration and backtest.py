"""
Key things to check:
Possible strategies to make money of the GLD ETF
Best times to hold Gold?

"""

import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ticker_list = ["GLD", "SPY", "^VIX"]
ticker_list_chart = ["GLD", "SPY"]
weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
# Get data and returns: for tickers
data_list = []
for ticker in ticker_list:
    temp = yf.download(ticker, start="1990-01-01", end="2030-01-01")
    temp.columns = temp.columns.droplevel(1)
    temp["c2c_return"] = (temp["Close"] - temp["Close"].shift(1)) / temp["Close"].shift(1)
    # add suffixes and to the data list.
    temp = temp.add_suffix(f"_{ticker}")
    data_list.append(temp)

data = pd.concat(data_list, axis=1)
# remove NAs?
data = data.dropna()
# Add dates and stuff
data["weekday"] = pd.to_datetime(data.index).day_name()


# What are the returns for each weekday?
for i in  weekday_list:
    print(f"Avg GLD Return for {i}:",
          round(data[data["weekday"] == i]["c2c_return_GLD"].mean()*100,2), "%",
          f"Cum. GLD Returns for {i}",
          round(((1+ data[data["weekday"] == i]["c2c_return_GLD"]).cumprod()[-1] -1) * 100, 2), "%",
          )
    print(f"Avg SPY Return for {i}:",
          round(data[data["weekday"] == i]["c2c_return_SPY"].mean()*100,2), "%",
          f"Cum. SPY Returns for {i}",
          round(((1 + data[data["weekday"] == i]["c2c_return_SPY"]).cumprod()[-1] -1) * 100, 2), "%",
          )


# What's the average return for days where SPY's return is positive vs negative?
print("Average GLD and SPY daily return:")
print("AVG GLD Return:", round(data["c2c_return_GLD"].mean()*100,2), "%")
print("AVG SPY Return:", round(data["c2c_return_GLD"].mean()*100,2), "%")
for ticker in ticker_list:
    print(f"Checking returns based on {ticker}'s prior-day returns")
    print(f"Positive Prior {ticker} Return:")
    print(f"Avg GLD Return:", round(data[data[f"c2c_return_{ticker}"].shift(1) > 0]["c2c_return_GLD"].mean()*100,2), "%",
          f"Avg SPY Return:", round(data[data[f"c2c_return_{ticker}"].shift(1) > 0]["c2c_return_SPY"].mean()*100,2), "%")
    print(f"Negative Prior {ticker} Return:")
    print(f"Avg GLD Return:", round(data[data[f"c2c_return_{ticker}"].shift(1) < 0]["c2c_return_GLD"].mean()*100,2), "%",
          f"Avg SPY Return:", round(data[data[f"c2c_return_{ticker}"].shift(1) < 0]["c2c_return_SPY"].mean()*100,2), "%")
    print("\n")


print(data.head().to_string())
print(data.tail().to_string())


# Plot normal returns for SPY and GLD
plt.figure()
plt.xticks(rotation=45)
#plt.suptitle("")
plt.title(f"GLD, SPY Buy+Hold Returns")
data_list = []
for ticker in ticker_list_chart:
    plt.plot(pd.to_datetime(data.index), (1 + data[f"c2c_return_{ticker}"]).cumprod(), label=f"{ticker} return")
    data_list.append(f"{ticker} return")
plt.legend(data_list)
plt.xlabel("Date")
plt.ylabel("Cumulative % Returns")
plt.yscale("log")
plt.show()


# Plot normal returns for SPY and GLD
plt.figure()
plt.xticks(rotation=45)
#plt.suptitle("")
plt.title(f"GLD, SPY Returns by Weekday held")
data_list = []
for i in weekday_list:
    _data = data[data["weekday"] == i]
    plt.plot(pd.to_datetime(_data.index), (1 + _data[f"c2c_return_GLD"]).cumprod(), label=f"{i} GLD return")
    data_list.append(f"{i} GLD return")
plt.legend(data_list)
plt.xlabel("Date")
plt.ylabel("Cumulative % Returns")
plt.yscale("log")
plt.show()

# Plot normal returns for SPY and GLD
plt.figure()
plt.xticks(rotation=45)
#plt.suptitle("")
plt.title(f"GLD, SPY Returns after changes in SPY, GLD, VIX")
data_list = []
for ticker in ticker_list:
    # Positive Return:
    _data = data[data[f"c2c_return_{ticker}"].shift(1) > 0]
    plt.plot(pd.to_datetime(_data.index), (1 + _data[f"c2c_return_GLD"]).cumprod(), label=f"GLD with prior {ticker} > 0 return")
    data_list.append(f"GLD with {ticker} > 0 return")

    # Negative Return:
    _data = data[data[f"c2c_return_{ticker}"].shift(1) < 0]
    plt.plot(pd.to_datetime(_data.index), (1 + _data[f"c2c_return_GLD"]).cumprod(), label=f"GLD with prior {ticker} < 0 return")
    data_list.append(f"GLD with {ticker} < 0 return")
plt.legend(data_list)
plt.xlabel("Date")
plt.ylabel("Cumulative % Returns")
plt.yscale("log")
plt.show()

