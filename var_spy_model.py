import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import datetime
import yfinance as yf
import warnings
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
warnings.filterwarnings("ignore")



# Functions:
maxlag = 5
def granger_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=5):
    #Check Granger Causality between each possible time series variabl.
    #The rows are the response variable, columns are predictors. The values in the table are the P-Values.
    # P-Values < significance level (0.05) imply the Null Hypothesis:
    # (Coeff of Xt-1 = 0, meaning X does not cause Y can be rejected.)
    #data: pandas dataframe, variables: list of time series columns in data.
    df_min = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    df_avg = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in df_min.columns:
        for row in df_min.index:
            test_result = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {row}, X = {col}, P Values = {p_values}')
            df_min.loc[row, col] = np.min(p_values)
            df_avg.loc[row, col] = np.mean(p_values)
    df_min.columns = [var + '_x' for var in variables]
    df_min.index = [var + '_y' for var in variables]
    df_avg.columns = [var + '_x' for var in variables]
    df_avg.index = [var + '_y' for var in variables]
    return df_min, df_avg

ticker = "SPY"
# Would have to use this to predict /ES daily.
ticker_list = [ticker, "/ES=F", "^SPX", "^VVIX", "^VIX", "USO", "SHY", "TLH", "GLD",]
interval = "1d"
start = "2006-1-1"
# Safe way to get the most up-to-date data.
end = "2030-1-1"
test = yf.download(ticker_list, start=start, end=end, interval=interval)[["Adj Close"]]["Adj Close"]
monthly = yf.download(ticker_list, start=start, end=end, interval="1mo")[["Adj Close"]]["Adj Close"]
test["Date"] = test.index
test["Year"] = pd.to_datetime(test["Date"]).dt.year
test["Month"] = pd.to_datetime(test["Date"]).dt.month
test["Day"] = pd.to_datetime(test["Date"]).dt.day

# What should our slippage be? 0.001%?
slippage = 0.00001


return_list = []
for tick in ticker_list:
    # Forward period return:
    test[f"f_{tick}_1p"] = (test[f"{tick}"].shift(-1) - test[f"{tick}"]) / test[f"{tick}"]# * 100
    monthly[f"f_{tick}_1p"] = (monthly[f"{tick}"].shift(-1) - monthly[f"{tick}"]) / monthly[f"{tick}"]  # * 100
    return_list.append(f"f_{tick}_1p")

temp = test.dropna()

print(temp.head().to_string())
print(temp.tail().to_string())

# Let's test for Granger Causality:
print("Granger Causality Test Minimum Lagged P-Value")
print(granger_causation_matrix(temp, variables=return_list)[0].to_string())
print("\n")
print("Granger Causality Test Avg Lagged P-Value")
print(granger_causation_matrix(temp, variables=return_list)[1].to_string())


#####
# Correlation Matrix
#####

# We want a Daily and Monthly Correlation Matrix.
# Daily
seaborn.heatmap(temp[return_list].corr(), vmin=-1, vmax=1, annot=True)\
    .set_title('Daily Return Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()

# Monthly
seaborn.heatmap(monthly[return_list].corr(), vmin=-1, vmax=1, annot=True)\
    .set_title('Monthly Return Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()

print(len(temp))
# We note that it outperforms the first month then kinda peters out.
test_size = int(len(temp)*0.20) # can plot out differences several times, replacing 0.05 with 0.01 to 0.20
# Should probably do an update every week or month then.
print(test_size)
var_data_train = temp[:test_size]
var_data_test = temp[-test_size+1:]
var_data_train = var_data_train[return_list
                          #+exog
                        ]
var_data_test = var_data_test[return_list
                        #+exog
                        ]

model = VAR(var_data_train[return_list],
            #exog=var_data_train[exog]
            )

order_type = None
order_size = 1000
for i in ['aic', 'bic', 'hqic']:
    #maxlags takes the number of lags we want to test
    #ic takes the information criterion metod based on which order would be suggested
    results = model.fit(maxlags=5,  ic=i)
    order = results.k_ar
    print(f"The suggested VAR order from {i} is {order}")
    if (order < order_size) & (order > 0):
        order_size = order
        order_type = i

print("Chose: ", order_type, order_size)

model = VAR(var_data_train[return_list],
            #exog=var_data_train[exog]
            )
model_fit = model.fit(maxlags = order_size, ic=order_type, trend='c')
print(model_fit.summary())

# Check the Coefficient values.
print("Coefficient Values:")
print(model_fit.params[f"f_{ticker_list[0]}_1p"].to_string())

# Test for residual normality:
print(f"Test for Residual Normality: {model_fit.test_normality().pvalue}")


#To test absence of significant residual autocorrelations one can use the test_whiteness method of VARResults
test_corr = model_fit.test_whiteness(nlags=order_size+2, signif=0.05, adjusted=False)
print(f"Whiteness Method Test: {round(test_corr.pvalue,4)}")


nobs = model_fit.nobs
idx = temp.index
idx = idx[-nobs:]
var_data_test = var_data_test[order_size-1:]
var_data_test.index = idx

# Only do the first 3-ish months?
#var_data_test = var_data_test[:60]

# {ticker}_fit Coefficients
var_data_test[f"{ticker}_pred"] = model_fit.params[f"f_{ticker}_1p"].loc["const"]

for i in return_list:
    for num in list(range(1, order_size+1)):
        var_data_test[f"{ticker}_pred"] += model_fit.params[f"f_{ticker}_1p"].loc[f"L{num}.{i}"] * var_data_test[i].shift(num)

var_data_test[f"Pred_error_{ticker}"] = np.where(np.sign(var_data_test[f"{ticker}_pred"]) != np.sign(var_data_test[f"f_{ticker}_1p"]),
    abs(var_data_test[f"{ticker}_pred"] - var_data_test[f"f_{ticker}_1p"]), 0)

# Abs error
var_data_test[f"Pred_error_{ticker}_AE"] = abs(var_data_test[f"{ticker}_pred"] - var_data_test[f"f_{ticker}_1p"])
# Need to use SQRT because it's value error is 1 > error > 0
var_data_test[f"Pred_error_{ticker}_SE"] = np.sqrt(abs(var_data_test[f"{ticker}_pred"] - var_data_test[f"f_{ticker}_1p"]))

# Make a ZN and TLT EMA?
# Seems that errors don't really help us.
ema = 10
for ticker in [ticker]:
    var_data_test[f"EMA-{ema}_{ticker}"] = var_data_test[f"f_{ticker}_1p"].shift(1).ewm(span=ema).mean()
    var_data_test[f"Pred_Error_EMA-{ema}_{ticker}"] = var_data_test[f"Pred_error_{ticker}"].shift(1).ewm(span=ema).mean()
    var_data_test[f"Pred_Error_growing"] = np.where(var_data_test[f"Pred_Error_EMA-{ema}_{ticker}"] > var_data_test[f"Pred_Error_EMA-{ema}_{ticker}"].shift(1), 1, 0)
    # For our 'average error' EMA
    var_data_test[f"Pred_Error_EMA-{ema}_{ticker}_AE"] = var_data_test[f"Pred_error_{ticker}_AE"].shift(1).ewm(span=ema).mean()
    var_data_test[f"Pred_Error_growing_AE"] = np.where(
        var_data_test[f"Pred_Error_EMA-{ema}_{ticker}_AE"] > var_data_test[f"Pred_Error_EMA-{ema}_{ticker}_AE"].shift(1), 1, 0)
    # For our 'squared error' EMA
    var_data_test[f"Pred_Error_EMA-{ema}_{ticker}_SE"] = var_data_test[f"Pred_error_{ticker}_SE"].shift(1).ewm(span=ema).mean()
    var_data_test[f"Pred_Error_growing_SE"] = np.where(
        var_data_test[f"Pred_Error_EMA-{ema}_{ticker}_SE"] > var_data_test[f"Pred_Error_EMA-{ema}_{ticker}_SE"].shift(1), 1, 0)


# How many data poitns do we have?
print("Number of Data Points to test:")
print(len(var_data_test))


# When do we buy and sell?
# Fade the EMA-5?
var_data_test["B/S"] = "None"
var_data_test["B/S"][(var_data_test[f"{ticker}_pred"] > 0)
    #((var_data_test[f"{ticker}_pred"] > 0) & (var_data_test["Pred_Error_growing_SE"] == 0))
    #| ((var_data_test[f"{ticker}_pred"] < 0) & (var_data_test["Pred_Error_growing_SE"] == 1))
            ] = "Buy"
var_data_test["B/S"][(var_data_test[f"{ticker}_pred"] < 0)
    #((var_data_test[f"{ticker}_pred"] < 0) & (var_data_test["Pred_Error_growing_SE"] == 0))
    #| ((var_data_test[f"{ticker}_pred"] > 0) & (var_data_test["Pred_Error_growing_SE"] == 1))
            ] = "Sell"

# It's an ETF, don't need to worry about Commission
var_data_test["Commission"] = 0


print(f"Trading From {var_data_test[:1].index[0]} to {var_data_test[-1:].index[0]}")

var_data_test["b+h_return"] = (1 + var_data_test[f"f_{ticker}_1p"]).cumprod() - 1
var_data_test["strategy_return"] = 0
var_data_test["strategy_return"][var_data_test["B/S"] == "Buy"] = var_data_test[f"f_{ticker}_1p"] - slippage
var_data_test["strategy_return"][var_data_test["B/S"] == "Sell"] = -var_data_test[f"f_{ticker}_1p"] - slippage

var_data_test["strat_return_total"] = (1 + var_data_test[f"strategy_return"]).cumprod() - 1
print(f"Hypothetical Overall Gains from Buy and Hold: {round(var_data_test['b+h_return'][-1]*100,2)}%")
print(f"Hypothetical Overall Gains from Strategy: {round(var_data_test['strat_return_total'][-1]*100,2)}%")
print(f"Percent of Trading Days Invested in Strategy: {round(len(var_data_test[var_data_test['B/S'] != 'None'])/len(var_data_test)*100,2)}%")
print(f"Trades done Buy or Sell: {var_data_test['B/S'].value_counts()}")
print(var_data_test.head(5).to_string())

# Now let's plot our data:
plt.figure(dpi=100)
plt.xticks(rotation=45)
plt.suptitle(f"Growth of ${100} Trading {ticker} with {ticker_list} factors")
plt.title(f"Return of ${100} Invested from {var_data_test[:1].index[0]} to {var_data_test[-1:].index[0]}")
plt.plot(var_data_test.index, var_data_test["strat_return_total"]*100, label=f"{ticker} trade prediction")
plt.plot(var_data_test.index, var_data_test["b+h_return"]*100, label=f"{ticker} Buy + Hold Return")
plt.legend([f"{ticker} trading strategy", f"{ticker} B+H"])
plt.show()



# Rolling VAR:
# We have a however-long context window (1000 trading days)
# then each day we make a prediction.
# And add the results to the model.
# Redo the model for each month.
# then link the data together.
window_size = 1000

# Data frame to hold the coefficients.
coef_list = ["const"] + return_list + ["Date"]
coef_frame = pd.DataFrame(columns=coef_list)

# Will need to change the start year if we use different ETFS that we introduced after 2006.
start_year = 2007
full_data = test[test["Year"].between(2006, start_year)]
for year in range(start_year+1, 2025):
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        print(f"Period Added: {year}-{month}")
        temp_data = test[(test["Year"] == year) & (test["Month"] == month)][return_list].dropna()
        if len(temp_data) < 12:
            break
        temp_full_data = full_data[return_list].dropna()
        # let's have at most f{window_size} pieces of data?
        temp_full_data = temp_full_data[-window_size:]
        model = VAR(temp_full_data[return_list],
                    # exog=temp_full_data[exog]
                    )
        order_type = None
        order_size = 1000
        for i in ['aic', 'bic', 'hqic']:
            # maxlags takes the number of lags we want to test
            # ic takes the information criterion metod based on which order would be suggested
            results = model.fit(maxlags=5, ic=i)
            order = results.k_ar
            #print(f"The suggested VAR order from {i} is {order}")
            if (order < order_size) & (order > 0):
                order_size = order
                order_type = i

        #print("Chose: ", order_type, order_size)

        if order_size == 1000:
            order_size = 1

        model = VAR(temp_full_data[return_list],
                    # exog=temp_full_data[exog]
                    )
        model_fit = model.fit(maxlags=order_size, ic=order_type, trend='c')
        #print(model_fit.summary())

        test_size = len(temp_data)
        #print(test_size)
        # To test absence of significant residual autocorrelations one can use the test_whiteness method of VARResults
        test_corr = model_fit.test_whiteness(nlags=order_size + 2, signif=0.05, adjusted=False)
        print(f"Whiteness Method Test: {round(test_corr.pvalue, 4)}")

        # Now let's merge the full and temp data
        temp_full_data = pd.concat([temp_full_data, temp_data], axis=0)

        # {ticker}_fit Coefficients
        temp_full_data[f"{ticker}_pred"] = model_fit.params[f"f_{ticker}_1p"].loc["const"]
        param_dict = {}
        param_dict["const"] = model_fit.params[f"f_{ticker}_1p"].loc["const"]
        for i in return_list:
            for num in list(range(1, order_size + 1)):
                temp_full_data[f"{ticker}_pred"] += model_fit.params[f"f_{ticker}_1p"].loc[f"L{num}.{i}"] * temp_full_data[i].shift(num)
                if num==1:
                    param_dict[i] = [model_fit.params[f"f_{ticker}_1p"].loc[f"L{num}.{i}"]]

        # Add Coef's to data frame
        param_dict = pd.DataFrame.from_dict(param_dict)
        param_dict["Date"] = pd.to_datetime(f"{year}-{month}-01")
        coef_frame = pd.concat([coef_frame, param_dict])

        temp_full_data[f"Pred_error_{ticker}"] = np.where(
            np.sign(temp_full_data[f"{ticker}_pred"]) != np.sign(temp_full_data[f"f_{ticker}_1p"]),
            abs(temp_full_data[f"{ticker}_pred"] - temp_full_data[f"f_{ticker}_1p"]), 0)

        # Abs error
        temp_full_data[f"Pred_error_{ticker}_AE"] = abs(temp_full_data[f"{ticker}_pred"] - temp_full_data[f"f_{ticker}_1p"])
        # Need to use SQRT because it's value error is 1 > error > 0
        temp_full_data[f"Pred_error_{ticker}_SE"] = np.sqrt(
            abs(temp_full_data[f"{ticker}_pred"] - temp_full_data[f"f_{ticker}_1p"]))

        temp_data = temp_full_data[-test_size:]
        if len(temp_data) > 10:
            #print(temp.tail(3).to_string())
            full_data = pd.concat([full_data, temp_data], axis=0)

print("Monthly Rebalanced Coefficients from the VAR:")
print(coef_frame.to_string())
coef_frame.index = coef_frame["Date"]

# Let's plot the COEF dataframe?
plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.title(f"Time Series of Monthly Coefficients")
plt.plot(coef_frame.index, coef_frame["const"], label=f"{ticker} Const")
legend_list = ["Const"]
for i in return_list:
    plt.plot(coef_frame.index, coef_frame[i], label=f"{i} Coefficient")
    legend_list.append(i)
plt.legend(legend_list, loc='lower left')
plt.show()


# Replace NAN with 0
full_data = full_data.fillna(0)

# moving Averages and Regimes for Errors.
ema = 10
for ticker in [ticker]:
    full_data[f"EMA-{ema}_{ticker}"] = full_data[f"f_{ticker}_1p"].shift(1).ewm(span=ema).mean()
    full_data[f"Pred_Error_EMA-{ema}_{ticker}"] = full_data[f"Pred_error_{ticker}"].shift(1).ewm(span=ema).mean()
    full_data[f"Pred_Error_growing"] = np.where(full_data[f"Pred_Error_EMA-{ema}_{ticker}"] > full_data[f"Pred_Error_EMA-{ema}_{ticker}"].shift(1), 1, 0)
    # For our 'average error' EMA
    full_data[f"Pred_Error_EMA-{ema}_{ticker}_AE"] = full_data[f"Pred_error_{ticker}_AE"].shift(1).ewm(span=ema).mean()
    full_data[f"Pred_Error_growing_AE"] = np.where(
        full_data[f"Pred_Error_EMA-{ema}_{ticker}_AE"] > full_data[f"Pred_Error_EMA-{ema}_{ticker}_AE"].shift(1), 1, 0)
    # For our 'squared error' EMA
    full_data[f"Pred_Error_EMA-{ema}_{ticker}_SE"] = full_data[f"Pred_error_{ticker}_SE"].shift(1).ewm(span=ema).mean()
    full_data[f"Pred_Error_growing_SE"] = np.where(
        full_data[f"Pred_Error_EMA-{ema}_{ticker}_SE"] > full_data[f"Pred_Error_EMA-{ema}_{ticker}_SE"].shift(1), 1, 0)



# Buy and sell orders
full_data["B/S"] = "None"
full_data["B/S"][(full_data[f"{ticker}_pred"] > 0)
                 #& (full_data[f"Pred_Error_growing"] == 0) # Error tracking doesn't seem to help
                ] = "Buy"
full_data["B/S"][(full_data[f"{ticker}_pred"] < 0)
                 #& (full_data[f"Pred_Error_growing"] == 0) # Error tracking doesn't seem to help
                ] = "Sell"

# Randomized Buy/Sell
full_data["random_B/S"] = np.random.choice(["Buy", "Sell"], size=len(full_data))

# Let's make sure we use the actual trading window.
full_data["Date"] = full_data.index
full_data = full_data[full_data["Date"] >= f"{start_year+1}-01-01"]


# Cumulative Returns from Buy and Hold
full_data["b+h_return"] = (1 + full_data[f"f_{ticker}_1p"]).cumprod() - 1


# Cumulative Returns from Our Buy/Short Stratgey
full_data["strategy_return"] = 0
full_data["strategy_return"][full_data["B/S"] == "Buy"] = full_data[f"f_{ticker}_1p"] - slippage
full_data["strategy_return"][full_data["B/S"] == "Sell"] = -full_data[f"f_{ticker}_1p"] - slippage
full_data["strat_return_total"] = (1 + full_data[f"strategy_return"]).cumprod() - 1

# Cumulative Returns from Random Buy/Short
full_data["random_strategy_return"] = 0
full_data["random_strategy_return"][full_data["random_B/S"] == "Buy"] = full_data[f"f_{ticker}_1p"] - slippage
full_data["random_strategy_return"][full_data["random_B/S"] == "Sell"] = -full_data[f"f_{ticker}_1p"] - slippage
full_data["random_strat_return_total"] = (1 + full_data[f"random_strategy_return"]).cumprod() - 1


# Printed out stats we might care about.
print(f"Number of Trading Days: {len(full_data)}")
print(f"Trading From {full_data[:1].index[0]} to {full_data[-1:].index[0]}")
print(f"Hypothetical Overall Gains from Buy and Hold: {round(full_data['b+h_return'][-1]*100,2)}%")
print(f"Hypothetical Overall Gains from Random Buy/Sell: {round(full_data['random_strat_return_total'][-1]*100,2)}%")
print(f"Hypothetical Overall Gains from Strategy: {round(full_data['strat_return_total'][-1]*100,2)}%")
print(f"Percent of Trading Days Invested in Strategy: {round(len(full_data[full_data['B/S'] != 'None'])/len(full_data)*100,2)}%")

# Trade breakdown (Buy/Sell/none)
print(f"Trades from Strategy Buy or Sell: {full_data['B/S'].value_counts()}")
print(f"Trades from Randomized Buy or Sell: {full_data['random_B/S'].value_counts()}")

# Average Daily Gains for Buy+Hold
print(f"Average Daily % Gain of B+H Strategy held entire period: {round(full_data[f'f_{ticker}_1p'].mean()*100,6)}%")
print(f"Average Daily % Gain of B+H Strategy On Traded Days: {round(full_data[f'f_{ticker}_1p'][full_data['B/S'] != 'None'].mean()*100,6)}%")
print(f"Percent of Buy and Hold Days in Correct Direction: {round(len(full_data.loc[full_data[f'f_{ticker}_1p'] > 0])/len(full_data)*100,2)}")


# Results from strategy Trading:
print(f"Average Daily % Gain of Strategy Longs: {round(full_data['strategy_return'][full_data['B/S'] == 'Buy'].mean()*100,6)}%")
print(f"Average Daily % Gain of Strategy Shorts: {round(full_data['strategy_return'][full_data['B/S'] == 'Sell'].mean()*100,6)}%")
print(f"Percent Predicted Move in Correct Direction: {round(len(full_data.loc[full_data['strategy_return'] > 0])/len(full_data)*100,2)}")

# Results from Random Trading:
print(f"Average Daily % Gain of Random Trading entire period: {round(full_data[f'random_strategy_return'].mean()*100,6)}%")
print(f"Average Daily % Gain of Random Trading Longs: {round(full_data['random_strategy_return'][full_data['B/S'] == 'Buy'].mean()*100,6)}%")
print(f"Average Daily % Gain of Random Trading Shorts: {round(full_data['random_strategy_return'][full_data['B/S'] == 'Sell'].mean()*100,6)}%")
print(f"Percent Random Move Chosen in Correct Direction: {round(len(full_data.loc[full_data['random_strategy_return'] > 0])/len(full_data)*100,2)}")

# Calculate Sharpe?
# What's our Risk Free Rate? Let's say 0% for now
print("Calculate Sharpe")
rfr = 0
print(f"'Sharpe' of {ticker} Buy+Hold: {np.sqrt(252)*(full_data[f'f_{ticker}_1p'].mean() - rfr) / full_data[f'f_{ticker}_1p'].std()}")
print(f"'Sharpe' of {ticker} Random Trading: {np.sqrt(252)*(full_data[f'random_strategy_return'].mean() - rfr) / full_data[f'random_strategy_return'].std()}")
print(f"'Sharpe' of {ticker} Strategy Trading: {np.sqrt(252)*(full_data[f'strategy_return'].mean() - rfr) / full_data[f'strategy_return'].std()}")



# Now let's plot our data:
plt.figure(dpi=150)
plt.xticks(rotation=45)
plt.suptitle(f"Growth of ${100} Trading {ticker} with {ticker_list} factors")
plt.title(f"Return of ${100} Invested from {full_data[:1].index[0]} to {full_data[-1:].index[0]}")
plt.plot(full_data.index, full_data["strat_return_total"]*100, label=f"{ticker} trade prediction")
plt.plot(full_data.index, full_data["b+h_return"]*100, label=f"{ticker} Buy + Hold Return")
plt.plot(full_data.index, full_data["random_strat_return_total"]*100, label=f"{ticker} Randomized Buy/Sell")
plt.legend([f"{ticker} trading strategy", f"{ticker} B+H", f"{ticker} Randomized Buy/Sell"])
plt.show()


