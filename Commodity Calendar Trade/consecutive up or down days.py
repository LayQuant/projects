"""

Given a stock or index, I want to know the statistics for the number of up or down days in a row.
Since inception?

If we're making a SPX or VIX filter to selectively trade Leveraged ETFs,
it would be good to know how often they'd need to be traded

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

ticker = "^VIX"


df = yf.download(ticker, start="1900-01-01", end="2030-01-01")
df.columns = df.columns.droplevel(1)

# 2. Identify if the VIX went up (1) or down (0)
df['up'] = (df['Close'] >= df['Close'].shift(1)).astype(int)
df['down'] = (df['Close'] < df['Close'].shift(1)).astype(int)


# 3. Create a unique identifier for each streak
# A new streak starts whenever 'up' changes from 1 to 0 or 0 to 1
df['streak_id_up'] = (df['up'] != df['up'].shift(1)).cumsum()
df['streak_id_down'] = (df['down'] != df['down'].shift(1)).cumsum()

# 4. Filter for only 'up' days and count the length of each streak
up_streaks = df[df['up'] == 1].groupby('streak_id_up')['up'].count()
down_streaks = df[df['down'] == 1].groupby('streak_id_down')['down'].count()


# 5. Calculate statistics Up:
average_up_streak = up_streaks.mean()
max_up_streak = up_streaks.max()
std_up_streak = up_streaks.std()
count_up_streak = len(up_streaks)
up_streak_days = up_streaks.sum()

#  And Down:
average_down_streak = down_streaks.mean()
max_down_streak = down_streaks.max()
std_down_streak = down_streaks.std()
count_down_streak = len(down_streaks)
down_streak_days = down_streaks.sum()


print(f"{ticker}")
print(f"From {df.index[0]} to {df.index[-1]}")
print("\nUp Days:")
print(f"Average {ticker} consecutive up days: {average_up_streak:.2f}")
print(f"Standard Deviation {ticker} consecutive up days: {std_up_streak:.2f}")
print(f"Maximum {ticker} consecutive up days: {max_up_streak}")
print(f"Number of {ticker} Up Streaks: {count_up_streak}")
print(f"Number of {ticker} trading days in Up Streaks: {up_streak_days}")
print("\nDown Days:")
print(f"Average {ticker} consecutive down days: {average_down_streak:.2f}")
print(f"Standard Deviation {ticker} consecutive down days: {std_down_streak:.2f}")
print(f"Maximum {ticker} consecutive down days: {max_down_streak}")
print(f"Number of {ticker} Down Streaks: {count_down_streak}")
print(f"Number of {ticker} trading days in Down Streaks: {down_streak_days}")

# Now Let's look at the Up and Down histograms to see the distributions:

# To plot a histogram for a specific column (Series)
plt.title(f"{ticker} Up Day Streak Historgram")
plt.ylabel("Instance Count")
plt.xlabel("Days Up Consecutively")
up_streaks.hist(bins=max_up_streak)
# Display the plot
plt.show()

plt.title(f"{ticker} Down Day Streak Historgram")
plt.ylabel("Instance Count")
plt.xlabel("Days Down Consecutively")
down_streaks.hist(bins=max_down_streak)
# Display the plot
plt.show()