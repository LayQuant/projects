"""
Want to see what St Louis Fed Data says on Uranium?
Can check monthly price growth:
"""


import pandas as pd
import numpy as np

data = pd.read_csv("./PURANUSDM.csv")
data["return_1m"] = (data["PURANUSDM"] - data["PURANUSDM"].shift(1)) / data["PURANUSDM"].shift(1)
data["date"] = pd.to_datetime(data["observation_date"])
data["month"] = data["date"].dt.month
for i in data["month"].unique():
    temp = data[data["month"] == i]
    print(i, round(temp["return_1m"].sum(),6), round(temp["return_1m"].mean(),6), round(temp["return_1m"].std(),6),
          )

# It seems that september is good for Uranium prices?!?
# And we can use UX to trade it.
print(data[data["month"] == 5].to_string())
# Only 3 of 11 Septembers were negative...
