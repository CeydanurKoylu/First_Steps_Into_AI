import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

df = pd.read_csv("./book_sales.csv", 
                 index_col = 'Date', 
                 parse_dates = ['Date'],
                 ).drop('Paperback', axis=1)

df["Time"] = np.arange(len(df.index))

df["Lag_1"] = df["Hardcover"].shift(1)

df = df.reindex(columns = ["Hardcover", "Lag_1","Time"])

import seaborn as sns
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
plt.show()
