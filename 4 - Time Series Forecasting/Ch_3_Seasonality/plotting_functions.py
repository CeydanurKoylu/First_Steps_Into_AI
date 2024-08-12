import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.signal import periodogram

class MyClass:
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True, figsize=(11, 5))
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=16,
        titlepad=10,
    )
    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
        legend=False,
    )

    def fourier_features(index, order,freq):
        time = np.arange(len(index), dtype=np.float32) # an array of numbers that go up from 0 to observation number
        k = 2*np.pi*(1/freq)*time # an array of frequencies 
        
        features = {} # a dictionary for fourier features upto the given order
        for i in range (1,order+1):
            features.update = ({
                f"sin_{freq}_{i}": np.sin(k*i), 
                f"cos_{freq}_{i}": np.cos(k*i),
            })
        return pd.DataFrame(features, index=index)
    # the order is decided through periodograms, where the graph tells you strength of certain frequencies in a time series


    def seasonal_plot(X,y,period,freq,ax=None):
        if ax is None:
            _,ax = plt.subplots()
        palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
        ax = sns.lineplot(
            x=freq,
            y=y,
            hue=period,
            data=X,
            errorbar=('ci',False),
            ax=ax,
            palette=palette,
            legend=False,
        )
        ax.set_title(f"Seasonal Plot ({period}/{freq})")
        for line, name in zip(ax.lines, X[period].unique()):
            y_ = line.get_ydata()[-1]
            ax.annotate(
                name,
                xy=(1, y_),
                xytext=(6, 0),
                color=line.get_color(),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                size=14,
                va="center",
            )
        return ax
    def plot_periodogram(ts, detrend='linear', ax = None):
        fs = pd.Timedelta("365D") / pd.Timedelta("1D")
        frequencies, spectrum = periodogram(ts, fs = fs, detrend = detrend, window = "boxcar", scaling = "spectrum")
        if ax is None:
            _, ax = plt.subplots()
        ax.step(frequencies, spectrum, color="purple")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
        ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)",
                "Weekly (52)",
                "Semiweekly (104)",
            ],
            rotation=30,
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Variance")
        ax.set_title("Periodogram")
        return ax