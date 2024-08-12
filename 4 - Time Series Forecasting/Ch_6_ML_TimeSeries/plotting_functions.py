import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.signal import periodogram
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

class Plots:
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True, figsize=(11, 4))
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
    )

    # Plots the wanted lag plot of y variable:
    def lagplot(self,x, y=None, lag=1, standardize=False, ax=None, **kwargs):
        from matplotlib.offsetbox import AnchoredText
        x_ = x.shift(lag)
        if standardize: # Normalize the X variable
            x_ = (x_ - x_.mean()) / x_.std()
        if y is not None: #Normalizes the y variable
            y_ = (y - y.mean()) / y.std() if standardize else y
        else:
            y_ = x
        corr = y_.corr(x_)
        if ax is None:
            fig, ax = plt.subplots()
        scatter_kws = dict(
            alpha=0.75,
            s=3,
        )
        line_kws = dict(color='C3', )
        ax = sns.regplot(x=x_,
                        y=y_,
                        scatter_kws=scatter_kws,
                        line_kws=line_kws,
                        lowess=True,
                        ax=ax,
                        **kwargs)
        at = AnchoredText(
            f"{corr:.2f}",
            prop=dict(size="large"),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("square, pad=0.0")
        ax.add_artist(at)
        ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
        return ax
    
    # This function creates a figure that contains multiple lag plots to compare them with each other:  
    def plot_lags(self, x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
        import math
        kwargs.setdefault('nrows', nrows)
        kwargs.setdefault('ncols', math.ceil(lags / nrows))
        kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
        fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
        for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
            if k + 1 <= lags:
                ax = self.lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
                ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel="")
            else:
                ax.axis('off')
        plt.setp(axs[-1, :], xlabel=x.name)
        plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
        fig.tight_layout(w_pad=0.1, h_pad=0.1)
        return fig
    
    def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
        palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
        if palette_kwargs is not None:
            palette_kwargs_.update(palette_kwargs)
        palette = sns.color_palette(**palette_kwargs_)
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_prop_cycle(plt.cycler('color', palette))
        for date, preds in y[::every].iterrows():
            preds.index = pd.period_range(start=date, periods=len(preds))
            preds.plot(ax=ax)
        return ax