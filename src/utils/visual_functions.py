import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from statsmodels.graphics.api import qqplot#
import math
import palettable
colors = [plt.cm.Blues(0.6), plt.cm.Reds(0.4), '#99ccff', '#ffcc99', plt.cm.Greys(0.6), plt.cm.Oranges(0.8), plt.cm.Greens(0.6), plt.cm.Purples(0.8)]
SPINE_COLOR="gray"
nice_fonts = {
        # Use LaTeX to write all text
        "font.family": "serif",
        # Always save as 'tight'
        "savefig.bbox" : "tight",
        "savefig.pad_inches" : 0.05,
        "ytick.right" : True,
        "font.serif" : "Times New Roman",
        "mathtext.fontset" : "dejavuserif",
        "axes.labelsize": 15,
        "font.size": 15,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Set line widths
        "axes.linewidth" : 0.5,
        "grid.linewidth" : 0.5,
        "lines.linewidth" : 1.,
        # Remove legend frame
        "legend.frameon" :True
}
matplotlib.rcParams.update(nice_fonts)
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import arviz as az
az.style.use(["science", "grid"])
import pandas as pd

nice_fonts = {
        # Use LaTeX to write all text
        "font.family": "serif",
        # Always save as 'tight'
        "savefig.bbox" : "tight",
        "savefig.pad_inches" : 0.05,
        "ytick.right" : True,
        "font.serif" : "Times New Roman",
        "mathtext.fontset" : "dejavuserif",
        "axes.labelsize": 15,
        "font.size": 15,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Set line widths
        "axes.linewidth" : 0.5,
        "grid.linewidth" : 0.5,
        "lines.linewidth" : 1.,
        # Remove legend frame
        "legend.frameon" : False
}
matplotlib.rcParams.update(nice_fonts)

def set_figure_size(fig_width=None, fig_height=None, columns=2):
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    return (fig_width, fig_height)


def format_axes(ax):
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax

def figure(fig_width=None, fig_height=None, columns=2):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig_width, fig_height =set_figure_size(fig_width, fig_height, columns)
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig



def legend(ax, ncol=3, loc=9, pos=(0.5, -0.1)):
    leg=ax.legend(loc=loc, bbox_to_anchor=pos, ncol=ncol)
    return leg

def savefig(filename, leg=None, format='.eps', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()

def get_label_distribution(ax, y, title=None, max_value=1):
    label_, counts_ = np.unique(y, return_counts=True)
    postion = np.arange(len(label_))
    plt.bar(postion, np.round(counts_*100/counts_.sum(0)), align='center', color='#a9a9a9')
    plt.xticks(postion, ["OFF", "ON"])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
    ax.set_title("{}".format(title))
    #ax.set_ylabel("");
    plt.yticks([])
    plt.tight_layout()
    for p in ax.patches:
        #ax.annotate('{:.0%}'.format(height), (p.get_x()+.15*width, p.get_y() + height + 0.01))
        ax.annotate("{}$\%$".format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    return ax

def plot_xy(ax, mu, true, q_pred, min_lim, max_lim):
    x  = np.arange(len(true))
    
    h1 = ax.plot(x, true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(x, mu,   '.-',  c="#1f77b4", alpha=0.8)
    
    ax.set_ylabel('Power $(KW)$')
    
    N = q_pred.shape[0]
    alpha = np.linspace(0.1, 0.9, N//2).tolist() + np.linspace(0.9, 0.2, 1+N//2).tolist()
    
    
    for i in range(N):
        y1 = q_pred[i, :]
        y2 = q_pred[-1-i, :]
        h3 = ax.fill_between(x, y1.flatten(), y2.flatten(), color="lightsteelblue", alpha=alpha[1])
    ax.autoscale(tight=True)
    ax.set_ylabel('Power $(KW)$')
    ax.autoscale(tight=True)
    ax.set_ylim(min_lim, max_lim)
    lines =[h1[0], h2[0], h3]
    label = ["True", "Pred mean", "$95\%$ Interval"]
    return ax, lines, label


def plot_intervals_ordered(ax, mu, std, true):
    
    
    order = np.argsort(true.flatten())
    mu, true, std = mu[order], true[order], std[order]
    xs = np.arange(len(order))
    
    _ = ax.errorbar(
        xs,
        mu,
        std,
        fmt="o",
        ls="none",
        linewidth=1.5,
        c="#1f77b4",
        alpha=0.5)
    h1 = ax.plot(xs, mu, "o", c="#1f77b4")
    h2 = ax.plot(xs, true, ".", linewidth=2.0, c="#ff7f0e")
    ax.legend([h1[0], h2[0]], ["Predicted Values", "Observed Values"], loc=4)
    
    intervals_lower_upper = [mu-std, mu+std]
    lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    _ = ax.set_ylim(lims_ext)
    _ = ax.set_xlabel("Index (Ordered by Observed Value)")
    _ = ax.set_ylabel("Predicted Values and Intervals")
    _ = ax.set_aspect("auto", "box")
    return ax



def plot_boxplot_per_encoder(ax, dict_results, metric, label):
    
    df = pd.DataFrame()
    
    encoders = list(dict_results.keys())
    for enc in encoders:
        df[enc]=dict_results[enc][metric]
    ax.boxplot(df, showmeans=True, manage_ticks=True, autorange=True)
    plt.xticks(range(1, len(encoders )+1), encoders , rotation=0, fontsize=12);
    ax.set_ylabel(label)
    ax.autoscale(tight=True)
    
    return ax



def plot_prediction_with_pi(ax, true, mu, q_pred, date, true_max=None):
  

    h1 = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(date, mu,   '--',  c="#1f77b4", alpha=0.8)
    ax.set_ylabel('Power $(W)$')
    N = q_pred.shape[0]
    alpha = np.linspace(0.1, 0.9, N//2).tolist() + np.linspace(0.9, 0.2, 1+N//2).tolist()
    
    for i in range(N):
        y1 = q_pred[i, :]
        y2 = q_pred[-1-i, :]
        h3 = ax.fill_between(date, y1.flatten(), y2.flatten(), color="lightsteelblue", alpha=alpha[1])

    #ax.plot(date, q_pred[-1,:], 'k--')
    #ax.plot(date, q_pred[0,:], 'k--')
    ax.autoscale(tight=True)
    if true_max is None:
        true_max = true.max()+1000
        
    ax.set_ylim(true.min(), true_max)
    
    locator = mdates.HourLocator()
    ax.xaxis.set_major_locator(locator)
    
    hfmt = mdates.DateFormatter('%m-%d %H')
    ax.xaxis.set_major_formatter(hfmt)
    
    lines =[h1[0], h2[0], h3]
    label = ["True", "Pred median", "$95\%$ Interval"]
    ax.legend(lines, label, loc=0)
    
    return ax


def plot_prediction(ax, true, mu, date, true_max=None):
  

    h1 = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(date, mu,   '--',  c="#1f77b4", alpha=0.8)
    ax.set_ylabel('Power $(W)$')
    ax.autoscale(tight=True)
    if true_max is None:
        true_max = true.max()+1000
        
    ax.set_ylim(true.min(), true_max)

    locator = mdates.HourLocator()
    ax.xaxis.set_major_locator(locator)
    
    hfmt = mdates.DateFormatter('%m-%d %H')
    ax.xaxis.set_major_formatter(hfmt)
    
    lines =[h1[0], h2[0]]
    label = ["True", "Pred median"]
    ax.legend(lines, label, loc=0)
    
    return ax