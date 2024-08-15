import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps



def create_color_categorical(num_colors):
    color_mag = np.arange(start=0, stop=1, step=1.0 / (num_colors))
    cmap = plt.get_cmap('rainbow')
    colors = cmap(color_mag)
    return colors


def play_add_kde_plot():
    num_data = 50
    x_data = np.random.normal(size=num_data)
    y_data = np.random.normal(size=num_data)

    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data)
    plt.show()


def add_kde_to_plot(ax):
    # data: n lists of 2d data
    """
    Create scatter plot with marginal KDE plots
    """

    # Set up 4 subplots and aspect ratios as axis objects using GridSpec:
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    fig = plt.figure() # Set background canvas colour to White instead of grey default
    fig.patch.set_facecolor('white')

    ax = plt.subplot(gs[0,1]) # Instantiate scatter plot area and axis range
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_xlabel(headers[0], fontsize = 14)
    # ax.set_ylabel(headers[1], fontsize = 14)
    ax.yaxis.labelpad = 10 # adjust space between x and y axes and their labels if needed

    axl = plt.subplot(gs[0,0], sharey=ax) # Instantiate left KDE plot area
    axl.get_xaxis().set_visible(False) # Hide tick marks and spines
    axl.get_yaxis().set_visible(False)
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

    axb = plt.subplot(gs[1,1], sharex=ax) # Instantiate bottom KDE plot area
    axb.get_xaxis().set_visible(False) # Hide tick marks and spines
    axb.get_yaxis().set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_visible(False)

    axc = plt.subplot(gs[1,0]) # Instantiate legend plot area
    axc.axis('off') # Hide tick marks and spines

    # For each category in the list...
    for n in range(0, len(category_list)):
        x = data_list[n][:, 0]
        y = data_list[n][:, 1]

        color = cl[n].copy()
        color[3] = 0.3
        # Plot data for each categorical variable as scatter and marginal KDE plots:
        ax.scatter(x,y, color=color, s=20, edgecolor= cl[n], label = category_list[n])

        kde = stats.gaussian_kde(x)
        xx = np.linspace(xmin, xmax, 1000)
        axb.plot(xx, kde(xx), color=cl[n])

        kde = stats.gaussian_kde(y)
        yy = np.linspace(ymin, ymax, 1000)
        axl.plot(kde(yy), yy, color=cl[n])

    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label
    handles, labels = ax.get_legend_handles_labels()
    axc.legend(handles, labels, scatterpoints = 1, loc = 'center', fontsize = 12)

    plt.show()

def marginal_kde(data_list):
    # data: n lists of 2d data
    """
    Create scatter plot with marginal KDE plots
    """
    # first create dataframe from datalist
    # data_dict = {}
    # for i in range(len(data_list)):
    #     data_dict[i] = data_list[i]
    # df = pd.DataFrame(data_dict)
    cl = create_color_categorical(len(data_list))

    all_data = np.concatenate(data_list, axis=0)

    # headers = list(df.columns) # Extract list of column headers
    # Find min and max values for all x (= col [0]) and y (= col [1]) in dataframe:
    xmin, xmax = all_data.min(axis=0)[0], all_data.max(axis=0)[0]
    ymin, ymax = all_data.min(axis=0)[1], all_data.max(axis=0)[1]
    # Create a list of all unique categories which occur in the right hand column (ie index '2'):
    category_list = list(np.arange(len(data_list)))

    # Set up 4 subplots and aspect ratios as axis objects using GridSpec:
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    fig = plt.figure() # Set background canvas colour to White instead of grey default
    fig.patch.set_facecolor('white')

    ax = plt.subplot(gs[0,1]) # Instantiate scatter plot area and axis range
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_xlabel(headers[0], fontsize = 14)
    # ax.set_ylabel(headers[1], fontsize = 14)
    ax.yaxis.labelpad = 10 # adjust space between x and y axes and their labels if needed

    axl = plt.subplot(gs[0,0], sharey=ax) # Instantiate left KDE plot area
    axl.get_xaxis().set_visible(False) # Hide tick marks and spines
    axl.get_yaxis().set_visible(False)
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

    axb = plt.subplot(gs[1,1], sharex=ax) # Instantiate bottom KDE plot area
    axb.get_xaxis().set_visible(False) # Hide tick marks and spines
    axb.get_yaxis().set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_visible(False)

    axc = plt.subplot(gs[1,0]) # Instantiate legend plot area
    axc.axis('off') # Hide tick marks and spines

    # For each category in the list...
    for n in range(0, len(category_list)):
        x = data_list[n][:, 0]
        y = data_list[n][:, 1]

        color = cl[n].copy()
        color[3] = 0.3
        # Plot data for each categorical variable as scatter and marginal KDE plots:
        ax.scatter(x,y, color=color, s=20, edgecolor= cl[n], label = category_list[n])

        kde = stats.gaussian_kde(x)
        xx = np.linspace(xmin, xmax, 1000)
        axb.plot(xx, kde(xx), color=cl[n])

        kde = stats.gaussian_kde(y)
        yy = np.linspace(ymin, ymax, 1000)
        axl.plot(kde(yy), yy, color=cl[n])

    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label
    handles, labels = ax.get_legend_handles_labels()
    axc.legend(handles, labels, scatterpoints = 1, loc = 'center', fontsize = 12)

    plt.show()

if __name__=="__main__":
    # data_list = []
    # min_bound = -1.0
    # max_bound = 1.0
    #
    # for i in range(3):
    #     center = np.random.uniform(low=min_bound, high=max_bound, size=2)
    #     distribution = np.random.uniform(low=min_bound, high=max_bound, size=2) / 4.0
    #     num_data = np.random.randint(low=50, high=100)
    #     data_i = np.random.multivariate_normal(mean=center, cov=np.diag(distribution), size=(num_data))
    #     data_list.append(data_i)
    #
    # marginal_kde(data_list)

    play_add_kde_plot()