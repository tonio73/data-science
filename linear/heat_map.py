from matplotlib import pyplot as plt
import numpy as np

def plot_heat_map(data, classes,
                  normalize=False,
                  title = None, 
                  xlabel = None,
                  yLabel = None,
                  ax = None,
                  cmap=plt.cm.Blues,
                  fmt='.2f'):
    """
    Plot a heatmap
    Normalization can be applied by setting `normalize=True`.
    """
    
    # Only use the labels that appear in the data
    if normalize:
        data = data.astype('float') / data.sum(axis=1)[:, np.newaxis]

    if not ax:
        fig, ax = plt.subplots(1, figsize=(5,4), subplot_kw = {'aspect':'auto'})
     
    # Plot map and legend color bar
    im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks
    ax.set_xticks(np.arange(data.shape[1]), minor=False)
    ax.set_xticklabels(classes)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_yticks(np.arange(data.shape[0]), minor=False)
    ax.set_yticklabels(classes)
    if yLabel:
        ax.set_ylabel(yLabel)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    if title:
        ax.set_title(title)

    # Workaround to center ticks and cells 
    # (https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#using-the-helper-function-code-style)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True) 
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = data.max() / 2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black")
    ax.figure.tight_layout()
    return ax
