
import matplotlib.pyplot as plt


def plot_training_curves(df, metric_col='accuracy'):
    """
    Plot training and validation loss and accuracy over 
    epochs. 
    
    Generates a two-panel figure showing loss and micro-accuracy 
    for training and validation sets across epochs. The epoch with the
    minimum validation loss is marked with a vertical dashed line.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame ordered by epoch containing columns for ``train loss``, 
        ``val loss``, ``train accuracy``, and ``val accuracy``.
    metric_col : str, default=accuracy
        Name of metric used in plots in addition to loss. Must be either
        `accuracy` or `dice score`

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the loss and accuracy subplots.
    """

    # setup figure and axes for two subplots
    fig, ax = plt.subplots(ncols=2, figsize=(10,6))

    # create generator for epochs
    epochs = range(1, len(df)+1)

    # plot loss subplot...
    ax[0].plot(epochs, df['train loss'], lw=0.75, label='Train',)
    ax[0].plot(epochs, df['val loss'], lw=0.75, label='Validation')
    ax[0].set_ylabel('Loss')

    # plot second metric subplot...
    if metric_col == 'accuracy':
        ax[1].plot(epochs, df['train accuracy'], lw=0.75, label='Train')
        ax[1].plot(epochs, df['val accuracy'], lw=0.75, label='Validation')
        ax[1].set_ylabel('Accuracy')

    if metric_col == 'dice':
        ax[1].plot(epochs, df['train dice score'], lw=0.75, label='Train')
        ax[1].plot(epochs, df['val dice score'], lw=0.75, label='Validation')
        ax[1].set_ylabel('Dice Score')

    # plot selected model at correct epoch
    for axes in ax:
        axes.axvline(x=df['val loss'].values.argmin()+1, linestyle='--', color='k', label='Selected')
        axes.legend(frameon=False)
        axes.set_xticks(epochs)
        axes.set_xticklabels([str(x) if x%5==0 else '' for x in epochs])
        axes.set_xlabel('Epochs')

    plt.suptitle(f"Training and Validation Curves", y=0.92)

    return fig