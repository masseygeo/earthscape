


class EarlyStopping:
    """
    Early stopping utility based on validation loss.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without improvement allowed before stopping.
    min_delta : float
        Minimum decrease in validation loss required to qualify as an improvement.
    warmup_epochs : int
        Number of initial epochs during which early stopping is disabled.

    Attributes
    ----------
    best_loss : float or None
        Best observed validation loss.
    bad_epochs : int
        Number of consecutive epochs without improvement.
    """
    def __init__(self, patience, min_delta, warmup_epochs):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_loss = None
        self.bad_epochs = 0


    def step(self, val_loss, epoch):
        """
        Evaluate early stopping criteria for a given epoch.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.
        epoch : int
            Current epoch index.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """

        # update best loss on first call & continue training...
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        # continue training for at least warmup_epochs epochs...
        if epoch <= self.warmup_epochs:
            return False

        # check validation loss & improvement...
        improved = val_loss < (self.best_loss - self.min_delta)

        # continue training if improved OR start counting number of epochs of no improvement...
        if improved:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        # return False to continue training / True to stop
        return self.bad_epochs > self.patience