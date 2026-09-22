from tqdm import tqdm
import tensorflow as tf


class ProgressBar(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to display a tqdm progress bar during training.
    Updates the progress bar after each epoch and displays loss and metric values.
    """

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training to initialise the tqdm progress bar.

        Args:
            logs (dict, optional): Contains information about the training process at the start.
                                   Defaults to None.
        """
        self.epochs = self.params["epochs"]
        self.pbar = tqdm(total=self.epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to update the progress bar and display current metrics.

        Args:
            epoch (int): The index of the completed epoch.
            logs (dict, optional): A dictionary containing the current training metrics.
                                   Defaults to None.
        """
        # Update the tqdm bar by one epoch
        self.pbar.update(1)

        # Update the progress bar postfix with training metrics
        if logs is not None:
            self.pbar.set_postfix(
                {
                    "loss": logs.get("loss", "N/A"),
                    "val_loss": logs.get("val_loss", "N/A"),
                    "rmse": logs.get("root_mean_squared_error", "N/A"),
                    "val_rmse": logs.get("val_root_mean_squared_error", "N/A"),
                }
            )
        else:
            self.pbar.set_postfix(
                {"loss": "N/A", "val_loss": "N/A", "rmse": "N/A", "val_rmse": "N/A"}
            )

    def on_train_end(self, logs=None):
        """
        Called at the end of training to close the tqdm progress bar.

        Args:
            logs (dict, optional): Contains information about the training process at the end.
                                   Defaults to None.
        """
        self.pbar.close()
