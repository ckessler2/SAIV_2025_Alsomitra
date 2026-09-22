
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
    """
    A utility class for evaluating and visualising the performance of machine learning models.
    Includes functions for plotting loss curves, predictions vs true values, and calculating metrics.
    """

    @staticmethod
    def plot_loss(history):
        """
        Plots the training and validation loss over epochs.

        Args:
            history (tf.keras.callbacks.History): The training history object containing loss values.
        """
        import os
        os.environ["XGD_SESSION_TYPE"] = "xcb"
        plt.rcParams["figure.dpi"] = 150
        plt.rc("font", family="sans-serif")
        plt.rcParams["font.family"] = "DejaVu Sans"


        # Extract training and validation loss
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)


        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, label="Training Loss", linewidth=2)
        plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        plt.title("Training and Validation Loss (RMSE)")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (Log Scale)")
        plt.yscale("log")
        plt.legend(loc="upper right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.ylim(1e-4,1)
        plt.show()

    @staticmethod
    def make_predictions(model, X_test):
        """
        Generates predictions for the test set using the provided model.

        Args:
            model (tf.keras.Model): The trained model to use for predictions.
            X_test (np.ndarray or pd.DataFrame): The test dataset features.

        Returns:
            np.ndarray: The predicted values for the test set.
        """
        return model.predict(X_test)

    @staticmethod
    def plot_predictions_vs_true(y_test, predictions):
        """
        Plots predicted values against true values for the test set.

        Args:
            y_test (np.ndarray or pd.Series): The true target values for the test set.
            predictions (np.ndarray): The predicted values from the model.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.6, s=15)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--", lw=2)
        plt.title("Predictions vs True Values")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.show()

    @staticmethod
    def calculate_metrics(y_test, predictions):
        """
        Calculates key regression metrics, including RMSE, MAE, and R^2.

        Args:
            y_test (np.ndarray or pd.Series): The true target values for the test set.
            predictions (np.ndarray): The predicted values from the model.

        Returns:
            dict: A dictionary containing RMSE, MAE, and R^2 values.
        """
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    @staticmethod
    def evaluate(history, model, X_test, y_test):
        """
        Provides a comprehensive evaluation report, including loss plots,
        predictions vs true values plots, and regression metrics.

        Args:
            history (tf.keras.callbacks.History): The training history object containing loss values.
            model (tf.keras.Model): The trained model to evaluate.
            X_test (np.ndarray or pd.DataFrame): The test dataset features.
            y_test (np.ndarray or pd.Series): The true target values for the test set.

        Returns:
            dict: A dictionary containing RMSE, MAE, and R^2 values.
        """
        
        # Plot training loss
        Evaluator.plot_loss(history)
        


        # Make predictions
        predictions = Evaluator.make_predictions(model, X_test)
        


        # Plot predictions vs true values
        Evaluator.plot_predictions_vs_true(y_test, predictions)
        
        # Calculate and return metrics
        metrics = Evaluator.calculate_metrics(y_test, predictions)

        
        return metrics


