 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

os.environ["APPDATA"] = ""
from pandasgui import show  # Optional for GUI display


class DataHandler:
    """
    A class to handle loading, processing, and preparing datasets for machine learning tasks.
    """

    def __init__(self, filepath):
        """
        Initialises the DataHandler with the file path to the dataset.

        Args:
            filepath (str): The file path to the CSV file containing the dataset.
        """
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """
        Loads the dataset from the specified file path and assigns appropriate column names.
        Additionally, applies min-max scaling to the target column.

        Returns:
            pd.DataFrame: The loaded and processed dataframe.
        """
        self.df = pd.read_csv(self.filepath, delim_whitespace=False, header=None)
        self.df.columns = [
            "dv_xpdt",
            "dv_ypdt",
            "domegadt",
            "dthetadt",
            "dx_dt",
            "dy_dt",
            "target",
        ]
        # self.df[['dx_dt', 'dy_dt']] = 0
        # Uncomment the next line to visualise the dataframe in a GUI
        # show(self.df)
        # self.min_max_scale_target()
        return self.df

    def min_max_scale_target(self, target_column="target"):
        """
        Applies min-max scaling to the specified target column within the dataframe.

        Args:
            target_column (str): The name of the column to scale. Defaults to 'target'.

        Returns:
            pd.DataFrame: The dataframe with the scaled target column.

        Raises:
            ValueError: If the target column is not found in the dataframe.
        """
        # if target_column in self.df.columns:
        #     min_val = self.df[target_column].min()
        #     max_val = self.df[target_column].max()
        #     self.df[target_column] = (self.df[target_column] - min_val) / (
        #         max_val - min_val
        #     )
        #     print(
        #         f"Min-max scaling applied to {target_column}: min={min_val}, max={max_val}"
        #     )
        # else:
        #     print(f"Column '{target_column}' not found in the DataFrame.")
        return self.df

    def prepare_datasets(
        self, target_column="target", test_size=0.1, random_state=20, batch_size=32
    ):
        """
        Prepares TensorFlow datasets for training and testing.

        Args:
            target_column (str): The name of the target column. Defaults to 'target'.
            test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.1.
            random_state (int): The seed for random operations to ensure reproducibility. Defaults to 20.
            batch_size (int): The batch size for the TensorFlow datasets. Defaults to 32.

        Returns:
            tuple: A tuple containing:
                - train_dataset (tf.data.Dataset): The training dataset.
                - test_dataset (tf.data.Dataset): The testing dataset.
                - X_test (pd.DataFrame): The feature set for the test dataset.
                - y_test (pd.Series): The target values for the test dataset.
        """
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=1024)
            .batch(batch_size)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            batch_size
        )

        return train_dataset, test_dataset, X_test, y_test

