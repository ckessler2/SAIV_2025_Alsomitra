
import tensorflow as tf

import tensorflow as tf



class ModelBuilder:
    """
    A class for constructing, compiling, and training a TensorFlow Keras model.
    """

    def __init__(self, input_size, learning_rate=0.1):
        """
        Initialises the ModelBuilder with the model's input size and training hyperparameters.

        Args:
            input_size (int): The size of the input features for the model.
            learning_rate (float): The learning rate for the Adam optimiser. Defaults to 0.001.
        """
        self.input_size = input_size
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.accuracy_function = tf.keras.metrics.RootMeanSquaredError()

    def build_model(self):
        """
        Builds a feedforward neural network with an input layer, two hidden layers,
        and an output layer. The layers use the sigmoid activation function.

        Returns:
            tf.keras.Model: The constructed Keras sequential model.
        """
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.input_size,), name="input_features"),
                tf.keras.layers.Dense(
                    6,
                    activation="relu",
                    kernel_initializer=initializer,
                    name="dense_1",
                ),
                tf.keras.layers.Dense(
                    4,
                    activation="relu",
                    kernel_initializer=initializer,
                    name="dense_2",
                ),
                # tf.keras.layers.Dense(
                #     4,
                #     activation="relu",
                #     kernel_initializer=initializer,
                #     name="dense_3",
                # ),
                tf.keras.layers.Dense(
                    1,
                    activation="relu",
                    kernel_initializer=initializer,
                    name="output_layer",
                ),
            ]
        )
        self.model.output_names=["output"]
        return self.model

    def compile_model(self):
        """
        Compiles the model with the Adam optimiser, Mean Squared Error loss function,
        and Root Mean Squared Error as the evaluation metric.
        """
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=[self.accuracy_function],
        )

    def train_model(
        self, train_dataset, test_dataset, epochs=1000, verbose=1, callbacks=None
    ):
        """
        Trains the model using the provided training and testing datasets.

        Args:
            train_dataset (tf.data.Dataset): The dataset for training the model.
            test_dataset (tf.data.Dataset): The dataset for validating the model during training.
            epochs (int): The number of epochs to train the model. Defaults to 1000.
            verbose (int): Verbosity level for training logs. Defaults to 1.
                - 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks (list): A list of callback instances for monitoring or customising
                the training process. Defaults to None.

        Returns:
            tf.keras.callbacks.History: The training history object containing metrics and loss values.
        """
        history = self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )
        return history

