
"""
This script trains a machine learning model, evaluates it, enhances it with adversarial training, 
compares performance, and saves the final models in ONNX format for platform interoperability.
"""

import argparse

from data_handler import DataHandler  # Handles data loading and preprocessing
from model_builder import ModelBuilder  # Constructs and compiles the machine learning model
from evaluator import Evaluator  # Evaluates model performance metrics
from progress_bar import ProgressBar  # Custom progress bar for tracking training
from onnx_saver import OnnxModelSaver  # Saves models in ONNX format
from adversarial_trainer import AdversarialTrainer  # Enhances model robustness with adversarial training
from evaluator2 import Evaluator2
import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def main(epsilon, mode="both"):
    """
    Main function to set up parameters, build and train a model, 
    evaluate the model's performance, apply adversarial training, 
    and save both the base and adversarially trained models.
    """
    
    # Parameters
    filepath = DATA_DIR / "Training_Data_Normalised.csv"  # Path to the dataset file
    target_column = "target"  # Column name for the labels in the dataset
    input_size = 6  # Number of input features for the model
    batch_size = 50  # Batch size for training
    epochs = 1000  # Number of epochs for training
    learning_rate = 0.06
    # epsilon = 0.000  # Epsilon for adversarial robustness (small perturbations)

    # Step 1: Data Handling
    data_handler = DataHandler(filepath)  # Initialises the data handler with the file path
    data_handler.load_data()  # Loads data from the specified file
    train_dataset, test_dataset, X_test, y_test = data_handler.prepare_datasets(
        target_column=target_column, batch_size=batch_size
    )  # Prepares training and test datasets, along with test data for evaluation

    # Step 2: Model Building
    progress_bar = ProgressBar()  # Custom progress bar for training
    base_metrics = None
    adversarial_metrics = None

    if mode in {"baseline", "both"}:
        model_builder = ModelBuilder(input_size=input_size, learning_rate=learning_rate)  # Initialises model builder with input size
        model = model_builder.build_model()  # Constructs the model structure
        model_builder.compile_model()  # Compiles the model with specified loss and optimizer

        naive_trainer = AdversarialTrainer(model, epsilon=epsilon)  # Sets up trainer
        model, history = naive_trainer.train(
            train_dataset, epochs=epochs, callbacks=[progress_bar]
        )  # Trains the model without adversarial examples

        evaluator = Evaluator()  # Initialises evaluator for performance metrics
        print("\nEvaluating the base model:")
        base_metrics = evaluator.evaluate(history, model, X_test, y_test)

        input_signature = [tf.TensorSpec([1,6], tf.float32, name='x')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        base_output = MODELS_DIR / 'final_base_model_norm.onnx'
        onnx.save(onnx_model, base_output)
        print(f"saved as {base_output}")

    if mode in {"adversarial", "both"}:
        print("\nStarting adversarial training with epsilon-ball robustness...")
        model_builder2 = ModelBuilder(input_size=input_size, learning_rate=learning_rate)  # Initialises model builder with input size
        model2 = model_builder2.build_model()  # Constructs the model structure
        model_builder2.compile_model()  # Compiles the model with specified loss and optimizer

        adversarial_trainer = AdversarialTrainer(model2, epsilon=epsilon)  # Sets up adversarial trainer
        adversarial_model, history2, adversarial_data = adversarial_trainer.train_with_adversarial_examples(
            train_dataset, epochs=epochs, alpha=1
        )  # Trains the model with adversarial examples for robustness

        print("\nEvaluating the adversarially trained model:")
        evaluator2 = Evaluator2()
        adversarial_metrics = evaluator2.evaluate(history2, adversarial_model, X_test, y_test)

        input_signature = [tf.TensorSpec([1,6], tf.float32, name='x')]
        onnx_model2, _ = tf2onnx.convert.from_keras(adversarial_model, input_signature, opset=13)
        adversarial_output = MODELS_DIR / ("final_adversarial_model_" + str(epsilon) + ".onnx")
        onnx.save(onnx_model2, adversarial_output)
        print(f"saved as {adversarial_output}")

    if mode == "both":
        print("\n--- Comparison of Base Model and Adversarial Model ---")
        print("Base Model Metrics:", base_metrics)
        print("Adversarial Model Metrics:", adversarial_metrics)

    return base_metrics, adversarial_metrics  # Returns metrics for further analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline and adversarial Alsomitra controllers.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "adversarial", "both"],
        default="both",
        help="Which model(s) to train.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.005,
        help="Epsilon-ball radius used for adversarial training.",
    )
    return parser.parse_args()


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args.epsilon, args.mode)
