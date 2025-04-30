
"""
This script trains a machine learning model, evaluates it, enhances it with adversarial training, 
compares performance, and saves the final models in ONNX format for platform interoperability.
"""

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

def main(epsilon,basetraining):
    """
    Main function to set up parameters, build and train a model, 
    evaluate the model's performance, apply adversarial training, 
    and save both the base and adversarially trained models.
    """
    
    # Parameters
    filepath = "Training_Data_Normalised.csv"  # Path to the dataset file
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
    progress_bar2 = ProgressBar()  
    model_builder = ModelBuilder(input_size=input_size, learning_rate=learning_rate)  # Initialises model builder with input size
    model = model_builder.build_model()  # Constructs the model structure
    model_builder.compile_model()  # Compiles the model with specified loss and optimizer
    
    # history = model_builder.train_model(
    #     train_dataset,
    #     test_dataset,
    #     epochs=epochs,
    #     verbose=0,  # Suppresses Keras verbose output
    #     callbacks=[progress_bar],  # Integrates custom progress bar
    # )  # Trains the model with training and validation data
    
    # %%
    naive_trainer = AdversarialTrainer(model, epsilon=epsilon)  # Sets up adversarial trainer
    model, history = naive_trainer.train(
        train_dataset, epochs=epochs, callbacks=[progress_bar]
    )  # Trains the model with adversarial examples for robustness

    # Step 3: Evaluate the Base Model
    evaluator = Evaluator()  # Initialises evaluator for performance metrics
    print("\nEvaluating the base model:")
    base_metrics = evaluator.evaluate(history, model, X_test, y_test)  # Evaluates base model
    
    saver = OnnxModelSaver()  # Initialises the model saver
    # saver.save(model, "base_model")  # Saves base model as ONNX
    
    # Export network as onnx
    input_signature = [tf.TensorSpec([1,6], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, 'final_base_model_norm.onnx')
    print("saved as final_base_model_norm.onnx")
        
    # Step 4: Adversarial Training for Robustness
    print("\nStarting adversarial training with epsilon-ball robustness...")
    model_builder2 = ModelBuilder(input_size=input_size, learning_rate=learning_rate)  # Initialises model builder with input size
    model2 = model_builder2.build_model()  # Constructs the model structure
    model_builder2.compile_model()  # Compiles the model with specified loss and optimizer
    
# %%
    adversarial_trainer = AdversarialTrainer(model2, epsilon=epsilon)  # Sets up adversarial trainer
    adversarial_model, history2, adversarial_data= adversarial_trainer.train_with_adversarial_examples(
        train_dataset, epochs=epochs, alpha=1
    )  # Trains the model with adversarial examples for robustness

    # %%
    # Step 5: Evaluate the Adversarially Trained Model
    print("\nEvaluating the adversarially trained model:")
    evaluator2 = Evaluator2()
    
    adversarial_metrics = evaluator2.evaluate(history2, adversarial_model, X_test, y_test)  # Evaluates adversarial model

    # Print Metrics Comparison
    print("\n--- Comparison of Base Model and Adversarial Model ---")
    print("Base Model Metrics:", base_metrics)  # Outputs base model metrics
    print("Adversarial Model Metrics:", adversarial_metrics)  # Outputs adversarial model metrics

    # Save Models in ONNX Format

    # saver.save(adversarial_model, "adversarial_model_" + str(epsilon))  # Saves adversarially trained model as ONNX
    # Export network as onnx
    input_signature = [tf.TensorSpec([1,6], tf.float32, name='x')]
    onnx_model2, _ = tf2onnx.convert.from_keras(adversarial_model, input_signature, opset=13)
    onnx.save(onnx_model2, "final_adversarial_model_" + str(epsilon) + ".onnx")
    
    # np.savetxt("adversarial_data_"+str(epsilon)+".csv", np.squeeze(np.array(adversarial_data)), delimiter=",", fmt="%.6f")


    return base_metrics, adversarial_metrics  # Returns metrics for further analysis


# Entry point of the script
if __name__ == "__main__":
    main(0.005,True)
    # main(0.01,True)
    # main(0.02,True)
    # main(0.04,True)
    # main(0.0025,True)
# 