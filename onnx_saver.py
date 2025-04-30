import tensorflow as tf
import tf2onnx
import onnx
import os


class OnnxModelSaver:
    def __init__(
        self, opset=13, input_name="input", output_name="output", save_dir="."
    ):
        """
        Initializes the ONNX model saver with configurable parameters.

        Parameters:
        - opset (int): The ONNX opset version. Default is 13.
        - input_name (str): Name for the input tensor. Default is 'input'.
        - output_name (str): Name for the output tensor. Default is 'output'.
        - save_dir (str): Directory to save the ONNX file. Default is the current directory.
        """
        self.opset = opset
        self.input_name = input_name
        self.output_name = output_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    def save(self, model, name):
        """
        Saves a TensorFlow Keras model as an ONNX file.

        Parameters:
        - model (tf.keras.Model): The Keras model to be converted.
        - name (str): The desired name for the ONNX file (without extension).

        Returns:
        - str: Path to the saved ONNX file, or None if the process fails.
        """
        try:
            # Set output names for the model
            model.output_names = [self.output_name]

            # Define input signature based on the model's input shape and type
            input_signature = [
                tf.TensorSpec(model.input_shape, tf.float64, name=self.input_name)
            ]

            # Convert the model to ONNX format
            onnx_model, _ = tf2onnx.convert.from_keras(
                model, input_signature=input_signature, opset=self.opset
            )

            # Define the path for saving the ONNX file
            onnx_path = os.path.join(self.save_dir, f"{name}.onnx")
            onnx.save(onnx_model, onnx_path)

            print(f"Model saved as {onnx_path}")
            return onnx_path

        except Exception as e:
            print(f"Failed to save model as ONNX: {e}")
            return None
