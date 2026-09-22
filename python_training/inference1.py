import onnx
from onnx2keras import onnx_to_keras
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VERIFY_DIR = BASE_DIR.parent / "vehicle_verification" / "property_verification"

# Load ONNX model
onnx_model1 = onnx.load(MODELS_DIR / 'base_model_norm.onnx')
# %%
onnx_model2 = onnx.load(VERIFY_DIR / 'merged_model_base.onnx')


# Print model graph for debugging
# print(onnx.helper.printable_graph(onnx_model.graph)) 

# Call the converter (input - is the main model input name, can be different for your model)
k_model1 = onnx_to_keras(onnx_model1,['x'], name_policy='short')
k_model2 = onnx_to_keras(onnx_model2,['x'], name_policy='short')

x = np.load(DATA_DIR / "X_Test.npy")
y = np.load(DATA_DIR / "Y_Test.npy")

# y1 = k_model1.predict(x)
# y2 = k_model2.predict([x,x])

# print(str(y1) + " end")
# print(str(y2))

input1 = np.expand_dims(x[1], axis=0)
output1 = k_model1.predict(input1)

input3 = np.expand_dims(x[3], axis=0)
output3 = k_model1.predict(input3)

input2 = np.concatenate((x[1],x[3]),axis=0)
input2 = np.expand_dims(input2, axis=0)
output2 = k_model2.predict(input2)

print(output1)
print(output3)
print(output2)
