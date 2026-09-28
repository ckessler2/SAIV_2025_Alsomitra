# ============================================================
# Orange .pkcls Neural Network -> ONNX
# ============================================================

# Install required packages if needed:
# !pip install -q onnx onnxruntime

import pickle
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


# ============================================================
# 1. LOAD ORANGE MODEL
# ============================================================

PKCLS_PATH = "model.pkcls"
ONNX_PATH = "model.onnx"

with open(PKCLS_PATH, "rb") as f:
    orange_model = pickle.load(f)

orange_mlp = orange_model.skl_model

print("Orange model:")
print(type(orange_model))

print("\nOrange neural network:")
print(type(orange_mlp))


# ============================================================
# 2. GET THE TRAINED WEIGHTS
# ============================================================

if not hasattr(orange_mlp, "coefs_"):
    raise RuntimeError(
        "The Orange neural network does not expose coefs_. "
        "This converter expects an Orange MLPRegressor."
    )

if not hasattr(orange_mlp, "intercepts_"):
    raise RuntimeError(
        "The Orange neural network does not expose intercepts_."
    )

weights = orange_mlp.coefs_
biases = orange_mlp.intercepts_

print("\nNetwork structure:")

for i, (w, b) in enumerate(zip(weights, biases)):
    print(
        f"Layer {i}: "
        f"{w.shape[0]} -> {w.shape[1]}"
    )


# ============================================================
# 3. DETERMINE INPUT/OUTPUT SIZE
# ============================================================

input_size = weights[0].shape[0]
output_size = weights[-1].shape[1]

print(f"\nInput features: {input_size}")
print(f"Output values:  {output_size}")


# ============================================================
# 4. DETERMINE ACTIVATION FUNCTIONS
# ============================================================

# Orange's MLPRegressor is based on sklearn's MLP implementation.
#
# sklearn MLP hidden-layer activations can be:
#
#   identity
#   tanh
#   logistic
#   relu
#
# Regression output uses identity activation.

activation = getattr(
    orange_mlp,
    "activation",
    "relu"
)

print(f"Hidden-layer activation: {activation}")


# ============================================================
# 5. CREATE ONNX INPUT
# ============================================================

input_tensor = helper.make_tensor_value_info(
    "input",
    TensorProto.FLOAT,
    [None, input_size]
)


# ============================================================
# 6. CREATE ONNX OUTPUT
# ============================================================

output_tensor = helper.make_tensor_value_info(
    "output",
    TensorProto.FLOAT,
    [None, output_size]
)


# ============================================================
# 7. BUILD ONNX GRAPH
# ============================================================

nodes = []
initializers = []

current = "input"


# ------------------------------------------------------------
# Hidden layers + output layer
# ------------------------------------------------------------

for layer_index, (W, B) in enumerate(zip(weights, biases)):

    # ONNX Gemm expects:
    #
    # Y = alpha * A * B + beta * C
    #
    # sklearn stores:
    #
    # X @ W + B
    #
    # Therefore W can be used directly with Gemm
    # because transB=0 means B is not transposed.

    W = np.asarray(W, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    weight_name = f"W_{layer_index}"
    bias_name = f"B_{layer_index}"
    gemm_output = f"gemm_{layer_index}"

    initializers.append(
        numpy_helper.from_array(
            W,
            name=weight_name
        )
    )

    initializers.append(
        numpy_helper.from_array(
            B,
            name=bias_name
        )
    )

    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[
                current,
                weight_name,
                bias_name
            ],
            outputs=[
                gemm_output
            ],
            name=f"Gemm_{layer_index}",
            transB=0
        )
    )

    # --------------------------------------------------------
    # Apply activation to hidden layers only
    # --------------------------------------------------------

    is_output_layer = (
        layer_index == len(weights) - 1
    )

    if is_output_layer:
        current = gemm_output

    else:

        activation_output = f"activation_{layer_index}"

        if activation == "relu":

            nodes.append(
                helper.make_node(
                    "Relu",
                    inputs=[gemm_output],
                    outputs=[activation_output],
                    name=f"Relu_{layer_index}"
                )
            )

        elif activation == "tanh":

            nodes.append(
                helper.make_node(
                    "Tanh",
                    inputs=[gemm_output],
                    outputs=[activation_output],
                    name=f"Tanh_{layer_index}"
                )
            )

        elif activation == "logistic":

            nodes.append(
                helper.make_node(
                    "Sigmoid",
                    inputs=[gemm_output],
                    outputs=[activation_output],
                    name=f"Sigmoid_{layer_index}"
                )
            )

        elif activation == "identity":

            # No activation required.
            activation_output = gemm_output

        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )

        current = activation_output


# ============================================================
# 8. CONNECT FINAL OUTPUT
# ============================================================

nodes.append(
    helper.make_node(
        "Identity",
        inputs=[current],
        outputs=["output"],
        name="Output"
    )
)


# ============================================================
# 9. CREATE ONNX GRAPH
# ============================================================

graph = helper.make_graph(
    nodes,
    "Orange_MLP",
    [input_tensor],
    [output_tensor],
    initializer=initializers
)


# ============================================================
# 10. CREATE ONNX MODEL
# ============================================================

model = helper.make_model(
    graph,
    producer_name="Orange-to-ONNX"
)

# Use a conservative ONNX version for compatibility.
model.opset_import[0].version = 13


# ============================================================
# 11. CHECK MODEL
# ============================================================

onnx.checker.check_model(model)


# ============================================================
# 12. SAVE
# ============================================================

onnx.save(
    model,
    ONNX_PATH
)

print("\n========================================")
print("SUCCESS")
print("========================================")
print(f"Input features : {input_size}")
print(f"Output values  : {output_size}")
print(f"Activation     : {activation}")
print(f"Saved to       : {ONNX_PATH}")
