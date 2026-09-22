# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 16:06:32 2025

@author: ck2049
"""

import onnx
from onnx import helper
from onnx import TensorProto

# model_path = 'base_model_norm.onnx'
model_path = 'adversarial_model_0.005.onnx'

model = onnx.load(model_path)

# Constants for normalization (example values, replace with your actual Cs and Ss)
Cs = [0.967568147250241,	-0.607397104011979,	-0.356972794479088,	-0.968478529510599,	0.482420778264618,	-41.6814052660887]  # Center values
Ss = [2.52232492135314,	0.564375712674976,	0.406153017236974,	0.900476013359655,	41.2322962227346,	45.8786387945936]  # Scale values

Cs2 = [0.181];
Ss2 =  [0.0120];

# Assume identification names for existing network nodes
first_relu_input_name = 'x'
last_relu_output_name = 'output'

Cs_init = helper.make_tensor('Cs', TensorProto.FLOAT, [6], Cs)
Ss_init = helper.make_tensor('Ss', TensorProto.FLOAT, [6], Ss)
Cs2_init = helper.make_tensor('Cs2', TensorProto.FLOAT, [1], Cs2)
Ss2_init = helper.make_tensor('Ss2', TensorProto.FLOAT, [1], Ss2)

for node in model.graph.node:
    if node.input[0] == first_relu_input_name :
        node.input[0] = 'normalized_input'
        
for node in model.graph.node:
    if node.output[0] == 'output' :
        node.output[0] = 'normalized_output'

# Normalization nodes
normalize_sub = helper.make_node('Sub', ['x', 'Cs'], ['sub_output'])
normalize_div = helper.make_node('Div', ['sub_output', 'Ss'], ['normalized_input'])

# Ensure normalization output feeds into the first ReLU
model.graph.node.insert(0, normalize_sub)
model.graph.node.insert(1, normalize_div)
model.graph.initializer.extend([Cs_init, Ss_init])


# Denormalization nodes (details omitted for brevity)
# Ensure the input for denormalization is the output of the last relevant layer
# Assuming this adjustment toward the end just before output.
denormalize_mul = helper.make_node('Mul', ['normalized_output', 'Ss2'], ['mul_output'])
denormalize_add = helper.make_node('Add', ['mul_output', 'Cs2'], ['output'])

# Append these nodes at the suitable positions, typically right before the output
model.graph.node.extend([denormalize_mul, denormalize_add])
model.graph.initializer.extend([Cs2_init, Ss2_init])

# Save the modified model
onnx.save(model, 'adversarial_model_005_denorm.onnx')