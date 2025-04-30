import onnx
import onnx.helper as helper
from onnx import helper, TensorProto, numpy_helper
import numpy as np
 
def merge_onnx_models(model_path, output_path):
    """
    Function to duplicate an ONNX model's nodes, inputs, outputs, and initialisers,
    renaming them to avoid conflicts, and then merge the duplicated elements into a new ONNX model.
 
    Parameters:
    - model_path (str): Path to the original ONNX model.
    - output_path (str): Path to save the merged ONNX model.
    """
 
    # Load the original ONNX model
    model = onnx.load(model_path)
   
    new_model = double_network(model)
    new_model = deeply_adjust_model_dtypes(new_model)
    for tensor_dtype in helper.get_all_tensor_dtypes():
        # print(helper.tensor_dtype_to_string(tensor_dtype))
        helper.tensor_dtype_to_np_dtype(1)
        tensor_dtype = 'TensorProto.FLOAT'
    for tensor_dtype in helper.get_all_tensor_dtypes():
        print(helper.tensor_dtype_to_string(tensor_dtype))

    onnx.save(new_model, output_path)
    print(f"Successfully saved merged ONNX model to {output_path}")


def double_network(orig_model):
    new_initializers = []
    new_nodes = []
    input_name = 'x'
    last_output_name = 'x'
    # 0 = 0
    
    for idx, node in enumerate(orig_model.graph.node):
        if node.op_type == 'Gemm':
            weight_tensor = next(t for t in orig_model.graph.initializer if t.name == node.input[1])
            bias_tensor = next(t for t in orig_model.graph.initializer if t.name == node.input[2])
            
            weight_array = numpy_helper.to_array(weight_tensor).astype(np.float32)
            bias_array = numpy_helper.to_array(bias_tensor).astype(np.float32)
            
            new_weight_array = np.zeros((weight_array.shape[0] * 2, weight_array.shape[1] * 2), dtype=np.float32)
            new_weight_array[:weight_array.shape[0], :weight_array.shape[1]] = weight_array + 0
            new_weight_array[weight_array.shape[0]:, weight_array.shape[1]:] = weight_array + 0
            
            new_bias_array = np.tile(bias_array, 2)
            
            new_weight = numpy_helper.from_array(new_weight_array, name=weight_tensor.name)
            new_bias = numpy_helper.from_array(new_bias_array, name=bias_tensor.name)
            new_initializers.extend([new_weight, new_bias])
            
            current_input = input_name if idx == 0 else last_output_name
            current_output = 'output' if idx == len(orig_model.graph.node) - 1 else node.output[0]
            
            # Add attributes transA and transB, both set to zero
            new_node = helper.make_node(
                'Gemm',
                [current_input, new_weight.name, new_bias.name],
                [current_output],
                name=node.name
            )
        else:
            current_input = input_name if idx == 0 else last_output_name
            current_output = 'output' if idx == len(orig_model.graph.node) - 1 else node.output[0]
            new_node = helper.make_node(
                node.op_type,
                [current_input] + node.input[1:],
                [current_output],
                name=node.name
            )
        new_nodes.append(new_node)
        last_output_name = current_output

    input_tensor = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 12])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    new_graph = helper.make_graph(
        new_nodes,
        "FullyConnectedModel",
        [input_tensor],
        [output_tensor],
        new_initializers
    )
    
    return helper.make_model(new_graph, producer_name='FixedOutputConnectionModel')


def convert_double_to_float_attribute(attr):
    """ Helper function to handle attribute conversion. """
    if attr.type == 11:
        attr.f = float(attr.d)
        attr.type = TensorProto.FLOAT
    else:
        if attr.t.data_type == 11:
            array = numpy_helper.to_array(attr.t)
            array = array.astype(np.float32)
            new_tensor = numpy_helper.from_array(array, name=attr.t.name)
            new_tensor.data_type = TensorProto.FLOAT
            attr.t.CopyFrom(new_tensor)

def deeply_adjust_model_dtypes(model):
    # Adjust all initializers
    for initializer in model.graph.initializer:
        if initializer.data_type == 11:
            print(f"Converting initializer {initializer.name} from DOUBLE to FLOAT.")
            data = numpy_helper.to_array(initializer).astype(np.float32)
            initializer.CopyFrom(numpy_helper.from_array(data, name=initializer.name))
            initializer.data_type = TensorProto.FLOAT

    # Adjust input and output tensor types
    for tensor_info in list(model.graph.input) + list(model.graph.output):
        if tensor_info.type.tensor_type.elem_type == 11:
            print(f"Converting {tensor_info.name} from DOUBLE to FLOAT.")
            tensor_info.type.tensor_type.elem_type = TensorProto.FLOAT

    # Adjust node attributes recursively if they contain numerical values or tensors
    for node in model.graph.node:
        for attribute in node.attribute:
            convert_double_to_float_attribute(attribute)

    return model

# Example Usage: Merge an ONNX model and save the output
merge_onnx_models("Property Verification 2/DL2_Lipschitz.onnx", "Property Verification 2/merged_DL2_Lipschitz.onnx")