import onnx
from onnx import helper
from onnx import TensorProto, shape_inference
import numpy as np
import onnx.numpy_helper as numpy_helper

# https://github.com/TexasInstruments/edgeai-tidl-tools/blob/08_06_00_05/scripts/osrt_model_tools/onnx_tools/onnx_model_opt.py#L65
def add_normalization_to_onnx_model(in_model_path, out_model_path, scaleList=[0.0078125, 0.0078125, 0.0078125], meanList=[128.0, 128.0, 128.0]):
    # Read Model
    meanList = [x * -1 for x in meanList]
    model = onnx.load_model(in_model_path)
    op = onnx.OperatorSetIdProto()
    # Track orginal opset:
    op.version = model.opset_import[0].version
    
    # Get Graph:
    originalGraph = model.graph
    # Get Nodes:
    originalNodes = originalGraph.node
    # Get Initializers:
    originalInitializers = originalGraph.initializer
    # Create Lists
    nodeList = [node for node in originalNodes]
    initList = [init for init in originalInitializers]

    nInCh = int(originalGraph.input[0].type.tensor_type.shape.dim[1].dim_value)

    # Input & Output Dimensions:
    inDims = tuple([x.dim_value for x in originalGraph.input[0].type.tensor_type.shape.dim])
    outDims = tuple([x.dim_value for x in originalGraph.output[0].type.tensor_type.shape.dim])

    # Construct bias & scale tensors
    biasTensor = helper.make_tensor("TIDL_preProc_Bias", TensorProto.FLOAT, [1, nInCh, 1, 1],
                                    np.array(meanList, dtype=np.float32))
    scaleTensor = helper.make_tensor("TIDL_preProc_Scale", TensorProto.FLOAT, [1, nInCh, 1, 1],
                                     np.array(scaleList, dtype=np.float32))

    # Add these tensors to initList:
    initList.append(biasTensor)
    initList.append(scaleTensor)

    # Cast Node:
    attrib_dict = {"to": TensorProto.FLOAT}
    cast = onnx.helper.make_node('Cast', inputs=[originalGraph.input[0].name + "Net_IN"], outputs=['TIDL_cast_in'],
                                 **attrib_dict)

    # Add Node:
    addNode = onnx.helper.make_node('Add', inputs=["TIDL_cast_in", "TIDL_preProc_Bias"], outputs=["TIDL_Scale_In"])

    # Scale Node:
    scaleNode = onnx.helper.make_node('Mul', inputs=["TIDL_Scale_In", "TIDL_preProc_Scale"], outputs=[
        originalGraph.input[0].name])  # Assumption that input[0].name is the input node

    nodeList = [cast, addNode, scaleNode] + nodeList  # Toplogically Sorted

    outSequence = originalGraph.output
    # Check for Argmax:
    for node in nodeList:
        if node.op_type == "ArgMax":
            # Check if it is final output:
            if node.output[0] == originalGraph.output[0].name:
                # Argmax Output is final output:
                attrib_dict_1 = {"to": TensorProto.UINT8}
                cast_out = onnx.helper.make_node('Cast', inputs=[originalGraph.output[0].name],
                                                 outputs=[originalGraph.output[0].name + 'TIDL_cast_out'],
                                                 **attrib_dict_1)
                nodeList = nodeList + [cast_out]  # Toplogically Sorted
                outSequence = [
                    helper.make_tensor_value_info(originalGraph.output[0].name + 'TIDL_cast_out', TensorProto.UINT8,
                                                  outDims)]

    # Construct Graph:
    newGraph = helper.make_graph(
        nodeList,
        'Rev_Model',
        [helper.make_tensor_value_info(originalGraph.input[0].name + "Net_IN", TensorProto.UINT8, inDims)],
        outSequence,
        initList
    )
    # Construct Model:
    # op.version = 11 # if hardcode opset version needed
    model_def_noShape = helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
    model_def = shape_inference.infer_shapes(model_def_noShape)

    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print('Converted model is invalid: %s' % e)
    else:
        print('Converted model is valid!')
        onnx.save_model(model_def, out_model_path)


def add_nv12_conversion_to_onnx_model(input_path, output_path):
    """
    Add proper NV12 to RGB preprocessing to an ONNX model.
    
    Args:
        input_path (str): Path to the input ONNX model
        output_path (str): Path to save the modified ONNX model with NV12 preprocessing
    """
    # Load the original model
    model = onnx.load(input_path)
    
    # Get the original input info
    original_input = model.graph.input[0]
    original_input_name = original_input.name
    original_input_shape = [dim.dim_value for dim in original_input.type.tensor_type.shape.dim]
    
    # Get dimensions
    batch_size = original_input_shape[0] if original_input_shape[0] > 0 else 1
    height = original_input_shape[2]
    width = original_input_shape[3]
    
    # Create new input for NV12 format (height * width * 1.5)
    nv12_input_shape = [batch_size, int(height * width * 1.5)]
    nv12_input = helper.make_tensor_value_info(
        'nv12_input',
        TensorProto.UINT8,
        nv12_input_shape
    )
    
    # Create proper NV12 to RGB conversion nodes
    nodes = []
    
    # Cast to float for processing
    cast_node = onnx.helper.make_node(
        'Cast',
        inputs=['nv12_input'],
        outputs=['nv12_float'],
        to=TensorProto.FLOAT,
        name='cast_to_float'
    )
    nodes.append(cast_node)
    
    # Extract Y plane (first height*width elements)
    y_slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['nv12_float', 'y_start', 'y_end', 'axes', 'steps'],
        outputs=['y_flat'],
        name='extract_y_plane'
    )
    nodes.append(y_slice_node)
    
    # Extract UV plane (remaining elements)
    uv_slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['nv12_float', 'uv_start', 'uv_end', 'axes', 'steps'],
        outputs=['uv_flat'],
        name='extract_uv_plane'
    )
    nodes.append(uv_slice_node)
    
    # Reshape Y plane to [batch, height, width]
    y_reshape_node = onnx.helper.make_node(
        'Reshape',
        inputs=['y_flat', 'y_shape'],
        outputs=['y_plane'],
        name='reshape_y'
    )
    nodes.append(y_reshape_node)
    
    # Reshape UV plane and upsample to match Y dimensions
    uv_reshape_node = onnx.helper.make_node(
        'Reshape',
        inputs=['uv_flat', 'uv_shape'],
        outputs=['uv_plane_small'],
        name='reshape_uv'
    )
    nodes.append(uv_reshape_node)
    
    # Upsample UV to full resolution using nearest neighbor
    uv_resize_node = onnx.helper.make_node(
        'Resize',
        inputs=['uv_plane_small', 'roi', 'scales'],
        outputs=['uv_plane'],
        mode='nearest',
        name='upsample_uv'
    )
    nodes.append(uv_resize_node)
    
    # Split UV into U and V channels
    uv_split_node = onnx.helper.make_node(
        'Split',
        inputs=['uv_plane'],
        outputs=['u_plane_full', 'v_plane_full'],
        axis=3,
        split=[1, 1],
        name='split_uv'
    )
    nodes.append(uv_split_node)
    
    # Squeeze to remove extra dimensions
    u_squeeze = onnx.helper.make_node('Squeeze', ['u_plane_full'], ['u_plane'], axes=[3], name='squeeze_u')
    v_squeeze = onnx.helper.make_node('Squeeze', ['v_plane_full'], ['v_plane'], axes=[3], name='squeeze_v')
    nodes.extend([u_squeeze, v_squeeze])
    
    # YUV to RGB conversion: 
    # R = Y + 1.402 * (V - 128)
    # G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
    # B = Y + 1.772 * (U - 128)
    
    # Subtract 128 from U and V
    u_sub = onnx.helper.make_node('Sub', ['u_plane', 'offset_128'], ['u_centered'], name='u_center')
    v_sub = onnx.helper.make_node('Sub', ['v_plane', 'offset_128'], ['v_centered'], name='v_center')
    nodes.extend([u_sub, v_sub])
    
    # Calculate RGB channels
    # R = Y + 1.402 * V_centered
    v_mul_r = onnx.helper.make_node('Mul', ['v_centered', 'coeff_1_402'], ['v_term_r'], name='v_mul_r')
    r_add = onnx.helper.make_node('Add', ['y_plane', 'v_term_r'], ['r_channel'], name='calc_r')
    
    # G = Y - 0.344 * U_centered - 0.714 * V_centered
    u_mul_g = onnx.helper.make_node('Mul', ['u_centered', 'coeff_0_344'], ['u_term_g'], name='u_mul_g')
    v_mul_g = onnx.helper.make_node('Mul', ['v_centered', 'coeff_0_714'], ['v_term_g'], name='v_mul_g')
    g_sub1 = onnx.helper.make_node('Sub', ['y_plane', 'u_term_g'], ['g_temp'], name='g_sub1')
    g_sub2 = onnx.helper.make_node('Sub', ['g_temp', 'v_term_g'], ['g_channel'], name='calc_g')
    
    # B = Y + 1.772 * U_centered
    u_mul_b = onnx.helper.make_node('Mul', ['u_centered', 'coeff_1_772'], ['u_term_b'], name='u_mul_b')
    b_add = onnx.helper.make_node('Add', ['y_plane', 'u_term_b'], ['b_channel'], name='calc_b')
    
    nodes.extend([v_mul_r, r_add, u_mul_g, v_mul_g, g_sub1, g_sub2, u_mul_b, b_add])
    
    # Clamp values to 0-255 range
    r_clip = onnx.helper.make_node('Clip', ['r_channel', 'min_val', 'max_val'], ['r_clipped'], name='clip_r')
    g_clip = onnx.helper.make_node('Clip', ['g_channel', 'min_val', 'max_val'], ['g_clipped'], name='clip_g')
    b_clip = onnx.helper.make_node('Clip', ['b_channel', 'min_val', 'max_val'], ['b_clipped'], name='clip_b')
    nodes.extend([r_clip, g_clip, b_clip])
    
    # Add channel dimension and stack RGB
    r_unsqueeze = onnx.helper.make_node('Unsqueeze', ['r_clipped'], ['r_chan'], axes=[1], name='r_unsqueeze')
    g_unsqueeze = onnx.helper.make_node('Unsqueeze', ['g_clipped'], ['g_chan'], axes=[1], name='g_unsqueeze')
    b_unsqueeze = onnx.helper.make_node('Unsqueeze', ['b_clipped'], ['b_chan'], axes=[1], name='b_unsqueeze')
    nodes.extend([r_unsqueeze, g_unsqueeze, b_unsqueeze])
    
    # Concatenate RGB channels
    rgb_concat = onnx.helper.make_node(
        'Concat',
        ['r_chan', 'g_chan', 'b_chan'],
        ['rgb_float'],
        axis=1,
        name='concat_rgb'
    )
    nodes.append(rgb_concat)
    
    # Cast back to uint8
    final_cast = onnx.helper.make_node(
        'Cast',
        inputs=['rgb_float'],
        outputs=[original_input_name],
        to=TensorProto.UINT8,
        name='cast_to_uint8'
    )
    nodes.append(final_cast)
    
    # Create initializers
    initializers = []
    
    # Slice parameters for Y plane
    y_size = height * width
    uv_size = height * width // 2
    
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name='y_start'))
    initializers.append(numpy_helper.from_array(np.array([y_size], dtype=np.int64), name='y_end'))
    initializers.append(numpy_helper.from_array(np.array([y_size], dtype=np.int64), name='uv_start'))
    initializers.append(numpy_helper.from_array(np.array([y_size + uv_size], dtype=np.int64), name='uv_end'))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='axes'))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name='steps'))
    
    # Reshape parameters
    initializers.append(numpy_helper.from_array(np.array([batch_size, height, width], dtype=np.int64), name='y_shape'))
    initializers.append(numpy_helper.from_array(np.array([batch_size, height//2, width//2, 2], dtype=np.int64), name='uv_shape'))
    
    # Resize parameters for UV upsampling
    initializers.append(numpy_helper.from_array(np.array([], dtype=np.float32), name='roi'))
    initializers.append(numpy_helper.from_array(np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float32), name='scales'))
    
    # YUV to RGB conversion coefficients
    initializers.append(numpy_helper.from_array(np.array(128.0, dtype=np.float32), name='offset_128'))
    initializers.append(numpy_helper.from_array(np.array(1.402, dtype=np.float32), name='coeff_1_402'))
    initializers.append(numpy_helper.from_array(np.array(0.344, dtype=np.float32), name='coeff_0_344'))
    initializers.append(numpy_helper.from_array(np.array(0.714, dtype=np.float32), name='coeff_0_714'))
    initializers.append(numpy_helper.from_array(np.array(1.772, dtype=np.float32), name='coeff_1_772'))
    
    # Clipping values
    initializers.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name='min_val'))
    initializers.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), name='max_val'))
    
    # Create new graph
    new_graph = onnx.helper.make_graph(
        nodes + list(model.graph.node),
        model.graph.name + '_with_nv12',
        [nv12_input] + list(model.graph.input)[1:],
        list(model.graph.output),
        initializers + list(model.graph.initializer)
    )
    
    # Create new model with same properties as original
    new_model = onnx.helper.make_model(new_graph)
    new_model.opset_import.extend(model.opset_import)
    new_model.ir_version = model.ir_version
    new_model.producer_name = model.producer_name
    new_model.producer_version = model.producer_version
    new_model.domain = model.domain
    new_model.model_version = model.model_version
    new_model.doc_string = model.doc_string
    
    # Save the modified model
    onnx.save(new_model, output_path)
    print(f"NV12 model saved to: {output_path}")


