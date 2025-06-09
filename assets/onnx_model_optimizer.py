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
                                 name='cast_input_to_float', **attrib_dict)

    # Add Node:
    addNode = onnx.helper.make_node('Add', inputs=["TIDL_cast_in", "TIDL_preProc_Bias"], outputs=["TIDL_Scale_In"], name='add_bias')

    # Scale Node:
    scaleNode = onnx.helper.make_node('Mul', inputs=["TIDL_Scale_In", "TIDL_preProc_Scale"], outputs=[
        originalGraph.input[0].name], name='multiply_scale')  # Assumption that input[0].name is the input node

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
                                                 name='cast_output_to_uint8', **attrib_dict_1)
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
    
    # Create two separate inputs for NV12 format in channels-last format
    # Y input: (1, h, w, 1)
    y_input_shape = [batch_size, height, width, 1]
    y_input = helper.make_tensor_value_info(
        'y_input',
        TensorProto.UINT8,
        y_input_shape
    )
    
    # UV input: (1, h/2, w/2, 2)
    uv_input_shape = [batch_size, height//2, width//2, 2]
    uv_input = helper.make_tensor_value_info(
        'uv_input',
        TensorProto.UINT8,
        uv_input_shape
    )
    
    # Create proper NV12 to RGB conversion nodes
    nodes = []
    
    # Cast inputs to float for processing
    y_cast = onnx.helper.make_node(
        'Cast',
        inputs=['y_input'],
        outputs=['y_float'],
        to=TensorProto.FLOAT,
        name='cast_y_to_float'
    )
    nodes.append(y_cast)
    
    uv_cast = onnx.helper.make_node(
        'Cast',
        inputs=['uv_input'],
        outputs=['uv_float'],
        to=TensorProto.FLOAT,
        name='cast_uv_to_float'
    )
    nodes.append(uv_cast)
    
    # Squeeze Y to remove channel dimension: (1, h, w, 1) -> (1, h, w)
    y_squeeze = onnx.helper.make_node(
        'Squeeze',
        inputs=['y_float'],
        outputs=['y_plane'],
        axes=[3],
        name='squeeze_y'
    )
    nodes.append(y_squeeze)
    
    # Transpose UV from (1, h/2, w/2, 2) to (1, 2, h/2, w/2) for easier processing
    uv_transpose = onnx.helper.make_node(
        'Transpose',
        inputs=['uv_float'],
        outputs=['uv_plane_small'],
        perm=[0, 3, 1, 2],  # (1, h/2, w/2, 2) -> (1, 2, h/2, w/2)
        name='transpose_uv'
    )
    nodes.append(uv_transpose)
    
    # Upsample UV to full resolution using nearest neighbor (BCHW format)
    uv_resize_node = onnx.helper.make_node(
        'Resize',
        inputs=['uv_plane_small', 'roi', 'scales'],
        outputs=['uv_plane'],
        mode='nearest',
        nearest_mode='round_prefer_ceil',
        name='upsample_uv'
    )
    nodes.append(uv_resize_node)
    
    # Split UV into U and V channels (along channel axis)
    uv_split_node = onnx.helper.make_node(
        'Split',
        inputs=['uv_plane'],
        outputs=['u_plane_full', 'v_plane_full'],
        axis=1,
        split=[1, 1],
        name='split_uv'
    )
    nodes.append(uv_split_node)
    
    # Squeeze to remove extra channel dimensions
    u_squeeze = onnx.helper.make_node('Squeeze', ['u_plane_full'], ['u_plane'], axes=[1], name='squeeze_u')
    v_squeeze = onnx.helper.make_node('Squeeze', ['v_plane_full'], ['v_plane'], axes=[1], name='squeeze_v')
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
    
    # G = Y + (-0.344) * U_centered + (-0.714) * V_centered (avoid Sub operations with two variables)
    u_mul_g = onnx.helper.make_node('Mul', ['u_centered', 'coeff_neg_0_344'], ['u_term_g'], name='u_mul_g')
    v_mul_g = onnx.helper.make_node('Mul', ['v_centered', 'coeff_neg_0_714'], ['v_term_g'], name='v_mul_g')
    g_add1 = onnx.helper.make_node('Add', ['y_plane', 'u_term_g'], ['g_temp'], name='g_add1')
    g_add2 = onnx.helper.make_node('Add', ['g_temp', 'v_term_g'], ['g_channel'], name='calc_g')
    
    # B = Y + 1.772 * U_centered
    u_mul_b = onnx.helper.make_node('Mul', ['u_centered', 'coeff_1_772'], ['u_term_b'], name='u_mul_b')
    b_add = onnx.helper.make_node('Add', ['y_plane', 'u_term_b'], ['b_channel'], name='calc_b')
    
    nodes.extend([v_mul_r, r_add, u_mul_g, v_mul_g, g_add1, g_add2, u_mul_b, b_add])
    
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
    
    # Keep as float (no casting needed)
    final_output = onnx.helper.make_node(
        'Identity',
        inputs=['rgb_float'],
        outputs=[original_input_name],
        name='final_output'
    )
    nodes.append(final_output)
    
    # Create initializers
    initializers = []
    
    # Resize parameters for UV upsampling (BCHW format: scales for [N, C, H, W])
    initializers.append(numpy_helper.from_array(np.array([], dtype=np.float32), name='roi'))
    initializers.append(numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name='scales'))
    
    # YUV to RGB conversion coefficients
    initializers.append(numpy_helper.from_array(np.array(128.0, dtype=np.float32), name='offset_128'))
    initializers.append(numpy_helper.from_array(np.array(1.402, dtype=np.float32), name='coeff_1_402'))
    initializers.append(numpy_helper.from_array(np.array(-0.344, dtype=np.float32), name='coeff_neg_0_344'))
    initializers.append(numpy_helper.from_array(np.array(-0.714, dtype=np.float32), name='coeff_neg_0_714'))
    initializers.append(numpy_helper.from_array(np.array(1.772, dtype=np.float32), name='coeff_1_772'))
    
    # Clipping values
    initializers.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name='min_val'))
    initializers.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), name='max_val'))
    
    # Create new graph
    new_graph = onnx.helper.make_graph(
        nodes + list(model.graph.node),
        model.graph.name + '_with_nv12',
        [y_input, uv_input] + list(model.graph.input)[1:],
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


def add_nv12_conversion_to_onnx_model_ti_optimized(input_path, output_path):
    """
    Add TI-optimized NV12 to RGB preprocessing to an ONNX model.
    This version avoids operations that are problematic on TI hardware:
    - No Sub operations with two variable inputs
    - Simplified UV upsampling without problematic Resize operations
    
    Args:
        input_path (str): Path to the input ONNX model
        output_path (str): Path to save the modified ONNX model with TI-optimized NV12 preprocessing
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
        TensorProto.FLOAT,
        nv12_input_shape
    )
    
    # Create TI-optimized NV12 to RGB conversion nodes
    nodes = []
    
    # Extract Y plane (first height*width elements)
    y_slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['nv12_input', 'y_start', 'y_end', 'axes', 'steps'],
        outputs=['y_flat'],
        name='extract_y_plane'
    )
    nodes.append(y_slice_node)
    
    # Extract UV plane (remaining elements)
    uv_slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['nv12_input', 'uv_start', 'uv_end', 'axes', 'steps'],
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
    
    # Reshape UV plane to [batch, height//2, width//2, 2]
    uv_reshape_node = onnx.helper.make_node(
        'Reshape',
        inputs=['uv_flat', 'uv_shape'],
        outputs=['uv_plane_small'],
        name='reshape_uv'
    )
    nodes.append(uv_reshape_node)
    
    # TI-optimized UV upsampling using simple nearest neighbor replication
    # Instead of Resize, use Tile and Reshape operations
    
    # First, split UV into U and V channels at half resolution
    uv_split_node = onnx.helper.make_node(
        'Split',
        inputs=['uv_plane_small'],
        outputs=['u_half', 'v_half'],
        axis=3,
        split=[1, 1],
        name='split_uv_half'
    )
    nodes.append(uv_split_node)
    
    # Remove extra dimension from U and V
    u_squeeze = onnx.helper.make_node('Squeeze', ['u_half'], ['u_half_2d'], axes=[3], name='squeeze_u_half')
    v_squeeze = onnx.helper.make_node('Squeeze', ['v_half'], ['v_half_2d'], axes=[3], name='squeeze_v_half')
    nodes.extend([u_squeeze, v_squeeze])
    
    # Upsample U and V by 2x using Tile (repeat each element)
    # Reshape to allow tiling along width and height
    u_reshape_for_tile = onnx.helper.make_node(
        'Reshape', ['u_half_2d', 'u_tile_shape'], ['u_for_tile'], 
        name='u_reshape_for_tile'
    )
    v_reshape_for_tile = onnx.helper.make_node(
        'Reshape', ['v_half_2d', 'v_tile_shape'], ['v_for_tile'], 
        name='v_reshape_for_tile'
    )
    nodes.extend([u_reshape_for_tile, v_reshape_for_tile])
    
    # Use simple duplication instead of complex resize
    # Repeat each pixel 2x along width to get full resolution
    u_tile = onnx.helper.make_node(
        'Tile', ['u_for_tile', 'tile_repeats'], ['u_tiled'], 
        name='u_tile'
    )
    v_tile = onnx.helper.make_node(
        'Tile', ['v_for_tile', 'tile_repeats'], ['v_tiled'], 
        name='v_tile'
    )
    nodes.extend([u_tile, v_tile])
    
    # Reshape back to full resolution
    u_reshape_final = onnx.helper.make_node(
        'Reshape', ['u_tiled', 'full_uv_shape'], ['u_plane'], 
        name='u_reshape_final'
    )
    v_reshape_final = onnx.helper.make_node(
        'Reshape', ['v_tiled', 'full_uv_shape'], ['v_plane'], 
        name='v_reshape_final'
    )
    nodes.extend([u_reshape_final, v_reshape_final])
    
    # TI-optimized YUV to RGB conversion avoiding Sub with two variables
    # Original formulas:
    # R = Y + 1.402 * (V - 128)  →  R = Y + 1.402 * V - 179.456
    # G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)  →  G = Y - 0.344 * U - 0.714 * V + 135.232  
    # B = Y + 1.772 * (U - 128)  →  B = Y + 1.772 * U - 226.816
    
    # Calculate coefficients with offsets to avoid Sub operations
    # R = Y + 1.402 * V + (-179.456)
    v_scaled_r = onnx.helper.make_node('Mul', ['v_plane', 'coeff_1_402'], ['v_term_r'], name='v_mul_r')
    r_add1 = onnx.helper.make_node('Add', ['y_plane', 'v_term_r'], ['r_temp'], name='r_add_v')
    r_final = onnx.helper.make_node('Add', ['r_temp', 'r_offset'], ['r_channel'], name='calc_r')
    
    # G = Y + (-0.344) * U + (-0.714) * V + 135.232
    u_scaled_g = onnx.helper.make_node('Mul', ['u_plane', 'coeff_neg_0_344'], ['u_term_g'], name='u_mul_g_neg')
    v_scaled_g = onnx.helper.make_node('Mul', ['v_plane', 'coeff_neg_0_714'], ['v_term_g'], name='v_mul_g_neg')
    g_add1 = onnx.helper.make_node('Add', ['y_plane', 'u_term_g'], ['g_temp1'], name='g_add_u')
    g_add2 = onnx.helper.make_node('Add', ['g_temp1', 'v_term_g'], ['g_temp2'], name='g_add_v')
    g_final = onnx.helper.make_node('Add', ['g_temp2', 'g_offset'], ['g_channel'], name='calc_g')
    
    # B = Y + 1.772 * U + (-226.816)
    u_scaled_b = onnx.helper.make_node('Mul', ['u_plane', 'coeff_1_772'], ['u_term_b'], name='u_mul_b')
    b_add1 = onnx.helper.make_node('Add', ['y_plane', 'u_term_b'], ['b_temp'], name='b_add_u')
    b_final = onnx.helper.make_node('Add', ['b_temp', 'b_offset'], ['b_channel'], name='calc_b')
    
    nodes.extend([
        v_scaled_r, r_add1, r_final,
        u_scaled_g, v_scaled_g, g_add1, g_add2, g_final,
        u_scaled_b, b_add1, b_final
    ])
    
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
    
    # Create initializers for TI-optimized approach
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
    
    # Tiling parameters for UV upsampling (simple 2x replication)
    initializers.append(numpy_helper.from_array(np.array([batch_size, height//2, width//2], dtype=np.int64), name='u_tile_shape'))
    initializers.append(numpy_helper.from_array(np.array([batch_size, height//2, width//2], dtype=np.int64), name='v_tile_shape'))
    initializers.append(numpy_helper.from_array(np.array([1, 2, 2], dtype=np.int64), name='tile_repeats'))
    initializers.append(numpy_helper.from_array(np.array([batch_size, height, width], dtype=np.int64), name='full_uv_shape'))
    
    # YUV to RGB conversion coefficients (avoiding Sub operations)
    initializers.append(numpy_helper.from_array(np.array(1.402, dtype=np.float32), name='coeff_1_402'))
    initializers.append(numpy_helper.from_array(np.array(-0.344, dtype=np.float32), name='coeff_neg_0_344'))  # Negative to avoid Sub
    initializers.append(numpy_helper.from_array(np.array(-0.714, dtype=np.float32), name='coeff_neg_0_714'))  # Negative to avoid Sub
    initializers.append(numpy_helper.from_array(np.array(1.772, dtype=np.float32), name='coeff_1_772'))
    
    # Offset constants (precomputed to avoid Sub operations)
    initializers.append(numpy_helper.from_array(np.array(-179.456, dtype=np.float32), name='r_offset'))  # -1.402 * 128
    initializers.append(numpy_helper.from_array(np.array(135.232, dtype=np.float32), name='g_offset'))   # 0.344 * 128 + 0.714 * 128
    initializers.append(numpy_helper.from_array(np.array(-226.816, dtype=np.float32), name='b_offset'))  # -1.772 * 128
    
    # Clipping values
    initializers.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name='min_val'))
    initializers.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), name='max_val'))
    
    # Create new graph
    new_graph = onnx.helper.make_graph(
        nodes + list(model.graph.node),
        model.graph.name + '_with_nv12_ti_opt',
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
    print(f"TI-optimized NV12 model saved to: {output_path}")


def remove_redundant_cast_nodes(input_path, output_path):
    """
    Remove redundant cast nodes from NV12+normalization models.
    Specifically removes:
    - 'cast_to_uint8' from NV12 conversion (float→uint8)
    - 'cast_input_to_float' from normalization (uint8→float)
    And connects them directly to eliminate unnecessary type conversions.
    
    Args:
        input_path (str): Path to the input ONNX model with redundant casts
        output_path (str): Path to save the optimized model without redundant casts
    """
    # Load the model
    model = onnx.load(input_path)
    
    # Get graph nodes
    nodes = list(model.graph.node)
    
    # Find the cast nodes to remove
    cast_to_uint8_node = None
    cast_input_to_float_node = None
    
    for node in nodes:
        if node.name == 'cast_to_uint8':
            cast_to_uint8_node = node
        elif node.name == 'cast_input_to_float':
            cast_input_to_float_node = node
    
    if not cast_to_uint8_node or not cast_input_to_float_node:
        print("Warning: Could not find both cast nodes to remove")
        # Just copy the model as-is
        onnx.save(model, output_path)
        return
    
    print(f"Found cast nodes to remove:")
    print(f"  - {cast_to_uint8_node.name}: {cast_to_uint8_node.input[0]} → {cast_to_uint8_node.output[0]}")
    print(f"  - {cast_input_to_float_node.name}: {cast_input_to_float_node.input[0]} → {cast_input_to_float_node.output[0]}")
    
    # Get the connection points
    nv12_output = cast_to_uint8_node.input[0]  # Output from NV12 conversion (before cast to uint8)
    cast_float_output = cast_input_to_float_node.output[0]  # Output of cast_input_to_float
    
    # Remove the cast nodes
    nodes_filtered = [node for node in nodes if node.name not in ['cast_to_uint8', 'cast_input_to_float']]
    
    # Update connections: replace all references to cast_float_output with nv12_output
    for node in nodes_filtered:
        # Update node inputs
        for i, input_name in enumerate(node.input):
            if input_name == cast_float_output:  # Was connected to cast_input_to_float output
                node.input[i] = nv12_output  # Connect directly to NV12 output
    
    print(f"Connecting {nv12_output} directly to normalization nodes (bypassing {cast_float_output})")
    
    # Create new graph with filtered nodes
    new_graph = onnx.helper.make_graph(
        nodes_filtered,
        model.graph.name + '_optimized',
        list(model.graph.input),
        list(model.graph.output),
        list(model.graph.initializer)
    )
    
    # Create new model
    new_model = onnx.helper.make_model(new_graph)
    new_model.opset_import.extend(model.opset_import)
    new_model.ir_version = model.ir_version
    new_model.producer_name = model.producer_name
    new_model.producer_version = model.producer_version
    new_model.domain = model.domain
    new_model.model_version = model.model_version
    new_model.doc_string = model.doc_string
    
    # Validate and save
    try:
        onnx.checker.check_model(new_model)
        print('Optimized model is valid!')
    except onnx.checker.ValidationError as e:
        print('Warning: Optimized model validation failed: %s' % e)
    
    onnx.save(new_model, output_path)
    print(f"Optimized model (redundant casts removed) saved to: {output_path}")


