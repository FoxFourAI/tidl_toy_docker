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


def add_nv12_normalization(input_path, output_path, scaleList=[0.0078125, 0.0078125, 0.0078125], meanList=[128.0, 128.0, 128.0]):
    """
    Add NV12 to RGB conversion + normalization in one step, following add_normalization_to_onnx_model pattern.
    This replaces the need for separate add_normalization_to_onnx_model + add_nv12_conversion_to_onnx_model_old + remove_redundant_cast_nodes.
    Uses only Cast/Transpose/Resize/Concat/Add/Conv operations compatible with TI compiler.
    
    Args:
        input_path (str): Path to the input ONNX model
        output_path (str): Path to save the modified ONNX model with NV12 preprocessing + normalization
        scaleList (list): Scale factors for normalization (default: 1/128 for each channel)
        meanList (list): Mean values for normalization (default: 128 for each channel, will be negated)
    """
    # Load the original model (exact copy from add_normalization_to_onnx_model)
    meanList = [x * -1 for x in meanList]
    model = onnx.load_model(input_path)
    op = onnx.OperatorSetIdProto()
    op.version = model.opset_import[0].version
    
    # Get Graph (exact copy from add_normalization_to_onnx_model)
    originalGraph = model.graph
    originalNodes = originalGraph.node
    originalInitializers = originalGraph.initializer
    
    # Create Lists (exact copy from add_normalization_to_onnx_model)
    nodeList = [node for node in originalNodes]
    initList = [init for init in originalInitializers]
    
    # Get input info
    original_input = originalGraph.input[0]
    original_input_name = original_input.name
    original_input_shape = [dim.dim_value for dim in original_input.type.tensor_type.shape.dim]
    
    # Get dimensions
    batch_size = original_input_shape[0] if original_input_shape[0] > 0 else 1
    height = original_input_shape[2]
    width = original_input_shape[3]
    nInCh = 3  # RGB output channels
    
    # Input & Output Dimensions (following add_normalization_to_onnx_model pattern)
    y_inDims = tuple([batch_size, height, width, 1])
    uv_inDims = tuple([batch_size, height//2, width//2, 2])
    outDims = tuple([x.dim_value for x in originalGraph.output[0].type.tensor_type.shape.dim])
    
    # Cast Y input (exact pattern from add_normalization_to_onnx_model)
    cast_y = onnx.helper.make_node('Cast', 
                                   inputs=[original_input_name + "Net_Y_IN"], 
                                   outputs=['TIDL_cast_y_in'],
                                   name='cast_y_input_to_float', 
                                   to=TensorProto.FLOAT)
    
    # Cast UV input
    cast_uv = onnx.helper.make_node('Cast', 
                                    inputs=[original_input_name + "Net_UV_IN"], 
                                    outputs=['TIDL_cast_uv_in'],
                                    name='cast_uv_input_to_float', 
                                    to=TensorProto.FLOAT)
    
    # Transpose Y: (B, H, W, 1) → (B, 1, H, W)
    transpose_y = onnx.helper.make_node('Transpose',
                                        inputs=['TIDL_cast_y_in'],
                                        outputs=['TIDL_y_transposed'],
                                        perm=[0, 3, 1, 2],
                                        name='transpose_y')
    
    # Transpose UV: (B, H/2, W/2, 2) → (B, 2, H/2, W/2)  
    transpose_uv = onnx.helper.make_node('Transpose',
                                         inputs=['TIDL_cast_uv_in'],
                                         outputs=['TIDL_uv_transposed'],
                                         perm=[0, 3, 1, 2],
                                         name='transpose_uv')
    
    # Upsample UV to match Y size
    resize_uv = onnx.helper.make_node('Resize',
                                      inputs=['TIDL_uv_transposed', 'TIDL_roi_empty', 'TIDL_uv_scales'],
                                      outputs=['TIDL_uv_resized'],
                                      mode='nearest',
                                      nearest_mode='round_prefer_ceil',
                                      name='resize_uv_to_full')
    
    # Concatenate Y and UV: (B, 1, H, W) + (B, 2, H, W) → (B, 3, H, W)
    concat_yuv = onnx.helper.make_node('Concat',
                                       inputs=['TIDL_y_transposed', 'TIDL_uv_resized'],
                                       outputs=['TIDL_yuv_concat'],
                                       axis=1,
                                       name='concat_yuv_channels')
    
    # ========================================================================
    # YUV to RGB conversion + normalization (following add_normalization_to_onnx_model)
    # ========================================================================
    
    # Add YUV bias (to handle -128 offset): [0, -128, -128] for Y, U, V channels
    addYuvBias = onnx.helper.make_node('Add', 
                                       inputs=["TIDL_yuv_concat", "TIDL_preProc_YUV_Bias"], 
                                       outputs=["TIDL_YUV_Biased"], 
                                       name='add_yuv_bias')

    # YUV to RGB conversion using 1x1 Conv 
    yuvToRgbConv = onnx.helper.make_node('Conv', 
                                         inputs=["TIDL_YUV_Biased", "TIDL_preProc_YUV2RGB_Weights", "TIDL_preProc_YUV2RGB_Bias"], 
                                         outputs=["TIDL_RGB_converted"], 
                                         name='yuv_to_rgb_conv')
    
    # Add normalization bias (following add_normalization_to_onnx_model pattern EXACTLY)
    addNode = onnx.helper.make_node('Add', 
                                    inputs=["TIDL_RGB_converted", "TIDL_preProc_Bias"], 
                                    outputs=["TIDL_Scale_In"], 
                                    name='add_bias')

    # Scale Node (following add_normalization_to_onnx_model pattern EXACTLY)
    scaleNode = onnx.helper.make_node('Mul', 
                                      inputs=["TIDL_Scale_In", "TIDL_preProc_Scale"], 
                                      outputs=[original_input_name], 
                                      name='multiply_scale')
    
    # YUV bias: subtract 128 from U,V channels, keep Y as-is
    yuvBiasTensor = helper.make_tensor("TIDL_preProc_YUV_Bias", TensorProto.FLOAT, [1, 3, 1, 1],
                                       np.array([0.0, -128.0, -128.0], dtype=np.float32))
    
    # YUV to RGB conversion weights (1x1 conv kernel): BT.601 full-range
    # R = Y + 1.402 * V      → weights [1.0,   0.0,   1.402]
    # G = Y - 0.344*U - 0.714*V → weights [1.0, -0.344, -0.714] 
    # B = Y + 1.772 * U      → weights [1.0,   1.772,  0.0]
    yuv2rgb_weights = np.array([
        [1.0,   0.0,   1.402],    # R channel weights
        [1.0, -0.344, -0.714],    # G channel weights  
        [1.0,   1.772,  0.0]      # B channel weights
    ], dtype=np.float32).reshape(3, 3, 1, 1)  # [out_ch, in_ch, kH, kW]
    
    yuvWeightsTensor = helper.make_tensor("TIDL_preProc_YUV2RGB_Weights", TensorProto.FLOAT, [3, 3, 1, 1],
                                          yuv2rgb_weights.flatten())
    
    # Conv bias (zero since bias is handled separately)
    yuvConvBiasTensor = helper.make_tensor("TIDL_preProc_YUV2RGB_Bias", TensorProto.FLOAT, [3],
                                           np.array([0.0, 0.0, 0.0], dtype=np.float32))
    
    # Normalization bias & scale tensors (exact copy from add_normalization_to_onnx_model)
    biasTensor = helper.make_tensor("TIDL_preProc_Bias", TensorProto.FLOAT, [1, nInCh, 1, 1],
                                    np.array(meanList, dtype=np.float32))
    scaleTensor = helper.make_tensor("TIDL_preProc_Scale", TensorProto.FLOAT, [1, nInCh, 1, 1],
                                     np.array(scaleList, dtype=np.float32))
    
    # Resize parameters
    roiTensor = helper.make_tensor("TIDL_roi_empty", TensorProto.FLOAT, [0], [])
    scalesTensor = helper.make_tensor("TIDL_uv_scales", TensorProto.FLOAT, [4], 
                                      np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    
    # Add tensors to initList (following add_normalization_to_onnx_model)
    initList.extend([yuvBiasTensor, yuvWeightsTensor, yuvConvBiasTensor, biasTensor, scaleTensor, roiTensor, scalesTensor])
    
    # Add nodes in topological order (following add_normalization_to_onnx_model)
    nodeList = [cast_y, cast_uv, transpose_y, transpose_uv, resize_uv, concat_yuv, addYuvBias, yuvToRgbConv, addNode, scaleNode] + nodeList
    
    # Input sequence (following add_normalization_to_onnx_model)
    y_input_info = helper.make_tensor_value_info(original_input_name + "Net_Y_IN", TensorProto.UINT8, y_inDims)
    uv_input_info = helper.make_tensor_value_info(original_input_name + "Net_UV_IN", TensorProto.UINT8, uv_inDims)
    
    # Check for Argmax (exact copy from add_normalization_to_onnx_model)
    outSequence = originalGraph.output
    for node in nodeList:
        if node.op_type == "ArgMax":
            if node.output[0] == originalGraph.output[0].name:
                attrib_dict_1 = {"to": TensorProto.UINT8}
                cast_out = onnx.helper.make_node('Cast', inputs=[originalGraph.output[0].name],
                                                 outputs=[originalGraph.output[0].name + 'TIDL_cast_out'],
                                                 name='cast_output_to_uint8', **attrib_dict_1)
                nodeList = nodeList + [cast_out]
                outSequence = [
                    helper.make_tensor_value_info(originalGraph.output[0].name + 'TIDL_cast_out', TensorProto.UINT8,
                                                  outDims)]
    
    # Construct Graph (exact copy from add_normalization_to_onnx_model)
    newGraph = helper.make_graph(
        nodeList,
        'NV12_Rev_Model',
        [y_input_info, uv_input_info],
        outSequence,
        initList
    )
    
    # Construct Model (exact copy from add_normalization_to_onnx_model)
    model_def_noShape = helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
    model_def = shape_inference.infer_shapes(model_def_noShape)

    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print('NV12 + normalization converted model is invalid: %s' % e)
    else:
        print('NV12 + normalization converted model is valid!')
        onnx.save_model(model_def, output_path)
