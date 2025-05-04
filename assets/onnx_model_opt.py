import onnx
from onnx import helper
from onnx import TensorProto, shape_inference
import numpy as np


# https://github.com/TexasInstruments/edgeai-tidl-tools/blob/08_06_00_05/scripts/osrt_model_tools/onnx_tools/onnx_model_opt.py#L65
def tidlOnnxModelOptimize(in_model_path, out_model_path, scaleList=[0.0078125, 0.0078125, 0.0078125], meanList=[128.0, 128.0, 128.0]):
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
    op.version = 11
    model_def_noShape = helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
    model_def = shape_inference.infer_shapes(model_def_noShape)

    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print('Converted model is invalid: %s' % e)
    else:
        print('Converted model is valid!')
        onnx.save_model(model_def, out_model_path)
