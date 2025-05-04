#!/bin/bash

# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################
TIDL_PYTHON_INTERPRETER_PATH=/opt/conda/envs/tidl-py310/bin

compile_armnn(){
    #requires tflite2.12 to be build first
    cd $HOME
    export BASEDIR=~/ArmNNDelegate
    mkdir $BASEDIR
    cd $BASEDIR
    apt-get update && apt-get install git wget unzip zip python git cmake scons

    git clone "https://review.mlplatform.org/ml/armnn"
    cd armnn
    git checkout branches/armnn_22_02

    #Buikd compute lib
    cd $BASEDIR
    git clone https://review.mlplatform.org/ml/ComputeLibrary
    cd ComputeLibrary/
    git checkout $(../armnn/scripts/get_compute_library.sh -p) # e.g. v21.11
    # The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if
    # your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
    sed -i 's/aarch64-linux-gnu-/aarch64-none-linux-gnu-/' SConstruct
    scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0


    #compile protobuff
    cd $BASEDIR
    git clone -b v3.12.0 https://github.com/google/protobuf.git protobuf
    cd protobuf
    git submodule update --init --recursive
    ./autogen.sh
    mkdir arm64_build
    cd arm64_build
    CC=aarch64-none-linux-gnu-gcc \
    CXX=aarch64-none-linux-gnu-g++ \
    ../configure --host=aarch64-linux \
    --prefix=$BASEDIR/google/arm64_pb_install \
    --with-protoc=$BASEDIR/google/x86_64_pb_install/bin/protoc
    make install -j16

    #build flatbuffers
    cd $BASEDIR
    wget --quiet  -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
    tar xf flatbuffers-1.12.0.tar.gz
    cd flatbuffers-1.12.0
    rm -f CMakeCache.txt
    mkdir build-arm64
    cd build-arm64
    # Add -fPIC to allow us to use the libraries in shared objects.
    CXXFLAGS="-fPIC" cmake .. -DCMAKE_C_COMPILER=aarch64-none-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-none-linux-gnu-g++ \
        -DFLATBUFFERS_BUILD_FLATC=1 \
        -DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-arm64 \
        -DFLATBUFFERS_BUILD_TESTS=0
    make all install

    # build tflite 2.5
    cd $BASEDIR
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow/
    git checkout v2.5.0
    cd ..
    mkdir -p tflite/build
    cd tflite/build
    ARMCC_PREFIX=$ARM64_GCC_PATH/bin/aarch64-none-linux-gnu-\
    ARMCC_FLAGS="-funsafe-math-optimizations" \
    cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
    -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
    -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DCMAKE_SYSTEM_NAME=Linux \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    $BASEDIR/tensorflow/tensorflow/lite/

    cmake --build .

    #finally build armnn
    cd $BASEDIR
    cd armnn
    rm -rf build # Remove any previous cmake build.
    mkdir build && cd build
    # if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
    CXX=aarch64-none-linux-gnu-g++
    CC=aarch64-none-linux-gnu-gcc
    cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
        -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build/ \
        -DARMCOMPUTENEON=1 \
        -DARMCOMPUTECL=1 \
        -DARMNNREF=1 \
        -DBUILD_TF_LITE_PARSER=1 \
        -DTENSORFLOW_ROOT=$BASEDIR/tensorflow/ \
        -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-1.12.0/ \
        -DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build \
        -DPROTOBUF_ROOT=$BASEDIR/google/x86_64_pb_install \
        -DPROTOBUF_ROOT=$BASEDIR/google/x86_64_pb_install/ \
        -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
        -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
        -DTF_LITE_SCHEMA_INCLUDE_PATH=$BASEDIR/tensorflow/tensorflow/lite/schema \
        -DTFLITE_LIB_ROOT=$BASEDIR/tflite/build/ \
        -DBUILD_ARMNN_TFLITE_DELEGATE=1
    make
}

download_armnn_repo(){
    export BASEDIR=~/ArmNNDelegate
    mkdir $BASEDIR
    git clone --depth 1 --single-branch -b branches/armnn_22_02 https://review.mlplatform.org/ml/armnn
    cd armnn
    git checkout branches/armnn_22_02
    cd $HOME
    ln -s $BASEDIR/armnn armnn
}

download_armnn_lib(){
    export TIDL_TARGET_LIBS=$SCRIPTDIR/tidl_target_libs
    mkdir $TIDL_TARGET_LIBS
    cd $TIDL_TARGET_LIBS
    wget --quiet  https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_03_00_19/libarmnnDelegate.so
    wget --quiet  https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_03_00_19/libarmnn.so
}

pip_install_local()
{
    if [ -f $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 ];then
        echo "Local file  found. Installing $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        ${TIDL_PYTHON_INTERPRETER_PATH}/pip install --force-reinstall $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    else
        echo "Local file not found at $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1. Installing default  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        ${TIDL_PYTHON_INTERPRETER_PATH}/pip install https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    fi
}

cp_tidl_tools()
{
    if [ -f $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz ];then
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Local file  found. Copying $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz"
            cp $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz .
        else
            echo "Local file  found. Copying $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz"
            cp $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz .
        fi
    else
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Local file not found at $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz . Downloading  default   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz"
            wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz
        else
            echo "Local file not found at $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz . Downloading  default   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools.tar.gz"
            wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools.tar.gz
        fi
    fi
}

cp_osrt_lib()
{
    if [ -f $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 ];then
        echo "Local file  found. Copying $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        cp  $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 .

    else
        echo "Local file not found at  $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 . Downloading  default https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    fi
}


SCRIPTDIR=`pwd`
REL=10_01_04_00
skip_cpp_deps=0
skip_arm_gcc_download=0
skip_x86_python_install=0
use_local=0
load_armnn=0
skip_model_optimizer=0


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --skip_cpp_deps)
    skip_cpp_deps=1
    ;;
    --skip_arm_gcc_download)
    skip_arm_gcc_download=1
    ;;
    --skip_x86_python_install)
    skip_x86_python_install=1
    ;;
    --use_local)
    use_local=1
    ;;
    --load_armnn)
    load_armnn=1
    ;;
    --skip_model_optimizer)
    skip_model_optimizer=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_cpp_deps            Skip Downloading or Compiling dependencies for CPP examples
    echo --skip_arm_gcc_download    Skip Downloading or setting environment variable  for ARM64_GCC_PATH
    echo --skip_x86_python_install  Skip installing of python packages
    echo --use_local                use OSRT packages and tidl_tools from localPath if present
    echo --load_armnn               load amrnn libs  for arm
    echo --skip_model_optimizer     skip installing model optimizer python package
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters

#Check if CPU or GPU tools
if [ -z "$TIDL_TOOLS_TYPE" ];then
    echo "Defaulting to CPU tools"
    tidl_gpu_tools=0
else
    echo "TIDL_TOOLS_TYPE set to :$TIDL_TOOLS_TYPE"
    if [ $TIDL_TOOLS_TYPE == GPU ];then
        tidl_gpu_tools=1
    else
        tidl_gpu_tools=0
    fi
fi


version_match=`${TIDL_PYTHON_INTERPRETER_PATH}/python -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
    echo 'python version must be >= 3.6'
return
fi

arch=$(uname -m)
if [[ $arch == x86_64 ]]; then
    echo "X64 Architecture"
elif [[ $arch == aarch64 ]]; then
    echo "ARM Architecture"
    $skip_arm_gcc_download=1
else
    echo 'Processor Architecture must be x86_64 or aarch64'
    echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

if [[ $use_local == 1 ]];then
    if [ -z "$LOCAL_PATH" ];then
        echo "LOCAL_PATH not defined. set LOCAL_PATH to/your/path/08_XX_XX_XX/"
        return
    else
        echo "using OSRT from LOCAL_PATH:$LOCAL_PATH"
    fi

fi

if [ -z "$SOC" ];then
    echo "SOC not defined. Run either of below commands"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68pa | j721e"
    echo "export SOC=am68a  | j721s2"
    echo "export SOC=am69a  | j784s4"
    echo "export SOC=am67a  | j722s"
    return
fi

case "$SOC" in
  am62|am62a|am68a|am68pa|am69a|am67a)
    ;;
  j721e)
    SOC=am68pa
    ;;
  j721s2)
    SOC=am68a
    ;; 
  j784s4)
    SOC=am69a
    ;;
  j722s)
    SOC=am67a
    ;;
  *)
    echo "Invalid SOC $SOC defined. Allowed values are"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68pa | j721e"
    echo "export SOC=am68a  | j721s2"
    echo "export SOC=am69a  | j784s4"
    echo "export SOC=am67a  | j722s"
    return
    ;;
esac

echo "SOC=${SOC}"

# ######################################################################
# # Installing dependencies
if [[ $arch == x86_64 && $skip_x86_python_install -eq 0 ]]; then
    echo 'Installing python packages...'
    ${TIDL_PYTHON_INTERPRETER_PATH}/pip install pybind11[global]
    # pip3 install -r ./requirements_pc.txt
fi
if [[ $arch == x86_64 ]]; then
    ${TIDL_PYTHON_INTERPRETER_PATH}/pip install pybind11[global]
    if [[ $use_local == 1 ]];then
        echo 'Installing python osrt packages from local...'
        pip_install_local dlr-1.13.0-py3-none-any.whl
        pip_install_local tvm-0.12.0-cp310-cp310-linux_x86_64.whl
        pip_install_local onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
        pip_install_local tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
    else
        echo 'Installing python osrt packages...'
        ${TIDL_PYTHON_INTERPRETER_PATH}/pip install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
        ${TIDL_PYTHON_INTERPRETER_PATH}/pip install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
        ${TIDL_PYTHON_INTERPRETER_PATH}/pip install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
        ${TIDL_PYTHON_INTERPRETER_PATH}/pip install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
    fi
fi

# Create tools directory
mkdir -p $SCRIPTDIR/tools

if [ -z "$TIDL_TOOLS_PATH" ]; then

    mkdir -p $SCRIPTDIR/tools/${SOC^^}/
    cd tools/${SOC^^}/

    if [ -f tidl_tools.tar.gz ];then
        rm tidl_tools.tar.gz
    fi
    if [ -f tidl_tools_gpu.tar.gz ];then
        rm tidl_tools_gpu.tar.gz
    fi
    if [ -d tidl_tools ];then
        rm -r tidl_tools
    fi

    if [[ $use_local == 1 ]];then
        cp_tidl_tools ${SOC^^}
    else
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Downloading GPU TIDL TOOLS for ${SOC^^} ..."
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/${SOC^^}/tidl_tools_gpu.tar.gz
        else
            echo "Downloading CPU TIDL TOOLS ${SOC^^} ..."
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/${SOC^^}/tidl_tools.tar.gz
        fi
    fi

    # Untar tidl tools & remove the tar ball
    if [ $tidl_gpu_tools -eq 1 ];then
        tar -xzf tidl_tools_gpu.tar.gz
        if [ -f tidl_tools_gpu.tar.gz ];then
            rm tidl_tools_gpu.tar.gz
        fi
    else
        tar -xzf tidl_tools.tar.gz
        if [ -f tidl_tools.tar.gz ];then
            rm tidl_tools.tar.gz
        fi
    fi
    cd tidl_tools
    if [[ ! -L libvx_tidl_rt.so.1.0 && ! -f libvx_tidl_rt.so.1.0 ]];then
         ln -s  libvx_tidl_rt.so libvx_tidl_rt.so.1.0
    fi
    export TIDL_TOOLS_PATH=$(pwd)
    cd $SCRIPTDIR
else
    echo "TIDL_TOOLS_PATH already set to ${TIDL_TOOLS_PATH}. Skipping..."
fi

# graph optimizer tool setup
if [[ $arch == x86_64 && $skip_model_optimizer -eq 0 ]]; then
    cd /home/workdir/osrt-model-tools
    source ./setup.sh
    cd $SCRIPTDIR
fi

if [[ $arch == x86_64 && $skip_arm_gcc_download -eq 0 ]]; then
    cd $SCRIPTDIR/tools/
    if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu ];then
        wget --quiet  https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    else
        echo "skipping gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu download: found $(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    fi
    cd $SCRIPTDIR
fi

if [[ $arch == x86_64 ]]; then
    cd $SCRIPTDIR/tools/
    if [ -f $CGT7X_ROOT/bin/cl7x ]; then
        echo "CGT7X_ROOT already set to $CGT7X_ROOT, skipping download"
    else
        if [ ! -d ti-cgt-c7000_3.1.0.LTS ];then
            wget --quiet https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/3.1.0.LTS/ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin
            chmod +x ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin
            ./ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin --mode unattended --installdir $(pwd)
            export CGT7X_ROOT=$(pwd)/ti-cgt-c7000_3.1.0.LTS
        else
            echo "skipping ti-cgt-c7000_3.1.0.LTS download: found $(pwd)/ti-cgt-c7000_3.1.0.LTS"
            export CGT7X_ROOT=$(pwd)/ti-cgt-c7000_3.1.0.LTS
        fi
    fi
    cd $SCRIPTDIR
fi

if [ $skip_cpp_deps -eq 0 ]; then
    if [[ $arch == x86_64 ]]; then
        cd $SCRIPTDIR/tools/
        if [ -d osrt_deps ];then
            rm -r osrt_deps
        fi
        mkdir -p osrt_deps
        cd osrt_deps
        # onnxruntime
        echo "Installing:onnxruntime"
        if [ -f onnx_1.15.0_x86_u22.tar.gz ];then
            rm onnx_1.15.0_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib onnx_1.15.0_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnx_1.15.0_x86_u22.tar.gz
        fi
        tar -xf onnx_1.15.0_x86_u22.tar.gz
        cd onnx_1.15.0_x86_u22
        if [ ! -f libonnxruntime.so ];then
            ln -s libonnxruntime.so.1.15.0 libonnxruntime.so
        fi
        if [ ! -f libonnxruntime.so.1.15.0 ];then
            ln -s libonnxruntime.so libonnxruntime.so.1.15.0
        fi
        cd ../
        rm onnx_1.15.0_x86_u22.tar.gz

        # tflite
        echo "Installing:tflite_2.12"
        if [ -f tflite_2.12_x86_u22.tar.gz ];then
            rm tflite_2.12_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib tflite_2.12_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_2.12_x86_u22.tar.gz
        fi
        tar -xf tflite_2.12_x86_u22.tar.gz
        rm tflite_2.12_x86_u22.tar.gz   -r

        # opencv
        echo "Installing:opencv"
        if [ -f opencv_4.2.0_x86_u22.tar.gz ];then
            rm opencv_4.2.0_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib opencv_4.2.0_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/opencv_4.2.0_x86_u22.tar.gz
        fi
        mkdir opencv_4.2.0_x86_u22 && tar xf opencv_4.2.0_x86_u22.tar.gz -C opencv_4.2.0_x86_u22 --strip-components 1
        rm opencv_4.2.0_x86_u22.tar.gz

        # dlr
        echo "Installing:dlr"
        if [ -f dlr_1.10.0_x86_u22.tar.gz ];then
            rm dlr_1.10.0_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib dlr_1.10.0_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr_1.10.0_x86_u22.tar.gz
        fi
        mkdir dlr_1.10.0_x86_u22 && tar xf dlr_1.10.0_x86_u22.tar.gz -C dlr_1.10.0_x86_u22 --strip-components 1
        rm dlr_1.10.0_x86_u22.tar.gz   -r

dlr_loc=$(${TIDL_PYTHON_INTERPRETER_PATH}/python  << EOF
import dlr
print(dlr.__file__)
EOF
)
        suffix="__init__.py"
        dlr_loc=${dlr_loc%"$suffix"}
        cp $dlr_loc/libdlr.so .

    fi

fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$SCRIPTDIR/tools/osrt_deps:$SCRIPTDIR/tools/osrt_deps/opencv_4.2.0_x86_u22/opencv/

if [ $load_armnn -eq 1 ]; then
    if [[ $arch == x86_64 ]]; then
        if [ ! -d armnn ];then
            download_armnn_repo
        fi
        if [ ! -f $SCRIPTDIR/tidl_target_libs/libarmnnDelegate.so ];then
            download_armnn_lib
        fi
    fi
fi

cd $TIDL_TOOLS_PATH
ln -s -r $SCRIPTDIR/tools/osrt_deps/ &> /dev/null
cd $SCRIPTDIR

echo "========================================================================="
echo "REL=$REL"
echo "SOC=$SOC"
echo "TIDL_TOOLS_PATH=$TIDL_TOOLS_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CGT7X_ROOT=$CGT7X_ROOT"
echo "ARM64_GCC_PATH=$ARM64_GCC_PATH"
echo "========================================================================="