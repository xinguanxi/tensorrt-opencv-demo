cmake_minimum_required (VERSION 2.8.11)
project (opencv_tensorRT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)

INCLUDE_DIRECTORIES(
    /usr/local/cuda-8.0/include
    /home/xgx/library/TensorRT-3.0.4/include
    )

cuda_add_executable (${PROJECT_NAME} src/sampleMNIST.cpp src/common.h src/common.cpp)   

set(BOOST_LIB boost_system boost_filesystem boost_thread)

target_link_libraries (${PROJECT_NAME}
      ${OpenCV_LIBS}
      ${BOOST_LIB}
      cuda
      /home/xgx/library/TensorRT-3.0.4/lib/libnvcaffe_parser.so
      /home/xgx/library/TensorRT-3.0.4/lib/libnvinfer.so
      /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so
      /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
      glog gflags protobuf m 
      cblas atlas
      stdc++
      )

