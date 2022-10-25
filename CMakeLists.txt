cmake_minimum_required(VERSION 3.15)

project(kernelTestBench CUDA CXX)
 
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_ARCH "53") # setting default cuda architecture. Please make sure detected cuda architecture is correct

set(PREFIX_PATH ${CMAKE_BINARY_DIR})
set(SOURCE_FILE_PATH ${PREFIX_PATH}/detectCudaArch.cu)

execute_process(COMMAND nvcc ${SOURCE_FILE_PATH} -o detectarch OUTPUT_VARIABLE CUDA_COMPILE)
message(STATUS "Output Of Compilation = ${CUDA_COMPILE}")
if(NOT ${CUDA_COMPILE} STREQUAL "")
        message(FATAL_ERROR "error in compilation output, please check") 
        EXIT()
endif()

execute_process(COMMAND ./detectarch OUTPUT_VARIABLE CUDA_ARCH)
message(STATUS "Detected Cuda Architecture = ${CUDA_ARCH}")

set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)



file(GLOB_RECURSE kernel_src CONFIGURE_DEPENDS kernels/*.cu)
file(GLOB_RECURSE kernel_header CONFIGURE_DEPENDS kernels/*.cuh)
file(GLOB_RECURSE kernel_camera_src CONFIGURE_DEPENDS kernels/cameraModule/*.cu)
file(GLOB_RECURSE kernel_camera_header CONFIGURE_DEPENDS kernels/cameraModule/*.cuh)
file(GLOB_RECURSE routines CONFIGURE_DEPENDS Routines/*.cuh)
file(GLOB_RECURSE routines_src CONFIGURE_DEPENDS Routines/*.cu)
file(GLOB_RECURSE util_src CONFIGURE_DEPENDS Utils/*.cu)
file(GLOB_RECURSE util_header_cuda CONFIGURE_DEPENDS Utils/*.cuh)
file(GLOB_RECURSE util_header CONFIGURE_DEPENDS Utils/*.h)

file(GLOB_RECURSE tests CONFIGURE_DEPENDS Tests/*.h)
file(GLOB_RECURSE includes CONFIGURE_DEPENDS Includes/*.h)


add_executable(kernelTestBench main.cpp ${kernel_src} ${kernel_header} ${kernel_camera_src} ${kernel_camera_header} 
                                        ${util_src}   ${util_header}   ${util_header_cuda}
                                        ${routines} ${routines_src}
                                        ${includes} ${test}
)

set_target_properties(kernelTestBench PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

target_link_libraries(kernelTestBench CUDA::cudart CUDA::toolkit CUDA::nppim)
