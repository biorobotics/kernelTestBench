cmake_minimum_required(VERSION 3.20)

project(kernelTestBench)

enable_language(CUDA)
enable_language(CXX)
 
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SOURCE_DIR .)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE kernel_src CONFIGURE_DEPENDS kernels/*.cu)
file(GLOB_RECURSE kernel_header CONFIGURE_DEPENDS kernels/*.cuh)
file(GLOB_RECURSE kernel_camera_src CONFIGURE_DEPENDS kernels/cameraModule/*.cu)
file(GLOB_RECURSE kernel_camera_header CONFIGURE_DEPENDS kernels/cameraModule/*.cuh)
file(GLOB_RECURSE routines CONFIGURE_DEPENDS Routines/*.cuh)
file(GLOB_RECURSE util_src CONFIGURE_DEPENDS Utils/*.cu)
file(GLOB_RECURSE util_header_cuda CONFIGURE_DEPENDS Utils/*.cuh)
file(GLOB_RECURSE util_header CONFIGURE_DEPENDS Utils/*.h)

file(GLOB_RECURSE tests CONFIGURE_DEPENDS Tests/*.h)
file(GLOB_RECURSE includes CONFIGURE_DEPENDS Includes/*.h)


set_source_files_properties(${kernel_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${kernel_header} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${kernel_camera_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${kernel_camera_header} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${routines} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${util_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${util_header_cuda} PROPERTIES LANGUAGE CUDA)

add_executable(kernelTestBench main.cpp ${kernel_src} ${kernel_header} ${kernel_camera_src} ${kernel_camera_header}
                            ${routines} ${util_src} ${util_header} ${util_header}  ${includes})

set_target_properties(kernelTestBench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(kernelTestBench ${OpenCV_LIBS} CUDA::toolkit CUDA::nppim)
