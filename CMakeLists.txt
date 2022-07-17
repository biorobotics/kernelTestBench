cmake_minimum_required(VERSION 3.20)

project(kernelTestBench)

enable_language(CUDA)
enable_language(CXX)
 
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

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


set_source_files_properties(${kernel_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${kernel_header} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${kernel_camera_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${kernel_camera_header} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${routines} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${routines_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${util_src} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${util_header_cuda} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${util_header} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${test} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${includes} PROPERTIES LANGUAGE CUDA)

add_library(kernelLib SHARED ${kernel_src} ${kernel_header} ${kernel_camera_src} ${kernel_camera_header})
add_library(utilLib SHARED   ${util_src}   ${util_header}   ${util_header_cuda})
add_library(routineLib STATIC ${routines} ${routines_src})
add_library(testLib SHARED ${includes} ${test}) 

add_dependencies(routineLib utilLib kernelLib ${OpenCV_LIBS} CUDA::toolkit CUDA::nppim)
add_dependencies(testLib kernelLib utilLib routineLib ${OpenCV_LIBS} CUDA::toolkit CUDA::nppim)

set_target_properties(kernelLib  PROPERTIES SOVERSION 1)
set_target_properties(utilLib    PROPERTIES SOVERSION 1)
set_target_properties(routineLib PROPERTIES SOVERSION 1)
set_target_properties(testLib    PROPERTIES SOVERSION 1)

target_link_libraries(routineLib kernelLib utilLib ) 
target_link_libraries(testLib routineLib utilLib kernelLib)

set_target_properties(kernelLib PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)

add_executable(kernelTestBench main.cpp)
target_link_libraries(kernelTestBench kernelLib utilLib routineLib testLib)
