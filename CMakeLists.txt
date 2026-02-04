cmake_minimum_required(VERSION 3.24)
project(tinyAI VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)


add_compile_options(-Wall -Wextra)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set(CMAKE_CUDA_ARCHITECTURES 80)
include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_Declare(
  libnpy
  GIT_REPOSITORY https://github.com/kstppd/libnpy.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(stb libnpy)
set(STB_INCLUDE_DIR "${stb_SOURCE_DIR}")
set(LIBNPY_INCLUDE_DIR "${libnpy_SOURCE_DIR}/include")

find_package(CUDAToolkit 11 REQUIRED)
find_package(MPI REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CURL REQUIRED)
find_package(spdlog REQUIRED)

# BLAS / OpenBLAS (try OpenBLAS first, then BLAS)
find_package(OpenBLAS QUIET)
if (OpenBLAS_FOUND)
  set(BLAS_LIB_TARGET OpenBLAS::OpenBLAS)
else()
  find_package(BLAS REQUIRED)
  set(BLAS_LIB_TARGET BLAS::BLAS)
endif()

find_package(GTest QUIET)
if (NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

function(tinyai_cuda_target_common tgt)
  target_compile_definitions(${tgt} PRIVATE USE_GPU)
endfunction()


add_executable(image_nn tests/image.cu)
target_include_directories(image_nn SYSTEM PRIVATE "${STB_INCLUDE_DIR}")
target_include_directories(image_nn SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
tinyai_cuda_target_common(image_nn)
target_link_libraries(image_nn PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(nn_io tests/nn_io.cu)
tinyai_cuda_target_common(nn_io)
target_link_libraries(nn_io PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(image_nn_ddp src/image_ddp.cu)
target_include_directories(image_nn_ddp SYSTEM PRIVATE "${STB_INCLUDE_DIR}")
target_include_directories(image_nn_ddp PRIVATE "${LIBNPY_INCLUDE_DIR}")
tinyai_cuda_target_common(image_nn_ddp)
target_link_libraries(image_nn_ddp PRIVATE CUDA::cudart CUDA::cublas MPI::MPI_CXX ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(minst_nn tests/mnist.cu)
target_link_libraries(minst_nn PRIVATE CUDA::cudart CUDA::cublas CURL::libcurl spdlog::spdlog GTest::gtest GTest::gtest_main ZLIB::ZLIB ${BLAS_LIB_TARGET})
target_link_libraries(minst_nn PRIVATE CUDA::nvToolsExt)

add_executable(matrix_test_host_fp32 tests/unit_matrix.cu)
target_compile_definitions(matrix_test_host_fp32 PRIVATE FP32P)
target_link_libraries(matrix_test_host_fp32 PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(matrix_test_device_fp32 tests/unit_matrix.cu)
target_compile_definitions(matrix_test_device_fp32 PRIVATE USE_GPU FP32P)
target_link_libraries(matrix_test_device_fp32 PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(matrix_test_host_fp64 tests/unit_matrix.cu)
target_link_libraries(matrix_test_host_fp64 PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(matrix_test_device_fp64 tests/unit_matrix.cu)
target_compile_definitions(matrix_test_device_fp64 PRIVATE USE_GPU)
target_link_libraries(matrix_test_device_fp64 PRIVATE CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET} spdlog::spdlog GTest::gtest GTest::gtest_main)

add_executable(pool tests/mempool.cpp)
target_link_libraries(pool PRIVATE CUDA::cudart ${BLAS_LIB_TARGET} GTest::gtest GTest::gtest_main)

set(VDF_SRC src/vdf_compressor_nn.cu)
add_library(vlasiator_vdf_compressor_nn_static STATIC ${VDF_SRC})
add_library(vlasiator_vdf_compressor_nn_shared SHARED ${VDF_SRC})
target_include_directories(vlasiator_vdf_compressor_nn_static PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(vlasiator_vdf_compressor_nn_shared PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(vlasiator_vdf_compressor_nn_static PUBLIC CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET})
target_link_libraries(vlasiator_vdf_compressor_nn_shared PUBLIC CUDA::cudart CUDA::cublas ${BLAS_LIB_TARGET})
target_link_libraries(vlasiator_vdf_compressor_nn_static PUBLIC CUDA::nvToolsExt)
target_link_libraries(vlasiator_vdf_compressor_nn_shared PUBLIC CUDA::nvToolsExt)

# Install the shared+static libraries
install(TARGETS
  vlasiator_vdf_compressor_nn_static
  vlasiator_vdf_compressor_nn_shared
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/genericTsPool.h"
        DESTINATION include/tinyAI)

add_test(NAME MatrixHost32   COMMAND matrix_test_host_fp32)
add_test(NAME MatrixDevice32 COMMAND matrix_test_device_fp32)
add_test(NAME MatrixHost64   COMMAND matrix_test_host_fp64)
add_test(NAME MatrixDevice64 COMMAND matrix_test_device_fp64)
add_test(NAME MempoolTets    COMMAND pool)

add_test(NAME ImageRegresion COMMAND image_nn "${CMAKE_CURRENT_SOURCE_DIR}/../assets/lena.png")
add_test(NAME IO   COMMAND nn_io)
add_test(NAME Mnist COMMAND minst_nn)

