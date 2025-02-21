project(
    'tinyAI3',
    'cpp',
    'cuda',
    version: '1.0.0',
    default_options: ['warning_level=3', 'cpp_std=c++20'],
)

CUSTOM_INSTALL_PATH = '/home/kstppd/software/'

add_global_arguments(['-Xcompiler', '-Wno-pedantic','-std=c++20'], language: 'cuda')
stb_image = include_directories(CUSTOM_INSTALL_PATH + 'stb/', is_system: true)
spdlog_include = include_directories(
    CUSTOM_INSTALL_PATH + '/spdlog/include/',
    is_system: true,
)
libspdlog = meson.get_compiler('cuda').find_library(
    'spdlog',
    dirs: [CUSTOM_INSTALL_PATH + 'spdlog/build'],
)
libcurl = meson.get_compiler('cuda').find_library('curl')
cuda_dep = dependency('cuda', version: '>=11', modules: ['cublas', 'nvToolsExt'])
zlib_dep = dependency('zlib')
blas_dep = dependency('openblas')
gtest_dep = dependency('gtest', fallback: ['gtest', 'gtest_dep'])

image_nn = executable(
    'image_nn',
    'src/image.cu',
    include_directories: [spdlog_include, stb_image],
    cuda_args: ['-DUSE_GPU'],
    dependencies: [cuda_dep, blas_dep, gtest_dep],
    install: true,
)

test_mnist = executable(
    'minst_nn',
    'src/mnist.cu',
    include_directories: [spdlog_include],
    dependencies: [cuda_dep, libcurl, zlib_dep, blas_dep],
    install: true,
)

matrix_unit_test_host_fp32 = executable(
    'matrix_test_host_fp32',
    'src/unit_matrix.cu',
    cuda_args: ['-DFP32P'],
    include_directories: [spdlog_include],
    dependencies: [cuda_dep, blas_dep, gtest_dep],
)

matrix_unit_test_device_fp32 = executable(
    'matrix_test_device_fp32',
    'src/unit_matrix.cu',
    cuda_args: ['-DUSE_GPU', '-DFP32P'],
    include_directories: [spdlog_include],
    dependencies: [cuda_dep, blas_dep, gtest_dep],
)

matrix_unit_test_host_fp64 = executable(
    'matrix_test_host_fp64',
    'src/unit_matrix.cu',
    include_directories: [spdlog_include],
    dependencies: [cuda_dep, blas_dep, gtest_dep],
)

matrix_unit_test_device_fp64 = executable(
    'matrix_test_device_fp64',
    'src/unit_matrix.cu',
    cuda_args: ['-DUSE_GPU'],
    include_directories: [spdlog_include],
    dependencies: [cuda_dep, blas_dep, gtest_dep],
)

mempool = executable(
    'pool',
    'src/mempool.cpp',
    dependencies: [cuda_dep, blas_dep, gtest_dep],
)

test('MatrixHost32', matrix_unit_test_host_fp32)
test('MatrixDevice32', matrix_unit_test_device_fp32)
test('MatrixHost64', matrix_unit_test_host_fp64)
test('MatrixDevice64', matrix_unit_test_device_fp64)
test('MempoolTets', mempool)
test('ImageRegresion', image_nn, args : ['../assets/lena.png'])
