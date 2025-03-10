project(
    'tinyAI',
    'cpp',
    'c',
    'cuda',
    version: '1.0.0',
    default_options: ['warning_level=3', 'cpp_std=c++20'],
)

# Download stb image
runner = run_command('mkdir', '-p' ,'external/stb', check: true)
runner = run_command('git', 'clone', 'https://github.com/nothings/stb.git', 'external/stb', check: false)
runner = run_command('git', 'clone', 'https://github.com/kstppd/libnpy.git', 'external/libnpy', check: false)
CUSTOM_INSTALL_PATH='./external/'

add_global_arguments(['-Xcompiler', '-Wno-pedantic','-std=c++20','--extended-lambda'], language: 'cuda')
stb_image = include_directories(CUSTOM_INSTALL_PATH + 'stb/', is_system: true)
libcurl = meson.get_compiler('cuda').find_library('curl')
cuda_dep = dependency('cuda', version: '>=11', modules: ['cublas', 'nvToolsExt'])
zlib_dep = dependency('zlib')
mpi_dep = dependency('mpi')
blas_dep = dependency('openblas')
gtest_dep = dependency('gtest', fallback: ['gtest', 'gtest_dep'])
spdlog_dep = dependency('spdlog')
include_npy=CUSTOM_INSTALL_PATH+'./libnpy/include'

image_nn = executable(
    'image_nn',
    'tests/image.cu',
    include_directories: [stb_image],
    cuda_args: ['-DUSE_GPU'],
    dependencies: [cuda_dep, blas_dep, gtest_dep,spdlog_dep],
    install: false,
)

image_nn_dpp = executable(
    'image_nn_ddp',
    'src/image_ddp.cu',
    include_directories: [stb_image,include_npy],
    cuda_args: ['-DUSE_GPU'],
    dependencies: [cuda_dep,mpi_dep,  blas_dep, gtest_dep,spdlog_dep],
    install: false,
)

surrogate= executable(
    'surrogate',
    'src/surrogate.cu',
    include_directories: [include_npy],
    cuda_args: ['-DUSE_GPU'],
    dependencies: [cuda_dep, blas_dep, spdlog_dep],
    install: false,
)

surrogate_ddp= executable(
    'surrogate_ddp',
    'src/surrogate_ddp.cu',
    include_directories: [include_npy],
    cuda_args: ['-DUSE_GPU'],
    dependencies: [cuda_dep,mpi_dep, blas_dep, spdlog_dep],
    install: false,
)

mnist = executable(
    'minst_nn',
    'tests/mnist.cu',
    dependencies: [cuda_dep, libcurl,spdlog_dep,gtest_dep, zlib_dep, blas_dep],
    install: false,
)

matrix_unit_test_host_fp32 = executable(
    'matrix_test_host_fp32',
    'tests/unit_matrix.cu',
    cuda_args: ['-DFP32P'],
    dependencies: [cuda_dep, blas_dep, gtest_dep,spdlog_dep],
)

matrix_unit_test_device_fp32 = executable(
    'matrix_test_device_fp32',
    'tests/unit_matrix.cu',
    cuda_args: ['-DUSE_GPU', '-DFP32P'],
    dependencies: [cuda_dep, blas_dep, gtest_dep,spdlog_dep],
)

matrix_unit_test_host_fp64 = executable(
    'matrix_test_host_fp64',
    'tests/unit_matrix.cu',
    dependencies: [cuda_dep, blas_dep, gtest_dep,spdlog_dep],
)

matrix_unit_test_device_fp64 = executable(
    'matrix_test_device_fp64',
    'tests/unit_matrix.cu',
    cuda_args: ['-DUSE_GPU'],
    dependencies: [cuda_dep, blas_dep, gtest_dep,spdlog_dep],
)

mempool = executable(
    'pool',
    'tests/mempool.cpp',
    dependencies: [cuda_dep, blas_dep, gtest_dep],
)

vlasiator_vdf_compressor = library(
    'vlasiator_vdf_compressor_nn',
    'src/vdf_compressor_nn.cu',
    include_directories: ['./include'],
    dependencies: [cuda_dep, blas_dep, spdlog_dep],
    install : true)
install_headers('include/genericTsPool.h', subdir : 'tinyAI') 

test('MatrixHost32', matrix_unit_test_host_fp32)
test('MatrixDevice32', matrix_unit_test_device_fp32)
test('MatrixHost64', matrix_unit_test_host_fp64)
test('MatrixDevice64', matrix_unit_test_device_fp64)
test('MempoolTets', mempool)
test('ImageRegresion', image_nn, args : ['../assets/lena.png'])
test('Mnist', mnist,is_parallel:false,)
