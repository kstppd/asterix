project('tinyAI', ['cpp', 'c', 'cuda'], version: '1.0.0', meson_version: '>=1.3.0',
  default_options: [
    'warning_level=3',
    'cpp_std=c++20',
    'cuda_std=c++20',
    'b_staticpic=true',
  ],
)

fs = import('fs')
if not fs.exists('external')
  run_command('mkdir', '-p', 'external', check: true)
  run_command('git','clone','--depth', '1','https://github.com/nothings/stb.git','external/stb',check: true)
  run_command('git','clone','--depth', '1','https://github.com/kstppd/libnpy.git','external/libnpy',check: true)
endif

#Cuda args
tiny_cuda_arch_code = get_option('tiny_cuda_arch_code')
cuda_arch_args = ['-gencode=arch=compute_' + tiny_cuda_arch_code +',code=sm_' + tiny_cuda_arch_code]
cuda_base_args = ['--extended-lambda'] + cuda_arch_args
cuda_gpu_args  = cuda_base_args + ['-DUSE_GPU']
cuda_host_args = cuda_base_args


#Includes
stb_inc    = include_directories('external/stb', is_system: true)
npy_inc    = include_directories('external/libnpy/include', is_system: true)
tinyai_inc = include_directories('include')

#Deps
cuda_dep      = dependency('cuda',version: '>=11',modules: ['cublas', 'nvToolsExt'])
zlib_dep      = dependency('zlib')
mpi_dep       = dependency('mpi')
blas_dep      = dependency('openblas', required: false)
gtest_dep     = dependency('gtest',fallback: ['gtest', 'gtest_dep'])
spdlog_proj   = subproject('spdlog', default_options: ['default_library=shared'])
spdlog_dep    = spdlog_proj.get_variable('spdlog_dep')
curl_dep      = dependency('libcurl', required: true, fallback: ['curl', 'libcurl_dep'])
test_deps     = [cuda_dep,blas_dep,gtest_dep,spdlog_dep]
app_deps      = [cuda_dep,blas_dep,spdlog_dep]
mpi_test_deps = [cuda_dep,mpi_dep,blas_dep,gtest_dep,spdlog_dep]
mpi_app_deps  = [cuda_dep,mpi_dep,blas_dep,spdlog_dep]
mempool_deps  = [cuda_dep,blas_dep,gtest_dep]
mnist_deps    = [cuda_dep,spdlog_dep,gtest_dep,zlib_dep,blas_dep,curl_dep]
add_project_arguments('-DTINYAILOG', language: ['c', 'cpp', 'cuda'])
if not blas_dep.found()
  message('BLAS not found, enabling SKIP_HOSTBLAS')
  add_project_arguments(
    '-DSKIP_HOSTBLAS',
    language: ['c', 'cpp', 'cuda'],
  )
endif

#Binaries
image_nn                     = executable('image_nn','tests/image.cu',include_directories: [stb_inc, tinyai_inc],cuda_args: cuda_gpu_args,dependencies: test_deps,install: false)
nn_io                        = executable('nn_io','tests/nn_io.cu',include_directories: [tinyai_inc],cuda_args: cuda_gpu_args,dependencies: test_deps,install: false)
image_nn_ddp                 = executable('image_nn_ddp','src/image_ddp.cu',include_directories: [stb_inc, npy_inc, tinyai_inc],cuda_args: cuda_gpu_args,dependencies: mpi_test_deps,install: false)
surrogate                    = executable('surrogate','src/surrogate.cu',include_directories: [npy_inc, tinyai_inc],cuda_args: cuda_gpu_args,dependencies: app_deps,install: false)
surrogate_ddp                = executable('surrogate_ddp','src/surrogate_ddp.cu',include_directories: [npy_inc, tinyai_inc],cuda_args: cuda_gpu_args,dependencies: mpi_app_deps,install: false)
mnist                        = executable('mnist_nn','tests/mnist.cu',include_directories: [tinyai_inc],cuda_args: cuda_gpu_args,dependencies: mnist_deps,install: false)
matrix_unit_test_host_fp32   = executable('matrix_test_host_fp32','tests/unit_matrix.cu',include_directories: [tinyai_inc],cuda_args: cuda_host_args + ['-DFP32P'],dependencies: test_deps,install: false)
matrix_unit_test_device_fp32 = executable('matrix_test_device_fp32','tests/unit_matrix.cu',include_directories: [tinyai_inc],cuda_args: cuda_gpu_args + ['-DFP32P'],dependencies: test_deps,install: false)
matrix_unit_test_host_fp64   = executable('matrix_test_host_fp64','tests/unit_matrix.cu',include_directories: [tinyai_inc],cuda_args: cuda_host_args,dependencies: test_deps,install: false)
matrix_unit_test_device_fp64 = executable('matrix_test_device_fp64','tests/unit_matrix.cu',include_directories: [tinyai_inc],cuda_args: cuda_gpu_args,dependencies: test_deps,install: false)
mempool                      = executable('pool','tests/mempool.cpp',include_directories: [tinyai_inc],dependencies: mempool_deps,install: false)

#Shared libs
vlasiator_vdf_compressor_both = both_libraries('vlasiator_vdf_compressor_nn','src/vdf_compressor_nn.cu',include_directories: [tinyai_inc],cuda_args: cuda_gpu_args,dependencies: app_deps,install: true)


install_headers('include/genericTsPool.h',subdir: 'tinyAI')

#Unit tests
test('MatrixHost32'   , matrix_unit_test_host_fp32)
test('MatrixDevice32' , matrix_unit_test_device_fp32)
test('MatrixHost64'   , matrix_unit_test_host_fp64)
test('MatrixDevice64' , matrix_unit_test_device_fp64)
test('MempoolTest'    , mempool)
test('ImageRegression', image_nn, args: ['../assets/lena.png'])
test('IO'             , nn_io)
test('Mnist'          , mnist, is_parallel: false)
