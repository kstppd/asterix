#include "genericTsPool.h"
#include "matrix.h"
#include "tinyAI.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

constexpr size_t MEMPOOL_BYTES = 4ul * 1024ul * 1024ul * 1024ul;
constexpr size_t BATCHSIZE = 32;
#define USE_GPU
#define NORM_PER_VDF

typedef double Real;
typedef float Realf;
using namespace NumericMatrix;
using namespace TINYAI;

template <typename T> T* check_ptr(T* ptr) {
   if (ptr) {
      return ptr;
   }
   std::cerr << "INVALID POINTER DETECTED" << std::endl;
   abort();
   return nullptr;
}

template <typename T>
NumericMatrix::HostMatrix<T> generate_fourier_features(const NumericMatrix::MatrixView<T>& input,
                                                       NumericMatrix::HostMatrix<T>& B, std::size_t num_features,
                                                       T scale) {
   if (num_features == 0) {
      return NumericMatrix::HostMatrix<T>(input);
   }
   assert(num_features % 2 == 0 && num_features > 0);
   const std::size_t input_dims = input.ncols();
   // Construct B
   if (B.isEmpty()) {
      B = NumericMatrix::HostMatrix<T>(input_dims, num_features);
      std::mt19937 rng(128);
      std::uniform_real_distribution<T> dist(-1.0, 1.0);
      for (std::size_t i = 0; i < input_dims; ++i) {
         for (std::size_t j = 0; j < num_features; ++j) {
            B(i, j) = scale * dist(rng); // rand_normal<T>();
         }
      }
   }

   // Apply mapping
   NumericMatrix::HostMatrix<T> output(input.nrows(), 2 * num_features);
   for (std::size_t i = 0; i < input.nrows(); ++i) {
      for (std::size_t j = 0; j < num_features; ++j) {
         T dot_product = 0.0;
         for (std::size_t k = 0; k < input.ncols(); ++k) {
            assert(input(i, k) >= -1.0 && input(i, k) <= 1.0);
            dot_product += input(i, k) * B(k, j);
         }
         output(i, j) = std::sin(2.0 * M_PI * dot_product);
         output(i, j + num_features) = std::cos(2.0 * M_PI * dot_product);
      }
   }
   return output;
}

std::size_t compress_and_reconstruct_vdf(const MatrixView<Real>& vcoords, const MatrixView<Real>& vspace,
                                         MatrixView<Real>& inference_coords, std::size_t fourier_order,
                                         std::size_t max_epochs, std::vector<int>& arch, Real tolerance,
                                         HostMatrix<Real>& reconstructed_vdf, float& error, int& status) {

#ifdef USE_GPU
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMallocManaged(&mem, MEMPOOL_BYTES);
#else
   constexpr auto HW = BACKEND::HOST;
   void* mem = (void*)malloc(MEMPOOL_BYTES);
#endif
   GENERIC_TS_POOL::MemPool p(mem, MEMPOOL_BYTES);
   std::size_t network_size = 0;
   {

      NumericMatrix::HostMatrix<Real> B;
      Real scale = 1.0;
      NumericMatrix::HostMatrix<Real> ff_input = generate_fourier_features<Real>(vcoords, B, fourier_order, scale);
      NumericMatrix::Matrix<Real, HW> vcoords_train(ff_input.nrows(), ff_input.ncols(), &p);
      NumericMatrix::get_from_host(vcoords_train, ff_input);
      printf("Vcoords train shape = [%zu,%zu]\n", vcoords_train.nrows(), vcoords_train.ncols());

      NumericMatrix::HostMatrix<Real> ff_inf_input =
          generate_fourier_features<Real>(inference_coords, B, fourier_order, scale);
      NumericMatrix::Matrix<Real, HW> vcoords_inference(ff_inf_input.nrows(), ff_inf_input.ncols(), &p);
      NumericMatrix::get_from_host(vcoords_inference, ff_inf_input);

      NumericMatrix::Matrix<Real, HW> vspace_train(vspace.nrows(), vspace.ncols(), &p);
      NumericMatrix::Matrix<Real, HW> vspace_inference(inference_coords.nrows(), vspace.ncols(), &p);
      // Actually read in the vspace for training
      if constexpr (HW == BACKEND::HOST) {
         vspace_train.copy_to_host_from_host_view(vspace);
      } else {
         vspace_train.copy_to_device_from_host_view(vspace);
      }

      NeuralNetwork<Real, HW, ACTIVATION::TANH> nn(arch, &p, vcoords_train, vspace_train, BATCHSIZE);
      network_size = nn.get_network_size();

      error = std::numeric_limits<float>::max();
      status = 0;
      Real lr = 1e-4;
      Real current_lr = lr;
      for (std::size_t i = 0; i < max_epochs; i++) {
         error = nn.train(BATCHSIZE, current_lr);
         if (i % 1 == 0) {
            printf("Loss at epoch %zu: %f\n", i, error);
         }
         if (error < tolerance) {
            status = 1;
            break;
         }
         current_lr = lr * std::exp(-0.1 * i);
      }
      tinyAI_gpuDeviceSynchronize();
      p.defrag();
      // nn.cast_to_float();
      nn.evaluate(vcoords_inference, vspace_inference);
      vspace_inference.export_to_host(reconstructed_vdf);
      std::cout << "Network  size in bytes is " << nn.get_network_size() << std::endl;
   }
   // p.defrag();
   // printf("Pool HWM = %f\n", p.memory_hwm());
   // p.stats();
#ifdef USE_GPU
   tinyAI_gpuFree(mem);
#else
   free(mem);
#endif
   return network_size;
}

std::size_t compress_vdf(const MatrixView<Real>& vcoords, const MatrixView<Real>& vspace, std::size_t fourier_order,
                         std::size_t max_epochs, std::vector<int>& arch, Real* bytes, Real tolerance, float& error,
                         int& status) {

#ifdef USE_GPU
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMallocManaged(&mem, MEMPOOL_BYTES);
#else
   constexpr auto HW = BACKEND::HOST;
   void* mem = (void*)malloc(MEMPOOL_BYTES);
#endif
   GENERIC_TS_POOL::MemPool p(mem, MEMPOOL_BYTES);
   std::size_t network_size = 0;
   {

      Real scale = 1.0;
      NumericMatrix::HostMatrix<Real> B;
      NumericMatrix::HostMatrix<Real> ff_input = generate_fourier_features<Real>(vcoords, B, fourier_order, scale);
      NumericMatrix::Matrix<Real, HW> vcoords_train(ff_input.nrows(), ff_input.ncols(), &p);
      NumericMatrix::get_from_host(vcoords_train, ff_input);
      NumericMatrix::Matrix<Real, HW> vspace_train(vspace.nrows(), vspace.ncols(), &p);

      // Actually read in the vspace for training
      if constexpr (HW == BACKEND::HOST) {
         vspace_train.copy_to_host_from_host_view(vspace);
      } else {
         vspace_train.copy_to_device_from_host_view(vspace);
      }

      NeuralNetwork<Real, HW, ACTIVATION::TANH> nn(arch, &p, vcoords_train, vspace_train, BATCHSIZE);
      network_size = nn.get_network_size();

      error = std::numeric_limits<float>::max();
      status = 0;
      Real lr = 1e-4;
      Real current_lr = lr;
      for (std::size_t i = 0; i < max_epochs; i++) {
         error = nn.train(BATCHSIZE, current_lr);
         if (i % 1 == 0) {
            printf("Loss at epoch %zu: %f\n", i, error);
         }
         if (error < tolerance) {
            status = 1;
            break;
         }
         current_lr = lr * std::exp(-0.1 * i);
      }
      tinyAI_gpuDeviceSynchronize();
      p.defrag();
      if (bytes != nullptr) {
         nn.get_weights(bytes);
      }
   }
#ifdef USE_GPU
   tinyAI_gpuFree(mem);
#else
   free(mem);
#endif
   return network_size;
}

void uncompress_vdf(const MatrixView<Real>& vcoords, MatrixView<Real>& vspace, std::size_t fourier_order,
                    std::vector<int>& arch, const Real* bytes) {

#ifdef USE_GPU
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMallocManaged(&mem, MEMPOOL_BYTES);
#else
   constexpr auto HW = BACKEND::HOST;
   void* mem = (void*)malloc(MEMPOOL_BYTES);
#endif
   GENERIC_TS_POOL::MemPool p(mem, MEMPOOL_BYTES);
   {
      Real scale = 1.0;
      NumericMatrix::HostMatrix<Real> B;
      NumericMatrix::HostMatrix<Real> ff_input = generate_fourier_features<Real>(vcoords, B, fourier_order, scale);
      NumericMatrix::Matrix<Real, HW> vcoords_train(ff_input.nrows(), ff_input.ncols(), &p);
      NumericMatrix::get_from_host(vcoords_train, ff_input);
      NumericMatrix::Matrix<Real, HW> vspace_train(vspace.nrows(), vspace.ncols(), &p);

      // Actually read in the vspace for training
      if constexpr (HW == BACKEND::HOST) {
         vspace_train.copy_to_host_from_host_view(vspace);
      } else {
         vspace_train.copy_to_device_from_host_view(vspace);
      }

      NeuralNetwork<Real, HW, ACTIVATION::TANH> nn(arch, &p, vcoords_train, vspace_train, BATCHSIZE);
      if (bytes == nullptr) {
         abort();
      }
      nn.load_weights(bytes);
      nn.evaluate(vcoords_train, vspace_train);
      NumericMatrix::export_to_host_view(vspace_train,vspace );
   }
#ifdef USE_GPU
   tinyAI_gpuFree(mem);
#else
   free(mem);
#endif
   return;
}

extern "C" {

size_t compress_and_reconstruct_vdf(std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr,
                                    std::size_t size, std::array<Real, 3>* inference_vcoords_ptr, Realf* new_vspace_ptr,
                                    std::size_t inference_size, std::size_t max_epochs, std::size_t fourier_order,
                                    size_t* hidden_layers_ptr, size_t n_hidden_layers, Real sparsity, Real tol,
                                    Real* weights_ptr, std::size_t weight_size, bool use_input_weights,
                                    uint32_t downsampling_factor, float& error, int& status) {

   PROFILE_START("Copy IN");
   std::vector<Real> vdf;
   vdf.reserve(size * nVDFS);
   for (std::size_t i = 0; i < nVDFS * size; ++i) {
      vdf.push_back(static_cast<Real>(vspace_ptr[i]));
   }
   PROFILE_END();

   const std::size_t vdf_size = vdf.size() * sizeof(Real);

   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back(nVDFS);

   PROFILE_START("Prepare VDF");
   MatrixView<Real> vcoords = get_view_from_raw(&(vcoords_ptr[0][0]), size, 3);
   MatrixView<Real> inference_coords = get_view_from_raw(&(inference_vcoords_ptr[0][0]), inference_size, 3);
   MatrixView<Real> vspace = get_view_from_raw(vdf.data(), size, nVDFS);
   HostMatrix<Real> vspace_inference_host(inference_coords.nrows(), nVDFS);

   HostMatrix<Real> downsampled_coords;
   HostMatrix<Real> downsampled_vdf;
   if (downsampling_factor >= 1) {
      PROFILE_START("Downsample VDF");
      std::size_t downsampled_rows = vcoords.nrows() / downsampling_factor;
      downsampled_coords = HostMatrix<Real>(downsampled_rows, vcoords.ncols());
      downsampled_vdf = HostMatrix<Real>(downsampled_rows, vspace.ncols());

      for (std::size_t i = 0; i < downsampled_coords.nrows(); ++i) {

         for (std::size_t j = 0; j < vcoords.ncols(); ++j) {
            downsampled_coords(i, j) = vcoords(i * downsampling_factor, j);
         }

         for (std::size_t j = 0; j < vspace.ncols(); ++j) {
            downsampled_vdf(i, j) = vspace(i * downsampling_factor, j);
         }
      }

      // Change the views to the downsample versions
      vcoords = get_view_from_raw(downsampled_coords.data(), downsampled_coords.nrows(), downsampled_coords.ncols());
      vspace = get_view_from_raw(downsampled_vdf.data(), downsampled_vdf.nrows(), downsampled_vdf.ncols());

      PROFILE_END();
   }
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   const std::size_t network_bytes_used = compress_and_reconstruct_vdf(
       vcoords, vspace, inference_coords, fourier_order, max_epochs, arch, tol, vspace_inference_host, error, status);
   PROFILE_END();

   PROFILE_START("Unscale  and copy VDF out");
   // Copy back
   for (std::size_t i = 0; i < vspace_inference_host.size(); ++i) {
      new_vspace_ptr[i] = static_cast<Realf>(vspace_inference_host(i));
   }
   PROFILE_END();
   return network_bytes_used;
}

size_t compress_vdf_union(std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr, std::size_t size,
                          std::size_t max_epochs, std::size_t fourier_order, size_t* hidden_layers_ptr,
                          size_t n_hidden_layers, Real sparsity, Real tol, Real* weights_ptr, std::size_t weight_size,
                          bool use_input_weights, uint32_t downsampling_factor, float& error, int& status) {

   PROFILE_START("Copy IN");
   std::vector<Real> vdf;
   vdf.reserve(size * nVDFS);
   for (std::size_t i = 0; i < nVDFS * size; ++i) {
      vdf.push_back(static_cast<Real>(vspace_ptr[i]));
   }
   PROFILE_END();

   const std::size_t vdf_size = vdf.size() * sizeof(Real);

   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back(nVDFS);

   PROFILE_START("Prepare VDF");
   MatrixView<Real> vcoords = get_view_from_raw(&(vcoords_ptr[0][0]), size, 3);
   MatrixView<Real> vspace = get_view_from_raw(vdf.data(), size, nVDFS);

   HostMatrix<Real> downsampled_coords;
   HostMatrix<Real> downsampled_vdf;
   if (downsampling_factor >= 1) {
      PROFILE_START("Downsample VDF");
      std::size_t downsampled_rows = vcoords.nrows() / downsampling_factor;
      downsampled_coords = HostMatrix<Real>(downsampled_rows, vcoords.ncols());
      downsampled_vdf = HostMatrix<Real>(downsampled_rows, vspace.ncols());

      for (std::size_t i = 0; i < downsampled_coords.nrows(); ++i) {

         for (std::size_t j = 0; j < vcoords.ncols(); ++j) {
            downsampled_coords(i, j) = vcoords(i * downsampling_factor, j);
         }

         for (std::size_t j = 0; j < vspace.ncols(); ++j) {
            downsampled_vdf(i, j) = vspace(i * downsampling_factor, j);
         }
      }

      // Change the views to the downsample versions
      vcoords = get_view_from_raw(downsampled_coords.data(), downsampled_coords.nrows(), downsampled_coords.ncols());
      vspace = get_view_from_raw(downsampled_vdf.data(), downsampled_vdf.nrows(), downsampled_vdf.ncols());

      PROFILE_END();
   }
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   const std::size_t network_bytes_used =
       compress_vdf(vcoords, vspace, fourier_order, max_epochs, arch, weights_ptr, tol, error, status);
   PROFILE_END();
   return network_bytes_used;
}

void uncompress_vdf_union(std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr, std::size_t size,
                          std::size_t fourier_order, size_t* hidden_layers_ptr, size_t n_hidden_layers,
                          Real* weights_ptr, std::size_t weight_size, bool use_input_weights) {

   PROFILE_START("Copy IN");
   std::vector<Real> vdf;
   vdf.reserve(size * nVDFS);
   PROFILE_END();

   const std::size_t vdf_size = vdf.size() * sizeof(Real);

   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back(nVDFS);

   PROFILE_START("Prepare VDF");
   MatrixView<Real> vcoords = get_view_from_raw(&(vcoords_ptr[0][0]), size, 3);
   MatrixView<Real> vspace = get_view_from_raw(vdf.data(), size, nVDFS);
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   uncompress_vdf(vcoords, vspace, fourier_order, arch, weights_ptr);
   PROFILE_END();


   
   PROFILE_START("Copy VDF out");
   // Copy back
   for (std::size_t i = 0; i < vspace.size(); ++i) {
      vspace_ptr[i] = static_cast<Realf>(vspace(i));
   }
   PROFILE_END();

   return;
}
}
