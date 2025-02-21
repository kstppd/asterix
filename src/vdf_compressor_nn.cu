 /*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 USA.
 */

#include "genericTsPool.h"
#include "matrix.h"
#include "tinyAI.h"
#include <array>
#include <vector>

#define ACT ACTIVATION::RELU
#define LR 5e-5
constexpr size_t MEMPOOL_BYTES = 5ul * 1024ul * 1024ul * 1024ul;
constexpr size_t BATCHSIZE = 64;
#define USE_GPU
#ifdef USE_GPU
constexpr auto HW = BACKEND::DEVICE;
#else
constexpr auto HW = BACKEND::HOST;
#endif
#define USE_PATIENCE

typedef double Real;
typedef float Realf;
using namespace NumericMatrix;
using namespace TINYAI;

template <typename T>
T* check_ptr(T* ptr) {
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

std::size_t compress_vdf(GENERIC_TS_POOL::MemPool* p, const MatrixView<Real>& vcoords, const MatrixView<Real>& vspace,
                         std::size_t fourier_order, std::size_t max_epochs, std::vector<int>& arch, Real* bytes,
                         Real tolerance, float& error, int& status) {
   std::size_t network_size = 0;
   {

      Real scale = 1.0;
      NumericMatrix::HostMatrix<Real> B;
      NumericMatrix::HostMatrix<Real> ff_input = generate_fourier_features<Real>(vcoords, B, fourier_order, scale);
      NumericMatrix::Matrix<Real, HW> vcoords_train(ff_input.nrows(), ff_input.ncols(), p);
      NumericMatrix::get_from_host(vcoords_train, ff_input);
      NumericMatrix::Matrix<Real, HW> vspace_train(vspace.nrows(), vspace.ncols(), p);

      // Actually read in the vspace for training
      if constexpr (HW == BACKEND::HOST) {
         vspace_train.copy_to_host_from_host_view(vspace);
      } else {
         vspace_train.copy_to_device_from_host_view(vspace);
      }

      NeuralNetwork<Real, HW, ACT, ACT, LOSSF::LOGCOSH> nn(arch, p, vcoords_train, vspace_train, BATCHSIZE);
      network_size = nn.get_network_size();

      error = std::numeric_limits<float>::max();
      status = 0;
      Real lr = LR;
      Real current_lr = lr;
      Real min_loss = std::numeric_limits<Real>::max();
      std::size_t patience_counter = 0;
      constexpr std::size_t patience = 8;
      tinyAI_gpuStream_t s;
      tinyAI_gpuStreamCreate(&s);
      for (std::size_t i = 0; i < max_epochs; i++) {
         error = nn.train(BATCHSIZE, current_lr, s);
         if (i % 1 == 0) {
            fprintf(stderr, "Loss at epoch %zu: [%f] patience counter= [%zu] \n", i, error, patience_counter);
         }
         if (i > 20 && error > 0.1) {
            fprintf(stderr, "NETWORK RESET\n");
            nn.reset();
            i = 0;
            patience_counter = 0;
         }
#ifdef USE_PATIENCE
         if (error < 0.995 * min_loss) {
            min_loss = error;
            patience_counter = 0;
            nn.get_weights(bytes);
         } else {
            patience_counter++;
         }
         if (patience_counter > patience && i > 30) {
            break;
         }
#endif
         if (error < tolerance && i > 20) {
            nn.get_weights(bytes);
            status = 1;
            break;
         }
         current_lr = lr * std::exp(-0.1 * i);
      }
      tinyAI_gpuDeviceSynchronize();
   }
   return network_size;
}

void uncompress_vdf(GENERIC_TS_POOL::MemPool* p, const MatrixView<Real>& vcoords, MatrixView<Real>& vspace,
                    std::size_t fourier_order, std::vector<int>& arch, const Real* bytes) {

   {
      Real scale = 1.0;
      NumericMatrix::HostMatrix<Real> B;
      NumericMatrix::HostMatrix<Real> ff_input = generate_fourier_features<Real>(vcoords, B, fourier_order, scale);
      NumericMatrix::Matrix<Real, HW> vcoords_train(ff_input.nrows(), ff_input.ncols(), p);
      NumericMatrix::get_from_host(vcoords_train, ff_input);
      NumericMatrix::Matrix<Real, HW> vspace_train(vspace.nrows(), vspace.ncols(), p);

      // Actually read in the vspace for training
      if constexpr (HW == BACKEND::HOST) {
         vspace_train.copy_to_host_from_host_view(vspace);
      } else {
         vspace_train.copy_to_device_from_host_view(vspace);
      }

      NeuralNetwork<Real, HW, ACT, ACT, LOSSF::LOGCOSH> nn(arch, p, vcoords_train, vspace_train, BATCHSIZE);
      if (bytes == nullptr) {
         abort();
      }
      nn.load_weights(bytes);
      nn.evaluate(vcoords_train, vspace_train);
      NumericMatrix::export_to_host_view(vspace_train, vspace);
   }
   return;
}

size_t compress_vdf_union(GENERIC_TS_POOL::MemPool* p, std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr,
                          Realf* vspace_ptr, std::size_t size, std::size_t max_epochs, std::size_t fourier_order,
                          size_t* hidden_layers_ptr, size_t n_hidden_layers, Real sparsity, Real tol, Real* weights_ptr,
                          std::size_t weight_size, bool use_input_weights, uint32_t downsampling_factor, float& error,
                          int& status) {

   TINYAI_UNUSED(use_input_weights);
   TINYAI_UNUSED(sparsity);
   TINYAI_UNUSED(weight_size);
   auto allocfunction = [](std::size_t bytes) {
#ifdef USE_GPU
      void* mem;
      tinyAI_gpuMalloc(&mem, bytes);
#else
      void* mem = (void*)malloc(bytes);
#endif
      return mem;
   };

   auto deallocfunction = [](void* ptr) {
#ifdef USE_GPU
      tinyAI_gpuFree(ptr);
#else
      free(ptr);
#endif
   };

   p->init(MEMPOOL_BYTES, allocfunction);

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
       compress_vdf(p, vcoords, vspace, fourier_order, max_epochs, arch, weights_ptr, tol, error, status);
   PROFILE_END();
   p->destroy_with(deallocfunction);
   return network_bytes_used;
}

void uncompress_vdf_union(GENERIC_TS_POOL::MemPool* p, std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr,
                          Realf* vspace_ptr, std::size_t size, std::size_t fourier_order, size_t* hidden_layers_ptr,
                          size_t n_hidden_layers, Real* weights_ptr, std::size_t weight_size, bool use_input_weights) {

   TINYAI_UNUSED(use_input_weights);
   TINYAI_UNUSED(weight_size);
   auto allocfunction = [&](std::size_t bytes) {
#ifdef USE_GPU
      void* mem;
      tinyAI_gpuMalloc(&mem, bytes);
#else
      void* mem = (void*)malloc(bytes);
#endif
      return mem;
   };

   auto deallocfunction = [&](void* ptr) {
#ifdef USE_GPU
      tinyAI_gpuFree(ptr);
#else
      free(ptr);
#endif
   };

   p->init(MEMPOOL_BYTES, allocfunction);

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
   uncompress_vdf(p, vcoords, vspace, fourier_order, arch, weights_ptr);
   PROFILE_END();

   PROFILE_START("Copy VDF out");
   // Copy back
   for (std::size_t i = 0; i < vspace.size(); ++i) {
      vspace_ptr[i] = static_cast<Realf>(vspace(i));
   }
   PROFILE_END();
   p->destroy_with(deallocfunction);
   return;
}
