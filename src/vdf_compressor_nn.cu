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
#define OUTACT ACTIVATION::NONE
#define LF LOSSF::LOGCOSH
#define LR 1e-4
constexpr size_t MEMPOOL_BYTES = 60ul * 1024ul * 1024ul * 1024ul;
constexpr size_t BATCHSIZE = 32;
#define USE_GPU
#ifdef USE_GPU
constexpr auto HW = BACKEND::DEVICE;
#else
constexpr auto HW = BACKEND::HOST;
#endif
#define USE_PATIENCE

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
      std::uniform_real_distribution<T> dist(0.0, 1.0);
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

template <typename T>
void decompress(GENERIC_TS_POOL::MemPool* p, const MatrixView<T>& x, MatrixView<T>& y,
                    std::size_t fourier_order, std::vector<int>& arch, const T* bytes) {

   {
      T scale = 1.0;
      NumericMatrix::HostMatrix<T> B;
      NumericMatrix::HostMatrix<T> ff_input = generate_fourier_features<T>(x, B, fourier_order, scale);
      NumericMatrix::Matrix<T, HW> x_train(ff_input.nrows(), ff_input.ncols(), p);
      NumericMatrix::get_from_host(x_train, ff_input);
      NumericMatrix::Matrix<T, HW> y_train(y.nrows(), y.ncols(), p);
       
      // Actually read in the y for training
      if constexpr (HW == BACKEND::HOST) {
         y_train.copy_to_host_from_host_view(y);
      } else {
         y_train.copy_to_device_from_host_view(y);
      }

      NeuralNetwork<T, HW, ACT,OUTACT,LF> nn(arch, p, x_train, y_train, BATCHSIZE);
      if (bytes == nullptr) {
         abort();
      }
      nn.load_weights(bytes);
      nn.evaluate(x_train, y_train);
      NumericMatrix::export_to_host_view(y_train, y);
   }
   return;
}


template<typename T>
std::size_t compress(GENERIC_TS_POOL::MemPool* p, const MatrixView<T>& x, const MatrixView<T>& y,
                         std::size_t fourier_order, std::size_t max_epochs, std::vector<int>& arch, T* bytes,
                         T tolerance, T& error, int& status) {
   std::size_t network_size = 0;
   {
      T scale = 1.0;
      NumericMatrix::HostMatrix<T> B;
      NumericMatrix::HostMatrix<T> ff_input = generate_fourier_features<T>(x, B, fourier_order, scale);
      NumericMatrix::Matrix<T, HW> x_train(ff_input.nrows(), ff_input.ncols(), p);
      NumericMatrix::get_from_host(x_train, ff_input);
      NumericMatrix::Matrix<T, HW> y_train(y.nrows(), y.ncols(), p);

      // Actually read in the vspace for training
      if constexpr (HW == BACKEND::HOST) {
         y_train.copy_to_host_from_host_view(y);
      } else {
         y_train.copy_to_device_from_host_view(y);
      }

      NeuralNetwork<T, HW, ACT,OUTACT,LF> nn(arch, p, x_train, y_train, BATCHSIZE);
      network_size = nn.get_network_size();

      error = std::numeric_limits<T>::max();
      status = 0;
      T lr = LR;
      T current_lr = lr;
      T min_loss = std::numeric_limits<T>::max();
      std::size_t patience_counter = 0;
      constexpr std::size_t patience = 10;
      tinyAI_gpuStream_t s;
      tinyAI_gpuStreamCreate(&s);
      for (std::size_t i = 0; i < max_epochs; i++) {
         error = nn.train(BATCHSIZE, current_lr, s);
         if (i % 1 == 0) {
            spdlog::info("-->Epoch [{0:d}] loss,patience=[{1:f}, {2:d}]", i, error, patience_counter);
         }
         if (i > 30 && error > 0.1) {
            spdlog::critical("NETWORK RESET");
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
            spdlog::info("EXIT(patience)=>Loss=[{0:f}]@({1:d})", error,i);
            break;
         }
#endif
         if (error < tolerance && i > 30) {
            spdlog::info("EXIT(normal)=>Loss=[{0:f}]@({1:d})", error,i);
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


size_t compress_phasespace6D_f32(GENERIC_TS_POOL::MemPool* p, std::size_t fin,std::size_t fout, float* coords_ptr, float* f_ptr,
                                 std::size_t size, std::size_t max_epochs, std::size_t fourier_order,
                                 size_t* hidden_layers_ptr, size_t n_hidden_layers, float sparsity, float tol,
                                 float* weights_ptr, std::size_t weight_size, bool use_input_weights,
                                 uint32_t downsampling_factor, float& error, int& status) {

   TINYAI_UNUSED(use_input_weights);
   TINYAI_UNUSED(sparsity);
   TINYAI_UNUSED(weight_size);
   TINYAI_UNUSED(downsampling_factor);
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
   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back((int)fout);

   PROFILE_START("Prepare VDF");
   MatrixView<float> vcoords = get_view_from_raw(coords_ptr, size, fin);
   MatrixView<float> vspace = get_view_from_raw(f_ptr, size, fout);
   PROFILE_END();

   PROFILE_START("Training Entry Point");
   const std::size_t network_bytes_used = compress<float>(p, vcoords, vspace, fourier_order, max_epochs, arch, weights_ptr, tol, error, status);
   PROFILE_END();
   p->destroy_with(deallocfunction);
   return network_bytes_used;
}


void decompress_phasespace6D_f32(GENERIC_TS_POOL::MemPool* p,std::size_t fin,std::size_t fout, float* vcoords_ptr,
                                float* vspace_ptr, std::size_t size, std::size_t fourier_order, size_t* hidden_layers_ptr,
                                size_t n_hidden_layers, float* weights_ptr, std::size_t weight_size, bool use_input_weights) {

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
   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back((int)fout);

   PROFILE_START("Prepare VDF");
   MatrixView<float> vcoords = get_view_from_raw(vcoords_ptr, size, fin);
   MatrixView<float> vspace = get_view_from_raw(vspace_ptr, size, fout);
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   decompress<float>(p, vcoords, vspace, fourier_order, arch, weights_ptr);
   PROFILE_END();

   PROFILE_START("Copy VDF out");
   // Copy back
   for (std::size_t i = 0; i < vspace.size(); ++i) {
      vspace_ptr[i] = vspace(i);
   }
   PROFILE_END();
   p->destroy_with(deallocfunction);
   return;
}




size_t compress_phasespace6D_f64(GENERIC_TS_POOL::MemPool* p, std::size_t fin,std::size_t fout, double* coords_ptr, double* f_ptr,
                                 std::size_t size, std::size_t max_epochs, std::size_t fourier_order,
                                 size_t* hidden_layers_ptr, size_t n_hidden_layers, double sparsity, double tol,
                                 double* weights_ptr, std::size_t weight_size, bool use_input_weights,
                                 uint32_t downsampling_factor, double& error, int& status) {

   TINYAI_UNUSED(use_input_weights);
   TINYAI_UNUSED(sparsity);
   TINYAI_UNUSED(weight_size);
   TINYAI_UNUSED(downsampling_factor);
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
   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back((int)fout);

   PROFILE_START("Prepare VDF");
   MatrixView<double> vcoords = get_view_from_raw(coords_ptr, size, fin);
   MatrixView<double> vspace = get_view_from_raw(f_ptr, size, fout);
   PROFILE_END();

   PROFILE_START("Training Entry Point");
   const std::size_t network_bytes_used = compress<double>(p, vcoords, vspace, fourier_order, max_epochs, arch, weights_ptr, tol, error, status);
   PROFILE_END();
   p->destroy_with(deallocfunction);
   return 0;
}


void decompress_phasespace6D_f64(GENERIC_TS_POOL::MemPool* p,std::size_t fin,std::size_t fout, double* vcoords_ptr,
                                double* vspace_ptr, std::size_t size, std::size_t fourier_order, size_t* hidden_layers_ptr,
                                size_t n_hidden_layers, double* weights_ptr, std::size_t weight_size, bool use_input_weights) {

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
   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back((int)fout);

   PROFILE_START("Prepare VDF");
   MatrixView<double> vcoords = get_view_from_raw(vcoords_ptr, size, fin);
   MatrixView<double> vspace = get_view_from_raw(vspace_ptr, size, fout);
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   decompress<double>(p, vcoords, vspace, fourier_order, arch, weights_ptr);
   PROFILE_END();

   PROFILE_START("Copy VDF out");
   // Copy back
   for (std::size_t i = 0; i < vspace.size(); ++i) {
      vspace_ptr[i] = vspace(i);
   }
   PROFILE_END();
   p->destroy_with(deallocfunction);
   return;
}
