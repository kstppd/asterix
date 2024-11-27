#include "genericTsPool.h"
#include "matrix.h"
#include "tinyAI.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

constexpr size_t MEMPOOL_BYTES = 10ul * 1024ul * 1024ul * 1024ul;
constexpr size_t BATCHSIZE = 64;
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

void scale_vdf(MatrixView<Real>& vspace, Real sparse) {
   std::for_each(vspace.begin(), vspace.end(),
                 [sparse](Real& value) { value = std::abs(std::log10(std::max(value, sparse))); });
}

std::array<Real, 2> normalize_vdf(MatrixView<Real>& vdf) {
   Real min_val = *(check_ptr(std::min_element(vdf.begin(), vdf.end())));
   Real max_val = *(check_ptr(std::max_element(vdf.begin(), vdf.end())));
   Real range = max_val - min_val;

   std::for_each(vdf.begin(), vdf.end(), [min_val, range](Real& value) { value = (value - min_val) / range; });
   return {min_val, max_val};
}

struct MinMaxValues {
   Real min = std::numeric_limits<Real>::lowest();
   Real max = std::numeric_limits<Real>::max();
   Real mean = 0.0;
};

void scale_vdfs(MatrixView<Real>& vdf, Real sparse) {
   const std::size_t nVDFS = vdf.ncols();
   for (std::size_t v = 0; v < nVDFS; ++v) {
      Real min_val = std::numeric_limits<Real>::max();
      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         const Real vdf_val = vdf(i, v);
         if (vdf_val <= 0.0) {
            continue;
         }
         min_val = std::min(min_val, vdf_val);
      }

      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         vdf(i, v) = std::abs(std::log10(std::max(vdf(i, v), 0.5 * sparse)));
      }
   }
}

std::vector<MinMaxValues> normalize_vdfs(MatrixView<Real>& vdf) {
   const std::size_t nVDFS = vdf.ncols();
   std::vector<MinMaxValues> retval(nVDFS);

   for (std::size_t v = 0; v < nVDFS; ++v) {
      Real sum = 0;
      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         sum += vdf(i, v);
      }
      Real mean_val = sum / vdf.nrows();

      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         vdf(i, v) -= mean_val;
      }

      Real min_val = std::numeric_limits<Real>::max();
      Real max_val = std::numeric_limits<Real>::lowest();
      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         min_val = std::min(min_val, vdf(i, v));
         max_val = std::max(max_val, vdf(i, v));
      }
      Real range = max_val - min_val;
      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         vdf(i, v) = (vdf(i, v) - min_val) / range;
      }
      retval[v] = MinMaxValues{.min = min_val, .max = max_val, .mean = mean_val};
   }

   return retval;
}

void unnormalize_vdfs(HostMatrix<Real>& vdf, const std::vector<MinMaxValues>& norms) {
   const std::size_t nVDFS = vdf.ncols();
   for (std::size_t v = 0; v < nVDFS; ++v) {
      const Real max_val = norms[v].max;
      const Real min_val = norms[v].min;
      const Real mean_val = norms[v].mean;
      const Real range = max_val - min_val;
      for (std::size_t i = 0; i < vdf.nrows(); ++i) {
         vdf(i, v) = vdf(i, v) * range + min_val + mean_val;
      }
   }
}

void unnormalize_vdf(HostMatrix<Real>& vdf, std::array<Real, 2> norm) {
   Real max_val = norm[1];
   Real min_val = norm[0];
   Real range = max_val - min_val;

   std::for_each(vdf.begin(), vdf.end(), [min_val, range](Real& value) { value = value * range + min_val; });
}

void unscale_vdf(HostMatrix<Real>& vdf) {
   std::for_each(vdf.begin(), vdf.end(), [](Real& value) { value = std::pow(10.0, -1.0 * value); });
}

void sparsify(HostMatrix<Real>& vdf, Real sparse) {
   std::for_each(vdf.begin(), vdf.end(), [sparse](Real& x) {
      if (x - sparse <= 0.0) {
         x = 0.0;
      }
   });
}

template <typename T>
NumericMatrix::HostMatrix<T> generate_fourier_features(const NumericMatrix::MatrixView<T>& input, NumericMatrix::HostMatrix<T>& B, std::size_t num_features,
                                                       T scale) {
   if (num_features == 0) {
      return NumericMatrix::HostMatrix<T>(input);
   }
   assert(num_features % 2 == 0 && num_features > 0);
   const std::size_t input_dims = input.ncols();
   // Construct B
   if (B.isEmpty()){
      B=NumericMatrix::HostMatrix<T>(input_dims, num_features);
      std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
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
      NumericMatrix::HostMatrix<Real> ff_input = generate_fourier_features<Real>(vcoords, B, fourier_order, 1);
      NumericMatrix::Matrix<Real, HW> vcoords_train(ff_input.nrows(), ff_input.ncols(), &p);
      NumericMatrix::get_from_host(vcoords_train, ff_input);
      printf("Vcoords train shape = [%zu,%zu]\n",vcoords_train.nrows(),vcoords_train.ncols());

      NumericMatrix::HostMatrix<Real> ff_inf_input = generate_fourier_features<Real>(inference_coords, B, fourier_order, 1);
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

      NeuralNetwork<Real,HW,ACTIVATION::TANH> nn(arch, &p, vcoords_train, vspace_train, BATCHSIZE);
      network_size = nn.get_network_size();

      error = std::numeric_limits<float>::max();
      status = 0;
      Real lr=1e-4;
      Real current_lr=lr;
      for (std::size_t i = 0; i < max_epochs; i++) {
         error = nn.train(BATCHSIZE, current_lr);
         if (i % 1 == 0) {
            printf("Loss at epoch %zu: %f\n", i, error);
         }
         if (error < tolerance) {
            status = 1;
            break;
         }
       current_lr  = lr * std::exp(-0.01 * i);
      }
      tinyAI_gpuDeviceSynchronize();
      p.defrag();
      // nn.cast_to_float();     
      nn.evaluate(vcoords_inference, vspace_inference);
      vspace_inference.export_to_host(reconstructed_vdf);
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

extern "C" {
Real compress_and_reconstruct_vdf_2(std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr, std::size_t size,
                                    std::array<Real, 3>* inference_vcoords_ptr, Realf* new_vspace_ptr,
                                    std::size_t inference_size, std::size_t max_epochs, std::size_t fourier_order,
                                    size_t* hidden_layers_ptr, size_t n_hidden_layers, Real sparsity, Real tol,
                                    Real* weights_ptr, std::size_t weight_size, bool use_input_weights,
                                    uint32_t downsampling_factor, float& error, int& status) {

   PROFILE_START("Copy IN");
   std::vector<Real> vdf;
   vdf.reserve(size);
   for (std::size_t i = 0; i < size; ++i) {
      vdf.push_back(static_cast<Real>(vspace_ptr[i]));
   }
   PROFILE_END();

   const std::size_t vdf_size = vdf.size() * sizeof(Real);

   std::vector<int> arch;
   arch.reserve(n_hidden_layers + 1);
   for (size_t i = 0; i < n_hidden_layers; ++i) {
      arch.push_back(static_cast<int>(hidden_layers_ptr[i]));
   }
   arch.push_back(1);

   PROFILE_START("Prepare VDF");
   MatrixView<Real> vcoords = get_view_from_raw(&(vcoords_ptr[0][0]), size, 3);
   MatrixView<Real> inference_coords = get_view_from_raw(&(inference_vcoords_ptr[0][0]), inference_size, 3);
   MatrixView<Real> vspace = get_view_from_raw(vdf.data(), vdf.size(), 1);
   HostMatrix<Real> vspace_inference_host(inference_coords.nrows(), 1);

   if (downsampling_factor > 1) {
      PROFILE_START("Downsample VDF");
      HostMatrix<Real> downsampled_coords(vcoords.nrows() / downsampling_factor, vcoords.ncols());
      HostMatrix<Real> downsampled_vdf(vspace.nrows() / downsampling_factor, vspace.ncols());

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

   // Scale and normalize
   scale_vdf(vspace, sparsity);
   std::array<Real, 2> norm = normalize_vdf(vspace);
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   error=std::numeric_limits<float>::max();
   const std::size_t bytes_used = compress_and_reconstruct_vdf(vcoords, vspace, inference_coords, fourier_order,
                                                               max_epochs, arch, tol, vspace_inference_host,error,status);
   PROFILE_END();

   PROFILE_START("Unscale  and copy VDF out");
   // Undo scalings
   unnormalize_vdf(vspace_inference_host, norm);
   unscale_vdf(vspace_inference_host);
   sparsify(vspace_inference_host, sparsity);

   // Copy back
   for (std::size_t i = 0; i < vspace_inference_host.nrows(); ++i) {
      new_vspace_ptr[i] = static_cast<Realf>(vspace_inference_host(i, 0));
   }
   PROFILE_END();
   return static_cast<float>(vdf_size) / static_cast<float>(bytes_used);
}

Real compress_and_reconstruct_vdf_2_multi(std::size_t nVDFS, std::array<Real, 3>* vcoords_ptr, Realf* vspace_ptr,
                                          std::size_t size, std::array<Real, 3>* inference_vcoords_ptr,
                                          Realf* new_vspace_ptr, std::size_t inference_size, std::size_t max_epochs,
                                          std::size_t fourier_order, size_t* hidden_layers_ptr, size_t n_hidden_layers,
                                          Real sparsity, Real tol, Real* weights_ptr, std::size_t weight_size,
                                          bool use_input_weights, uint32_t downsampling_factor, float& error,
                                          int& status) {

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
      downsampled_coords =HostMatrix<Real>(downsampled_rows, vcoords.ncols());
      downsampled_vdf =HostMatrix<Real>(downsampled_rows, vspace.ncols());

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

   // Scale and normalize
   scale_vdfs(vspace, sparsity);
#ifdef NORM_PER_VDF
   auto norms = normalize_vdfs(vspace);
#else
   auto norms = normalize_vdf(vspace);
#endif
   PROFILE_END();

   // Reconstruct
   PROFILE_START("Training Entry Point");
   const std::size_t bytes_used = compress_and_reconstruct_vdf(vcoords, vspace, inference_coords, fourier_order,
                                                               max_epochs, arch, tol, vspace_inference_host,error,status);
   PROFILE_END();

   PROFILE_START("Unscale  and copy VDF out");
// Undo scalings
#ifdef NORM_PER_VDF
   unnormalize_vdfs(vspace_inference_host, norms);
#else
   unnormalize_vdf(vspace_inference_host, norms);
#endif
   unscale_vdf(vspace_inference_host);
   sparsify(vspace_inference_host, sparsity);

   // Copy back
   for (std::size_t i = 0; i < vspace_inference_host.size(); ++i) {
      new_vspace_ptr[i] = static_cast<Realf>(vspace_inference_host(i));
   }
   PROFILE_END();
   return static_cast<float>(bytes_used);
}
}
