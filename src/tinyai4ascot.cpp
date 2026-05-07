#if 0
#This produces the full thing
nvcc -ccbin=mpicxx -DUSE_GPU \
  -std=c++20 -O3 --use_fast_math -x cu tinai4ascot.cpp \
  -isystem=/home/kstppd/dev/asterix/external/libnpy/include \
  -I/home/kstppd/dev/asterix/include \
  -I/home/kstppd/software/spdlog/include/ \
  -L/home/kstppd/software/spdlog/build/ \
  -o interpolator -lcublas -lopenblas

#This is only to produce the prediciton shared lib with minimal deps
nvcc -ccbin=mpicxx -DLIB_PREDICT_ONLY   \
  -std=c++20 -O3 --use_fast_math -x cu \
  -Xcompiler -fPIC -shared tinai4aspect.cpp \
  -I/home/kstppd/dev/asterix/include \
  -o libtinyai_predict.so \
  -lopenblas
exit
#endif
// INFO=1 mpirun -n 1 ./bin train coordinates.npy b_field.npy ; INFO=1 mpirun -n
// 1 predict coordinates.npy interpolator.bin ;

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../include/tinyAI.h"
#ifndef LIB_PREDICT_ONLY
#include "npy.hpp"
#endif

#define MASTER 0
#define ARCH 256, 256, 256
#define Network NeuralNetwork<float, HW, ACTIVATION::TANH, ACTIVATION::NONE>
constexpr std::size_t BATCHSIZE = 32;
constexpr std::size_t NFOURIER = 128;
constexpr float STDD = 1.0;
constexpr std::size_t EPOCHS = 100;
using namespace TINYAI;
using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;
constexpr std::size_t N = 24ul * 1024ul * 1024ul * 1024ul;

struct NormStats {
   std::vector<float> minv;
   std::vector<float> maxv;
};

float normalize_value(float x, float mn, float mx) {
   const float denom = mx - mn;
   return denom > 0.0f ? 2.0f * (x - mn) / denom - 1.0f : 0.0f;
}

float denormalize_value(float x_norm, float mn, float mx) {
   return 0.5f * (x_norm + 1.0f) * (mx - mn) + mn;
}

NormStats minmax_normalize_mpi(HostMatrix<float>& m, const char* name,
                               int myRank) {
   const std::size_t rows = m.nrows();
   const std::size_t cols = m.ncols();
   std::vector<float> local_min(cols, std::numeric_limits<float>::max());
   std::vector<float> local_max(cols, -std::numeric_limits<float>::max());
   std::vector<float> global_min(cols);
   std::vector<float> global_max(cols);
   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         const float v = m(i, j);
         local_min[j] = std::min(local_min[j], v);
         local_max[j] = std::max(local_max[j], v);
      }
   }

   MPI_Allreduce(local_min.data(), global_min.data(), cols, MPI_FLOAT, MPI_MIN,
                 MPI_COMM_WORLD);
   MPI_Allreduce(local_max.data(), global_max.data(), cols, MPI_FLOAT, MPI_MAX,
                 MPI_COMM_WORLD);
   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         m(i, j) = normalize_value(m(i, j), global_min[j], global_max[j]);
         // assert(m(i, j) >= -1.0001f && m(i, j) <= 1.0001f);
      }
   }
   return {global_min, global_max};
}

void normalize_matrix_with_stats(HostMatrix<float>& m, const NormStats& s) {
   assert(m.ncols() == s.minv.size());
   assert(m.ncols() == s.maxv.size());
   for (std::size_t i = 0; i < m.nrows(); ++i) {
      for (std::size_t j = 0; j < m.ncols(); ++j) {
         m(i, j) = normalize_value(m(i, j), s.minv[j], s.maxv[j]);
      }
   }
}

void denormalize_matrix_with_stats(HostMatrix<float>& m, const NormStats& s) {
   assert(m.ncols() == s.minv.size());
   assert(m.ncols() == s.maxv.size());
   for (std::size_t i = 0; i < m.nrows(); ++i) {
      for (std::size_t j = 0; j < m.ncols(); ++j) {
         m(i, j) = denormalize_value(m(i, j), s.minv[j], s.maxv[j]);
      }
   }
}

HostMatrix<float> generate_fourier_features(const HostMatrix<float>& input,
                                            HostMatrix<float>& B,
                                            std::size_t num_features,
                                            float scale) {
   if (num_features == 0) {
      return HostMatrix<float>(input);
   }

   assert(num_features % 2 == 0 && num_features > 0);
   const std::size_t input_dims = input.ncols();
   if (B.isEmpty()) {
      B = HostMatrix<float>(input_dims, num_features);
      std::mt19937 rng(128);
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);

      for (std::size_t i = 0; i < input_dims; ++i) {
         for (std::size_t j = 0; j < num_features; ++j) {
            B(i, j) = scale * dist(rng);
         }
      }
   }

   HostMatrix<float> output(input.nrows(), 2 * num_features);

   for (std::size_t i = 0; i < input.nrows(); ++i) {
      for (std::size_t j = 0; j < num_features; ++j) {
         float dot_product = 0.0f;
         for (std::size_t k = 0; k < input.ncols(); ++k) {
            // assert(input(i, k) >= -1.0001f && input(i, k) <= 1.0001f);
            dot_product += input(i, k) * B(k, j);
         }
         output(i, j) = std::sin(2.0f * M_PI * dot_product);
         output(i, j + num_features) = std::cos(2.0f * M_PI * dot_product);
      }
   }
   return output;
}

#ifndef LIB_PREDICT_ONLY
HostMatrix<float> read_npy_to_matrix(const npy::npy_data<float>& data) {
   const auto dims = data.shape;
   const std::size_t rows = dims[0];
   const std::size_t cols = dims[1];
   spdlog::info("Dims = {0:d},{1:d}", rows, cols);
   HostMatrix<float> mat(rows, cols);
   for (std::size_t row = 0; row < mat.nrows(); ++row) {
      for (std::size_t col = 0; col < mat.ncols(); ++col) {
         mat.set_value(row, col, data.data.at(row * cols + col));
      }
   }

   return mat;
}

HostMatrix<float> read_npy_to_matrix_mpi(const char* filename) {
   int myRank;
   int size;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   auto shape = npy::read_npy_shape(filename);
   const std::size_t rows = shape[0];
   const std::size_t cols = shape[1];
   const std::size_t rows_per_rank = rows / size;
   const std::size_t left_over_rows = rows % size;
   const std::size_t local_rows =
       rows_per_rank + left_over_rows * (myRank == size - 1);
   const std::size_t byte_offset =
       myRank * rows_per_rank * cols * sizeof(float);
   const npy::shape_t local_shape = {local_rows, cols};
   npy::npy_data<float> data =
       npy::read_npy_partial<float>(filename, local_shape, byte_offset);

   const auto dims = data.shape;
   const std::size_t local_nrows = dims[0];
   const std::size_t local_ncols = dims[1];
   HostMatrix<float> mat(local_nrows, local_ncols);
   for (std::size_t row = 0; row < mat.nrows(); ++row) {
      for (std::size_t col = 0; col < mat.ncols(); ++col) {
         mat.set_value(row, col, data.data.at(row * local_ncols + col));
      }
   }

   return mat;
}
#endif

bool serialize_to_file(const char* filename, const std::vector<float>& weights,
                       const NormStats& xnorm, const NormStats& ynorm) {
   FILE* f = fopen(filename, "wb");
   if (!f) return false;

   auto write_size = [&](size_t v) {
      uint64_t x = (uint64_t)v;
      fwrite(&x, sizeof(uint64_t), 1, f);
   };

   auto write_vec = [&](const std::vector<float>& v) {
      write_size(v.size());
      if (!v.empty()) {
         fwrite(v.data(), sizeof(float), v.size(), f);
      }
   };

   write_vec(weights);
   write_vec(xnorm.minv);
   write_vec(xnorm.maxv);
   write_vec(ynorm.minv);
   write_vec(ynorm.maxv);
   fclose(f);
   return true;
}

bool deserialize_from_file(const char* filename, std::vector<float>& weights,
                           NormStats& xnorm, NormStats& ynorm) {
   FILE* f = fopen(filename, "rb");
   if (!f) return false;

   auto read_size = [&](size_t& v) {
      uint64_t x;
      auto _r = fread(&x, sizeof(uint64_t), 1, f);
      (void)_r;
      v = (size_t)x;
   };

   auto read_vec = [&](std::vector<float>& v) {
      size_t n;
      read_size(n);
      v.resize(n);
      if (n > 0) {
         auto _r = fread(v.data(), sizeof(float), n, f);
         (void)_r;
      }
   };
   read_vec(weights);
   read_vec(xnorm.minv);
   read_vec(xnorm.maxv);
   read_vec(ynorm.minv);
   read_vec(ynorm.maxv);
   fclose(f);
   return true;
}

bool deserialize_from_memory(const void* buffer, size_t buffer_size,
                             std::vector<float>& weights, NormStats& xnorm,
                             NormStats& ynorm) {
   const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
   const uint8_t* end = ptr + buffer_size;

   auto read_bytes = [&](void* dst, size_t n) -> bool {
      if ((size_t)(end - ptr) < n) return false;
      std::memcpy(dst, ptr, n);
      ptr += n;
      return true;
   };

   auto read_size = [&](size_t& v) -> bool {
      uint64_t x = 0;
      if (!read_bytes(&x, sizeof(uint64_t))) return false;
      v = static_cast<size_t>(x);
      return true;
   };

   auto read_vec = [&](std::vector<float>& v) -> bool {
      size_t n = 0;
      if (!read_size(n)) return false;
      if ((size_t)(end - ptr) < n * sizeof(float)) return false;
      v.resize(n);
      if (n > 0) {
         std::memcpy(v.data(), ptr, n * sizeof(float));
         ptr += n * sizeof(float);
      }
      return true;
   };

   return read_vec(weights) && read_vec(xnorm.minv) && read_vec(xnorm.maxv) &&
          read_vec(ynorm.minv) && read_vec(ynorm.maxv);
}

void train(const char* filex, const char* filey) {
#ifndef LIB_PREDICT_ONLY
   int myRank;
   int size;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   constexpr auto HW = BACKEND::DEVICE;

   void* mem = nullptr;
   // tinyAI_gpuMalloc(&mem, N);
   cudaMallocManaged(&mem, N);
   assert(mem && "Could not allocate memory!");

   MemPool p(mem, N);

   HostMatrix<float> B;

   auto space_tmp = read_npy_to_matrix_mpi(filex);
   const NormStats x_norm = minmax_normalize_mpi(space_tmp, "X", myRank);

   auto space = generate_fourier_features(space_tmp, B, NFOURIER, STDD);

   auto val = read_npy_to_matrix_mpi(filey);
   const NormStats y_norm = minmax_normalize_mpi(val, "Y", myRank);

   const std::size_t train_size = space.nrows();
   const std::size_t fin = space.ncols();
   const std::size_t fout = val.ncols();
   Matrix<float, HW> xtrain(train_size, space.ncols(), &p);
   Matrix<float, HW> ytrain(train_size, val.ncols(), &p);
   get_from_host(xtrain, space);
   get_from_host(ytrain, val);

   std::vector<int> arch{ARCH, static_cast<int>(fout)};
   Network nn(arch, &p, xtrain.nrows(), xtrain.ncols(), ytrain.ncols(),
              BATCHSIZE);

   const std::size_t nnsize = nn.get_network_size() / sizeof(float);
   std::vector<float> local_weights(nnsize);
   std::vector<float> global_weights(nnsize);
   std::vector<float> local_grads(nnsize);
   std::vector<float> global_grads(nnsize);

   nn.get_weights(local_weights.data());
   MPI_Bcast(local_weights.data(), local_weights.size(), MPI_FLOAT, MASTER,
             MPI_COMM_WORLD);

   nn.load_weights(local_weights.data());
   MPI_Barrier(MPI_COMM_WORLD);
   spdlog::info("Network Size = {0:d}", nnsize);

   Matrix<float, HW> sample(BATCHSIZE, xtrain.ncols(), &p);
   Matrix<float, HW> target(BATCHSIZE, ytrain.ncols(), &p);
   Matrix<float, HW> error(BATCHSIZE, ytrain.ncols(), &p);

   std::size_t* dperm = p.allocate<std::size_t>(BATCHSIZE);

   for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch) {
      float l = 0.0f;

      for (std::size_t b = 0; b < xtrain.nrows(); b += BATCHSIZE) {
         nn.get_permutation_indices(dperm, BATCHSIZE, 0);
         nn.shuffle_into(xtrain, sample, dperm, 0);
         nn.shuffle_into(ytrain, target, dperm, 0);
         nn.forward(sample, 0);
         l += nn.loss<HW, LOSSF::MSE>(error, target, 0);
         nn.backward(sample, target, 0);
         MPI_Barrier(MPI_COMM_WORLD);
         nn.get_grads(local_grads.data());
         MPI_Allreduce(local_grads.data(), global_grads.data(),
                       local_grads.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

         for (auto& g : global_grads) {
            g /= static_cast<float>(size);
         }
         nn.load_grads(global_grads.data());
         MPI_Barrier(MPI_COMM_WORLD);
         nn.update_weights_adamw(epoch + 1, 1e-3, 0);
         tinyAI_gpuStreamSynchronize(0);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      // l /= static_cast<float>(xtrain.nrows() * ytrain.ncols());
      if (myRank == MASTER) {
         nn.get_weights(local_weights.data());
         serialize_to_file("interpolator.bin", local_weights, x_norm, y_norm);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      float global_l = 0.0f;
      MPI_Allreduce(&l, &global_l, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      uint64_t local_steps = (xtrain.nrows() + BATCHSIZE - 1) / BATCHSIZE;
      uint64_t global_steps = 0;
      MPI_Allreduce(&local_steps, &global_steps, 1, MPI_UINT64_T, MPI_SUM,
                    MPI_COMM_WORLD);
      global_l /= static_cast<float>(global_steps);
      if (myRank == MASTER) {
         spdlog::info("Epoch {0:d} Global Loss {1:.8e}.", epoch, global_l);
      }
      // spdlog::info("[{0:d}] Epoch {1:d} Loss {2:f}.", myRank, epoch, l);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   spdlog::info("Pool HW = {0:f}", p.memory_hwm());
#endif
}

void predict(const char* filex, const char* model_file) {
#ifndef LIB_PREDICT_ONLY
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   if (myRank != MASTER) return;
   constexpr auto HW = BACKEND::HOST;
   std::vector<float> weights;
   NormStats x_norm;
   NormStats y_norm;

   if (!deserialize_from_file(model_file, weights, x_norm, y_norm)) {
      fprintf(stderr, "Could not read tinyAI file: %s\n", model_file);
      return;
   }

   void* mem = malloc(N);
   assert(mem && "Could not allocate memory!");

   GENERIC_TS_POOL::MemPool p(mem, N);
   auto datax = npy::read_npy<float>(filex);
   auto x_tmp = read_npy_to_matrix(datax);
   normalize_matrix_with_stats(x_tmp, x_norm);
   HostMatrix<float> B;
   auto x_feat = generate_fourier_features(x_tmp, B, NFOURIER, STDD);
   std::vector<int> arch{ARCH, static_cast<int>(y_norm.minv.size())};
   Network nn(arch, &p, x_feat.nrows(), x_feat.ncols(), y_norm.minv.size(),
              BATCHSIZE);
   nn.load_weights(weights.data());

   Matrix<float, HW> xdev(x_feat.nrows(), x_feat.ncols(), &p);
   Matrix<float, HW> ydev(x_feat.nrows(), y_norm.minv.size(), &p);
   get_from_host(xdev, x_feat);
   nn.evaluate(xdev, ydev);
   HostMatrix<float> yhost(ydev);
   denormalize_matrix_with_stats(yhost, y_norm);
   npy::npy_data_ptr<float> d;
   d.data_ptr = yhost.data();
   d.shape = {yhost.nrows(), yhost.ncols()};
   npy::write_npy("prediction.npy", d);
   spdlog::info("Prediction written to prediction.npy");
   free(mem);
#endif
}

extern "C" int tinyai_predict_into(void** persistent_mem,
                                   const void* model_buffer, size_t model_size,
                                   const float* coords, size_t nrows,
                                   size_t ncols, float* out, size_t out_cols) {
   constexpr auto HW = BACKEND::HOST;

   std::vector<float> weights;
   NormStats x_norm, y_norm;

   if (!deserialize_from_memory(model_buffer, model_size, weights, x_norm,
                                y_norm)) {
      return -1;
   }

   const size_t ycols = y_norm.minv.size();
   if (out_cols != ycols) return -2;
   void* mem = malloc(N);
   if (!mem) return -3;
   GENERIC_TS_POOL::MemPool p(mem, N);
   HostMatrix<float> x(nrows, ncols);
   for (size_t i = 0; i < nrows; ++i) {
      for (size_t j = 0; j < ncols; ++j) {
         x(i, j) = coords[i * ncols + j];
      }
   }

   normalize_matrix_with_stats(x, x_norm);

   HostMatrix<float> B;
   auto xf = generate_fourier_features(x, B, NFOURIER, STDD);

   std::vector<int> arch{ARCH, (int)ycols};
   Network nn(arch, &p, xf.nrows(), xf.ncols(), ycols, BATCHSIZE);
   nn.load_weights(weights.data());

   Matrix<float, HW> xdev(xf.nrows(), xf.ncols(), &p);
   Matrix<float, HW> ydev(xf.nrows(), ycols, &p);

   get_from_host(xdev, xf);
   nn.evaluate(xdev, ydev);

   HostMatrix<float> y(ydev);
   denormalize_matrix_with_stats(y, y_norm);
   memcpy(out, y.data(), nrows * ycols * sizeof(float));
   free(mem);
   return 0;
}

extern "C" int tinyai_predict_into_persistent(void** persistent_mem,
                                              const void* model_buffer,
                                              size_t model_size,
                                              const float* coords, size_t nrows,
                                              size_t ncols, float* out,
                                              size_t out_cols) {
   constexpr auto HW = BACKEND::HOST;
   if (persistent_mem == nullptr) return -3;
   if (model_buffer == nullptr) return -1;
   if (coords == nullptr || out == nullptr) return -4;

   std::vector<float> weights;
   NormStats x_norm, y_norm;

   if (!deserialize_from_memory(model_buffer, model_size, weights, x_norm,
                                y_norm)) {
      return -1;
   }

   const size_t ycols = y_norm.minv.size();
   if (out_cols != ycols) return -2;

   constexpr size_t POOL_OFFSET =
       (sizeof(Network) + alignof(GENERIC_TS_POOL::MemPool) - 1) &
       ~(alignof(GENERIC_TS_POOL::MemPool) - 1);

   constexpr size_t DATA_OFFSET =
       (POOL_OFFSET + sizeof(GENERIC_TS_POOL::MemPool) +
        alignof(std::max_align_t) - 1) &
       ~(alignof(std::max_align_t) - 1);

   if (N <= DATA_OFFSET) return -3;

   char* base = nullptr;
   Network* nn = nullptr;
   GENERIC_TS_POOL::MemPool* pool = nullptr;
   const bool first_init = (*persistent_mem == nullptr);

   if (first_init) {
      *persistent_mem = std::malloc(N);
      if (*persistent_mem == nullptr) return -3;

      base = static_cast<char*>(*persistent_mem);

      void* network_mem = static_cast<void*>(base);
      void* pool_mem = static_cast<void*>(base + POOL_OFFSET);
      void* data_mem = static_cast<void*>(base + DATA_OFFSET);
      const size_t data_size = N - DATA_OFFSET;
      pool = new (pool_mem) GENERIC_TS_POOL::MemPool(data_mem, data_size);
      std::vector<int> arch{ARCH, static_cast<int>(ycols)};
      nn = new (network_mem)
          Network(arch, pool, nrows, NFOURIER * 2, out_cols, nrows);
   } else {
      base = static_cast<char*>(*persistent_mem);
      nn = reinterpret_cast<Network*>(base);
      pool = reinterpret_cast<GENERIC_TS_POOL::MemPool*>(base + POOL_OFFSET);
   }

   HostMatrix<float> x(nrows, ncols);
   for (size_t i = 0; i < nrows; ++i) {
      for (size_t j = 0; j < ncols; ++j) {
         x(i, j) = coords[i * ncols + j];
      }
   }

   normalize_matrix_with_stats(x, x_norm);
   HostMatrix<float> B;
   auto xf = generate_fourier_features(x, B, NFOURIER, STDD);
   nn->load_weights(weights.data());
   Matrix<float, HW> xdev(xf.nrows(), xf.ncols(), pool);
   Matrix<float, HW> ydev(xf.nrows(), ycols, pool);
   get_from_host(xdev, xf);
   nn->evaluate(xdev, ydev);
   HostMatrix<float> y(ydev);
   denormalize_matrix_with_stats(y, y_norm);
   std::memcpy(out, y.data(), nrows * ycols * sizeof(float));
   
   return 0;
}

int main(int argc, char** argv) {
   int myRank;
   int provided;

   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   if (argc != 4) {
      if (myRank == MASTER) {
         fprintf(stdout,
                 "Usage:\n"
                 "  %s train   <xfile.npy> <yfile.npy>\n"
                 "  %s predict <xfile.npy> <interpolator.bin>\n",
                 argv[0], argv[0]);
      }
      MPI_Finalize();
      return 1;
   }

   if (std::strcmp(argv[1], "train") == 0) {
      train(argv[2], argv[3]);
   } else if (std::strcmp(argv[1], "predict") == 0) {
      predict(argv[2], argv[3]);
   } else {
      if (myRank == MASTER) {
         fprintf(stderr, "Unknown mode: %s\n", argv[1]);
      }
      MPI_Finalize();
      return 1;
   }

   MPI_Finalize();
   return 0;
}

/*
Example code for using the prediction code.
Requires interpolator_128gh200_100epochs_tanh_1em3.bin to be in the PWD
Paste it in a driver.c file

compile this main.cpp with
   nvcc -ccbin=mpicxx -DLIB_PREDICT_ONLY   \
     -std=c++20 -O3 --use_fast_math -x cu \
     -Xcompiler -fPIC -shared main.cpp \
     -I/home/kstppd/dev/asterix/include \
     -o libtinyai_predict.so \
     -lopenblas

bash driver.c and then run it with ./bin

#if 0
gcc -O3  driver.c -o driver libtinyai_predict.so -o bin
exit
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void *persistent_memory = NULL;

extern int tinyai_predict_into(void **persistent_memory,
                               const void *model_buffer, size_t model_size,
                               const float *coords, size_t nrows, size_t ncols,
                               float *out, size_t out_cols);

int main(void) {
   FILE *f = fopen("interpolator_128gh200_100epochs_tanh_1em3.bin", "rb");
   if (!f) {
      fprintf(stderr, "Failed to load model file");
      return 1;
   }

   fseek(f, 0, SEEK_END);
   size_t model_size = ftell(f);
   rewind(f);

   void *model_buf = malloc(model_size);
   fread(model_buf, 1, model_size, f);
   fclose(f);

   // Define dataset sizes
   size_t nrows = 10;
   size_t ncols = 3;
   float coords[] = {0.2835f, 0.0f,  -1.1f,        0.2835f, 2.0f,  0.13819095f,
                     0.2835f, 5.0f,  -0.83467335f, 0.2835f, 7.0f,  0.40351757f,
                     0.2835f, 10.0f, -0.5693467f,  0.2835f, 12.0f, 0.6688442f,
                     0.2835f, 15.0f, -0.3040201f,  0.2835f, 17.0f, 0.93417084f,
                     0.2835f, 20.0f, -0.03869347f, 0.2835f, 23.0f, -1.0115578f};
   // Out cols is fout Bx By Bz = 3
   size_t out_cols = 3;
   float *out = malloc(nrows * out_cols * sizeof(float));

   int err = tinyai_predict_into(&persistent_memory, model_buf, model_size,
                                 coords, nrows, ncols, out, out_cols);

   if (err != 0) {
      fprintf(stderr, "Prediction failed: %d\n", err);
      return 1;
   }

   for (size_t i = 0; i < nrows; ++i) {
      for (size_t j = 0; j < out_cols; ++j) {
         printf("%f ", out[i * out_cols + j]);
      }
      printf("\n");
   }

   free(out);
   free(model_buf);
   return 0;
}




*/
