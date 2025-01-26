/* Author: Kostis Papadakis
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
 * USA.
 * */
#pragma once
#include "linear_layer.h"
#include "matrix.h"
#include <cmath>
#include <driver_types.h>
#include <random>
#include <ranges>
#include <stdlib.h>
#include <tuple>
#include <vector>

#ifndef NOPROFILE
#ifdef __CUDACC__
#include <nvToolsExt.h>
#define PROFILE_START(msg) nvtxRangePushA((msg))
#define PROFILE_END() nvtxRangePop()
#else
#include <roctx.h>
#define PROFILE_START(msg) roctxRangePush((msg))
#define PROFILE_END() roctxRangePop()
#endif
#else
#define PROFILE_START(msg)
#define PROFILE_END()
#endif

namespace TINYAI {

template <typename T, BACKEND Backend = BACKEND::HOST, ACTIVATION Activation = ACTIVATION::TANH> class NeuralNetwork {
public:
   NeuralNetwork(std::vector<int>& arch, GENERIC_TS_POOL::MemPool* pool, const NumericMatrix::Matrix<T, Backend>& input,
                 const NumericMatrix::Matrix<T, Backend>& output, size_t batchSize, int seed = 42)
       : arch(arch), _pool(pool), batchSize_in_use(batchSize) {

      // Bind layers to the pool
      layers.resize(arch.size());
      for (size_t i = 0; i < layers.size(); ++i) {
         layers.at(i) = LinearLayer<T, Activation, Backend>(arch.at(i), _pool);
      }
      // Bind all objects to the memory pool
      inputData = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = input.ncols(), .rows = input.nrows()};
      outputData = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = output.ncols(), .rows = output.nrows()};
      input.getView(inputData, 0);
      output.getView(outputData, 0);
      sample = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = input.ncols(), .rows = batchSize};
      target = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = output.ncols(), .rows = batchSize};
      sample_t = NumericMatrix::Matrix<T, Backend>(input.ncols(), batchSize, _pool);
      batchedInput = NumericMatrix::Matrix<T, Backend>(batchSize, inputData.ncols(), _pool);
      batchedOutput = NumericMatrix::Matrix<T, Backend>(batchSize, outputData.ncols(), _pool);
      layers[0].setup(arch.front(), inputData.ncols(), batchSize, 0);
      for (size_t l = 1; l < layers.size(); ++l) {
         auto* curr_layer = &layers[l];
         auto* prev_layer = &layers[l - 1];
         curr_layer->setup(arch[l], arch[l - 1], batchSize, l);
      }
      spdlog::debug("TinyAI Initalized on CPU.");
      std::cerr << "TINY AI INITIALIZED" << std::endl;

      if constexpr (Backend == BACKEND::DEVICE) {
         spdlog::debug("TinyAI Initalized on GPU");
         tinyAI_gpuStreamCreate(&s[0]);
         tinyAI_gpuStreamCreate(&s[1]);
         auto stat = tinyAI_blasCreate(&handle);
         if (stat != BLAS_SUCCESS) {
            std::cerr << "Stat = " << stat << std::endl;
            spdlog::error("Failed to initialize CUBLAS.");
            throw std::runtime_error("Failed to initialize CUBLAS");
         } else {
            spdlog::debug("CUBLAS initialized succesfully.");
         }
      } else {
         spdlog::debug("TinyAI Initalized on CPU.");
      }
      set_log_level();
      generator();
      dist = std::uniform_int_distribution<std::size_t>(static_cast<std::size_t>(0), inputData.nrows() - 1);
   }

   NeuralNetwork(const NeuralNetwork& other) = delete;
   NeuralNetwork(NeuralNetwork&& other) = delete;
   NeuralNetwork operator=(const NeuralNetwork& other) = delete;
   NeuralNetwork operator=(NeuralNetwork&& other) = delete;
   ~NeuralNetwork() {
      if constexpr (Backend == BACKEND::DEVICE) {
         for (const auto& str : s) {
            tinyAI_gpuStreamDestroy(str);
         }
         spdlog::debug("TinyAI Destroyed on GPU");
         tinyAI_blasDestroy(handle);
      } else {
         spdlog::debug("TinyAI Desctroyed on CPU.");
      }
   }

   void forward(const NumericMatrix::Matrix<T, Backend>& in) noexcept {
      spdlog::stopwatch timer;
      layers[0].forward(in, &handle);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l].forward(layers[l - 1].a, &handle);
      }
      spdlog::debug("Feed Forward {:.3}s", timer);
   }

   void forward(const NumericMatrix::MatrixView<T>& in) noexcept {
      spdlog::stopwatch timer;
      layers[0].forward(in, &handle);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l].forward(layers[l - 1].a, &handle);
      }
      spdlog::debug("Feed Forward {:.3}s", timer);
   }

   void forward(const NumericMatrix::ConstMatrixView<T>& in) noexcept {
      spdlog::stopwatch timer;
      layers[0].forward(in, &handle);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l].forward(layers[l - 1].a, &handle);
      }
      spdlog::debug("Feed Forward {:.3}s", timer);
   }

   void backward(const NumericMatrix::ConstMatrixView<T>& sample,
                 const NumericMatrix::ConstMatrixView<T>& target) noexcept {
      spdlog::stopwatch timer;
      auto& curr_layer = layers.back();
      NumericMatrix::matsub(curr_layer.a, target, curr_layer.delta_store, &handle);
      NumericMatrix::mat_pointwise_activate_prime<T, Activation>(curr_layer.z, curr_layer.a_prime, curr_layer.wmega);
      NumericMatrix::mat_pointwise_mul(curr_layer.delta_store, curr_layer.a_prime, curr_layer.delta);
      NumericMatrix::transpose_into(sample, sample_t);
      // curr_layer.dw.zero_out();
      // curr_layer.db.zero_out();
      if (layers.size() == 1) {
         NumericMatrix::matmul(sample_t, curr_layer.delta, curr_layer.dw, &handle);
      } else {
         NumericMatrix::transpose_into(layers[layers.size() - 2].a, layers[layers.size() - 2].a_t);
         NumericMatrix::matmul(layers[layers.size() - 2].a_t, curr_layer.delta, curr_layer.dw, &handle);
      }
      NumericMatrix::matsum_rows(curr_layer.delta, curr_layer.db);

      for (int i = layers.size() - 2; i >= 0; i--) {

         auto& next_layer = layers[i + 1];
         auto& curr_layer = layers[i];
         next_layer.buffer.zero_out();
         NumericMatrix::transpose_into(next_layer.w, next_layer.w_t);
         NumericMatrix::matmul(next_layer.delta, next_layer.w_t, next_layer.buffer, &handle);
         NumericMatrix::mat_pointwise_activate_prime<T, Activation>(curr_layer.z, curr_layer.a_prime, curr_layer.wmega);
         NumericMatrix::mat_pointwise_mul(next_layer.buffer, curr_layer.a_prime, curr_layer.delta);
         // curr_layer.dw.zero_out();
         // curr_layer.db.zero_out();
         if (i == 0) {
            NumericMatrix::matmul(sample_t, curr_layer.delta, curr_layer.dw, &handle);
         } else {
            NumericMatrix::transpose_into(layers[i - 1].a, layers[i - 1].a_t);
            NumericMatrix::matmul(layers[i - 1].a_t, curr_layer.delta, curr_layer.dw, &handle);
         }
         NumericMatrix::matsum_rows(curr_layer.delta, curr_layer.db);
      }
      spdlog::debug("Backward {:.3}s", timer);
   }

   void update_weights() noexcept {
      spdlog::stopwatch timer;
      for (std::size_t i = 0; i < layers.size(); i++) {
         auto& current = layers[i];
         T lr = 1e-1;
         NumericMatrix::matscale(current.dw, static_cast<T>(lr / batchSize_in_use), &handle);
         NumericMatrix::matscale(current.db, static_cast<T>(lr / batchSize_in_use), &handle);
         NumericMatrix::matsub(current.w, current.dw, current.w, &handle);
         NumericMatrix::matsub(current.b, current.db, current.b, &handle);
      }
      spdlog::debug("Weight Update {:.3}s", timer);
   }

   T train(std::size_t batchSize, T lr = 1e-3) {
      // We need to check whether wed need to reconfigure our internal data
      // structures now due to a batchsize change
      assert(batchSize > 0 && batchSize <= inputData.nrows() &&
             "Batchsize cannot be bigger than your dataset you fool!");
      if (batchSize_in_use != batchSize) {
         migrate_to_batchsize(batchSize);
      }
      NumericMatrix::Matrix<T, Backend> error =
          NumericMatrix::Matrix<T, Backend>(target.nrows(), target.ncols(), _pool);

      T loss = 0.0;
      PROFILE_START("Epoch Training");
      PROFILE_START("Pool allocation");
      PROFILE_END();

      std::vector<std::size_t> perm(batchSize_in_use, 0);
      std::size_t* dperm = _pool->allocate<std::size_t>(batchSize_in_use);
      PROFILE_START("Permutation Indices");
      for (std::size_t k = 0; k < batchSize_in_use; ++k) {
         perm[k] = dist(generator);
      }
      tinyAI_gpuMemcpyAsync(dperm, perm.data(), batchSize * sizeof(std::size_t), tinyAI_gpuMemcpyHostToDevice, s[0]);
      PROFILE_END();
      for (size_t i = 0; i < inputData.nrows(); i += batchSize) {

         PROFILE_START("BATCH PASS");
         PROFILE_START("IO");
         if constexpr (Backend == BACKEND::DEVICE) {
            if (batchSize_in_use > 1024) {
               throw std::runtime_error("TinyAI unable to shuffle rows on the GPU when running with batchsizes larger "
                                        "than the max blocksize of 1024");
            }
            NumericMatrix::shuffle_rows<<<1, batchSize_in_use, 0, s[0]>>>(inputData.data(), dperm, batchedInput.data(),
                                                                          inputData.ncols());
            tinyAI_gpuStreamSynchronize(s[0]);
            NumericMatrix::shuffle_rows<<<1, batchSize_in_use, 0, s[1]>>>(outputData.data(), dperm,
                                                                          batchedOutput.data(), outputData.ncols());
            tinyAI_gpuStreamSynchronize(s[1]);
         } else {
            for (std::size_t k = 0; k < batchSize_in_use; ++k) {
               const std::size_t index = dist(generator);
               std::memcpy(&batchedInput(k, 0), &inputData(index, 0), inputData.ncols() * sizeof(T));
               std::memcpy(&batchedOutput(k, 0), &outputData(index, 0), outputData.ncols() * sizeof(T));
            }
         }
         PROFILE_END();

         PROFILE_START("Permutation Indices");
         for (std::size_t k = 0; k < batchSize_in_use; ++k) {
            perm[k] = dist(generator);
         }
         // Launch this copy here and we wait it in the next loop
         tinyAI_gpuMemcpyAsync(dperm, perm.data(), batchSize * sizeof(std::size_t), tinyAI_gpuMemcpyHostToDevice, s[0]);
         PROFILE_END();
         // Collect input-output
         batchedInput.getView(sample, 0);
         batchedOutput.getView(target, 0);
         PROFILE_START("Forward");
         forward(sample);
         PROFILE_END();
         PROFILE_START("Error calculation");
         // Get loss
         NumericMatrix::matsub_error_mse(layers.back().a, target, error, &handle);
         if constexpr (Backend == BACKEND::HOST) {
            loss += NumericMatrix::matreduce_add(error, &handle);
         } else {
            loss += NumericMatrix::matreduce_add_gpu(error, _pool, &handle);
         }
         PROFILE_END();
         PROFILE_START("Backward");
         backward(sample, target);
         PROFILE_END();
         PROFILE_START("Weight Update AdamW");
         update_weights_adamw(iter, lr);
         PROFILE_END();
         iter++;
         PROFILE_END();
      }
      PROFILE_START("Pool deallocation");
      PROFILE_END();
      PROFILE_END();
      spdlog::debug("Epoch done");
      return loss / (inputData.nrows() * outputData.ncols());
   }

   void update_weights_adamw(size_t iteration, T lr, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8,
                             T decay = 1e-4) noexcept {
      spdlog::stopwatch timer;
      for (auto& curr_layer : layers) {
         NumericMatrix::matscale(curr_layer.m_w, beta1, &handle);
         NumericMatrix::matscale(curr_layer.dw, static_cast<T>(1.0 - beta1), &handle);
         NumericMatrix::matadd(curr_layer.m_w, curr_layer.dw, curr_layer.m_w, &handle);

         NumericMatrix::matscale(curr_layer.v_w, beta2, &handle);
         NumericMatrix::mat_pointwise_mul(curr_layer.dw, curr_layer.dw, curr_layer.tmp);
         NumericMatrix::matscale(curr_layer.tmp, static_cast<T>(1.0 - beta2), &handle);
         NumericMatrix::matadd(curr_layer.v_w, curr_layer.tmp, curr_layer.v_w, &handle);

         T m_hat_scale = static_cast<T>(1.0) / (1 - std::pow(beta1, iteration));
         T v_hat_scale = static_cast<T>(1.0) / (1 - std::pow(beta2, iteration));

         NumericMatrix::matscale_to(curr_layer.m_w, curr_layer.mw_hat, m_hat_scale, &handle);
         NumericMatrix::matscale_to(curr_layer.v_w, curr_layer.vw_hat, v_hat_scale, &handle);
         NumericMatrix::mat_pointwise_sqrt(curr_layer.vw_hat, curr_layer.vw_hat);
         NumericMatrix::matadd_scalar(curr_layer.vw_hat, curr_layer.vw_hat, epsilon, &handle);

         NumericMatrix::matscale_to(curr_layer.m_w, curr_layer.w_decay, decay, &handle);

         NumericMatrix::mat_pointwise_div(curr_layer.mw_hat, curr_layer.vw_hat, curr_layer.tmp);
         NumericMatrix::matadd(curr_layer.tmp, curr_layer.w_decay, curr_layer.tmp, &handle);
         NumericMatrix::matscale(curr_layer.tmp, lr, &handle);
         NumericMatrix::matsub(curr_layer.w, curr_layer.tmp, curr_layer.w, &handle);

         NumericMatrix::matscale(curr_layer.db, lr, &handle);
         NumericMatrix::matsub(curr_layer.b, curr_layer.db, curr_layer.b, &handle);
      }
      spdlog::debug("AdamW Weight Update {:.3}s", timer);
   }

   void evaluate(const NumericMatrix::Matrix<T, Backend>& eval_samples,
                 NumericMatrix::Matrix<T, Backend>& eval_output) noexcept {
      std::size_t eval_batchsize = eval_samples.nrows();
      assert(eval_batchsize > 0 && "Invalid batchsize!");
      if (batchSize_in_use != eval_batchsize) {
         migrate_to_batchsize(eval_batchsize);
      }
      NumericMatrix::Matrix<T, Backend> eval_samples_device(_pool);
      eval_samples_device = eval_samples;
      forward(eval_samples_device);
      tinyAI_gpuDeviceSynchronize();
      eval_output = layers.back().a;
   }

   // Returns the number of bytes written
   size_t get_weights(T* dst) const noexcept {
      size_t write_index = 0;
      for (const auto& layer : layers) {
         // Weights
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(&dst[write_index], layer.w.data(), layer.w.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(&dst[write_index], layer.w.data(), layer.w.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToHost);
         }
         write_index += layer.w.size();
         // Biases
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(&dst[write_index], layer.b.data(), layer.b.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(&dst[write_index], layer.b.data(), layer.b.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToHost);
         }
         write_index += layer.b.size();
      }
      return write_index * sizeof(T);
   }

   // Returns the number of bytes read
   size_t load_weights(const T* src) noexcept {
      size_t read_index = 0;
      for (auto& layer : layers) {
         // Weights
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(layer.w.data(), &src[read_index], layer.w.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(layer.w.data(), &src[read_index], layer.w.size() * sizeof(T),
                             tinyAI_gpuMemcpyHostToDevice);
         }
         read_index += layer.w.size();
         // Biases
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(layer.b.data(), &src[read_index], layer.b.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(layer.b.data(), &src[read_index], layer.b.size() * sizeof(T),
                             tinyAI_gpuMemcpyHostToDevice);
         }
         read_index += layer.b.size();
      }
      return read_index * sizeof(T);
   }

   // Returns the number of bytes needed to store the network's weights (W,B)
   size_t get_network_size() const noexcept {
      size_t total_size = 0;
      for (auto& layer : layers) {
         total_size += layer.w.size() * sizeof(T);
         total_size += layer.b.size() * sizeof(T);
      }
      return total_size;
   }

   std::tuple<T, T, std::vector<size_t>> quantize(const std::vector<T>& inputVector, int bits) {
      assert(!inputVector.empty());
      auto [minVal, maxVal] = minmax(inputVector);
      size_t numLevels = 1u << bits;
      T delta = (maxVal - minVal) / (numLevels - 1);
      T scale = 1.0f / delta;

      std::vector<size_t> kappa;
      kappa.reserve(inputVector.size());

      transform(inputVector, std::back_inserter(kappa), [&](T val) {
         T clamped = std::clamp(val, minVal, maxVal);
         return static_cast<size_t>((clamped - minVal) * scale + 0.5f);
      });
      return std::make_tuple(delta, minVal, kappa);
   }

   std::vector<T> dequantize(T delta, T minVal, const std::vector<size_t>& kappa) {
      std::vector<T> outputVector;
      outputVector.reserve(kappa.size());
      transform(kappa, std::back_inserter(outputVector), [&](size_t q) { return static_cast<T>(q) * delta + minVal; });
      return outputVector;
   }

private:
   void migrate_to_batchsize(std::size_t new_batchsize) {
      spdlog::debug("Migrating to a batch size of {0:d} ", new_batchsize);
      sample = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = inputData.ncols(), .rows = new_batchsize};
      target = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = outputData.ncols(), .rows = new_batchsize};
      sample_t = NumericMatrix::Matrix<T, Backend>(inputData.ncols(), new_batchsize, _pool);
      batchSize_in_use = new_batchsize;
      auto stored_layers = layers;
      layers[0].setup(arch.front(), inputData.ncols(), new_batchsize, 0);
      layers[0].w = stored_layers[0].w;
      layers[0].b = stored_layers[0].b;
      for (size_t l = 1; l < layers.size(); ++l) {
         auto* curr_layer = &layers[l];
         auto* prev_layer = &layers[l - 1];
         curr_layer->setup(arch[l], arch[l - 1], new_batchsize, l);
         curr_layer->w = stored_layers[l].w;
         curr_layer->b = stored_layers[l].b;
      }
   }

   void set_log_level() const noexcept {
      if (const char* env_p = std::getenv("DEBUG")) {
         if (strncmp(env_p, "1", 1) == 0) {
            spdlog::debug("Setting log level to DEBUG");
            spdlog::set_level(spdlog::level::debug);
         } else {
            spdlog::set_level(spdlog::level::info);
            spdlog::debug("Setting log level to INFO");
         }
      }
   }

   std::vector<int> arch;
   std::vector<TINYAI::LinearLayer<T, Activation, Backend>> layers;
   GENERIC_TS_POOL::MemPool* _pool;
   NumericMatrix::ConstMatrixView<T> sample, target;
   NumericMatrix::ConstMatrixView<T> inputData, outputData;
   NumericMatrix::Matrix<T, Backend> sample_t;
   NumericMatrix::Matrix<T, Backend> batchedInput, batchedOutput;
   NumericMatrix::HostMatrix<T> buffer;
   std::size_t batchSize_in_use = 0;
   std::size_t iter = 1;
   tinyAI_blasHandle_t handle;
   std::array<tinyAI_gpuStream_t, 2> s;
   std::mt19937 generator;
   std::uniform_int_distribution<std::size_t> dist;
};
} // namespace TINYAI
