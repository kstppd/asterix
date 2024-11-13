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
#include <ranges>
#include <stdlib.h>
#include <tuple>
#include <vector>

#ifndef NOPROFILE
   #ifdef __CUDACC__
      #include <nvToolsExt.h>
      #define PROFILE_START(msg)   nvtxRangePushA((msg))
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

template <typename T, BACKEND Backend>
class NeuralNetwork {
public:
   NeuralNetwork(std::vector<int>& arch, GENERIC_TS_POOL::MemPool* pool,
                          NumericMatrix::Matrix<T, Backend>& input, NumericMatrix::Matrix<T, Backend>& output,
                          size_t batchSize, int seed = 42)
       : arch(arch), _pool(pool), batchSize_in_use(batchSize) {

      // Bind layers to the pool
      layers.resize(arch.size());
      for (size_t i = 0; i < layers.size(); ++i) {
         layers.at(i) = LinearLayer<T, Backend>(arch.at(i), _pool);
      }
      // Bind all objects to the memory pool
      inputData = NumericMatrix::MatrixView<T>{._data = nullptr, .cols = input.ncols(), .rows = input.nrows()};
      outputData = NumericMatrix::MatrixView<T>{._data = nullptr, .cols = output.ncols(), .rows = output.nrows()};
      input.getView(inputData, 0);
      output.getView(outputData, 0);
      sample = NumericMatrix::MatrixView<T>{._data = nullptr, .cols = input.ncols(), .rows = batchSize};
      target = NumericMatrix::MatrixView<T>{._data = nullptr, .cols = output.ncols(), .rows = batchSize};
      sample_t = NumericMatrix::Matrix<T, Backend>(input.ncols(), batchSize, _pool);
      layers[0].setup(arch.front(), inputData.ncols(), batchSize);
      for (size_t l = 1; l < layers.size(); ++l) {
         auto* curr_layer = &layers[l];
         auto* prev_layer = &layers[l - 1];
         curr_layer->setup(arch[l], arch[l - 1], batchSize);
      }
      if constexpr (Backend == BACKEND::DEVICE) {
         spdlog::debug("TinyAI Initalized on GPU");
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
      this->rng.seed(seed);
   }
   NeuralNetwork(const NeuralNetwork& other) = delete;
   NeuralNetwork(NeuralNetwork&& other) = delete;
   NeuralNetwork operator=(const NeuralNetwork& other) = delete;
   NeuralNetwork operator=(NeuralNetwork&& other) = delete;
   ~NeuralNetwork() {
      if constexpr (Backend == BACKEND::DEVICE) {
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

   void backward(const NumericMatrix::MatrixView<T>& sample, const NumericMatrix::MatrixView<T>& target) noexcept {
      spdlog::stopwatch timer;
      auto& curr_layer = layers.back();
      NumericMatrix::matsub(curr_layer.a, target, curr_layer.delta_store, &handle);
      NumericMatrix::mat_pointwise_activate_prime(curr_layer.z, curr_layer.a_prime);
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
         NumericMatrix::mat_pointwise_activate_prime(curr_layer.z, curr_layer.a_prime);
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

   T train(std::size_t batchSize, T lr = 1e-3) noexcept {
      // We need to check whether wed need to reconfigure our internal data
      // structures now due to a batchsize change
      assert(batchSize > 0 && batchSize <= inputData.nrows() &&
             "Batchsize cannot be bigger than your dataset you fool!");
      if (batchSize_in_use != batchSize) {
         migrate_to_batchsize(batchSize);
      }
      NumericMatrix::Matrix<T, Backend> error =
          NumericMatrix::Matrix<T, Backend>(target.nrows(), target.ncols(), _pool);

      NumericMatrix::Matrix<T, Backend> data_in =
          NumericMatrix::Matrix<T, Backend>(batchSize, inputData.ncols(), _pool);
      NumericMatrix::Matrix<T, Backend> data_out =
          NumericMatrix::Matrix<T, Backend>(batchSize, outputData.ncols(), _pool);

      // This block does the data shuffling once per epoch -- LUMI hackathon 2024
      PROFILE_START("IO");
      if constexpr (Backend == BACKEND::DEVICE) {

         NumericMatrix::HostMatrix<T> shuffle_in(inputData.nrows(), inputData.ncols());
         NumericMatrix::HostMatrix<T> shuffle_out(outputData.nrows(), outputData.ncols());
         NumericMatrix::export_to_host(inputData, shuffle_in);
         NumericMatrix::export_to_host(outputData, shuffle_out);
         const size_t numberOfInputRows = shuffle_in.nrows();
         const size_t numberOfInputCols = shuffle_in.ncols();
         const size_t numberOfOutputCols = shuffle_out.ncols();
         std::random_device rd;
         std::mt19937 gen(rd());
         T* buffer_in = new T[numberOfInputCols];
         T* buffer_out = new T[numberOfOutputCols];
         for (uint64_t i = numberOfInputRows - 1; i > 0; --i) {
            std::uniform_int_distribution<int> dist(0, i);
            auto target = dist(gen);
            // Input
            std::memcpy(buffer_in, &shuffle_in(target, 0), numberOfInputCols * sizeof(T));
            std::memcpy(&shuffle_in(target, 0), &shuffle_in(i, 0), numberOfInputCols * sizeof(T));
            std::memcpy(&shuffle_in(i, 0), buffer_in, numberOfInputCols * sizeof(T));
            // Output
            std::memcpy(buffer_out, &shuffle_out(target, 0), numberOfOutputCols * sizeof(T));
            std::memcpy(&shuffle_out(target, 0), &shuffle_out(i, 0), numberOfOutputCols * sizeof(T));
            std::memcpy(&shuffle_out(i, 0), buffer_out, numberOfOutputCols * sizeof(T));
         }
         delete[] buffer_in;
         delete[] buffer_out;
         NumericMatrix::get_from_host(inputData, shuffle_in);
         NumericMatrix::get_from_host(outputData, shuffle_out);

      } else {

         NumericMatrix::HostMatrix<T> shuffle_in(inputData.nrows(), inputData.ncols());
         NumericMatrix::HostMatrix<T> shuffle_out(outputData.nrows(), outputData.ncols());
         NumericMatrix::export_to_host_from_host(inputData, shuffle_in);
         NumericMatrix::export_to_host_from_host(outputData, shuffle_out);
         const size_t numberOfInputRows = shuffle_in.nrows();
         const size_t numberOfInputCols = shuffle_in.ncols();
         const size_t numberOfOutputCols = shuffle_out.ncols();
         std::random_device rd;
         std::mt19937 gen(rd());
         T* buffer_in = new T[numberOfInputCols];
         T* buffer_out = new T[numberOfOutputCols];
         for (uint64_t i = numberOfInputRows - 1; i > 0; --i) {
            std::uniform_int_distribution<int> dist(0, i);
            auto target = dist(gen);
            // Input
            std::memcpy(buffer_in, &shuffle_in(target, 0), numberOfInputCols * sizeof(T));
            std::memcpy(&shuffle_in(target, 0), &shuffle_in(i, 0), numberOfInputCols * sizeof(T));
            std::memcpy(&shuffle_in(i, 0), buffer_in, numberOfInputCols * sizeof(T));
            // Output
            std::memcpy(buffer_out, &shuffle_out(target, 0), numberOfOutputCols * sizeof(T));
            std::memcpy(&shuffle_out(target, 0), &shuffle_out(i, 0), numberOfOutputCols * sizeof(T));
            std::memcpy(&shuffle_out(i, 0), buffer_out, numberOfOutputCols * sizeof(T));
         }
         delete[] buffer_in;
         delete[] buffer_out;
         NumericMatrix::get_from_host_from_host(inputData, shuffle_in);
         NumericMatrix::get_from_host_from_host(outputData, shuffle_out);
      }
      PROFILE_END();

      T loss = 0.0;
      PROFILE_START("Epoch Training");
      PROFILE_START("Pool allocation");
      T* dev_loss = _pool->allocate<T>(1);
      PROFILE_END();
      for (size_t i = 0; i < inputData.nrows(); i += batchSize) {
         // Collect input-output
         inputData.getView(sample, i);
         outputData.getView(target, i);
         PROFILE_START("Forward");
         forward(sample);
         PROFILE_END();
         PROFILE_START("Error calculation");
         // Get loss
         NumericMatrix::matsub(layers.back().a, target, error, &handle);
         NumericMatrix::matreduce_mse(error, dev_loss, &handle);
         if constexpr (Backend == BACKEND::HOST) {
            loss += *dev_loss;
         } else {
            T tmp = 0.0;
            tinyAI_gpuMemcpy(&tmp, dev_loss, sizeof(T), tinyAI_gpuMemcpyDeviceToHost);
            loss += tmp;
         }
         PROFILE_END();
         PROFILE_START("Backward");
         backward(sample, target);
         PROFILE_END();
         PROFILE_START("Weight Update AdamW");
         update_weights_adamw(iter, lr);
         PROFILE_END();
         iter++;
      }
      PROFILE_START("Pool deallocation");
      _pool->deallocate(dev_loss);
      PROFILE_END();
      PROFILE_END();
      spdlog::debug("Epoch done");
      return loss / inputData.nrows();
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
      const auto target_cols=eval_output.ncols();
      const auto target_rows=eval_output.nrows();
      if (target_cols!=layers.back().a.ncols() || target_rows!=layers.back().a.nrows()){\
         std::cerr<<"Buffer is wrongly sized in inference copy-out!"<<std::endl;
         abort();
      }
      eval_output = layers.back().a;
   }

   // Returns the number of bytes written
   size_t get_weights(T* dst) const noexcept {
      size_t write_index = 0;
      for (const auto& layer : layers) {
         // Weights
         for (size_t i = 0; i < layer.w.size(); ++i) {
            dst[write_index] = layer.w(i);
            write_index++;
         }
         // Biases
         for (size_t i = 0; i < layer.b.size(); ++i) {
            dst[write_index] = layer.b(i);
            write_index++;
         }
      }
      return write_index * sizeof(T);
   }
   
   // Returns the number of bytes written
   void cast_to_float() noexcept {
      for (auto& layer : layers) {
         // Weights
         for (size_t i = 0; i < layer.w.size(); ++i) {
            layer.w(i)=static_cast<T>( static_cast<float>(layer.w(i))  );
         }
         // Biases
         for (size_t i = 0; i < layer.b.size(); ++i) {
            layer.b(i)=static_cast<T>( static_cast<float>(layer.b(i))  );
         }
      }
   }

   // Returns the number of bytes read
   size_t load_weights(T* src) noexcept {
      size_t read_index = 0;
      for (auto& layer : layers) {
         // Weights
         for (size_t i = 0; i < layer.w.size(); ++i) {
            layer.w(i) = src[read_index];
            read_index++;
         }
         // Biases
         for (size_t i = 0; i < layer.b.size(); ++i) {
            layer.b(i) = src[read_index];
            read_index++;
         }
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
      transform(kappa, std::back_inserter(outputVector),
                             [&](size_t q) { return static_cast<T>(q) * delta + minVal; });
      return outputVector;
   }

private:
   void migrate_to_batchsize(std::size_t new_batchsize) {
      spdlog::debug("Migrating to a batch size of {0:d} ", new_batchsize);
      sample = NumericMatrix::MatrixView<T>{._data = nullptr, .cols = inputData.ncols(), .rows = new_batchsize};
      target = NumericMatrix::MatrixView<T>{._data = nullptr, .cols = outputData.ncols(), .rows = new_batchsize};
      sample_t = NumericMatrix::Matrix<T, Backend>(inputData.ncols(), new_batchsize, _pool);
      batchSize_in_use = new_batchsize;
      auto stored_layers = layers;
      layers[0].setup(arch.front(), inputData.ncols(), new_batchsize);
      layers[0].w = stored_layers[0].w;
      layers[0].b = stored_layers[0].b;
      for (size_t l = 1; l < layers.size(); ++l) {
         _pool->defrag();
         auto* curr_layer = &layers[l];
         auto* prev_layer = &layers[l - 1];
         curr_layer->setup(arch[l], arch[l - 1], new_batchsize);
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
   std::vector<TINYAI::LinearLayer<T, Backend>> layers;
   GENERIC_TS_POOL::MemPool* _pool;
   NumericMatrix::MatrixView<T> sample, target;
   NumericMatrix::Matrix<T, Backend> sample_t;
   NumericMatrix::MatrixView<T> inputData, outputData;
   NumericMatrix::HostMatrix<T> buffer;
   std::size_t batchSize_in_use = 0;
   std::size_t iter = 1;
   tinyAI_blasHandle_t handle;
   std::default_random_engine rng;
};
} // namespace TINYAI
