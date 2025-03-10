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

template <typename T, BACKEND Backend = BACKEND::HOST, ACTIVATION Activation = ACTIVATION::TANH,
          ACTIVATION OutputActivation = ACTIVATION::TANH, LOSSF LossF = LOSSF::MSE>
class NeuralNetwork {
public:
   NeuralNetwork(std::vector<int>& arch, GENERIC_TS_POOL::MemPool* pool, const NumericMatrix::Matrix<T, Backend>& input,
                 const NumericMatrix::Matrix<T, Backend>& output, size_t batchSize, int seed = 42)
       : arch(arch), _pool(pool), batchSize_in_use(batchSize) {

      set_log_level();
      TINYAI_UNUSED(seed);
      // Bind layers to the pool
      layers.resize(arch.size());
      for (size_t i = 0; i < layers.size() - 1; ++i) {
         layers.at(i) = std::make_unique<LinearLayer<T, Activation, Backend>>(arch.at(i), _pool);
      }
      layers.back() = std::make_unique<LinearLayer<T, OutputActivation, Backend>>(arch.back(), _pool);
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
      layers[0]->setup(arch.front(), inputData.ncols(), batchSize, 0);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l]->setup(arch[l], arch[l - 1], batchSize, l);
      }

      if constexpr (Backend == BACKEND::DEVICE) {
         spdlog::info("TinyAI Initalized on GPU");
         tinyAI_gpuStreamCreate(&s[0]);
         tinyAI_gpuStreamCreate(&s[1]);
         auto stat = tinyAI_blasCreate(&handle);
         if (stat != BLAS_SUCCESS) {
            std::cerr << "Stat = " << stat << std::endl;
            spdlog::error("Failed to initialize CUBLAS.");
            throw std::runtime_error("Failed to initialize CUBLAS");
         } else {
            spdlog::info("CUBLAS initialized succesfully.");
         }
      } else {
         spdlog::info("TinyAI Initalized on CPU");
      }
      set_log_level();
      generator();
      dist = std::uniform_int_distribution<std::size_t>(static_cast<std::size_t>(0), inputData.nrows() - 1);
   }

   NeuralNetwork(std::vector<int>& arch, GENERIC_TS_POOL::MemPool* pool, std::size_t samples, std::size_t fin,
                 std::size_t fout, size_t batchSize, int seed = 42)
       : arch(arch), _pool(pool), batchSize_in_use(batchSize) {

      set_log_level();
      TINYAI_UNUSED(seed);
      // Bind layers to the pool
      layers.resize(arch.size());
      for (size_t i = 0; i < layers.size() - 1; ++i) {
         layers.at(i) = std::make_unique<LinearLayer<T, Activation, Backend>>(arch.at(i), _pool);
      }
      layers.back() = std::make_unique<LinearLayer<T, OutputActivation, Backend>>(arch.back(), _pool);
      // Bind all objects to the memory pool
      inputData = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = fin, .rows = samples};
      outputData = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = fout, .rows = samples};
      sample = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = fin, .rows = batchSize};
      target = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = fout, .rows = batchSize};
      sample_t = NumericMatrix::Matrix<T, Backend>(fin, batchSize, _pool);
      batchedInput = NumericMatrix::Matrix<T, Backend>(batchSize, fin, _pool);
      batchedOutput = NumericMatrix::Matrix<T, Backend>(batchSize, fout, _pool);
      layers[0]->setup(arch.front(), inputData.ncols(), batchSize, 0);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l]->setup(arch[l], arch[l - 1], batchSize, l);
      }

      if constexpr (Backend == BACKEND::DEVICE) {
         spdlog::info("TinyAI Initalized on GPU");
         tinyAI_gpuStreamCreate(&s[0]);
         tinyAI_gpuStreamCreate(&s[1]);
         auto stat = tinyAI_blasCreate(&handle);
         if (stat != BLAS_SUCCESS) {
            std::cerr << "Stat = " << stat << std::endl;
            spdlog::error("Failed to initialize CUBLAS.");
            throw std::runtime_error("Failed to initialize CUBLAS");
         } else {
            spdlog::info("CUBLAS initialized succesfully.");
         }
      } else {
         spdlog::info("TinyAI Initalized on CPU");
      }
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

   void reset() {
      layers[0]->reset(inputData.ncols(), 0);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l]->reset(arch[l - 1], l);
      }
   }

   void forward(const NumericMatrix::Matrix<T, Backend>& in, tinyAI_gpuStream_t stream = 0) noexcept {
      spdlog::stopwatch timer;
      layers[0]->forward(in, &handle, stream);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l]->forward(layers[l - 1]->a, &handle, stream);
      }
      spdlog::debug("Feed Forward {:.3}s", timer);
   }

   void forward(const NumericMatrix::MatrixView<T>& in, tinyAI_gpuStream_t stream = 0) noexcept {
      spdlog::stopwatch timer;
      layers[0]->forward(in, &handle, stream);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l]->forward(layers[l - 1]->a, &handle, stream);
      }
      spdlog::debug("Feed Forward {:.3}s", timer);
   }

   void forward(const NumericMatrix::ConstMatrixView<T>& in, tinyAI_gpuStream_t stream = 0) noexcept {
      spdlog::stopwatch timer;
      layers[0]->forward(in, &handle, stream);
      for (size_t l = 1; l < layers.size(); ++l) {
         layers[l]->forward(layers[l - 1]->a, &handle, stream);
      }
      spdlog::debug("Feed Forward {:.3}s", timer);
   }

   void backward(const NumericMatrix::ConstMatrixView<T>& sample, const NumericMatrix::ConstMatrixView<T>& target,
                 tinyAI_gpuStream_t stream) noexcept {
      spdlog::stopwatch timer;
      auto& curr_layer = layers.back();
      NumericMatrix::loss_derivative<T, LossF>(curr_layer->a, target, curr_layer->delta_store, &handle, stream);
      NumericMatrix::mat_pointwise_activate_prime<T, Activation>(curr_layer->z, curr_layer->a_prime, curr_layer->wmega,
                                                                 stream);
      NumericMatrix::mat_pointwise_mul(curr_layer->delta_store, curr_layer->a_prime, curr_layer->delta, stream);
      NumericMatrix::transpose_into(sample, sample_t, stream);
      if (layers.size() == 1) {
         NumericMatrix::matmul(sample_t, curr_layer->delta, curr_layer->dw, &handle);
      } else {
         NumericMatrix::transpose_into(layers[layers.size() - 2]->a, layers[layers.size() - 2]->a_t, stream);
         NumericMatrix::matmul(layers[layers.size() - 2]->a_t, curr_layer->delta, curr_layer->dw, &handle);
      }
      NumericMatrix::matsum_rows(curr_layer->delta, curr_layer->db, stream);

      for (int i = layers.size() - 2; i >= 0; i--) {

         auto& next_layer = layers[i + 1];
         auto& curr_layer = layers[i];
         NumericMatrix::transpose_into(next_layer->w, next_layer->w_t, stream);
         NumericMatrix::matmul(next_layer->delta, next_layer->w_t, next_layer->buffer, &handle);
         NumericMatrix::mat_pointwise_activate_prime<T, Activation>(curr_layer->z, curr_layer->a_prime,
                                                                    curr_layer->wmega, stream);
         NumericMatrix::mat_pointwise_mul(next_layer->buffer, curr_layer->a_prime, curr_layer->delta, stream);
         if (i == 0) {
            NumericMatrix::matmul(sample_t, curr_layer->delta, curr_layer->dw, &handle);
         } else {
            NumericMatrix::transpose_into(layers[i - 1]->a, layers[i - 1]->a_t, stream);
            NumericMatrix::matmul(layers[i - 1]->a_t, curr_layer->delta, curr_layer->dw, &handle);
         }
         NumericMatrix::matsum_rows(curr_layer->delta, curr_layer->db, stream);
      }
      for (auto& l : layers) {
         NumericMatrix::matscale(l->dw, T(1.0) / batchSize_in_use, &handle, stream);
         NumericMatrix::matscale(l->db, T(1.0) / batchSize_in_use, &handle, stream);
      }
      spdlog::debug("Backward {:.3}s", timer);
   }

   void backward(const NumericMatrix::Matrix<T, Backend>& sample, const NumericMatrix::Matrix<T, Backend>& target,
                 tinyAI_gpuStream_t stream) noexcept {
      NumericMatrix::ConstMatrixView<T> sample_view;
      NumericMatrix::ConstMatrixView<T> target_view;
      sample.getFullView(sample_view);
      target.getFullView(target_view);
      return backward(sample_view, target_view, stream);
   }

   void setStream(tinyAI_gpuStream_t stream) noexcept {
      if constexpr (Backend == BACKEND::DEVICE) {
         tinyAI_cuSetStream(handle, stream);
      }
   }

   T train(std::size_t batchSize, T lr, tinyAI_gpuStream_t stream = 0) {

      // We need to check whether wed need to reconfigure our internal data
      // structures now due to a batchsize change
      assert(batchSize > 0 && batchSize <= inputData.nrows() &&
             "Batchsize cannot be bigger than your dataset you fool!");
      if (batchSize_in_use != batchSize) {
         migrate_to_batchsize(batchSize);
      }
      if (inputData.data() == nullptr || outputData.data() == nullptr) {
         throw std::runtime_error("ERROR: input and/or output data views have not been set!");
      }

      setStream(stream);
      NumericMatrix::Matrix<T, Backend> error =
          NumericMatrix::Matrix<T, Backend>(target.nrows(), target.ncols(), _pool);

      T loss = 0.0;
      PROFILE_START("Epoch Training");

      std::vector<std::size_t> perm(batchSize_in_use, 0);
      std::size_t* dperm = _pool->allocate<std::size_t>(batchSize_in_use);
      for (size_t i = 0; i < inputData.nrows(); i += batchSize) {

         PROFILE_START("BATCH PASS");

         PROFILE_START("Permutation Indices");
         for (std::size_t k = 0; k < batchSize_in_use; ++k) {
            perm[k] = dist(generator);
         }
         // Launch this copy here and we wait it in the next loop
         if constexpr (Backend == BACKEND::DEVICE) {
            tinyAI_gpuMemcpyAsync(dperm, perm.data(), batchSize * sizeof(std::size_t), tinyAI_gpuMemcpyHostToDevice,
                                  stream);
         }
         tinyAI_gpuStreamSynchronize(stream);
         PROFILE_END();

         PROFILE_START("IO");
         if constexpr (Backend == BACKEND::DEVICE) {
            if (batchSize_in_use > 1024) {
               throw std::runtime_error(
                   "ERROR: TinyAI unable to shuffle rows on the GPU when running with batchsizes larger "
                   "than the max blocksize of 1024");
            }

#ifdef __NVCC__
            NumericMatrix::shuffle_rows_warpwide(inputData.data(), dperm, batchSize_in_use, batchedInput.data(),
                                                 inputData.ncols(), stream);
            NumericMatrix::shuffle_rows_warpwide(outputData.data(), dperm, batchSize_in_use, batchedOutput.data(),
                                                 outputData.ncols(), stream);
#else
            NumericMatrix::shuffle_rows<<<1, batchSize_in_use, 0, stream>>>(inputData.data(), dperm,
                                                                            batchedInput.data(), inputData.ncols());
            NumericMatrix::shuffle_rows<<<1, batchSize_in_use, 0, stream>>>(outputData.data(), dperm,
                                                                            batchedOutput.data(), outputData.ncols());
#endif
            tinyAI_gpuStreamSynchronize(stream);

         } else {
            for (std::size_t k = 0; k < batchSize_in_use; ++k) {
               const std::size_t index = dist(generator);
               std::memcpy(&batchedInput(k, 0), &inputData(index, 0), inputData.ncols() * sizeof(T));
               std::memcpy(&batchedOutput(k, 0), &outputData(index, 0), outputData.ncols() * sizeof(T));
            }
         }
         PROFILE_END();

         // Collect input-output
         batchedInput.getView(sample, 0);
         batchedOutput.getView(target, 0);

         PROFILE_START("Forward");
         forward(sample, stream);
         tinyAI_gpuStreamSynchronize(stream);
         PROFILE_END();

         PROFILE_START("Error calculation");
         // Get loss
         NumericMatrix::loss<T, LossF>(layers.back()->a, target, error, &handle, stream);
         if constexpr (Backend == BACKEND::HOST) {
            loss += NumericMatrix::matreduce_add(error, &handle);
         } else {
            loss += NumericMatrix::matreduce_add_gpu(error, _pool, &handle, stream);
         }
         tinyAI_gpuStreamSynchronize(stream);
         PROFILE_END();

         PROFILE_START("Backward");
         backward(sample, target, stream);
         tinyAI_gpuStreamSynchronize(stream);
         PROFILE_END();

         PROFILE_START("Weight Update AdamW");
         update_weights_adamw(iter, lr, stream);
         tinyAI_gpuStreamSynchronize(stream);
         PROFILE_END();

         iter++;
         PROFILE_END();
      }
      PROFILE_END();
      tinyAI_gpuStreamSynchronize(stream);
      spdlog::debug("Epoch done");
      setStream(stream);
      return loss / (inputData.nrows() * outputData.ncols());
   }

   T train(const NumericMatrix::Matrix<T, Backend>& x, const NumericMatrix::Matrix<T, Backend>& y,
           std::size_t batchSize, T lr, tinyAI_gpuStream_t stream = 0) {

      inputData = NumericMatrix::ConstMatrixView<T>{._data = x.data(), .cols = x.ncols(), .rows = x.nrows()};
      outputData = NumericMatrix::ConstMatrixView<T>{._data = y.data(), .cols = y.ncols(), .rows = y.nrows()};
      return train(batchSize, lr, stream);
   }

   void update_weights_adamw(size_t iteration, T lr, tinyAI_gpuStream_t stream, T beta1 = 0.9, T beta2 = 0.999,
                             T epsilon = 1e-8, T decay = 1e-4) noexcept {
      spdlog::stopwatch timer;
      const T m_hat_scale = static_cast<T>(1.0) / (1 - std::pow(beta1, iteration));
      const T v_hat_scale = static_cast<T>(1.0) / (1 - std::pow(beta2, iteration));
      for (auto& curr_layer : layers) {

         // Weights
         NumericMatrix::adamw(curr_layer->w, curr_layer->m_w, curr_layer->v_w, curr_layer->dw, m_hat_scale, v_hat_scale,
                              beta1, beta2, decay, lr, epsilon, stream);
         // Biases
         NumericMatrix::adamw(curr_layer->b, curr_layer->m_b, curr_layer->v_b, curr_layer->db, m_hat_scale, v_hat_scale,
                              beta1, beta2, decay, lr, epsilon, stream);
      }
      spdlog::debug("AdamW Weight Update {:.3}s", timer);
   }

   // TODO verify this works correclty with the batched inferense
   void evaluate(NumericMatrix::Matrix<T, Backend>& eval_samples,
                 NumericMatrix::Matrix<T, Backend>& eval_output) noexcept {

      const std::size_t total_samples = eval_samples.nrows();
      for (std::size_t i = 0; i < eval_samples.nrows(); i += batchSize_in_use) {

         NumericMatrix::MatrixView<T> x{._data = nullptr, .cols = eval_samples.ncols(), .rows = batchSize_in_use};
         NumericMatrix::MatrixView<T> y{._data = nullptr, .cols = eval_output.ncols(), .rows = batchSize_in_use};
         eval_samples.getView(x, i);
         eval_output.getView(y, i);
         forward(x);
         tinyAI_gpuMemcpy(y.data(), layers.back()->a.data(), layers.back()->a.size() * sizeof(T),
                          tinyAI_gpuMemcpyDeviceToDevice);

         std::size_t left_over = total_samples - (i + batchSize_in_use);
         if (left_over > 0 && left_over < batchSize_in_use) {
            _pool->defrag();
            migrate_to_batchsize(left_over);

            NumericMatrix::MatrixView<T> x_last{._data = nullptr, .cols = eval_samples.ncols(), .rows = left_over};
            NumericMatrix::MatrixView<T> y_last{._data = nullptr, .cols = eval_output.ncols(), .rows = left_over};
            eval_samples.getView(x_last, i + batchSize_in_use);
            eval_output.getView(y_last, i + batchSize_in_use);

            forward(x_last);
            tinyAI_gpuMemcpy(y_last.data(), layers.back()->a.data(), layers.back()->a.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToDevice);
            break;
         }
      }
   }

   void evaluate_at_once(const NumericMatrix::Matrix<T, Backend>& eval_samples,
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
      eval_output = layers.back()->a;
   }

   // Returns the number of bytes written
   size_t get_weights(T* dst) const noexcept {
      size_t write_index = 0;
      for (const auto& layer : layers) {
         // Weights
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(&dst[write_index], layer->w.data(), layer->w.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(&dst[write_index], layer->w.data(), layer->w.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToHost);
         }
         write_index += layer->w.size();
         // Biases
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(&dst[write_index], layer->b.data(), layer->b.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(&dst[write_index], layer->b.data(), layer->b.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToHost);
         }
         write_index += layer->b.size();
      }
      return write_index * sizeof(T);
   }

   // Returns the number of bytes read
   size_t load_weights(const T* src) noexcept {
      size_t read_index = 0;
      for (auto& layer : layers) {
         // Weights
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(layer->w.data(), &src[read_index], layer->w.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(layer->w.data(), &src[read_index], layer->w.size() * sizeof(T),
                             tinyAI_gpuMemcpyHostToDevice);
         }
         read_index += layer->w.size();
         // Biases
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(layer->b.data(), &src[read_index], layer->b.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(layer->b.data(), &src[read_index], layer->b.size() * sizeof(T),
                             tinyAI_gpuMemcpyHostToDevice);
         }
         read_index += layer->b.size();
      }
      return read_index * sizeof(T);
   }
   
   size_t get_grads(T* dst) const noexcept {
      size_t write_index = 0;
      for (const auto& layer : layers) {
         // Weights
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(&dst[write_index], layer->dw.data(), layer->dw.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(&dst[write_index], layer->dw.data(), layer->dw.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToHost);
         }
         write_index += layer->dw.size();
         // Biases
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(&dst[write_index], layer->db.data(), layer->db.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(&dst[write_index], layer->db.data(), layer->db.size() * sizeof(T),
                             tinyAI_gpuMemcpyDeviceToHost);
         }
         write_index += layer->db.size();
      }
      return write_index * sizeof(T);
   }
   
   size_t load_grads(const T* src) noexcept {
      size_t read_index = 0;
      for (auto& layer : layers) {
         // Weights
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(layer->dw.data(), &src[read_index], layer->dw.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(layer->dw.data(), &src[read_index], layer->dw.size() * sizeof(T),
                             tinyAI_gpuMemcpyHostToDevice);
         }
         read_index += layer->dw.size();
         // Biases
         if constexpr (Backend == BACKEND::HOST) {
            std::memcpy(layer->db.data(), &src[read_index], layer->db.size() * sizeof(T));
         } else {
            tinyAI_gpuMemcpy(layer->db.data(), &src[read_index], layer->db.size() * sizeof(T),
                             tinyAI_gpuMemcpyHostToDevice);
         }
         read_index += layer->db.size();
      }
      return read_index * sizeof(T);
   }
   

   // Returns the number of bytes needed to store the network's weights (W,B)
   std::size_t get_network_size() const noexcept {
      std::size_t total_size = 0;
      for (auto& layer : layers) {
         total_size += layer->w.size() * sizeof(T);
         total_size += layer->b.size() * sizeof(T);
      }
      return total_size;
   }
   
   std::size_t get_network_weight_count() const noexcept {
      std::size_t total_size = 0;
      for (auto& layer : layers) {
         total_size += layer->w.size();
         total_size += layer->b.size();
      }
      return total_size;
   }

   void migrate_to_batchsize(std::size_t new_batchsize) {
      spdlog::debug("Migrating to a batch size of {0:d} ", new_batchsize);
      sample = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = inputData.ncols(), .rows = new_batchsize};
      target = NumericMatrix::ConstMatrixView<T>{._data = nullptr, .cols = outputData.ncols(), .rows = new_batchsize};
      sample_t = NumericMatrix::Matrix<T, Backend>(inputData.ncols(), new_batchsize, _pool);
      batchSize_in_use = new_batchsize;
      std::vector<std::unique_ptr<BaseLayer<T, Backend>>> stored_layers;
      for (std::size_t i = 0; i < layers.size() - 1; ++i) {
         stored_layers.push_back(std::make_unique<LinearLayer<T, Activation, Backend>>(arch.at(i), _pool));
         stored_layers.back()->w = layers[i]->w;
         stored_layers.back()->b = layers[i]->b;
      }
      stored_layers.push_back(std::make_unique<LinearLayer<T, OutputActivation, Backend>>(arch.back(), _pool));
      stored_layers.back()->w = layers.back()->w;
      stored_layers.back()->b = layers.back()->b;

      layers[0]->setup(arch.front(), inputData.ncols(), new_batchsize, 0);
      layers[0]->w = stored_layers[0]->w;
      layers[0]->b = stored_layers[0]->b;
      for (size_t l = 1; l < layers.size(); ++l) {
         auto& curr_layer = layers[l];
         auto& prev_layer = layers[l - 1];
         curr_layer->setup(arch[l], arch[l - 1], new_batchsize, l);
         curr_layer->w = stored_layers[l]->w;
         curr_layer->b = stored_layers[l]->b;
      }
   }

   void set_log_level() {
      spdlog::set_level(spdlog::level::info);
      if (const char* env_p = std::getenv("DEBUG")) {
         if (strncmp(env_p, "1", 1) == 0) {
            spdlog::set_level(spdlog::level::debug);
            return;
         }
      }
      if (const char* env_p = std::getenv("INFO")) {
         if (strncmp(env_p, "0", 0) == 0) {
            spdlog::set_level(spdlog::level::off);
         }
      }
   }

   constexpr BACKEND get_backend() const noexcept { return Backend; }

   template <BACKEND HW, LOSSF LOSSFUNCTION>
   T loss(NumericMatrix::Matrix<T, HW>& error, NumericMatrix::ConstMatrixView<T>& target,
          tinyAI_gpuStream_t stream = 0) {
      NumericMatrix::loss<T, LOSSFUNCTION>(layers.back()->a, target, error, &handle, stream);
      T loss = T(0.0);
      if constexpr (Backend == BACKEND::HOST) {
         loss += NumericMatrix::matreduce_add(error, &handle);
      } else {
         loss += NumericMatrix::matreduce_add_gpu(error, _pool, &handle, stream);
      }
      tinyAI_gpuStreamSynchronize(stream);
      return loss;
   }

   template <BACKEND HW, LOSSF LOSSFUNCTION>
   T loss(NumericMatrix::Matrix<T, HW>& error, NumericMatrix::Matrix<T, HW>& target, tinyAI_gpuStream_t stream = 0) {
      NumericMatrix::ConstMatrixView<T> target_view;
      target.getFullView(target_view);
      return loss<HW, LOSSFUNCTION>(error, target_view, stream);
   }

   void shuffle_into(NumericMatrix::Matrix<T, Backend>& src, NumericMatrix::Matrix<T, Backend>& dst,
                     std::size_t* perm,tinyAI_gpuStream_t stream = 0) {
      if constexpr (Backend == BACKEND::DEVICE) {
#ifdef __NVCC__
         NumericMatrix::shuffle_rows_warpwide(src.data(), perm, dst.nrows(), dst.data(),
                                              src.ncols(), stream);
#else
         NumericMatrix::shuffle_rows<<<1, batchSize_in_use, 0, stream>>>(src.data(), perm, dst.data(),
                                                                         src.ncols());
#endif
      } else {
         for (std::size_t k = 0; k < dst.nrows(); ++k) {
            std::memcpy(&dst(k, 0), &src(perm[k], 0), src.ncols() * sizeof(T));
         }
      }
   }

   void get_permutation_indices(std::size_t *perm,std::size_t n,tinyAI_gpuStream_t stream = 0){
      std::vector<std::size_t> stage(n,0);
      for (std::size_t k = 0; k < n; ++k) {
          stage[k] = dist(generator);
      }
      if constexpr (Backend == BACKEND::DEVICE) {
         tinyAI_gpuMemcpyAsync(perm, stage.data(),n * sizeof(std::size_t), tinyAI_gpuMemcpyHostToDevice, stream);
      } else {
         std::memcpy(perm,stage.data(),n * sizeof(std::size_t));
      }
      return;
   }

   void print_weights_statistics()const noexcept{
      for (std::size_t i=0;i<layers.size();++i){
         layers[i]->print_stats(i);
      }
   }

   std::vector<int> arch;
   std::vector<std::unique_ptr<BaseLayer<T, Backend>>> layers;
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
