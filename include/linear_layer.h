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
#include "genericTsPool.h"
#include "matrix.h"
#include "spdlog/stopwatch.h"

namespace TINYAI {

template <typename T, BACKEND Backend>
class BaseLayer {
public:
   size_t neurons = 0;
   T wmega = 1.0;
   GENERIC_TS_POOL::MemPool* _pool;
   NumericMatrix::Matrix<T, Backend> buffer, w, w_t, b, b_broadcasted, z, a, a_prime, a_t, dw, db, delta, delta_store;
   NumericMatrix::Matrix<T, Backend> m_w, m_b, v_w, v_b;
   virtual ~BaseLayer() = default;
   virtual void forward(const NumericMatrix::Matrix<T, Backend>& input, tinyAI_blasHandle_t* handle,
                        tinyAI_gpuStream_t stream) noexcept = 0;

   virtual void forward(const NumericMatrix::MatrixView<T>& input, tinyAI_blasHandle_t* handle,
                        tinyAI_gpuStream_t stream) noexcept = 0;

   virtual void forward(const NumericMatrix::ConstMatrixView<T>& input, tinyAI_blasHandle_t* handle,
                        tinyAI_gpuStream_t stream) noexcept = 0;

   virtual void setup(size_t neurons, size_t input, size_t batchSize, size_t layer_id) = 0;
   virtual void reset(size_t input, size_t layer_id) = 0;
   void print_stats(std::size_t layer_id){
      const std::size_t sz=this->w.size();
      NumericMatrix::HostMatrix<T> w_host(this->w);
      const auto [min,max] = std::minmax_element(w_host.begin(),w_host.end());
      const T sum = std::accumulate(w_host.begin(),w_host.end(),T(0));
      const T mu = sum/w_host.size();
      const T variance = std::accumulate(w_host.begin(), w_host.end(), T(0),
                                  [mu](T acc, T x) { return acc + (x - mu) * (x - mu); }) / w_host.size();
      const T sigma = std::sqrt(variance);
      printf("Layer {%zu}\n\t Min,Max,Mu,Sigma=[%f,%f,%f,%f]\n",layer_id,*min,*max,mu,sigma);
   }
};

template <typename T, ACTIVATION Activation, BACKEND Backend>
class LinearLayer : public BaseLayer<T, Backend> {
public:
   LinearLayer() { this->_pool = nullptr; }
   LinearLayer(GENERIC_TS_POOL::MemPool* p) { this->_pool = p; }
   LinearLayer(size_t n, GENERIC_TS_POOL::MemPool* p) {
      this->neurons = n;
      this->_pool = p;
   }

   void setup(size_t neurons, size_t input, size_t batchSize, size_t layer_id) override {
      assert(neurons > 0 && "This layer has 0 neurons!");
      this->w = NumericMatrix::Matrix<T, Backend>(input, neurons, this->_pool);
      this->m_w = NumericMatrix::Matrix<T, Backend>(input, neurons, this->_pool);
      this->v_w = NumericMatrix::Matrix<T, Backend>(input, neurons, this->_pool);
      this->b = NumericMatrix::Matrix<T, Backend>(1, neurons, this->_pool);
      this->m_b = NumericMatrix::Matrix<T, Backend>(1, neurons, this->_pool);
      this->v_b = NumericMatrix::Matrix<T, Backend>(1, neurons, this->_pool);
      this->z = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, this->_pool);
      this->a = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, this->_pool);
      this->b_broadcasted = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, this->_pool);
      this->a_prime = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, this->_pool);
      this->w_t = NumericMatrix::Matrix<T, Backend>(neurons, input, this->_pool);
      this->a_t = NumericMatrix::Matrix<T, Backend>(neurons, batchSize, this->_pool);
      this->delta = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, this->_pool);
      this->delta_store = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, this->_pool);
      this->buffer = NumericMatrix::Matrix<T, Backend>(batchSize, input, this->_pool);
      this->dw = NumericMatrix::Matrix<T, Backend>(input, neurons, this->_pool);
      this->db = NumericMatrix::Matrix<T, Backend>(1, neurons, this->_pool);

      const T fan_in = input;
      const T fan_out = neurons;
      T std = std::sqrt(2.0 / (fan_in));
      if constexpr (Activation == ACTIVATION::SIN) {
         if (layer_id == 0) {
            this->wmega = 10;
            std = std::sqrt(6.0f / (T)input);
         } else {
            this->wmega = 1;
            std = std::sqrt(6.0f / ((T)input));
         }
      }
      // printf("Layer init with std=%f\n",std);
      if constexpr (Backend == BACKEND::DEVICE) {
         NumericMatrix::HostMatrix<T> _w(this->w);
         NumericMatrix::mat_randomise(_w, std);
         NumericMatrix::get_from_host(this->w, _w);
      } else {
         NumericMatrix::mat_randomise(this->w, std);
      }
      this->b.zero_out();
   }

   void reset(size_t input, size_t layer_id) override {
      const T fan_in = input;
      const T fan_out = this->neurons;
      T std = std::sqrt(2.0 / (fan_in + fan_out));
      if constexpr (Activation == ACTIVATION::SIN) {
         if (layer_id == 0) {
            this->wmega = 10;
            std = std::sqrt(6.0f / (T)input);
         } else {
            this->wmega = 1;
            std = std::sqrt(6.0f / ((T)input));
         }
      }

      if constexpr (Backend == BACKEND::DEVICE) {
         NumericMatrix::HostMatrix<T> _w(this->w.nrows(), this->w.ncols());
         NumericMatrix::HostMatrix<T> _b(this->b.nrows(), this->b.ncols());
         NumericMatrix::export_to_host(this->w, _w);
         NumericMatrix::export_to_host(this->b, _b);
         NumericMatrix::mat_randomise(_w, std);
         NumericMatrix::mat_randomise(_b, std);
         NumericMatrix::get_from_host(this->w, _w);
         NumericMatrix::get_from_host(this->b, _b);
      } else {
         NumericMatrix::mat_randomise(this->w, std);
         NumericMatrix::mat_randomise(this->b, std);
      }
   }

   void forward(const NumericMatrix::Matrix<T, Backend>& input, tinyAI_blasHandle_t* handle,
                tinyAI_gpuStream_t stream) noexcept override {
      assert(this->neurons > 0 && "This layer has 0 neurons!");
      NumericMatrix::matmul(input, this->w, this->z, handle);
      NumericMatrix::matbroadcast(this->b, this->b_broadcasted, stream);
      NumericMatrix::matadd_and_activate<T, Activation>(this->z, this->b_broadcasted, this->z, this->a, this->wmega,
                                                        handle, stream);
   }

   void forward(const NumericMatrix::MatrixView<T>& input, tinyAI_blasHandle_t* handle,
                tinyAI_gpuStream_t stream) noexcept override {
      assert(this->neurons > 0 && "This layer has 0 neurons!");
      NumericMatrix::matmul(input, this->w, this->z, handle);
      NumericMatrix::matbroadcast(this->b, this->b_broadcasted, stream);
      NumericMatrix::matadd_and_activate<T, Activation>(this->z, this->b_broadcasted, this->z, this->a, this->wmega,
                                                        handle, stream);
   }

   void forward(const NumericMatrix::ConstMatrixView<T>& input, tinyAI_blasHandle_t* handle,
                tinyAI_gpuStream_t stream) noexcept override {
      assert(this->neurons > 0 && "This layer has 0 neurons!");
      NumericMatrix::matmul(input, this->w, this->z, handle);
      NumericMatrix::matbroadcast(this->b, this->b_broadcasted, stream);
      NumericMatrix::matadd_and_activate<T, Activation>(this->z, this->b_broadcasted, this->z, this->a, this->wmega,
                                                        handle, stream);
   }
};
} // namespace TINYAI
