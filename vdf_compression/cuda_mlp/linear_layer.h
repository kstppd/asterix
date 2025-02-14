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
template <typename T,ACTIVATION Activation, BACKEND Backend> class LinearLayer {

public:
  size_t neurons = 0;
  T wmega=1.0;
  GENERIC_TS_POOL::MemPool *_pool;
  NumericMatrix::Matrix<T, Backend> buffer, w, w_t, b, b_broadcasted, z,
       a, a_prime, a_t, dw, db, delta, delta_store;
  NumericMatrix::Matrix<T, Backend> m_w, m_b, v_w, v_b, tmp, mw_hat, vw_hat, mb_hat, vb_hat,
      w_decay,dw_copy,db_copy,db_tmp;
  LinearLayer() : _pool(nullptr) {}
  LinearLayer(GENERIC_TS_POOL::MemPool *p) : _pool(p) {}
  LinearLayer(size_t n, GENERIC_TS_POOL::MemPool *p) : neurons(n), _pool(p) {}

  void setup(size_t neurons, size_t input, size_t batchSize,size_t layer_id) {
    assert(neurons > 0 && "This layer has 0 neurons!");
    spdlog::debug("Layer setup [batchsize,inputsize]= [{0:d} x {0:d}]",
                  batchSize, input);
    w = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    w_decay = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    m_w = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    v_w = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    mw_hat = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    vw_hat = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    mb_hat = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    vb_hat = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    b = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    m_b = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    v_b = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    z = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, _pool);
    a = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, _pool);
    b_broadcasted =
        NumericMatrix::Matrix<T, Backend>(batchSize, neurons, _pool);
    a_prime = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, _pool);
    w_t = NumericMatrix::Matrix<T, Backend>(neurons, input, _pool);
    a_t = NumericMatrix::Matrix<T, Backend>(neurons, batchSize, _pool);
    delta = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, _pool);
    delta_store = NumericMatrix::Matrix<T, Backend>(batchSize, neurons, _pool);
    buffer = NumericMatrix::Matrix<T, Backend>(batchSize, input, _pool);
    dw = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    dw_copy = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    tmp = NumericMatrix::Matrix<T, Backend>(input, neurons, _pool);
    db = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    db_copy = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    db_tmp = NumericMatrix::Matrix<T, Backend>(1, neurons, _pool);
    
    const T fan_in=input;
    const T fan_out=neurons;
    T std=std::sqrt(2.0 / (fan_in + fan_out));
    if constexpr (Activation==ACTIVATION::SIN){
      if(layer_id==0){
        wmega=10;
        std=std::sqrt(6.0f / (T)input);      
      }else{
        wmega=1;
        std=std::sqrt(6.0f / ((T)input));      
      }
    }
    
    if constexpr (Backend==BACKEND::DEVICE){
      NumericMatrix::HostMatrix<T>_w(w.nrows(),w.ncols()); 
      NumericMatrix::HostMatrix<T>_b(b.nrows(),b.ncols()); 
      NumericMatrix::export_to_host(w,_w);
      NumericMatrix::export_to_host(b,_b);
      NumericMatrix::mat_randomise(_w,std);
      NumericMatrix::mat_randomise(_b,std);
      NumericMatrix::get_from_host(w,_w);
      NumericMatrix::get_from_host(b,_b);
    }else{
      NumericMatrix::mat_randomise(w,std);
      NumericMatrix::mat_randomise(b,std);
    }
 }

 void reset(size_t input,size_t layer_id){
    const T fan_in=input;
    const T fan_out=neurons;
    T std=std::sqrt(2.0 / (fan_in + fan_out));
    if constexpr (Activation==ACTIVATION::SIN){
      if(layer_id==0){
        wmega=10;
        std=std::sqrt(6.0f / (T)input);      
      }else{
        wmega=1;
        std=std::sqrt(6.0f / ((T)input));      
      }
    }
    
    if constexpr (Backend==BACKEND::DEVICE){
      NumericMatrix::HostMatrix<T>_w(w.nrows(),w.ncols()); 
      NumericMatrix::HostMatrix<T>_b(b.nrows(),b.ncols()); 
      NumericMatrix::export_to_host(w,_w);
      NumericMatrix::export_to_host(b,_b);
      NumericMatrix::mat_randomise(_w,std);
      NumericMatrix::mat_randomise(_b,std);
      NumericMatrix::get_from_host(w,_w);
      NumericMatrix::get_from_host(b,_b);
    }else{
      NumericMatrix::mat_randomise(w,std);
      NumericMatrix::mat_randomise(b,std);
    }
  }
  
  void forward(const NumericMatrix::Matrix<T, Backend> &input,
               tinyAI_blasHandle_t *handle) noexcept {
    assert(neurons > 0 && "This layer has 0 neurons!");
    NumericMatrix::matmul(input, w, z, handle);
    NumericMatrix::matbroadcast(b, b_broadcasted);
    NumericMatrix::matadd_and_activate<T, Activation>(z, b_broadcasted, z, a, wmega, handle);
  }

  void forward(const NumericMatrix::MatrixView<T> &input,
               tinyAI_blasHandle_t *handle) noexcept {
    assert(neurons > 0 && "This layer has 0 neurons!");
    NumericMatrix::matmul(input, w, z, handle);
    NumericMatrix::matbroadcast(b, b_broadcasted);
    NumericMatrix::matadd_and_activate<T, Activation>(z, b_broadcasted, z, a, wmega, handle);
  }
  
  void forward(const NumericMatrix::ConstMatrixView<T> &input,
               tinyAI_blasHandle_t *handle) noexcept {
    assert(neurons > 0 && "This layer has 0 neurons!");
    NumericMatrix::matmul(input, w, z, handle);
    NumericMatrix::matbroadcast(b, b_broadcasted);
    NumericMatrix::matadd_and_activate<T, Activation>(z, b_broadcasted, z, a, wmega, handle);
  }
};
} // namespace TINYAI
