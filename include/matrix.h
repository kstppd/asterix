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
#include "spdlog/spdlog.h"
#include <cassert> //assert
#ifndef SKIP_HOSTBLAS
#include <cblas.h>
#endif
#include <cstring> //std::memcpy
#include <memory>  //allocator
#include <numeric>
#include <random> //allocator
#include <sys/ucontext.h>

#ifdef __NVCC__
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#define __m_WARPSIZE__ 32ul
#endif

#ifdef __HIP__
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <hiprand/hiprand_kernel.h>
#define __m_WARPSIZE__ 64ul
#endif

#define __m_BLOCKSIZE__ 128ul
#include "tiny_arch_macros.h"
#include <array>

#define TINYAI_UNUSED(x) (void)(x)
#define TINYAI_ASSERT assert

#ifdef __NVCC__
/* Define the CUDA error checking macro */
#define CHECK_ERR(err) (cuda_error(err, __FILE__, __LINE__))
static void cuda_error(cudaError_t err, const char* file, int line) {
   if (err != cudaSuccess) {
      std::cerr << "\n\n" << cudaGetErrorString(err) << " in " << file << " at line " << line << "\n";
      abort();
   }
}
#endif
#ifdef __HIP__
/* Define the HIP error checking macro */
#define CHECK_ERR(err) (hip_error(err, __FILE__, __LINE__))
static void hip_error(hipError_t err, const char* file, int line) {
   if (err != hipSuccess) {
      std::cerr << "\n\n" << hipGetErrorString(err) << " in " << file << " at line " << line << "\n";
      abort();
   }
}
#endif

//Some common operations
template<typename T>
struct OP_ELEMENTWISE_MUL {
    __host__ __device__ T operator()(T a, T b) const noexcept{
        return a * b;
    }
};

template<typename T>
struct OP_ELEMENTWISE_ADD {
    __host__ __device__ T operator()(T a, T b) const noexcept{
        return a + b;
    }
};

template<typename T>
struct OP_ELEMENTWISE_SUB {
    __host__ __device__ T operator()(T a, T b) const noexcept{
        return a - b;
    }
};

template<typename T>
struct OP_ELEMENTWISE_DIV {
    __host__ __device__ T operator()(T a, T b) const noexcept{
        return a / b;
    }
};

template<typename T>
struct OP_ELEMENTWISE_SCALE {
    __host__ __device__ T operator()(T a, T factor) const noexcept{
        return factor*a;
    }
};

template<typename T>
struct OP_ELEMENTWISE_SQUARE {
    __host__ __device__ T operator()(T a) const noexcept{
        return a*a;
    }
};

template<typename T>
struct OP_ELEMENTWISE_SQRT {
    __host__ __device__ T operator()(T a) const noexcept{
        return std::sqrt(a);
    }
};


// Used to distringuish residency at compile time
enum class BACKEND { HOST, DEVICE };
enum class ACTIVATION { TANH, RELU, SIN, ELU, NONE };
enum class LOSSF { MSE, LOGCOSH };

namespace NumericMatrix {
template <typename T, BACKEND backend>
class Matrix;

// A non-owning Matrix Container
template <typename T>
struct MatrixView {
   T* _data;
   std::size_t cols;
   std::size_t rows;
   inline constexpr std::size_t id(std::size_t row, std::size_t col) const noexcept {
      TINYAI_ASSERT(row < rows && "Exceeded matrix rows");
      TINYAI_ASSERT(col < cols && "Exceeded matrix cols");
      return row * cols + col;
   }
   inline T& operator()(std::size_t index) noexcept { return _data[index]; }
   inline const T& operator()(std::size_t index) const noexcept { return _data[index]; }
   inline T& operator()(std::size_t row, std::size_t col) noexcept { return _data[id(row, col)]; }
   inline const T& operator()(std::size_t row, std::size_t col) const noexcept { return _data[id(row, col)]; }
   const T* data() const noexcept { return _data; }
   T* data() noexcept { return _data; }
   inline std::size_t nrows() const noexcept { return rows; }
   inline std::size_t ncols() const noexcept { return cols; }
   inline std::size_t size() const noexcept { return cols * rows; }

   T* begin() noexcept { return &_data[0]; }

   T* end() noexcept { return &_data[size()]; }

   void getView(MatrixView<T>& view, size_t row) { view._data = &_data[id(row, 0)]; }

   void copy_row_to(std::size_t row_index, T* dst, BACKEND other_backend) const noexcept {
      const std::size_t len = ncols();
      const T* src = &(this->operator()(row_index, 0));
      if (other_backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(dst, src, len * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
      }
      if (other_backend == BACKEND::HOST) {
         std::memcpy(dst, src, len * sizeof(T));
      }
   }
};

template <typename T>
struct ConstMatrixView {
   const T* _data;
   std::size_t cols;
   std::size_t rows;
   inline constexpr std::size_t id(std::size_t row, std::size_t col) const noexcept {
      TINYAI_ASSERT(row < rows && "Exceeded matrix rows");
      TINYAI_ASSERT(col < cols && "Exceeded matrix cols");
      return row * cols + col;
   }
   inline const T& operator()(std::size_t index) const noexcept { return _data[index]; }
   inline const T& operator()(std::size_t row, std::size_t col) const noexcept { return _data[id(row, col)]; }
   const T* data() const noexcept { return _data; }
   inline std::size_t nrows() const noexcept { return rows; }
   inline std::size_t ncols() const noexcept { return cols; }
   inline std::size_t size() const noexcept { return cols * rows; }

   const T* begin() noexcept { return &_data[0]; }

   const T* end() noexcept { return &_data[size()]; }

   void getConstView(ConstMatrixView<T>& view, size_t row) { view._data = &_data[id(row, 0)]; }

   void copy_row_to(std::size_t row_index, T* dst, BACKEND other_backend) const noexcept {
      const std::size_t len = ncols();
      const T* src = &(this->operator()(row_index, 0));
      if (other_backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(dst, src, len * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
      }
      if (other_backend == BACKEND::HOST) {
         std::memcpy(dst, src, len * sizeof(T));
      }
   }
};

// A simple host only matrix usually used for data loading
// Resources are managed by the Allocator
template <typename T, typename Allocator = std::allocator<T>>
class HostMatrix {
private:
   T* _data;
   std::size_t rows;
   std::size_t cols;
   Allocator _allocator;
   inline constexpr std::size_t id(std::size_t row, std::size_t col) const noexcept {
      TINYAI_ASSERT(row < rows && "Exceeded matrix rows");
      TINYAI_ASSERT(col < cols && "Exceeded matrix cols");
      return row * cols + col;
   }

public:
   explicit HostMatrix() : _data(nullptr), rows(0), cols(0) {}

   explicit HostMatrix(std::size_t nrows, std::size_t ncols) : _data(nullptr), rows(nrows), cols(ncols) {
      static_assert(std::is_trivial<T>::value, "HostMatrix is for trivial types only!");
      const std::size_t len = nrows * ncols;
      _data = _allocator.allocate(len);
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
   }

   HostMatrix<T, Allocator>(const HostMatrix<T, Allocator>& other)
       : _data(nullptr), rows(other.rows), cols(other.cols) {
      const std::size_t len = rows * cols;
      _data = _allocator.allocate(len);
      std::memcpy(_data, other._data, len * sizeof(T));
   }

   template<BACKEND Backend>
   HostMatrix<T, Allocator>(const Matrix<T, Backend>& other)
       : _data(nullptr), rows(other.nrows()), cols(other.ncols()) {
      const std::size_t len = rows * cols;
      _data = _allocator.allocate(len);
      if constexpr(Backend ==BACKEND::HOST){
         std::memcpy(_data, other.data(), len * sizeof(T));
      }
      if constexpr(Backend ==BACKEND::DEVICE){
         tinyAI_gpuMemcpy(_data, other.data(), len * sizeof(T),tinyAI_gpuMemcpyDeviceToHost);
      }
   }

   HostMatrix<T, Allocator>(const MatrixView<T>& other) : _data(nullptr), rows(other.rows), cols(other.cols) {
      const std::size_t len = rows * cols;
      _data = _allocator.allocate(len);
      std::memcpy(_data, other._data, len * sizeof(T));
   }

   HostMatrix<T, Allocator>(HostMatrix<T, Allocator>&& other) noexcept : rows(other.rows), cols(other.cols) {
      _data = other._data;
      other._data = nullptr;
   }

   ~HostMatrix() noexcept {
      if (_data) {
         _allocator.deallocate(_data, cols * rows);
      }
   }

   bool isEmpty() const noexcept { return ((rows == 0) && (cols == 0)); }

   void set_value(std::size_t row, std::size_t col, T val) noexcept {
      std::memcpy(&_data[id(row, col)], &val, sizeof(T));
   }

   T get_value(std::size_t row, std::size_t col) const noexcept {
      T val{};
      std::memcpy(&val, &_data[id(row, col)], sizeof(T));
      return val;
   }

   T* begin() noexcept { return &_data[0]; }

   T* end() noexcept { return &_data[size()]; }

   HostMatrix<T, Allocator>& operator=(HostMatrix<T, Allocator>&& other) noexcept {
      if (this == &other) {
         return *this;
      }
      if (_data) {
         _allocator.deallocate(_data, cols * rows);
      }
      _data = other._data;
      other._data = nullptr;
      cols = other.cols;
      rows = other.rows;
      return *this;
   }

   template <BACKEND backend>
   HostMatrix<T, Allocator>& operator=(const Matrix<T, backend>& other) {

      const size_t other_len = other.ncols() * other.nrows();
      if (other.size() != size()) {
         _allocator.deallocate(_data, size());
         _data = _allocator.allocate(other.size());
         cols = other.ncols();
         rows = other.nrows();
      }
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
      size_t len = size();
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(_data, other.data(), len * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(_data, other.data(), len * sizeof(T));
      }
      return *this;
   }

   inline T& operator()(std::size_t i) noexcept { return _data[i]; }
   inline const T& operator()(std::size_t i) const noexcept { return _data[i]; }
   inline T& operator()(std::size_t row, std::size_t col) noexcept { return _data[id(row, col)]; }
   inline const T& operator()(std::size_t row, std::size_t col) const noexcept { return _data[id(row, col)]; }
   const T* data() const noexcept { return _data; }
   T* data() noexcept { return _data; }
   inline std::size_t nrows() const noexcept { return rows; }
   inline std::size_t ncols() const noexcept { return cols; }
   inline std::size_t size() const noexcept { return cols * rows; }
};

// A flexible matrix that uses a memory pool to store its contents
// Data can be resident in either CPU or GPU memory
// Resources are handled by the pool only
template <typename T, BACKEND backend>
class Matrix {
private:
   T* _data;
   std::size_t rows;
   std::size_t cols;
   GENERIC_TS_POOL::MemPool* _pool;
   inline constexpr std::size_t id(std::size_t row, std::size_t col) const noexcept {
      TINYAI_ASSERT(row < rows && "Exceeded matrix rows");
      TINYAI_ASSERT(col < cols && "Exceeded matrix cols");
      return row * cols + col;
   }

public:
   explicit Matrix() : _data(nullptr), rows(0), cols(0), _pool(nullptr) {}
   explicit Matrix(GENERIC_TS_POOL::MemPool* p) : _data(nullptr), rows(0), cols(0), _pool(p) {}

   explicit Matrix(std::size_t nrows, std::size_t ncols, GENERIC_TS_POOL::MemPool* p)
       : _data(nullptr), rows(nrows), cols(ncols), _pool(p) {
      static_assert(std::is_trivial<T>::value, "Matrix is for trivial types only!");
      const std::size_t len = nrows * ncols;
      _data = _pool->allocate<T>(len);
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
      zero_out();
   }

   Matrix(const Matrix& other) {
      const size_t other_len = other.ncols() * other.nrows();
      _pool = other._pool;
      cols = other.cols;
      rows = other.rows;
      _data = _pool->allocate<T>(cols * rows);
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(_data, other._data, other_len * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(_data, other._data, other_len * sizeof(T));
      }
   }

   Matrix(const HostMatrix<T>& other) {
      const size_t other_len = other.ncols() * other.nrows();
      _pool = other._pool;
      cols = other.cols;
      rows = other.rows;
      _data = _pool->allocate<T>(cols * rows);
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(_data, other._data, other_len * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(_data, other._data, other_len * sizeof(T));
      }
   }

   Matrix(Matrix&& other) {
      const size_t other_len = other.ncols() * other.nrows();
      _pool = other._pool;
      cols = other.cols;
      rows = other.rows;
      _data = other._data;
      other._data = nullptr;
   }

   void set_value(std::size_t row, std::size_t col, T val) noexcept {
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(&_data[id(row, col)], &val, sizeof(T), tinyAI_gpuMemcpyDefault));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(&_data[id(row, col)], &val, sizeof(T));
      }
   }

   T get_value(std::size_t row, std::size_t col) const noexcept {
      T val{};
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(&val, &_data[id(row, col)], sizeof(T), tinyAI_gpuMemcpyDefault));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(&val, &_data[id(row, col)], sizeof(T));
      }
      return val;
   }

   void copy_to_host_from_host_view(const MatrixView<T>& view) {
      TINYAI_ASSERT(size() == view.size());
      std::memcpy(_data, view._data, sizeof(T) * size());
   }

   void copy_to_host_from_host_view(const ConstMatrixView<T>& view) {
      TINYAI_ASSERT(size() == view.size());
      std::memcpy(_data, view._data, sizeof(T) * size());
   }

   void copy_to_device_from_host_view(const MatrixView<T>& view) {
      TINYAI_ASSERT(size() == view.size());
      tinyAI_gpuMemcpy(_data, view._data, sizeof(T) * size(), tinyAI_gpuMemcpyHostToDevice);
   }

   void copy_to_device_from_host_view(const ConstMatrixView<T>& view) {
      TINYAI_ASSERT(size() == view.size());
      tinyAI_gpuMemcpy(_data, view._data, sizeof(T) * size(), tinyAI_gpuMemcpyHostToDevice);
   }

   Matrix& operator=(const Matrix& other) {
      if (this == &other) {
         return *this;
      }

      // Same shape?
      if (cols == other.cols && rows == other.rows) {
         if constexpr (backend == BACKEND::DEVICE) {
            CHECK_ERR(tinyAI_gpuMemcpy(_data, other._data, other.rows * other.cols * sizeof(T),
                                       tinyAI_gpuMemcpyDeviceToDevice));
         }
         if constexpr (backend == BACKEND::HOST) {
            std::memcpy(_data, other._data, other.rows * other.cols * sizeof(T));
         }
         return *this;
      }

      //This can occur so we just piggyback on the same pool
      if (_pool==nullptr){
         _pool=other._pool;
      }
      const size_t other_len = other.ncols() * other.nrows();
      _pool->deallocate(_data);
      cols = other.cols;
      rows = other.rows;
      _data = _pool->allocate<T>(cols * rows);
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(_data, other._data, other_len * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(_data, other._data, other_len * sizeof(T));
      }
      return *this;
   };

   Matrix<T, backend>& operator=(Matrix<T, backend>&& other) noexcept {
      if (this == &other) {
         return *this;
      }
      if (_data) {
         _pool->deallocate(_data);
      }
      _data = other._data;
      other._data = nullptr;
      cols = other.cols;
      rows = other.rows;
      _pool = other._pool;
      return *this;
   }

   Matrix<T, backend>& operator=(const HostMatrix<T>& other) {

      const size_t other_len = other.ncols() * other.nrows();

      if (_data) {
         _pool->deallocate(_data);
      }
      _data = _pool->allocate<T>(other_len);
      cols = other.ncols();
      rows = other.nrows();
      if (_data == nullptr) {
         throw std::bad_alloc();
      }
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(_data, other.data(), other_len * sizeof(T), tinyAI_gpuMemcpyHostToDevice));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(_data, other.data(), other_len * sizeof(T));
      }
      return *this;
   }

   ~Matrix() noexcept { _pool->deallocate(_data); }

   // Member Functions
   void zero_out() {
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemset(_data, 0, size() * sizeof(T)));
      }
      if constexpr (backend == BACKEND::HOST) {
         memset(_data, 0, size() * sizeof(T));
      }
   }
   void getView(MatrixView<T>& view, size_t row) { view._data = &_data[id(row, 0)]; }
   void getView(ConstMatrixView<T>& view, size_t row) const { view._data = &_data[id(row, 0)]; }
   void getFullView(MatrixView<T>& view) {
      view._data = _data;
      view.rows = nrows();
      view.cols = ncols();
   }
   void getFullView(ConstMatrixView<T>& view) const {
      view._data = _data;
      view.rows = nrows();
      view.cols = ncols();
   }
   inline T& operator()(std::size_t row, std::size_t col) noexcept { return _data[id(row, col)]; }
   inline const T& operator()(std::size_t row, std::size_t col) const noexcept { return _data[id(row, col)]; }
   inline T& operator()(std::size_t index) noexcept { return _data[index]; }
   inline const T& operator()(std::size_t index) const noexcept { return _data[index]; }

   void export_to_host(HostMatrix<T>& out) const noexcept {
      TINYAI_ASSERT(ncols() == out.ncols());
      TINYAI_ASSERT(nrows() == out.nrows());
      if constexpr (backend == BACKEND::DEVICE) {
         CHECK_ERR(tinyAI_gpuMemcpy(out.data(), _data, size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
      }
      if constexpr (backend == BACKEND::HOST) {
         std::memcpy(out.data(), _data, size() * sizeof(T));
      }
   }

   const T* data() const noexcept { return _data; }
   T* data() noexcept { return _data; }
   inline std::size_t nrows() const noexcept { return rows; }
   inline std::size_t ncols() const noexcept { return cols; }
   inline std::size_t size() const noexcept { return cols * rows; }
   inline void print_shape() const noexcept { std::cout << "[" << nrows() << " x " << ncols() << "]\n"; }
};

template <typename T>
inline void mat_pointwise_mul(Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                              tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < C.size(); ++i) {
      C(i) = A(i) * B(i);
   }
}

template <typename T>
inline void mat_pointwise_sqrt(Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.size() == B.size());
   for (size_t i = 0; i < A.size(); ++i) {
      B(i) = std::sqrt(A(i));
   }
}

template <typename T>
inline void mat_pointwise_div(Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                              tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < C.size(); ++i) {
      C(i) = A(i) / B(i);
   }
}

template <typename T>
inline void matmul(const Matrix<T, BACKEND::HOST>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   constexpr T alpha = 1.0;
   constexpr T beta = 0.0;
   #ifndef SKIP_HOSTBLAS
   if constexpr (sizeof(T) == sizeof(float)) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   } else {
      TINYAI_ASSERT(B.data());
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   }
   #endif
}

template <typename T>
inline void matmul(const MatrixView<T>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   constexpr T alpha = 1.0;
   constexpr T beta = 0.0;
   #ifndef SKIP_HOSTBLAS
   if constexpr (sizeof(T) == sizeof(float)) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   } else {
      TINYAI_ASSERT(B.data());
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   }
   #endif
}

template <typename T>
inline void matmul(const Matrix<T, BACKEND::HOST>& A, const MatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   constexpr T alpha = 1.0;
   constexpr T beta = 0.0;
   #ifndef SKIP_HOSTBLAS
   if constexpr (sizeof(T) == sizeof(float)) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   } else {
      TINYAI_ASSERT(B.data());
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   }
   #endif
}

template <typename T>
inline void matmul(const ConstMatrixView<T>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   constexpr T alpha = 1.0;
   constexpr T beta = 0.0;
   #ifndef SKIP_HOSTBLAS
   if constexpr (sizeof(T) == sizeof(float)) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   } else {
      TINYAI_ASSERT(B.data());
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   }
   #endif
}

template <typename T>
inline void matmul(const Matrix<T, BACKEND::HOST>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   constexpr T alpha = 1.0;
   constexpr T beta = 0.0;
   #ifndef SKIP_HOSTBLAS
   if constexpr (sizeof(T) == sizeof(float)) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   } else {
      TINYAI_ASSERT(B.data());
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, B.ncols(), A.nrows(), A.ncols(), alpha, B.data(),
                  B.ncols(), A.data(), A.ncols(), beta, C.data(), C.ncols());
   }
   #endif
}

template <typename T>
inline void transpose_into(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& C, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(j, i) = A(i, j);
      }
   }
}

template <typename T>
inline void transpose_into(const MatrixView<T>& A, Matrix<T, BACKEND::HOST>& C, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(j, i) = A(i, j);
      }
   }
}

template <typename T>
inline void transpose_into(const ConstMatrixView<T>& A, Matrix<T, BACKEND::HOST>& C, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(j, i) = A(i, j);
      }
   }
}

template <typename T>
inline void matadd(const Matrix<T, BACKEND::HOST>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_UNUSED(cublasHandle);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) + B(i, j);
      }
   }
}

template <typename T>
inline void matadd_scalar(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, T scalar, void* cublasHandle,
                          tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_UNUSED(cublasHandle);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         B(i, j) = A(i, j) + scalar;
      }
   }
}

template <typename T>
inline T matreduce_add(const Matrix<T, BACKEND::HOST>& A, void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   T sum = T(0.0);
   for (size_t i = 0; i < A.size(); i++) {
      sum += A(i);
   }
   return sum;
}

template <typename T>
inline T matreduce_sum(const Matrix<T, BACKEND::HOST>& A, void* cublasHandle) {
   TINYAI_UNUSED(cublasHandle);
   T sum = T(0.0);
   for (size_t i = 0; i < A.size(); i++) {
      sum += A(i);
   }
   return sum;
}

template <typename T>
inline void matscale(Matrix<T, BACKEND::HOST>& A, T factor, void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < A.size(); i++) {
      A(i) *= factor;
   }
}

template <typename T>
inline void matscale_to(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, T factor, void* cublasHandle,
                        tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_UNUSED(cublasHandle);
   TINYAI_ASSERT(A.size() == B.size() && "Cmooon'");
   for (size_t i = 0; i < A.size(); i++) {
      B(i) = A(i) * factor;
   }
}

template <typename T>
inline void matsub(const Matrix<T, BACKEND::HOST>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_UNUSED(cublasHandle);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) - B(i, j);
      }
   }
}

template <typename T, LOSSF LossF>
inline void loss_derivative(const Matrix<T, BACKEND::HOST>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                            tinyAI_blasHandle_t* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_UNUSED(cublasHandle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         if constexpr (LossF == LOSSF::LOGCOSH) {
            C(i, j) = std::tanh(A(i, j) - B(i, j));
         }
         if constexpr (LossF == LOSSF::MSE) {
            C(i, j) = A(i, j) - B(i, j);
         }
      }
   }
}

template <typename T, LOSSF LossF>
inline void loss(const Matrix<T, BACKEND::HOST>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                 tinyAI_blasHandle_t* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         if constexpr (LossF == LOSSF::LOGCOSH) {
            auto abs_diff = std::abs(A(i, j) - B(i, j));
            C(i, j) = (abs_diff + std::log1p(std::exp(-2 * abs_diff)) - M_LN2);
         }
         if constexpr (LossF == LOSSF::MSE) {
            auto diff = (A(i, j) - B(i, j));
            C(i, j) = diff * diff;
         }
      }
   }
}

template <typename T>
inline void matsub(const MatrixView<T>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) - B(i, j);
      }
   }
}

template <typename T>
inline void matsub(const Matrix<T, BACKEND::HOST>& A, const MatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) - B(i, j);
      }
   }
}

template <typename T>
inline void matsub(const ConstMatrixView<T>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) - B(i, j);
      }
   }
}

template <typename T>
inline void matsub(const Matrix<T, BACKEND::HOST>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                   void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) - B(i, j);
      }
   }
}

template <typename T>
inline void matsub_error_mse(const Matrix<T, BACKEND::HOST>& A, const Matrix<T, BACKEND::HOST>& B,
                             Matrix<T, BACKEND::HOST>& C, void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         T tmp = A(i, j) - B(i, j);
         C(i, j) = tmp * tmp;
      }
   }
}

template <typename T>
inline void matsub_error_mse(const MatrixView<T>& A, const Matrix<T, BACKEND::HOST>& B, Matrix<T, BACKEND::HOST>& C,
                             void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         T tmp = A(i, j) - B(i, j);
         C(i, j) = tmp * tmp;
      }
   }
}

template <typename T>
inline void matsub_error_mse(const Matrix<T, BACKEND::HOST>& A, const MatrixView<T>& B, Matrix<T, BACKEND::HOST>& C,
                             void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         T tmp = A(i, j) - B(i, j);
         C(i, j) = tmp * tmp;
      }
   }
}

template <typename T>
inline void matsub_error_mse(const ConstMatrixView<T>& A, const Matrix<T, BACKEND::HOST>& B,
                             Matrix<T, BACKEND::HOST>& C, void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         T tmp = A(i, j) - B(i, j);
         C(i, j) = tmp * tmp;
      }
   }
}

template <typename T>
inline void matsub_error_mse(const Matrix<T, BACKEND::HOST>& A, const ConstMatrixView<T>& B,
                             Matrix<T, BACKEND::HOST>& C, void* cublasHandle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         T tmp = A(i, j) - B(i, j);
         C(i, j) = tmp * tmp;
      }
   }
}

template <typename T>
inline void matsum_rows(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(B.ncols() == A.ncols() && B.nrows() == 1 &&
                 "Result matrix must have the same number of columns as A and exactly "
                 "1 row.");
   for (size_t i = 0; i < A.ncols(); ++i) {
      T sum = 0;
      for (size_t j = 0; j < A.nrows(); ++j) {
         sum += A(j, i); // Summing elements column-wise for each row
      }
      B(0, i) = sum; // Store the sum in B
   }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const NumericMatrix::Matrix<T, BACKEND::HOST>& A) {
   printf("\tRows= %zu, Cols=%zu\n", A.nrows(), A.ncols());
   for (size_t i = 0; i < A.nrows(); ++i) {
      for (size_t j = 0; j < A.ncols(); ++j) {
         std::cout << A(i, j) << ",";
      }
      std::cout << std::endl;
   }
   return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const NumericMatrix::Matrix<T, BACKEND::DEVICE>& A) {
   printf("\tRows= %zu, Cols=%zu\n", A.nrows(), A.ncols());
   NumericMatrix::HostMatrix<T> hostA(A.nrows(), A.ncols());
   CHECK_ERR(tinyAI_gpuMemcpy(hostA.data(), A.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
   for (size_t i = 0; i < hostA.nrows(); ++i) {
      for (size_t j = 0; j < hostA.ncols(); ++j) {
         std::cout << hostA(i, j) << ",";
      }
      std::cout << std::endl;
   }
   return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const MatrixView<T>& A) {
   printf("\tRows= %zu, Cols=%zu\n", A.nrows(), A.ncols());
   for (size_t i = 0; i < A.nrows(); ++i) {
      for (size_t j = 0; j < A.ncols(); ++j) {
         std::cout << A(i, j) << ",";
      }
      std::cout << std::endl;
   }
   return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const ConstMatrixView<T>& A) {
   printf("\tRows= %zu, Cols=%zu\n", A.nrows(), A.ncols());
   for (size_t i = 0; i < A.nrows(); ++i) {
      for (size_t j = 0; j < A.ncols(); ++j) {
         std::cout << A(i, j) << ",";
      }
      std::cout << std::endl;
   }
   return os;
}

template <typename T, ACTIVATION Activation>
__host__ __device__ T activate(T val, T w) {
   if constexpr (Activation == ACTIVATION::TANH) {
      return std::tanh(val);
   }
   if constexpr (Activation == ACTIVATION::SIN) {
      return std::sin(w * val);
   }
   if constexpr (Activation == ACTIVATION::RELU) {
      return (val > 0) ? val : 0.01 * val;
   }
   if constexpr (Activation == ACTIVATION::ELU) {
      return (val >= 0) ? val : 1.0 * (std::exp(val) - 1);
   }
   if constexpr (Activation == ACTIVATION::NONE) {
      return val;
   }
}

template <typename T, ACTIVATION Activation>
__host__ __device__ T activate_prime(T val, T w) {
   if constexpr (Activation == ACTIVATION::TANH) {
      T tanh_val = std::tanh(val);
      return 1. - tanh_val * tanh_val;
   }
   if constexpr (Activation == ACTIVATION::SIN) {
      return w * std::cos(w * val);
   }
   if constexpr (Activation == ACTIVATION::RELU) {
      return (val > 0) ? static_cast<T>(1) : 0.01;
   }
   if constexpr (Activation == ACTIVATION::ELU) {
      return (val >= 0) ? 1.0 : 1.0 * std::exp(val);
   }
   if constexpr (Activation == ACTIVATION::NONE) {
      return T(1.0);
   }
}

template <typename T, ACTIVATION Activation>
inline void mat_pointwise_activate(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, T w,
                                   tinyAI_gpuStream_t stream) {
   for (size_t i = 0; i < A.size(); ++i) {
      B(i) = activate<T, Activation>(A(i), w);
   }
}

template <typename T, ACTIVATION Activation>
inline void mat_pointwise_activate_prime(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, T w,
                                         tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && "Matrix A and B shouuld have the same shape!");
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < A.size(); ++i) {
      B(i) = activate_prime<T, Activation>(A(i), w);
   }
}

template <typename T>
inline void mat_randomise(Matrix<T, BACKEND::HOST>& A, T stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::normal_distribution<T> dist(0, stddev); 
    for (size_t i = 0; i < A.size(); ++i) {
        A(i) = dist(gen);
    }
}

template <typename T>
inline void mat_randomise(HostMatrix<T>& A, T stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::normal_distribution<T> dist(0, stddev); 
    for (size_t i = 0; i < A.size(); ++i) {
        A(i) = dist(gen);
    }
}

template <typename T>
inline void matbroadcast(const Matrix<T, BACKEND::HOST>& A, Matrix<T, BACKEND::HOST>& B, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < B.nrows(); i++) {
      for (size_t j = 0; j < B.ncols(); j++) {
         B(i, j) = A(0, j);
      }
   }
}

template <typename T>
inline void adamw(Matrix<T, BACKEND::HOST>& w, Matrix<T, BACKEND::HOST>& mw, Matrix<T, BACKEND::HOST>& vw,
                  Matrix<T, BACKEND::HOST>& dw, T m_hat_scale, T v_hat_scale, T beta_1, T beta_2, T weight_decay, T lr,
                  T epsilon, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   for (size_t i = 0; i < w.size(); i++) {
      T dw_sq_val = dw(i) * dw(i);
      mw(i) = beta_1 * mw(i) + (T(1.0) - beta_1) * dw(i);
      vw(i) = beta_2 * vw(i) + (T(1.0) - beta_2) * dw_sq_val;
      T mw_hat = mw(i) * m_hat_scale;
      T vw_hat = vw(i) * v_hat_scale;
      w(i) -= lr * (mw_hat / (std::sqrt(vw_hat) + epsilon) + weight_decay * w(i));
   }
}

template <typename T, ACTIVATION Activation>
inline void matadd_and_activate(const Matrix<T, BACKEND::HOST>& A, const Matrix<T, BACKEND::HOST>& B,
                                Matrix<T, BACKEND::HOST>& C, Matrix<T, BACKEND::HOST>& D, T w, void* cublasHandle,
                                tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(cublasHandle);
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.ncols() == B.ncols() && A.nrows() == B.nrows());
   for (size_t i = 0; i < A.nrows(); i++) {
      for (size_t j = 0; j < A.ncols(); j++) {
         C(i, j) = A(i, j) + B(i, j);
         D(i, j) = activate<T, Activation>(C(i, j), w);
      }
   }
}

template <typename T, typename FF>
void apply(const T *const start_a, const T *const start_b, T*const start_c, std::size_t len, FF&& f) {
   for (std::size_t i=0; i<len;++i){
      start_c[i]=f(start_a[i],start_b[i]);
   }
}

template <typename T,typename FF>
inline void matapply_to(const Matrix<T, BACKEND::HOST>& A,
                        const Matrix<T, BACKEND::HOST>& B,
                        Matrix<T, BACKEND::HOST>& C,
                        FF&& f,
                        tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(stream);
   TINYAI_ASSERT(A.size() == B.size() && A.size() == C.size());
   apply<T>(A.data(), B.data(),C.data(),A.size(),f);
}


//~ BACKEND::HOST Functionality

template <typename T, typename FF>
__global__ void apply_kernel(const T *const start_a, const T *const start_b, T*const start_c, std::size_t len, FF&& f) {
   const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid<len){
      start_c[tid]=f(start_a[tid],start_b[tid]);
   }
}

template <typename T>
inline void matmul(const Matrix<T, BACKEND::DEVICE>& A, const Matrix<T, BACKEND::DEVICE>& B,
                   Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle) {
   constexpr T alpha = 1.0f;
   constexpr T beta = 0.0f;
   if constexpr (sizeof(T) == sizeof(float)) {
      tinyAI_blasStatus_t status =
          tinyAI_blasSgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   } else {
      tinyAI_blasStatus_t status =
          tinyAI_blasDgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   }
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T>
inline void matmul(const Matrix<T, BACKEND::DEVICE>& A, const MatrixView<T>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle) {
   constexpr T alpha = 1.0f;
   constexpr T beta = 0.0f;
   if constexpr (sizeof(T) == sizeof(float)) {
      tinyAI_blasStatus_t status =
          tinyAI_blasSgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   } else {
      tinyAI_blasStatus_t status =
          tinyAI_blasDgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   }
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T>
inline void matmul(const MatrixView<T>& A, const Matrix<T, BACKEND::DEVICE>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle) {
   constexpr T alpha = 1.0f;
   constexpr T beta = 0.0f;
   if constexpr (sizeof(T) == sizeof(float)) {
      tinyAI_blasStatus_t status =
          tinyAI_blasSgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   } else {
      tinyAI_blasStatus_t status =
          tinyAI_blasDgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());

      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   }
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T>
inline void matmul(const Matrix<T, BACKEND::DEVICE>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle) {
   constexpr T alpha = 1.0f;
   constexpr T beta = 0.0f;
   if constexpr (sizeof(T) == sizeof(float)) {
      tinyAI_blasStatus_t status =
          tinyAI_blasSgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   } else {
      tinyAI_blasStatus_t status =
          tinyAI_blasDgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   }
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T>
inline void matmul(const ConstMatrixView<T>& A, const Matrix<T, BACKEND::DEVICE>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle) {
   constexpr T alpha = 1.0f;
   constexpr T beta = 0.0f;
   if constexpr (sizeof(T) == sizeof(float)) {
      tinyAI_blasStatus_t status =
          tinyAI_blasSgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());
      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   } else {
      tinyAI_blasStatus_t status =
          tinyAI_blasDgemm(*handle, tinyAI_blas_OP_N, tinyAI_blas_OP_N, B.ncols(), A.nrows(), A.ncols(), &alpha,
                           B.data(), B.ncols(), A.data(), A.ncols(), &beta, C.data(), C.ncols());

      TINYAI_ASSERT(status == BLAS_SUCCESS && "Cublas matmul failed");
   }
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T>
__global__ void transpose_matrix_kernel(const T* A, T* B, size_t Arows, size_t Acols) {
   const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t row = tid / Acols;
   const size_t col = tid % Acols;
   const std::size_t target = col * Arows + row; // hacky
   B[target] = A[tid];
}

template <typename T>
inline void transpose_into(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& C,
                           tinyAI_gpuStream_t stream) {

   TINYAI_ASSERT(A.size() == C.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   transpose_matrix_kernel<<<blocks, threads, 0, stream>>>(A.data(), C.data(), A.nrows(), A.ncols());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
   spdlog::debug("Transpose matrix kernel [blocks,threads]= [{0:d} x {1:d} for "
                 "matrix size {2:d} ]",
                 blocks, threads, A.size());
}

template <typename T>
inline void transpose_into(const MatrixView<T>& A, Matrix<T, BACKEND::DEVICE>& C, tinyAI_gpuStream_t stream) {

   TINYAI_ASSERT(A.size() == C.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   transpose_matrix_kernel<<<blocks, threads, 0, stream>>>(A.data(), C.data(), A.nrows(), A.ncols());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
   spdlog::debug("Transpose matrix kernel [blocks,threads]= [{0:d} x {1:d} for "
                 "matrix size {2:d} ]",
                 blocks, threads, A.size());
}

template <typename T>
inline void transpose_into(const ConstMatrixView<T>& A, Matrix<T, BACKEND::DEVICE>& C, tinyAI_gpuStream_t stream) {

   TINYAI_ASSERT(A.size() == C.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   transpose_matrix_kernel<<<blocks, threads, 0, stream>>>(A.data(), C.data(), A.nrows(), A.ncols());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
   spdlog::debug("Transpose matrix kernel [blocks,threads]= [{0:d} x {1:d} for "
                 "matrix size {2:d} ]",
                 blocks, threads, A.size());
}

template <typename T>
__global__ void matadd(const T* A, const T* B, T* C, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      C[tid] = A[tid] + B[tid];
   }
}

template <typename T>
__global__ void matadd_scalar(const T* A, T* B, T scalar, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      B[tid] = A[tid] + scalar;
   }
}

template <typename T>
__global__ void matscale(T* A, T factor, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      A[tid] *= factor;
   }
}

template <typename T>
__global__ void matscale_to(const T* A, T* B, T factor, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      B[tid] = A[tid] * factor;
   }
}

template <typename T>
__global__ void matsub(const T* A, const T* B, T* C, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      C[tid] = A[tid] - B[tid];
   }
}

template <typename T>
__global__ void matsub_error_mse(const T* A, const T* B, T* C, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      T tmp = A[tid] - B[tid];
      C[tid] = tmp * tmp;
   }
}

template <typename T, LOSSF LossF>
__global__ void loss(const T* A, const T* B, T* C, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      if constexpr (LossF == LOSSF::LOGCOSH) {
         T diff = A[tid] - B[tid];
         T abs_diff = std::abs(diff);
         C[tid] = (abs_diff + std::log1p(std::exp(-2 * abs_diff)) - M_LN2);
      }
      if constexpr (LossF == LOSSF::MSE) {
         T tmp = A[tid] - B[tid];
         C[tid] = tmp * tmp;
      }
   }
}

template <typename T, LOSSF LossF>
__global__ void loss_derivative(const T* A, const T* B, T* C, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      if constexpr (LossF == LOSSF::LOGCOSH) {
         C[tid] = std::tanh(A[tid] - B[tid]);
      }
      if constexpr (LossF == LOSSF::MSE) {
         C[tid] = A[tid] - B[tid];
      }
   }
}

template <typename T>
__global__ void matreduce_add(const T* A, size_t len, T* output) {
   *output = T(0.0);
   for (size_t i = 0; i < len; i++) {
      *output += A[i] * A[i];
   }
}

template <typename T>
inline void matadd(const Matrix<T, BACKEND::DEVICE>& A, const Matrix<T, BACKEND::DEVICE>& B,
                   Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matadd<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matadd kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matadd_scalar(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B, T scalar,
                          tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matadd_scalar<<<blocks, threads, 0, stream>>>(A.data(), B.data(), scalar, A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matadd kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub(const Matrix<T, BACKEND::DEVICE>& A, const Matrix<T, BACKEND::DEVICE>& B,
                   Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub(const MatrixView<T>& A, const Matrix<T, BACKEND::DEVICE>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub(const Matrix<T, BACKEND::DEVICE>& A, const MatrixView<T>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub(const ConstMatrixView<T>& A, const Matrix<T, BACKEND::DEVICE>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub(const Matrix<T, BACKEND::DEVICE>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::DEVICE>& C,
                   tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T, LOSSF LossF>
inline void loss_derivative(const Matrix<T, BACKEND::DEVICE>& A, const ConstMatrixView<T>& B,
                            Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   loss_derivative<T, LossF><<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Loss derivative kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads,
                 A.size());
}

template <typename T, LOSSF LossF>
inline void loss(const Matrix<T, BACKEND::DEVICE>& A, const ConstMatrixView<T>& B, Matrix<T, BACKEND::DEVICE>& C,
                 tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   loss<T, LossF><<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Loss derivative kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads,
                 A.size());
}

template <typename T>
inline void matsub_error_mse(const Matrix<T, BACKEND::DEVICE>& A, const Matrix<T, BACKEND::DEVICE>& B,
                             Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub_error_mse<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub_error_mse(const MatrixView<T>& A, const Matrix<T, BACKEND::DEVICE>& B, Matrix<T, BACKEND::DEVICE>& C,
                             tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub_error_mse<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub_error_mse(const Matrix<T, BACKEND::DEVICE>& A, const MatrixView<T>& B, Matrix<T, BACKEND::DEVICE>& C,
                             tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub_error_mse<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub_error_mse(const ConstMatrixView<T>& A, const Matrix<T, BACKEND::DEVICE>& B,
                             Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub_error_mse<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matsub_error_mse(const Matrix<T, BACKEND::DEVICE>& A, const ConstMatrixView<T>& B,
                             Matrix<T, BACKEND::DEVICE>& C, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matsub_error_mse<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matsub kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T, typename U>
__device__ __forceinline__ T s_shuffle_down(T variable, unsigned int delta, U mask = 0) noexcept {
#ifdef __NVCC__
   return __shfl_down_sync(mask, variable, delta);
#endif
#ifdef __HIP__
   return __shfl_down(variable, delta);
#endif
}

template <typename T>
__device__ T warp_reduce_sum(T val) {
   for (int offset = __m_WARPSIZE__ / 2; offset > 0; offset /= 2) {
      val += s_shuffle_down(val, offset, -1);
   }
   return val;
}

inline constexpr std::array<std::size_t, 2> launch_params(std::size_t arraySize, std::size_t blockSize) {
   std::size_t gridSize = (arraySize + blockSize - 1) / blockSize;
   return {gridSize, blockSize};
}

template <typename T>
__global__ void reduce_sum_kernel(const T* data, T* block_sums, std::size_t len) {
   __shared__ T shared_sum[__m_WARPSIZE__];
   std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   std::size_t lane = threadIdx.x % __m_WARPSIZE__;
   std::size_t warp_id = threadIdx.x / __m_WARPSIZE__;

   if (warp_id * __m_WARPSIZE__ + __m_WARPSIZE__ > len) {
      return;
   }

   if (warp_id == 0) {
      shared_sum[lane] = 0;
   }
   __syncthreads();

   T cand = T(0);
   if (tid < len) {
      cand = data[tid];
   }

   T sum = warp_reduce_sum(cand);

   if (lane == 0) {
      shared_sum[warp_id] = sum;
   }
   __syncthreads();

   if (warp_id == 0) {
      sum = shared_sum[lane];
      sum = warp_reduce_sum(sum);
   }

   if (threadIdx.x == 0) {
      block_sums[blockIdx.x] = sum;
   }
}

template <typename T>
inline T matreduce_add_gpu(const Matrix<T, BACKEND::DEVICE>& A, GENERIC_TS_POOL::MemPool* _pool,
                           tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   const std::size_t len = A.size();
   const auto lp = launch_params(len, 1024ul);
   const std::size_t nblocks = lp[0];
   T* d_block_sums = _pool->allocate<T>(nblocks);
   // std::cout<<A.size()<<" "<<nblocks<<" "<<lp[1]<<std::endl;
   reduce_sum_kernel<<<nblocks, lp[1], 0, stream>>>(A.data(), d_block_sums, len);
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
   std::vector<T> h_block_sums(nblocks);
   CHECK_ERR(tinyAI_gpuMemcpyAsync(h_block_sums.data(), d_block_sums, nblocks * sizeof(T), tinyAI_gpuMemcpyDeviceToHost,
                                   stream));
   tinyAI_gpuStreamSynchronize(stream);
   const T total_sum = std::accumulate(h_block_sums.cbegin(), h_block_sums.cend(), T(0.0));
   _pool->deallocate(d_block_sums);
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
   spdlog::debug("Matsub reduce add kernel");
   return total_sum;
}

template <typename T>
inline void matscale(Matrix<T, BACKEND::DEVICE>& A, T factor, tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matscale<<<blocks, threads, 0, stream>>>(A.data(), factor, A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Scale kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void matscale_to(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B, T factor,
                        tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size());
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matscale_to<<<blocks, threads, 0, stream>>>(A.data(), B.data(), factor, A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Scale kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T, ACTIVATION Activation>
__global__ void pointwise_activate(const T* A, T* B, const size_t len, T w) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   B[tid] = activate<T, Activation>(A[tid], w);
}

template <typename T, ACTIVATION Activation>
__global__ void pointwise_activate_prime(const T* A, T* B, const size_t len, T w) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      B[tid] = activate_prime<T, Activation>(A[tid], w);
   }
}

template <typename T>
__global__ void pointwise_mul(const T* A, const T* B, T* C, const size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      C[tid] = A[tid] * B[tid];
   }
}

template <typename T, typename F>
__global__ void pointwise_F(const T* A, const T* B, T* C, const size_t len, F f) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      C[tid] = f(A[tid], B[tid]);
   }
}

template <typename T, typename F>
__global__ void pointwise_F_inplace(T* A, const size_t len, F f) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      f(A[tid]);
   }
}

template <typename T>
__global__ void pointwise_sqrt(const T* A, T* B, const size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      B[tid] = std::sqrt(A[tid]);
   }
}

template <typename T>
__global__ void pointwise_div(const T* A, const T* B, T* C, const size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      C[tid] = A[tid] / B[tid];
   }
}

template <typename T>
__global__ void broadcast(const T* A, T* B, size_t Brows, size_t Bcols) {
   const size_t len = Brows * Bcols;
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      const size_t row = tid / Bcols;
      const size_t col = tid % Bcols;
      B[row * Bcols + col] = A[col];
   }
}

template <typename T>
__global__ void matsum_rows(const T* A, T* B, size_t Arows, size_t Acols) {
   const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
   if (col < Acols) {
      T col_sum = 0;

      for (size_t row = 0; row < Arows; ++row) {
         col_sum += A[row * Acols + col];
      }

      B[col] = col_sum;
   }
}

template <typename T>
inline void matbroadcast(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B,
                         tinyAI_gpuStream_t stream) {
   const size_t threads = std::min(__m_BLOCKSIZE__, B.size());
   const size_t blocks = B.size() / __m_BLOCKSIZE__ + (B.size() % __m_BLOCKSIZE__ != 0);
   broadcast<<<blocks, threads, 0, stream>>>(A.data(), B.data(), B.nrows(), B.ncols());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Broadcast kernel [blocks,threads]= [{0:d} x {1:d} for matrix "
                 "size {2:d} ]",
                 blocks, threads, B.size());
}

template <typename T>
inline void matsum_rows(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B, tinyAI_gpuStream_t stream) {
   const size_t threads = std::min(__m_BLOCKSIZE__, B.size());
   const size_t blocks = B.size() / __m_BLOCKSIZE__ + (B.size() % __m_BLOCKSIZE__ != 0);
   matsum_rows<<<blocks, threads, 0, stream>>>(A.data(), B.data(), A.nrows(), A.ncols());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Broadcast kernel [blocks,threads]= [{0:d} x {1:d} for matrix "
                 "size {2:d} ]",
                 blocks, threads, B.size());
}

template <typename T, ACTIVATION Activation>
inline void mat_pointwise_activate(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B, T w,
                                   tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   pointwise_activate<T, Activation><<<blocks, threads, 0, stream>>>(A.data(), B.data(), A.size(), w);
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Activation kernel [blocks,threads]= [{0:d} x {1:d} for matrix "
                 "size {2:d} ]",
                 blocks, threads, A.size());
}

template <typename T, ACTIVATION Activation>
inline void mat_pointwise_activate_prime(const Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B, T w,
                                         tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   pointwise_activate_prime<T, Activation><<<blocks, threads, 0, stream>>>(A.data(), B.data(), A.size(), w);
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Activation prime kernel [blocks,threads]= [{0:d} x {1:d} for "
                 "matrix size {2:d} ]",
                 blocks, threads, A.size());
}

template <typename T>
inline void mat_pointwise_mul(Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B,
                              Matrix<T, BACKEND::DEVICE>& C, tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   pointwise_mul<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), C.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Pointwise mul [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void mat_pointwise_div(Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B,
                              Matrix<T, BACKEND::DEVICE>& C, tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   pointwise_div<<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), C.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Pointwise div [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void mat_pointwise_sqrt(Matrix<T, BACKEND::DEVICE>& A, Matrix<T, BACKEND::DEVICE>& B,
                               tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   pointwise_sqrt<<<blocks, threads, 0, stream>>>(A.data(), B.data(), A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Pointwise mul [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T>
inline void export_to_host(const Matrix<T, BACKEND::DEVICE>& A, HostMatrix<T>& B) {
   CHECK_ERR(tinyAI_gpuMemcpy(B.data(), A.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
}

template <typename T, BACKEND Backend>
inline void export_to_host_view(const Matrix<T, Backend>& A, MatrixView<T>& B) {
   if constexpr (Backend == BACKEND::DEVICE) {
      CHECK_ERR(tinyAI_gpuMemcpy(B.data(), A.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
   } else {
      std::memcpy(B.data(), A.data(), A.size() * sizeof(T));
   }
}

template <typename T, BACKEND Backend>
inline void get_from_host(Matrix<T, Backend>& A, const HostMatrix<T>& B) {
   TINYAI_ASSERT(A.size() == B.size() && "Size mismatch");
   if constexpr (Backend == BACKEND::DEVICE) {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyHostToDevice));
   } else {
      std::memcpy(A.data(), B.data(), A.size() * sizeof(T));
   }
}

template <typename T, BACKEND Backend>
inline void get_from_host_view(Matrix<T, Backend>& A, const MatrixView<T>& B) {
   TINYAI_ASSERT(A.size() == B.size() && "Size mismatch");
   if constexpr (Backend == BACKEND::DEVICE) {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyHostToDevice));
   } else {
      std::memcpy(A.data(), B.data(), A.size() * sizeof(T));
   }
}

template <typename T, BACKEND Backend>
inline void get_from_device_view(Matrix<T, Backend>& A, const MatrixView<T>& B) {
   TINYAI_ASSERT(A.size() == B.size() && "Size mismatch");
   if constexpr (Backend == BACKEND::DEVICE) {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
   } else {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
   }
}

template <typename T, BACKEND Backend>
inline void get_from_host_view(Matrix<T, Backend>& A, const ConstMatrixView<T>& B) {
   TINYAI_ASSERT(A.size() == B.size() && "Size mismatch");
   if constexpr (Backend == BACKEND::DEVICE) {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyHostToDevice));
   } else {
      std::memcpy(A.data(), B.data(), A.size() * sizeof(T));
   }
}

template <typename T, BACKEND Backend>
inline void get_from_device_view(Matrix<T, Backend>& A, const ConstMatrixView<T>& B) {
   TINYAI_ASSERT(A.size() == B.size() && "Size mismatch");
   if constexpr (Backend == BACKEND::DEVICE) {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToDevice));
   } else {
      CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
   }
}

template <typename T>
inline void export_to_host(const MatrixView<T>& A, HostMatrix<T>& B) {
   CHECK_ERR(tinyAI_gpuMemcpy(B.data(), A.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
}

template <typename T>
inline void export_to_host_from_host(const MatrixView<T>& A, HostMatrix<T>& B) {
   std::memcpy(B.data(), A.data(), A.size() * sizeof(T));
}

template <typename T>
inline void export_to_host(const ConstMatrixView<T>& A, HostMatrix<T>& B) {
   CHECK_ERR(tinyAI_gpuMemcpy(B.data(), A.data(), A.size() * sizeof(T), tinyAI_gpuMemcpyDeviceToHost));
}

template <typename T>
inline void export_to_host_from_host(const ConstMatrixView<T>& A, HostMatrix<T>& B) {
   std::memcpy(B.data(), A.data(), A.size() * sizeof(T));
}

template <typename T>
inline void get_from_host(MatrixView<T>& A, const HostMatrix<T>& B) {
   CHECK_ERR(tinyAI_gpuMemcpy(A.data(), B.data(), B.size() * sizeof(T), tinyAI_gpuMemcpyHostToDevice));
}

template <typename T>
inline void get_from_host_from_host(MatrixView<T>& A, const HostMatrix<T>& B) {
   std::memcpy(A.data(), B.data(), B.size() * sizeof(T));
}

template <typename T>
inline MatrixView<T> get_view_from_raw(T* ptr, std::size_t rows, std::size_t cols) {
   return MatrixView<T>{._data = ptr, .cols = cols, .rows = rows};
}

template <typename T>
__global__ void shuffle_rows(const T* matrix, const std::size_t* perm, T* output, std::size_t ncols) {
   const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
   const std::size_t target_row = perm[row];
   for (std::size_t col = 0; col < ncols; ++col) {
      output[row * ncols + col] = matrix[target_row * ncols + col];
   }
}

template <typename T>
__global__ void shuffle_rows_warp_wide_kernel(const T* matrix, const std::size_t* perm, T* output, std::size_t ncols,
                                              std::size_t warps_per_column) {
   const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   const std::size_t wid = tid / __m_WARPSIZE__;
   const std::size_t w_tid = tid % __m_WARPSIZE__;
   const std::size_t row = wid / warps_per_column;
   const std::size_t target_row = perm[row];
   const std::size_t target_col = (wid % warps_per_column) * __m_WARPSIZE__ + w_tid;
   if (target_col < ncols) {
      output[row * ncols + target_col] = matrix[target_row * ncols + target_col];
   }
}

template <typename T>
void shuffle_rows_warpwide(const T* data_in, const std::size_t* dperm, std::size_t batchsize, T* data_out,
                           std::size_t ncols_in, tinyAI_gpuStream_t s = 0) {
   const std::size_t warps_per_col = ncols_in / __m_WARPSIZE__ + (ncols_in % __m_WARPSIZE__ != 0);
   const std::size_t total_warps = warps_per_col * batchsize;
   if (warps_per_col <= 1) {
      NumericMatrix::shuffle_rows<<<1, batchsize, 0, s>>>(data_in, dperm, data_out, ncols_in);
      return;
   }
   const std::size_t blockSize = std::min(total_warps * __m_WARPSIZE__, __m_BLOCKSIZE__);
   const std::size_t blocks =
       total_warps * __m_WARPSIZE__ / __m_BLOCKSIZE__ + ((total_warps * __m_WARPSIZE__) % __m_BLOCKSIZE__ != 0);
   shuffle_rows_warp_wide_kernel<<<blocks, blockSize, 0, s>>>(data_in, dperm, data_out, ncols_in, warps_per_col);
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T>
__global__ void adamw_kernel(T* w, T* mw, T* vw, T* dw, T m_hat_scale, T v_hat_scale, T beta_1, T beta_2,
                             T weight_decay, T lr, T epsilon, std::size_t len) {
   const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      T dw_val = dw[tid];
      T dw_sq_val = dw_val * dw_val;
      T mw_new = beta_1 * mw[tid] + (T(1.0) - beta_1) * dw_val;
      T vw_new = beta_2 * vw[tid] + (T(1.0) - beta_2) * dw_sq_val;
      mw[tid] = mw_new;
      vw[tid] = vw_new;
      T mw_hat = mw_new * m_hat_scale;
      T vw_hat = vw_new * v_hat_scale;
      w[tid] -= lr * (mw_hat / (std::sqrt(vw_hat) + epsilon) + weight_decay * w[tid]);
   }
}

template <typename T>
inline void adamw(Matrix<T, BACKEND::DEVICE>& w, Matrix<T, BACKEND::DEVICE>& mw, Matrix<T, BACKEND::DEVICE>& vw,
                  Matrix<T, BACKEND::DEVICE>& dw, T m_hat_scale, T v_hat_scale, T beta_1, T beta_2, T weight_decay,
                  T lr, T epsilon, tinyAI_gpuStream_t stream) {
   const size_t threads = std::min(__m_BLOCKSIZE__, w.size());
   const size_t blocks = w.size() / __m_BLOCKSIZE__ + (w.size() % __m_BLOCKSIZE__ != 0);
   adamw_kernel<<<blocks, threads, 0, stream>>>(w.data(), mw.data(), vw.data(), dw.data(), m_hat_scale, v_hat_scale,
                                                beta_1, beta_2, weight_decay, lr, epsilon, w.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
}

template <typename T, ACTIVATION Activation>
__global__ void matadd_and_activate(const T* A, const T* B, T* C, T* D, T w, size_t len) {
   const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < len) {
      C[tid] = A[tid] + B[tid];
      D[tid] = activate<T, Activation>(C[tid], w);
   }
}

template <typename T, ACTIVATION Activation>
inline void matadd_and_activate(const Matrix<T, BACKEND::DEVICE>& A, const Matrix<T, BACKEND::DEVICE>& B,
                                Matrix<T, BACKEND::DEVICE>& C, Matrix<T, BACKEND::DEVICE>& D, T w,
                                tinyAI_blasHandle_t* handle, tinyAI_gpuStream_t stream) {
   TINYAI_UNUSED(handle);
   TINYAI_ASSERT(A.size() == B.size() && "Dimension mismatch");
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   matadd_and_activate<T, Activation>
       <<<blocks, threads, 0, stream>>>(A.data(), B.data(), C.data(), D.data(), w, A.size());
   CHECK_ERR(tinyAI_gpuPeekAtLastError());

   spdlog::debug("Matadd kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}

template <typename T,typename FF>
inline void matapply_to(const Matrix<T, BACKEND::DEVICE>& A,
                        const Matrix<T, BACKEND::DEVICE>& B,
                        Matrix<T, BACKEND::DEVICE>& C,
                        FF&& f,
                        tinyAI_gpuStream_t stream) {
   TINYAI_ASSERT(A.size() == B.size() && A.size() == C.size());
   const size_t threads = std::min(__m_BLOCKSIZE__, A.size());
   const size_t blocks = A.size() / __m_BLOCKSIZE__ + (A.size() % __m_BLOCKSIZE__ != 0);
   apply_kernel<<<blocks, threads, 0, stream>>>(A.data(), B.data(),C.data(),A.size(),f);
   CHECK_ERR(tinyAI_gpuPeekAtLastError());
   spdlog::debug("Scale kernel [blocks,threads]= [{0:d} x {1:d} for matrix size {2:d} ]", blocks, threads, A.size());
}
//~BACKEND::DEVICE Functionality
} // namespace NumericMatrix
