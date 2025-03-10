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

#include "../include/genericTsPool.h"
#include "../include/matrix.h"
#include <gtest/gtest.h>
#include <limits>

using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;
constexpr std::size_t NB = 2ul * 1024ul * 1024ul * 1024ul;
static cublasHandle_t handle;
#ifdef USE_GPU
constexpr auto HW = BACKEND::DEVICE;
#else
constexpr auto HW = BACKEND::HOST;
#endif
#ifdef FP32P
using type_t = float;
#define expect_eq_numerics EXPECT_FLOAT_EQ
#else
using type_t = double;
#define expect_eq_numerics EXPECT_DOUBLE_EQ
#endif

static tinyAI_gpuStream_t s;
void* allocate_unit(std::size_t bytes) {
   (void)bytes;
#ifdef USE_GPU
   void* mem;
   tinyAI_gpuMallocManaged(&mem, NB);
   std::cout << mem << std::endl;
#else
   void* mem = (void*)malloc(NB);
#endif
   return mem;
}

void free_unit(void* mem) {
#ifdef USE_GPU
   tinyAI_gpuFree(mem);
#else
   free(mem);
#endif
}

template <typename T>
T random_number(T min, T max) {
   return min + (T)rand() / (T)RAND_MAX * (max - min);
   ;
};

TEST(Matrix, Construction) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);
   constexpr std::size_t rows = 60000;
   constexpr std::size_t cols = 2;
   NumericMatrix::HostMatrix<type_t> x_train_host(rows, cols);
   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);
   EXPECT_EQ(x_train_host.nrows(), rows);
   EXPECT_EQ(x_train_host.ncols(), cols);
   free_unit(mem);
}

TEST(Matrix, Add) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(i + j);
         B(i, j) = static_cast<type_t>((i + 1) * j);
      }
   }

   NumericMatrix::matadd(A, B, C, &handle, s);
   tinyAI_gpuDeviceSynchronize();

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         EXPECT_EQ(C(i, j), A(i, j) + B(i, j)) << "Mismatch at (" << i << ", " << j << ")";
      }
   }

   free_unit(mem);
}

TEST(Matrix, Subtract) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(i + j);
         B(i, j) = static_cast<type_t>((i + 1) * j);
      }
   }

   NumericMatrix::matsub(A, B, C, &handle, s);
   tinyAI_gpuDeviceSynchronize();

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         EXPECT_EQ(C(i, j), A(i, j) - B(i, j)) << "Mismatch at (" << i << ", " << j << ")";
      }
   }

   free_unit(mem);
}

TEST(Matrix, ScaleTo) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   type_t factor = random_number<type_t>(1, 100);
   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(std::rand() % 100);
      }
   }

   NumericMatrix::matscale_to(A, B, factor, &handle, s);
   tinyAI_gpuDeviceSynchronize();
   ;

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         EXPECT_EQ(B(i, j), A(i, j) * factor) << "Mismatch at (" << i << ", " << j << ")";
      }
   }

   free_unit(mem);
}

TEST(Matrix, Scale) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   type_t factor = random_number<type_t>(1, 100);
   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(std::rand() % 100);
         B(i, j) = A(i, j);
      }
   }

   NumericMatrix::matscale(B, factor, &handle, s);
   tinyAI_gpuDeviceSynchronize();
   ;

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         EXPECT_EQ(B(i, j), A(i, j) * factor) << "Mismatch at (" << i << ", " << j << ")";
      }
   }

   free_unit(mem);
}

TEST(Matrix, Matmul) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);
   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;
   constexpr std::size_t inner_dim = 1 << 10;

   NumericMatrix::Matrix<type_t, HW> A(rows, inner_dim, &p);
   NumericMatrix::Matrix<type_t, HW> B(inner_dim, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < inner_dim; ++j) {
         A(i, j) = static_cast<type_t>(std::rand() % 100);
      }
   }

   for (std::size_t i = 0; i < inner_dim; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         B(i, j) = static_cast<type_t>(std::rand() % 100);
      }
   }

   NumericMatrix::matmul(A, B, C, &handle);
   tinyAI_gpuDeviceSynchronize();
   ;

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         type_t expected_value = 0;
         for (std::size_t k = 0; k < inner_dim; ++k) {
            expected_value += A(i, k) * B(k, j);
         }
         expect_eq_numerics(C(i, j), expected_value) << "Mismatch at (" << i << ", " << j << ")"
                                                     << "\nExpected: " << expected_value << "\nActual: " << C(i, j);
      }
   }

   free_unit(mem);
}

TEST(Matrix, ElementWiseMultiply) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   std::srand(static_cast<unsigned int>(std::time(nullptr)));

   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(std::rand() % 100);
         B(i, j) = static_cast<type_t>(std::rand() % 100);
      }
   }

   NumericMatrix::mat_pointwise_mul(A, B, C, s);
   tinyAI_gpuDeviceSynchronize();
   ;

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         type_t expected_value = A(i, j) * B(i, j);

         expect_eq_numerics(C(i, j), expected_value) << "Mismatch at (" << i << ", " << j << ")"
                                                     << "\nExpected: " << expected_value << "\nActual: " << C(i, j);
      }
   }

   free_unit(mem);
}

bool test_apply() {
   void* mem = allocate_unit(NB);
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   std::srand(static_cast<unsigned int>(std::time(nullptr)));

   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(std::rand() % 100);
         B(i, j) = static_cast<type_t>(std::rand() % 100);
      }
   }

   NumericMatrix::matapply_to<type_t>(A,B,C,OP_ELEMENTWISE_MUL<type_t>{},s);
   tinyAI_gpuDeviceSynchronize();
   
   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         type_t expected_value = A(i, j) * B(i, j);

         if (std::abs(C(i, j) - expected_value)>std::numeric_limits<type_t>::epsilon()){
            return false;
         }
      }
   }
  
   tinyAI_gpuDeviceSynchronize();
   free_unit(mem);
   return true;
}

TEST(Matrix, ElementWiseDivide) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   constexpr std::size_t rows = 1 << 10;
   constexpr std::size_t cols = 1 << 9;

   std::srand(static_cast<unsigned int>(std::time(nullptr)));

   NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);
   NumericMatrix::Matrix<type_t, HW> C(rows, cols, &p);

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         A(i, j) = static_cast<type_t>(1.0 + std::rand() % 100);
         B(i, j) = static_cast<type_t>(1.0 + std::rand() % 100);
      }
   }

   NumericMatrix::mat_pointwise_div(A, B, C, s);
   tinyAI_gpuDeviceSynchronize();
   ;

   for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
         type_t expected_value = A(i, j) / B(i, j);

         expect_eq_numerics(C(i, j), expected_value) << "Mismatch at (" << i << ", " << j << ")"
                                                     << "\nExpected: " << expected_value << "\nActual: " << C(i, j);
      }
   }

   free_unit(mem);
}

#ifdef USE_GPU
TEST(Matrix, Reductions) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   for (std::size_t i = 1; i < 16; i++) {
      std::size_t rows = i * 32;
      std::size_t cols = i * 32;

      std::srand(static_cast<unsigned int>(std::time(nullptr)));
      NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);

      type_t res = 0.0;
      for (std::size_t i = 0; i < rows; ++i) {
         for (std::size_t j = 0; j < cols; ++j) {
            type_t val = static_cast<type_t>(1.0 + std::rand() % 100);
            A(i, j) = val;
            res += val;
         }
      }
      type_t sum = NumericMatrix::matreduce_add_gpu(A, &p, &handle, s);
      expect_eq_numerics(sum, res) << "Broken reduction: ";
   }

   free_unit(mem);
}

TEST(Matrix, Shuffler) {
   void* mem = allocate_unit(NB);
   ASSERT_NE(mem, nullptr) << "Memory allocation failed!";
   GENERIC_TS_POOL::MemPool p(mem, NB);

   for (std::size_t w = 0; w < 10; ++w) {
      std::size_t rows = 1280;
      std::size_t cols = 5120;
      NumericMatrix::Matrix<type_t, HW> A(rows, cols, &p);
      NumericMatrix::Matrix<type_t, HW> B(rows, cols, &p);

      for (std::size_t i = 0; i < rows; ++i) {
         type_t val = static_cast<type_t>(1.0 + std::rand() % 100);
         for (std::size_t j = 0; j < cols; ++j) {
            A(i, j) = val;
         }
      }

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<std::size_t> dist(0, rows);
      std::vector<type_t> perm(rows);
      std::size_t* dperm = p.allocate<std::size_t>(rows);
      for (std::size_t k = 0; k < rows; ++k) {
         perm[k] = dist(gen);
      }

      NumericMatrix::shuffle_rows_warpwide(A.data(), dperm, rows, B.data(), A.ncols(), s);
      tinyAI_gpuStreamSynchronize(s);

      for (std::size_t i = 0; i < rows; ++i) {
         type_t target = B(i, 0);
         for (std::size_t j = 0; j < cols; ++j) {
            expect_eq_numerics(B(i, j), target) << "Broken shuffler. ";
         }
      }
   }
   free_unit(mem);
}
#endif

int main(int argc, char* argv[]) {

   if constexpr (HW == BACKEND::DEVICE) {
      tinyAI_blasCreate(&handle);
      tinyAI_gpuStreamCreate(&s);
   }
   srand(time(NULL));
   if (!test_apply()){
      return 1;
   }
   ::testing::InitGoogleTest(&argc, argv);
   auto ok = RUN_ALL_TESTS();
   if constexpr (HW == BACKEND::DEVICE) {
      tinyAI_blasDestroy(handle);
      tinyAI_gpuStreamDestroy(s);
   }
   return ok;
}
