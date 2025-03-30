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

#include "../include/tinyAI.h"
#include <gtest/gtest.h>
#include <iomanip>

using namespace TINYAI;
using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;

#define expect_true EXPECT_TRUE
#define expect_false EXPECT_FALSE
#define expect_eq EXPECT_EQ
constexpr size_t N = 2ul * 1024ul * 1024ul * 1024ul;
GENERIC_TS_POOL::MemPool p;
constexpr BACKEND HW= BACKEND::DEVICE;

TEST(NN, IO_F32) {
  size_t BATCHSIZE = 128;
  std::vector<int> arch{200, 200, 1};
  NumericMatrix::Matrix<float, HW> x_train(1024, 128, &p);
  NumericMatrix::Matrix<float, HW> y_train(1024, 64, &p);
  NeuralNetwork<float, HW> nn(arch, &p, x_train, y_train, BATCHSIZE);
  auto sz=nn.get_network_weight_count();
  std::vector<float> w(sz);
  nn.get_weights(w.data());
  nn.save("tmp.tiny");

  NeuralNetwork<float, HW> nn2("tmp.tiny", &p);
  auto sz2=nn2.get_network_weight_count();
  std::vector<float> w2(sz2);
  nn2.get_weights(w2.data());
  expect_true(sz==sz2);
  expect_true(w==w2);
}

TEST(NN, IO_F64) {
  size_t BATCHSIZE = 128;
  std::vector<int> arch{200, 200, 1};
  NumericMatrix::Matrix<double, HW> x_train(1024, 128, &p);
  NumericMatrix::Matrix<double, HW> y_train(1024, 64, &p);
  NeuralNetwork<double, HW> nn(arch, &p, x_train, y_train, BATCHSIZE);
  auto sz=nn.get_network_weight_count();
  std::vector<double> w(sz);
  nn.get_weights(w.data());
  nn.save("tmp.tiny");

  NeuralNetwork<double, HW> nn2("tmp.tiny", &p);
  auto sz2=nn2.get_network_weight_count();
  std::vector<double> w2(sz2);
  nn2.get_weights(w2.data());
  expect_true(sz==sz2);
  expect_true(w==w2);
}

int main(int argc, char* argv[]){
    
#ifdef USE_GPU
   void* mem;
   tinyAI_gpuMallocManaged(&mem, N);
   std::cout << mem << std::endl;
#else
   void* mem = (void*)malloc(N);
#endif
    p.resize(mem,N);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
