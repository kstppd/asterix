#include "../include/tinyAI.h"
#include "npy.hpp"
#include "qdargparser.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#define MASTER 0
using namespace TINYAI;
using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;
constexpr std::size_t N = 1ul * 1024ul * 1024ul * 1024ul;
enum class MODE { TRAIN, PREDICT };

template <typename T>
void dumpVector(const std::string& filename, const std::vector<T>& data) {
   std::ofstream outFile(filename, std::ios::binary);
   if (!outFile) {
      std::cerr << "ERROR: Could not open file: " << filename << std::endl;
      return;
   }
   outFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
   outFile.close();
   return;
}

template <typename T>
std::vector<T> loadVector(const std::string& filename) {
   std::ifstream inFile(filename, std::ios::binary);
   if (!inFile) {
      std::cerr << "ERROR: Could not open file: " << filename << std::endl;
      return {};
   }
   inFile.seekg(0, std::ios::end);
   std::size_t fileSize = inFile.tellg();
   inFile.seekg(0, std::ios::beg);
   std::vector<T> data(fileSize / sizeof(T));
   inFile.read(reinterpret_cast<char*>(data.data()), fileSize);
   inFile.close();
   return data;
}

NumericMatrix::HostMatrix<float> generate_fourier_features(const NumericMatrix::HostMatrix<float> &input,
                                                       NumericMatrix::HostMatrix<float>& B, std::size_t num_features,
                                                       float scale) {
   if (num_features == 0) {
      return NumericMatrix::HostMatrix<float>(input);
   }
   assert(num_features % 2 == 0 && num_features > 0);
   const std::size_t input_dims = input.ncols();
   // Construct B
   if (B.isEmpty()) {
      B = NumericMatrix::HostMatrix<float>(input_dims, num_features);
      std::mt19937 rng(128);
      std::uniform_real_distribution<float> dist(0.0, 1.0);
      for (std::size_t i = 0; i < input_dims; ++i) {
         for (std::size_t j = 0; j < num_features; ++j) {
            B(i, j) = scale * dist(rng); // rand_normal<float>();
         }
      }
   }

   // Apply mapping
   NumericMatrix::HostMatrix<float> output(input.nrows(), 2 * num_features);
   for (std::size_t i = 0; i < input.nrows(); ++i) {
      for (std::size_t j = 0; j < num_features; ++j) {
         float dot_product = 0.0;
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

NumericMatrix::HostMatrix<float> read_npy_to_matrix(const npy::npy_data<float>& data) {
   const auto dims = data.shape;
   const std::size_t train_size = dims[0];
   const std::size_t fin = dims[1];
   spdlog::info("Dims = {0:d},{1:d}", train_size, fin);
   NumericMatrix::HostMatrix<float> space(train_size, fin);
   for (std::size_t row = 0; row < space.nrows(); ++row) {
      for (std::size_t col = 0; col < space.ncols(); ++col) {
         space.set_value(row, col, data.data.at(row * fin + col));
      }
   }
   return space;
}

NumericMatrix::HostMatrix<float> read_npy_to_matrix_1d(const npy::npy_data<float>& data) {
   const auto dims = data.shape;
   const std::size_t train_size = dims[0];
   const std::size_t fin = 1;
   spdlog::info("Dims = {0:d},{1:d}", train_size, fin);
   NumericMatrix::HostMatrix<float> space(train_size, fin);
   for (std::size_t row = 0; row < space.nrows(); ++row) {
      for (std::size_t col = 0; col < space.ncols(); ++col) {
         space.set_value(row, col, data.data.at(row * fin + col));
      }
   }
   return space;
}

NumericMatrix::HostMatrix<float> read_npy_to_matrix_mpi(const char* filename) {
   int myRank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   spdlog::info("Chucking {0:d}", myRank);
   auto shape = npy::read_npy_shape(filename);
   const auto rows = shape[0];
   const auto cols = shape[1];
   const std::size_t rows_per_rank = rows / size;
   const std::size_t left_over_rows = rows % size;
   std::size_t byte_offset = 0 + myRank * rows_per_rank * cols * sizeof(float);
   const npy::shape_t local_shape = {rows_per_rank + left_over_rows * (myRank == size - 1), cols};
   npy::npy_data<float> data = npy::read_npy_partial<float>(filename, local_shape, byte_offset);
   const auto dims = data.shape;
   const std::size_t train_size = dims[0];
   const std::size_t fin = dims[1];
   NumericMatrix::HostMatrix<float> mat(train_size, fin);
   for (std::size_t row = 0; row < mat.nrows(); ++row) {
      for (std::size_t col = 0; col < mat.ncols(); ++col) {
         mat.set_value(row, col, data.data.at(row * fin + col));
      }
   }
   return mat;
}

NumericMatrix::HostMatrix<float> read_npy_to_matrix_mpi_1d(const char* filename,std::size_t& offset,std::size_t& total_size) {
   int myRank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   spdlog::info("Chucking {0:d}", myRank);
   auto shape = npy::read_npy_shape(filename);
   const auto rows = shape[0];
   const auto cols = 1;
   const std::size_t rows_per_rank = rows / size;
   const std::size_t left_over_rows = rows % size;
   offset=0 + myRank * rows_per_rank * cols ;
   total_size=rows;
   std::size_t byte_offset = 0 + myRank * rows_per_rank * cols * sizeof(float);
   const npy::shape_t local_shape = {rows_per_rank + left_over_rows * (myRank == size - 1)};
   npy::npy_data<float> data = npy::read_npy_partial<float>(filename, local_shape, byte_offset);
   const auto dims = data.shape;
   const std::size_t train_size = dims[0];
   const std::size_t fin = 1;
   NumericMatrix::HostMatrix<float> mat(train_size, fin);
   for (std::size_t row = 0; row < mat.nrows(); ++row) {
      for (std::size_t col = 0; col < mat.ncols(); ++col) {
         mat.set_value(row, col, data.data.at(row * fin + col));
      }
   }
   return mat;
}

void train(const char* filex, const char* filey) {
   int myRank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   spdlog::info("Hello from rank {0:d}", myRank);
   // tinyAI_gpuSetDevice(myRank);

   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMalloc(&mem, N);
   assert(mem && "Could not allocate memory !");
   GENERIC_TS_POOL::MemPool p(mem, N);

   NumericMatrix::HostMatrix<float> B;
   std::size_t rank_offset=0;
   std::size_t total_size=0;
   auto space_tmp = read_npy_to_matrix_mpi(filex);
   auto space=generate_fourier_features(space_tmp,B,128,10.0);
   auto val = read_npy_to_matrix_mpi_1d(filey,rank_offset,total_size);
   
   spdlog::info("Rank {0:d}-->  x[{1:d},{2:d}], y[{3:d},{4:d}]", myRank, space.nrows(), space.ncols(), val.nrows(),
                val.ncols());
   const std::size_t train_size = space.nrows();
   const std::size_t fin = space.ncols();
   const std::size_t fout = val.ncols();

   // Copy to HW
   NumericMatrix::Matrix<float, HW> xtrain(train_size, space.ncols(), &p);
   NumericMatrix::Matrix<float, HW> ytrain(train_size, val.ncols(), &p);

   NumericMatrix::get_from_host(xtrain, space);
   NumericMatrix::get_from_host(ytrain, val);
   std::vector<int> arch{256,256,256, (int)fout};
   constexpr std::size_t batchsize = 32;
   NeuralNetwork<float, HW, ACTIVATION::RELU, ACTIVATION::NONE> nn(arch, &p, xtrain.nrows(), xtrain.ncols(),
                                                                   ytrain.ncols(), batchsize);
   const std::size_t nnsize = nn.get_network_size() / sizeof(float);
   std::vector<float> local_weights(nnsize);
   std::vector<float> global_weights(nnsize);
   std::vector<float> local_grads(nnsize);
   std::vector<float> global_grads(nnsize);
   nn.get_weights(local_weights.data());
   // Bcast the weights
   MPI_Bcast(local_weights.data(), local_weights.size(), MPI_FLOAT, MASTER, MPI_COMM_WORLD);
   nn.load_weights(local_weights.data());
   MPI_Barrier(MPI_COMM_WORLD);

   spdlog::info("Network Size = {0:d}", nnsize);
   NumericMatrix::Matrix<float, HW> sample(batchsize, xtrain.ncols(), &p);
   NumericMatrix::Matrix<float, HW> target(batchsize, ytrain.ncols(), &p);
   NumericMatrix::Matrix<float, HW> error(batchsize, ytrain.ncols(), &p);
   std::size_t* dperm = p.allocate<std::size_t>(batchsize);
   for (size_t i = 0; i < 5; i++) {
      float l = 0.0;
      for (size_t b = 0; b < xtrain.nrows(); b += batchsize) {
         nn.get_permutation_indices(dperm, batchsize, 0);
         nn.shuffle_into(xtrain, sample, dperm, 0);
         nn.shuffle_into(ytrain, target, dperm, 0);
         nn.forward(sample, 0);
         l += nn.loss<HW, LOSSF::MSE>(error, target, 0);
         nn.backward(sample, target, 0);

         // Communicate gradients
         MPI_Barrier(MPI_COMM_WORLD);
         nn.get_grads(local_grads.data());
         MPI_Allreduce(local_grads.data(), global_grads.data(), local_grads.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
         for (auto& g : global_grads) {
            g /= size;
         }
         nn.load_grads(global_grads.data());
         MPI_Barrier(MPI_COMM_WORLD);

         nn.update_weights_adamw(i + 1, 1e-3, 0);
         tinyAI_gpuStreamSynchronize(0);
      }
      // // Sync weights
      // MPI_Barrier(MPI_COMM_WORLD);
      // nn.get_weights(local_weights.data());
      // MPI_Allreduce(local_weights.data(), global_weights.data(), local_weights.size(), MPI_FLOAT, MPI_SUM,
      //               MPI_COMM_WORLD);
      // for (auto& w : global_weights) {
      //    w /= size;
      // }
      // nn.load_weights(global_weights.data());
      MPI_Barrier(MPI_COMM_WORLD);
      l /= (xtrain.nrows() * ytrain.ncols());
      spdlog::info("[{0:d}]Epoch {1:d} Loss {2:f}.", myRank, i, l);
   }
   
   MPI_Barrier(MPI_COMM_WORLD);
   if (myRank == MASTER+1) {
      npy::npy_data<float> datax = npy::read_npy<float>(filex);
      npy::npy_data<float> datay = npy::read_npy<float>(filey);

      auto space_inf_tmp = read_npy_to_matrix(datax);
      auto space_inf = generate_fourier_features(space_inf_tmp, B, 128, 10.0);
      auto val_inf = read_npy_to_matrix_1d(npy::read_npy<float>(filey));

      NumericMatrix::Matrix<float, HW> xinf(space_inf.nrows(), space_inf.ncols(), &p);
      NumericMatrix::Matrix<float, HW> yinf(val_inf.nrows(), val_inf.ncols(), &p);
      NumericMatrix::get_from_host(xinf, space_inf);
      NumericMatrix::get_from_host(yinf, val_inf);
      
      nn.evaluate(xinf,yinf);
      NumericMatrix::HostMatrix<float>yinf_host(yinf);
      npy::npy_data_ptr<float> d;
      d.data_ptr = yinf_host.data();
      d.shape = {512,512};
      const std::string path{"output.npy"};
      npy::write_npy(path, d);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   spdlog::info("Pool HW = {0:f}", p.memory_hwm());
   return;
}


int main(int argc, char** argv) {
   int myRank;
   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   if (argc != 3) {
      fprintf(stdout, "Usage: %s <npy file x> <npy file y> \n", argv[0]);
      MPI_Finalize();
      return 1;
   }
   train(argv[1],argv[2]);
   MPI_Finalize();
   return 0;
}
