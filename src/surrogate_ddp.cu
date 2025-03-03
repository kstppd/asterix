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
constexpr std::size_t N = 2ul * 1024ul * 1024ul * 1024ul;
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

NumericMatrix::HostMatrix<float> read_npy_to_matrix_mpi(const char* filename) {
   int myRank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   auto shape = npy::read_npy_shape(filename);
   const auto rows = shape[0];
   const auto cols = shape[1];
   const std::size_t rows_per_rank = rows / size;
   const std::size_t left_over_rows = rows % size;
   std::size_t byte_offset = 0 + myRank * rows_per_rank * cols * sizeof(float);
   ;
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

void train(const char* filex, const char* filey) {
   int myRank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   tinyAI_gpuSetDevice(myRank);

   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMalloc(&mem, N);
   assert(mem && "Could not allocate memory !");
   GENERIC_TS_POOL::MemPool p(mem, N);

   auto space = read_npy_to_matrix_mpi(filex);
   auto val = read_npy_to_matrix_mpi(filey);
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
   std::vector<int> arch{256, 256, 256, 256, 256, (int)fout};
   constexpr std::size_t batchsize = 64;
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
   for (size_t i = 0; i < 10; i++) {
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

         nn.update_weights_adamw(i + 1, 1e-4, 0);
         tinyAI_gpuStreamSynchronize(0);
      }
      // Sync weights
      MPI_Barrier(MPI_COMM_WORLD);
      nn.get_weights(local_weights.data());
      MPI_Allreduce(local_weights.data(), global_weights.data(), local_weights.size(), MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      for (auto& w : global_weights) {
         w /= size;
      }
      nn.load_weights(global_weights.data());
      MPI_Barrier(MPI_COMM_WORLD);
      l /= (xtrain.nrows() * ytrain.ncols());
      spdlog::info("[{0:d}]Epoch {1:d} Loss {2:f}.", myRank, i, l);
   }
   spdlog::info("Pool HW = {0:f}", p.memory_hwm());
   return;
}

void predict(const npy::npy_data<float>& datax) {
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMalloc(&mem, N);
   assert(mem && "Could not allocate memory !");
   GENERIC_TS_POOL::MemPool p(mem, N);

   const auto dimsx = datax.shape;
   const std::size_t train_size = dimsx[0];
   const std::size_t fin = dimsx[1];
   const std::size_t fout = 5;

   // IO
   auto space = read_npy_to_matrix(datax);
   NumericMatrix::HostMatrix<float> val(train_size, fout);
   auto weights = loadVector<float>("weights.tiny");

   spdlog::info("X Dims = {0:d},{1:d}", space.nrows(), space.ncols());
   spdlog::info("Y Dims = {0:d},{1:d}", val.nrows(), val.ncols());

   // Copy to HW
   NumericMatrix::Matrix<float, HW> xtrain(train_size, space.ncols(), &p);
   NumericMatrix::Matrix<float, HW> ytrain(train_size, val.ncols(), &p);

   NumericMatrix::get_from_host(xtrain, space);
   NumericMatrix::get_from_host(ytrain, val);
   std::vector<int> arch{256, 256, 256, 256, 256, (int)fout};
   constexpr std::size_t batchsize = 512;
   NeuralNetwork<float, HW, ACTIVATION::RELU, ACTIVATION::NONE> nn(arch, &p, xtrain, ytrain, batchsize);
   nn.load_weights(weights.data());
   nn.evaluate_at_once(xtrain, ytrain);
   tinyAI_gpuDeviceSynchronize();
   ytrain.export_to_host(val);

   npy::npy_data<float> d;
   d.shape = {train_size, fout};
   d.data = std::move(std::vector<float>(val.begin(), val.end()));
   npy::write_npy("prediction.npy", d);
   return;
}

void help(const char* bin) {
   fprintf(stdout, "Usage: %s --m MODE <xtrain/xtest> <ytrain> \n", bin);
   fprintf(stdout, "\tMODE  can be either train or predict.\n");
   return;
}

bool parse_input(int argc, char** argv, MODE& mode) {
   if (argc < 2) {
      fprintf(stderr, "ERROR: wrong usage!\n");
      help(argv[0]);
      return false;
   }
   QdArgParser<' '> cli_args(argc, argv);

   auto it = cli_args.begin(); // binary name
   it = it.next();
   if (*it != "--m") {
      help(argv[0]);
      return false;
   }

   // Either train or predict;
   it = it.next();
   if (*it == "train") {
      mode = MODE::TRAIN;
   } else if (*it == "predict") {
      mode = MODE::PREDICT;
   } else {
      help(argv[0]);
      return false;
   }

   if (mode == MODE::PREDICT) {
      if (argc != 4) {
         help(argv[0]);
         return false;
      }
   }

   if (mode == MODE::TRAIN) {
      if (argc != 5) {
         help(argv[0]);
         return false;
      }
   }
   return true;
}

void train(char** argv) {
   const char* filenamex = argv[3];
   const char* filenamey = argv[4];
   spdlog::info("Training on file {0:s} {1:s}.", filenamex, filenamey);
   train(filenamex, filenamey);
   return;
}

void predict(char** argv) {
   const char* filenamex = argv[3];
   npy::npy_data<float> datax = npy::read_npy<float>(filenamex);
   predict(datax);
   return;
}

int main(int argc, char** argv) {

   int myRank;
   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MODE mode;
   if (!parse_input(argc, argv, mode)) {
      return 1;
   }

   switch (mode) {
   case MODE::TRAIN:
      train(argv);
      break;
   case MODE::PREDICT:
      predict(argv);
      break;
   default:
      throw std::runtime_error("Parsing failed and caused catastrophic failure!");
   }

   MPI_Finalize();

   return 0;
}
