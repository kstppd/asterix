#include "npy.hpp"
#include "qdargparser.h"
#include "spdlog/spdlog.h"
#include "../include/tinyAI.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <nvToolsExt.h>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace TINYAI;
using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;
constexpr std::size_t N = 5ul * 1024ul * 1024ul * 1024ul;
enum class MODE { TRAIN, PREDICT };

template <typename T>
void dumpVector(const std::string &filename, const std::vector<T> &data) {
  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "ERROR: Could not open file: " << filename << std::endl;
    return;
  }
  outFile.write(reinterpret_cast<const char *>(data.data()),
                data.size() * sizeof(T));
  outFile.close();
  return;
}

template <typename T> std::vector<T> loadVector(const std::string &filename) {
  std::ifstream inFile(filename, std::ios::binary);
  if (!inFile) {
    std::cerr << "ERROR: Could not open file: " << filename << std::endl;
    return {};
  }
  inFile.seekg(0, std::ios::end);
  std::size_t fileSize = inFile.tellg();
  inFile.seekg(0, std::ios::beg);
  std::vector<T> data(fileSize / sizeof(T));
  inFile.read(reinterpret_cast<char *>(data.data()), fileSize);
  inFile.close();
  return data;
}

NumericMatrix::HostMatrix<float>
read_npy_to_matrix(const npy::npy_data<float> &data) {
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

void train(const npy::npy_data<float> &datax,
           const npy::npy_data<float> &datay) {
  constexpr auto HW = BACKEND::DEVICE;
  void *mem;
  tinyAI_gpuMalloc(&mem, N);
  assert(mem && "Could not allocate memory !");
  GENERIC_TS_POOL::MemPool p(mem, N);

  const auto dimsx = datax.shape;
  const auto dimsy = datay.shape;
  const std::size_t train_size = dimsx[0];
  const std::size_t fin = dimsx[1];
  const std::size_t fout = dimsy[1];

  auto space = read_npy_to_matrix(datax);
  auto val = read_npy_to_matrix(datay);

  spdlog::info("X Dims = {0:d},{1:d}", space.nrows(), space.ncols());
  spdlog::info("Y Dims = {0:d},{1:d}", val.nrows(), val.ncols());

  // Copy to HW
  NumericMatrix::Matrix<float, HW> xtrain(train_size, space.ncols(), &p);
  NumericMatrix::Matrix<float, HW> ytrain(train_size, val.ncols(), &p);

  NumericMatrix::get_from_host(xtrain, space);
  NumericMatrix::get_from_host(ytrain, val);
  std::vector<int> arch{256, 256, 256, 256, 256, (int)fout};
  constexpr std::size_t batchsize = 64;
  NeuralNetwork<float, HW, ACTIVATION::RELU, ACTIVATION::NONE> nn(
      arch, &p, xtrain.nrows(),xtrain.ncols(), ytrain.ncols(), batchsize);
  const std::size_t nnsize = nn.get_network_size();
  spdlog::info("Network Size = {0:d}", nnsize);
  NumericMatrix::Matrix<float, HW> sample(batchsize, xtrain.ncols(),&p);
  NumericMatrix::Matrix<float, HW> target(batchsize, ytrain.ncols(),&p);
  NumericMatrix::Matrix<float, HW> error (batchsize, ytrain.ncols(),&p);
  std::size_t *dperm = p.allocate<std::size_t>(batchsize);
  for (size_t i = 0; i < 10; i++) {
    float l = 0.0;
    for (size_t b = 0; b < xtrain.nrows(); b += batchsize) {
      nn.get_permutation_indices(dperm,batchsize, 0);
      nn.shuffle_into(xtrain,sample,dperm,0);
      nn.shuffle_into(ytrain,target,dperm,0);
      nn.forward(sample, 0);
      l+=nn.loss<HW, LOSSF::MSE>(error,target,0);
      nn.backward(sample, target, 0);
      nn.update_weights_adamw(i + 1, 1e-4, 0);
      tinyAI_gpuStreamSynchronize(0);
    }
    l/= (xtrain.nrows() * ytrain.ncols());
    if (l < 1.0e-6) {
      spdlog::info("Breaking at epoch [{0:d}] with  loss [{1:f}]", i, l);
      break;
    }
    if (true) {
      std::vector<float> w(nnsize / sizeof(float), 0.0f);
      nn.get_weights(w.data());
      dumpVector("weights.tiny", w);
    }
    spdlog::info("Epochg {0:d} Loss {1:f}.",i,l);
  }
  spdlog::info("Pool HW = {0:f}", p.memory_hwm());
  return;
}

void predict(const npy::npy_data<float> &datax) {
  constexpr auto HW = BACKEND::DEVICE;
  void *mem;
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
  NeuralNetwork<float, HW, ACTIVATION::RELU, ACTIVATION::NONE> nn(
      arch, &p, xtrain, ytrain, batchsize);
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


void help(const char *bin) {
  fprintf(stdout, "Usage: %s --m MODE <xtrain/xtest> <ytrain> \n", bin);
  fprintf(stdout, "\tMODE  can be either train or predict.\n");
  return;
}

bool parse_input(int argc, char **argv, MODE &mode) {
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

void train(char **argv) {
  const char *filenamex = argv[3];
  const char *filenamey = argv[4];
  spdlog::info("Training on file {0:s} {1:s}.", filenamex, filenamey);
  npy::npy_data<float> datax = npy::read_npy<float>(filenamex);
  npy::npy_data<float> datay = npy::read_npy<float>(filenamey);
  train(datax, datay);
  return;
}

void predict(char **argv) {
  const char *filenamex = argv[3];
  npy::npy_data<float> datax = npy::read_npy<float>(filenamex);
  predict(datax);
  return;
}

int main(int argc, char **argv) {

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
  return 0;
}
