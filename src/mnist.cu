#include "../include/genericTsPool.h"
#include "../include/tinyAI.h"
#include "include/matrix.h"
#include <chrono>
#include <curl/curl.h>
#include <gtest/gtest.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <nvToolsExt.h>
#include <vector>
#include <zlib.h>

using namespace TINYAI;
using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;
using type_t = float;
using byte = unsigned char;

static const char* TEST_IMAGES_MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz";
static const char* TEST_LABELS_MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz";
static const char* TRAIN_IMAGES_MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
static const char* TRAIN_LABELS_MNIST = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";

struct MNIST_DATA {
   std::vector<float> train_images;
   std::vector<float> train_labels;
   std::vector<float> test_images;
   std::vector<float> test_labels;
};

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::vector<byte>* userp) {
   size_t totalSize = size * nmemb;
   userp->insert(userp->end(), (byte*)contents, (byte*)contents + totalSize);
   return totalSize;
}

std::vector<byte> download(const char* url) {
   std::vector<byte> buffer;
   CURL* curl = curl_easy_init();
   if (!curl) {
      std::cerr << "CURLinit failed: " << std::endl;
      abort();
   }
   CURLcode res;
   curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
   curl_easy_setopt(curl, CURLOPT_URL, url);
   curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
   res = curl_easy_perform(curl);
   if (res != CURLE_OK) {
      std::cerr << "Request failed: " << curl_easy_strerror(res) << std::endl;
      ;
      abort();
   }
   curl_easy_cleanup(curl);
   return buffer;
}

std::vector<byte> decompressGzip(const std::vector<uint8_t>& compressedData) {
   z_stream strm = {};
   strm.total_in = strm.avail_in = compressedData.size();
   strm.total_out = strm.avail_out = compressedData.size() * 2;
   strm.next_in = const_cast<Bytef*>(compressedData.data());

   std::vector<uint8_t> decompressedData(strm.avail_out);
   inflateInit2(&strm, 16 + MAX_WBITS);
   int ret;
   do {
      strm.avail_out = decompressedData.size() - strm.total_out;
      strm.next_out = decompressedData.data() + strm.total_out;

      ret = inflate(&strm, Z_NO_FLUSH);
      if (ret == Z_STREAM_END)
         break;

      if (ret == Z_OK && strm.avail_out == 0) {
         decompressedData.resize(decompressedData.size() * 2);
      } else if (ret != Z_OK) {
         inflateEnd(&strm);
         throw std::runtime_error("Gzip decompression failed");
      }
   } while (ret != Z_STREAM_END);
   inflateEnd(&strm);
   decompressedData.resize(strm.total_out);
   return decompressedData;
}

std::vector<float> parseIdx(const std::vector<byte>& idxData) {
   if (idxData.size() < 8) {
      throw std::runtime_error("Invalid IDX file format");
   }

   int magicNumber = (idxData[0] << 24) | (idxData[1] << 16) | (idxData[2] << 8) | idxData[3];
   int numItems = (idxData[4] << 24) | (idxData[5] << 16) | (idxData[6] << 8) | idxData[7];

   std::vector<float> parsedData;
   if (magicNumber == 0x803) {
      int numRows = (idxData[8] << 24) | (idxData[9] << 16) | (idxData[10] << 8) | idxData[11];
      int numCols = (idxData[12] << 24) | (idxData[13] << 16) | (idxData[14] << 8) | idxData[15];
      parsedData.resize(numItems * numRows * numCols);

      size_t dataOffset = 16;
      for (size_t i = 0; i < parsedData.size(); ++i) {
         parsedData[i] = static_cast<float>(idxData[dataOffset + i]) / 255.0f; // Normalize to 0-1
      }
   } else if (magicNumber == 0x801) { // This is for labels
      parsedData.resize(numItems);
      for (size_t i = 0; i < (size_t)numItems; ++i) {
         parsedData[i] = static_cast<float>(idxData[8 + i]);
      }
   } else {
      throw std::runtime_error("Unknown IDX format");
   }
   return parsedData;
}

// Comment in/out
#define USE_GPU
#define N 784

int main() {
   MNIST_DATA mnist;
   mnist.train_images = parseIdx(decompressGzip(download(TRAIN_IMAGES_MNIST)));
   mnist.train_labels = parseIdx(decompressGzip(download(TRAIN_LABELS_MNIST)));
   mnist.test_images = parseIdx(decompressGzip(download(TEST_IMAGES_MNIST)));
   mnist.test_labels = parseIdx(decompressGzip(download(TEST_LABELS_MNIST)));

   size_t NB = 4ul * 1024ul * 1024ul * 1024ul;
#ifdef USE_GPU
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   tinyAI_gpuMalloc(&mem, NB);
   std::cout << mem << std::endl;
#else
   constexpr auto HW = BACKEND::HOST;
   void* mem = (void*)malloc(NB);
#endif
   assert(mem && "Could not allocate memory !");
   GENERIC_TS_POOL::MemPool p(mem, NB);
   NumericMatrix::HostMatrix<type_t> x_train_host(60000, N);
   NumericMatrix::HostMatrix<type_t> y_train_host(60000, 1);
   NumericMatrix::HostMatrix<type_t> x_test_host(10000, N);
   NumericMatrix::HostMatrix<type_t> y_test_host(10000, 1);

   for (auto& val : mnist.train_labels) {
      val /= 10.0;
   }
   for (auto& val : mnist.test_labels) {
      val /= 10.0;
   }

   size_t cnt = 0;
   for (size_t i = 0; i < mnist.train_images.size(); i += 784) {
      for (size_t j = 0; j < N; ++j) {
         x_train_host(cnt, j) = mnist.train_images[i + j];
      }
      cnt++;
   }

   cnt = 0;
   for (size_t i = 0; i < mnist.test_images.size(); i += 784) {
      for (size_t j = 0; j < N; ++j) {
         x_test_host(cnt, j) = mnist.test_images[i + j];
      }
   }

   for (size_t i = 0; i < mnist.train_labels.size(); ++i) {
      y_train_host(i, 0) = mnist.train_labels[i];
   }

   for (size_t i = 0; i < mnist.test_labels.size(); ++i) {
      y_test_host(i, 0) = mnist.test_labels[i];
   }

   NumericMatrix::Matrix<type_t, HW> xtrain(60000, N, &p);
   NumericMatrix::Matrix<type_t, HW> xtest(10000, N, &p);
   NumericMatrix::Matrix<type_t, HW> ytrain(60000, 1, &p);
   NumericMatrix::Matrix<type_t, HW> ytest(10000, 1, &p);
   NumericMatrix::get_from_host(xtrain, x_train_host);
   NumericMatrix::get_from_host(xtest, x_test_host);
   NumericMatrix::get_from_host(ytrain, y_train_host);
   NumericMatrix::get_from_host(ytest, y_test_host);

   NumericMatrix::Matrix<type_t, HW> recon = ytest;
   NumericMatrix::Matrix<type_t, HW> xtrain2 = xtest;
   NumericMatrix::Matrix<type_t, HW> ytrain2 = ytest;
   std::vector<int> arch{400, 400, 1};
   size_t BATCHSIZE = 32;
   NeuralNetwork<type_t, HW, ACTIVATION::RELU> nn(arch, &p, xtrain, ytrain, BATCHSIZE);

   auto start = std::chrono::steady_clock::now();
   for (size_t i = 0; i < 10; i++) {
      auto l = nn.train(BATCHSIZE, 1e-3);
      printf("Epoch %zu done.Loss =%f \n", i, l);
   }
   auto end = std::chrono::steady_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
   std::cout << "Time for 10 epochs =" << duration.count() << " seconds." << std::endl;

   tinyAI_gpuDeviceSynchronize();
   nn.evaluate(xtrain2, recon);
   tinyAI_gpuDeviceSynchronize();

   size_t hits = 0;
   for (size_t i = 0; i < recon.size(); ++i) {
      float val = recon.get_value(i, 0);
      float correct = ytrain2.get_value(i, 0);
      if (std::abs(val - correct) < 0.5f) {
         hits++;
      }
   }
   float train_success = 100.0 * hits / 10000.0;
   EXPECT_TRUE(train_success >80.0);
   return 0;
}
