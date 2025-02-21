#include <gtest/gtest.h>
#include "../include/genericTsPool.h"
#include "../include/tinyAI.h"
#include <algorithm>
#include <chrono>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iomanip>
#include <nvToolsExt.h>
#include <random>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace TINYAI;
using namespace GENERIC_TS_POOL;
using namespace NumericMatrix;
using type_t = float;
using pixel = uint8_t;


struct Image {
   int width;
   int height;
   int channels;
   void* data;
};

template <typename T>
T rand_normal() {
   return (T)rand() / RAND_MAX * T(2.0) - T(1.0);
}

template <typename T>
NumericMatrix::HostMatrix<T> generate_fourier_features(NumericMatrix::HostMatrix<T>& input,

                                                       std::size_t num_features, T scale) {
   if (num_features == 0) {
      return NumericMatrix::HostMatrix<T>(input);
   }
   assert(num_features % 2 == 0);
   const std::size_t input_dims = input.ncols();
   // Construct B
   NumericMatrix::HostMatrix<T> B(input_dims, num_features);

   std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
   std::uniform_real_distribution<T> dist(-1.0, 1.0);
   for (std::size_t i = 0; i < input_dims; ++i) {
      for (std::size_t j = 0; j < num_features; ++j) {
         B(i, j) = scale * dist(rng); // rand_normal<T>();
      }
   }

   // Apply mapping
   NumericMatrix::HostMatrix<T> output(input.nrows(), 2 * num_features);
   for (std::size_t i = 0; i < input.nrows(); ++i) {
      for (std::size_t j = 0; j < num_features; ++j) {
         T dot_product = 0.0;
         for (std::size_t k = 0; k < input.ncols(); ++k) {
            dot_product += input(i, k) * B(k, j);
         }
         output(i, j) = std::sin(2.0 * M_PI * dot_product);
         output(i, j + num_features) = std::cos(2.0 * M_PI * dot_product);
      }
   }
   return output;
}

double calculate_mse(const unsigned char* original, const unsigned char* reconstructed, int width, int height) {
   double mse = 0.0;
   int num_pixels = width * height;
   for (int i = 0; i < num_pixels; i++) {
      int diff = original[i] - reconstructed[i];
      mse += diff * diff;
   }
   mse /= num_pixels;
   return mse;
}

double calculate_psnr(const unsigned char* original, const unsigned char* reconstructed, int width, int height) {
   double mse = calculate_mse(original, reconstructed, width, height);
   if (mse == 0) {
      return INFINITY;
   }
   double max_pixel_value = 255.0;
   double psnr = 20.0 * std::log10(max_pixel_value / std::sqrt(mse));
   return psnr;
}

int main(int argc, char** argv) {

   if (argc != 2) {
      fprintf(stderr, "ERROR: usage ./%s <image file>\n", argv[0]);
      return 1;
   }

   size_t N = 1ul * 1024ul * 1024ul * 1024ul;
#ifdef USE_GPU
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   cudaMalloc(&mem, N);
#else
   constexpr auto HW = BACKEND::HOST;
   void* mem = (void*)malloc(N);
#endif
   assert(mem && "Could not allocate memory !");
   GENERIC_TS_POOL::MemPool p(mem, N);

   size_t BATCHSIZE = 128;
   {

      const char* image_name = argv[1];
      Image img, rec_img;
      img.data = stbi_load(image_name, &img.width, &img.height, &img.channels, 0);
      rec_img.data = stbi_load(image_name, &rec_img.width, &rec_img.height, &rec_img.channels, 0);
      const size_t n_samples = img.width * img.height;
      NumericMatrix::HostMatrix<type_t> pos_temp(n_samples, 2);
      NumericMatrix::Matrix<type_t, HW> val(n_samples, 1, &p);
      NumericMatrix::Matrix<type_t, HW> recon(n_samples, 1, &p);
      size_t cnt = 0;
      for (size_t j = 0; j < static_cast<size_t>(img.height); j++) {
         for (size_t i = 0; i < static_cast<size_t>(img.width); i++) {
            float y = 2.0 * ((type_t)j / (type_t)img.height) - 1.0;
            float x = 2.0 * ((type_t)i / (type_t)img.width) - 1.0;
            pos_temp.set_value(cnt, 0, y);
            pos_temp.set_value(cnt, 1, x);
            val.set_value(cnt, 0, reinterpret_cast<pixel*>(img.data)[i + j * img.width] / 255.0);
            cnt++;
         }
      }

      NumericMatrix::HostMatrix<type_t> ff_input = generate_fourier_features<type_t>(pos_temp, 32, 10.0);

      NumericMatrix::Matrix<type_t, HW> pos(n_samples, ff_input.ncols(), &p);

      NumericMatrix::get_from_host(pos, ff_input);

      NumericMatrix::Matrix<type_t, HW> pos2 = pos;
      NumericMatrix::Matrix<type_t, HW> val2 = val;
      std::vector<int> arch{200, 200, 1};

      NeuralNetwork<type_t, HW, ACTIVATION::RELU> nn(arch, &p, pos, val, BATCHSIZE);
      auto V = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < 5; i++) {
         auto l = nn.train(BATCHSIZE, 1e-3);
      }

      auto Y = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Y - V).count();
      cudaDeviceSynchronize();

      nn.evaluate(pos2, recon);
      cnt = 0;
      for (size_t j = 0; j < static_cast<size_t>(img.height); j++) {
         for (size_t i = 0; i < static_cast<size_t>(img.width); i++) {
            auto val = std::clamp(recon.get_value(cnt, 0), 0.0f, 1.0f);
            reinterpret_cast<pixel*>(rec_img.data)[i + j * img.width] = val * 255.0;
            cnt++;
         }
      }
      double psnr =
          calculate_psnr((unsigned char*)img.data, (unsigned char*)rec_img.data, rec_img.width, rec_img.height);
      EXPECT_TRUE(psnr>20.0 );

   }
   p.defrag();

// CleanUp
#ifdef USE_GPU
   cudaFree(mem);
#else
   free(mem);
#endif
   return 0;
}
