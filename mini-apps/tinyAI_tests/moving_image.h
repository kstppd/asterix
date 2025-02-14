#pragma once
#include "genericTsPool.h"
#include "tinyAI.h"
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <omp.h>
#include <vector>
#include "stb_image.h"
#include "stb_image_write.h"
using type_t = float;
using pixel = uint8_t;

inline std::string padInt(int num, int pad) {
   std::stringstream ss;
   ss << std::setw(pad) << std::setfill('0') << num;
   return ss.str();
}

struct MovingImage {
   struct Image {
      int width;
      int height;
      int channels;
      void* data;
   };

   Image img;
   std::size_t n_shifts;
   std::array<std::size_t, 2> shift_size = {0, 0};
   NumericMatrix::HostMatrix<type_t> xtrain, ytrain;

   MovingImage(const char* image_file_name, std::size_t n_shifts, std::size_t shift_step_x, std::size_t shift_step_y)
       : n_shifts(n_shifts), shift_size({shift_step_x, shift_step_y}) {
      img.data = stbi_load(image_file_name, &img.width, &img.height, &img.channels, 0);
      spdlog::info("Loaded image {0:s}: [{1:d} x {2:d}].", image_file_name, img.width, img.height);
      spdlog::info("Building training data");
      const std::size_t xrows = img.height * img.width;
      const std::size_t xcols = n_shifts;
      const std::size_t yrows = img.height * img.width;
      const std::size_t ycols = n_shifts;

      xtrain = NumericMatrix::HostMatrix<type_t>(xrows, xcols);
      ytrain = NumericMatrix::HostMatrix<type_t>(yrows, ycols);

      for (std::size_t s = 0; s < n_shifts; ++s) {
         std::size_t cnt = 0;
         for (std::size_t j = 0; j < static_cast<std::size_t>(img.height); j++) {
            for (std::size_t i = 0; i < static_cast<std::size_t>(img.width); i++) {
               type_t y = 2.0 * ((type_t)j / (type_t)img.height) - 1.0;
               type_t x = 2.0 * ((type_t)i / (type_t)img.width) - 1.0;
               xtrain.set_value(cnt, 0, y);
               xtrain.set_value(cnt, 1, x);
               const std::size_t target_i = (i + s * shift_size[0]) % img.width;
               const std::size_t target_j = (j + s * shift_size[1]) % img.height;
               ytrain.set_value(cnt, s, reinterpret_cast<pixel*>(img.data)[target_i + target_j * img.width] / 255.0);
               cnt++;
            }
         }
      }
      spdlog::info("Built training data!");
   }
   MovingImage() = delete;
   MovingImage(const MovingImage& other) = delete;
   MovingImage(MovingImage&& other) = delete;
   MovingImage& operator=(const MovingImage& other) = delete;
   MovingImage& operator=(MovingImage&& other) = delete;
   ~MovingImage() {}

   void save() const {
      spdlog::info("Saving moving image");
      std::vector<pixel> buffer_data(img.height * img.height, 0);
      for (std::size_t s = 0; s < n_shifts; ++s) {
         std::size_t cnt = 0;
         for (std::size_t j = 0; j < static_cast<std::size_t>(img.height); j++) {
            for (std::size_t i = 0; i < static_cast<std::size_t>(img.width); i++) {
               buffer_data.at(cnt) = std::max(0.0,ytrain.get_value(cnt, s) * 255.0);
               cnt++;
            }
         }
         const auto name = "output." + padInt(s, 5) + ".png";
         stbi_write_png(name.c_str(), img.width, img.height, 1, buffer_data.data(), img.width);
      }
   }
};


template <typename T>
NumericMatrix::HostMatrix<T>
generate_fourier_features(NumericMatrix::HostMatrix<T> &input,
                          
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
 for (std::size_t j = 0; j < num_features; ++j) {
     for (std::size_t i = 0; i < input_dims; ++i) {
      B(i, j) = scale * dist(rng) ;// rand_normal<T>();
    }
  }
  
  
  // Apply mapping
  NumericMatrix::HostMatrix<T> output(input.nrows(), 2 * num_features);
  #pragma omp parallel for collapse(2)
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
