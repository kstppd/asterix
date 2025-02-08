#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "moving_image.h"
#include "train.h"

int main(int argc, char** argv) {

   if (argc != 2) {
      spdlog::error("Usage {0:s} <image file> ", argv[0]);
      return 1;
   }

   const char* image_filename = argv[1];
   spdlog::info("Training on image: {0:s} ", image_filename);

   constexpr std::size_t n_shifts = 32;
   constexpr std::size_t shift_step_x = 32;
   constexpr std::size_t shift_step_y = 32;
   constexpr std::size_t ff_mapping = 512;
   constexpr std::size_t neurons = 512;
   constexpr std::size_t epochs = 10;
   constexpr std::size_t batchSize = 256;
   constexpr type_t lr = 1e-3;

   MovingImage img(image_filename, n_shifts, shift_step_x, shift_step_y);
   learn(img, epochs, batchSize, neurons, ff_mapping, /*fourier scale read the paper-->*/ 10.0,lr);
   img.save();
   return 0;
}
