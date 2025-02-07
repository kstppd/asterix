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

   MovingImage img(image_filename, 32, 32, 0);
   learn(img, 100,256,400,128,10.0);
   img.save();
   return 0;
}
