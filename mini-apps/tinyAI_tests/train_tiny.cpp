#include "moving_image.h"
#include "train.h"
#include <chrono>
#include <driver_types.h>
#define USE_GPU

void learn(MovingImage& img, std::size_t max_epochs, std::size_t batchsize, std::size_t neurons, std::size_t ff,
           type_t scale) {

   size_t N = 5ul * 1024ul * 1024ul * 1024ul;
#ifdef USE_GPU
   constexpr auto HW = BACKEND::DEVICE;
   void* mem;
   cudaMalloc(&mem, N);
   std::cout << mem << std::endl;
#else
   constexpr auto HW = BACKEND::HOST;
   void* mem = (void*)malloc(N);
#endif

   assert(mem && "Could not allocate memory !");
   GENERIC_TS_POOL::MemPool p(mem, N);

   NumericMatrix::HostMatrix<type_t> ff_input = generate_fourier_features<type_t>(img.xtrain, ff, scale);

   const int fin = ff_input.ncols();
   const int fout = img.ytrain.ncols();
   const int train_size = ff_input.nrows();

   NumericMatrix::Matrix<type_t, HW> xtrain(train_size, fin, &p);
   NumericMatrix::Matrix<type_t, HW> ytrain(train_size, fout, &p);

   NumericMatrix::get_from_host(xtrain, ff_input);
   NumericMatrix::get_from_host(ytrain, img.ytrain);

   std::vector<int> arch{ (int)neurons, (int)neurons, fout};
   TINYAI::NeuralNetwork<type_t, HW, ACTIVATION::RELU> nn(arch, &p, xtrain, ytrain, batchsize);

   auto V = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < max_epochs; i++) {

      auto l = nn.train(batchsize, 1e-4);
      if (i % 1 == 0) {
         printf("Loss at epoch %zu: %f\n", i, l);
      }
   }
   auto Y = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = Y-V;
   std::cout <<" Tiny training took "<< duration.count() << " seconds";
   nn.evaluate(xtrain, ytrain);
   cudaDeviceSynchronize();
   NumericMatrix::export_to_host(ytrain,img.ytrain);
}
