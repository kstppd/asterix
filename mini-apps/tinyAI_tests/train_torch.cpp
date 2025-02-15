#include "moving_image.h"
#include "train.h"
#include <chrono>
#include <driver_types.h>
#include <torch/torch.h>

class MLP : public torch::nn::Module {
public:
   MLP(int input_size, int hidden_size, int output_size) {
      fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
      fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
      fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
   }

   torch::Tensor forward(torch::Tensor x) {
      x = torch::relu(fc1->forward(x));
      x = torch::relu(fc2->forward(x));
      x = fc3->forward(x);
      return x;
   }

private:
   torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

double learn(MovingImage& img, std::size_t max_epochs, std::size_t batchsize, std::size_t neurons, std::size_t ff,
           type_t scale, type_t lr) {

   NumericMatrix::HostMatrix<type_t> ff_input = generate_fourier_features<type_t>(img.xtrain, ff, scale);

   const int fin = ff_input.ncols();
   const int fout = img.ytrain.ncols();
   const int train_size = ff_input.nrows();

   type_t* datax;
   type_t* datay;
   cudaMalloc(&datax, ff_input.size() * sizeof(type_t));
   cudaMalloc(&datay, img.ytrain.size() * sizeof(type_t));
   cudaMemcpy(datax, ff_input.data(), ff_input.size() * sizeof(type_t), cudaMemcpyHostToDevice);
   cudaMemcpy(datay, img.ytrain.data(), img.ytrain.size() * sizeof(type_t), cudaMemcpyHostToDevice);

   auto xtrain =
       torch::from_blob(datax, {train_size, fin}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

   auto ytrain =
       torch::from_blob(datay, {train_size, fout}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

   auto model = std::make_shared<MLP>(fin, neurons, fout);
   model->to(torch::kCUDA, torch::kFloat);
   auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
   auto optimizer = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(lr).weight_decay(1e-4));

   model->train();
   const int batch_size = batchsize;
   auto start = std::chrono::high_resolution_clock::now();
   for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
      type_t epoch_loss = 0.0;
      auto indices = torch::randperm(train_size, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

      for (int i = 0; i < train_size; i += batch_size) {
         auto batch_indices = indices.slice(0, i, std::min(i + batch_size, train_size));
         auto batch_inputs = xtrain.index_select(0, batch_indices);
         auto batch_targets = ytrain.index_select(0, batch_indices);
         auto outputs = model->forward(batch_inputs);
         auto loss = criterion(outputs, batch_targets);
         optimizer.zero_grad();
         loss.backward();
         optimizer.step();
         epoch_loss += loss.item<type_t>();
      }

      epoch_loss /= static_cast<type_t>(train_size) / batch_size;
      printf("[TORCH]: [%zu , %f]\n", epoch, epoch_loss);

   }
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;

   model->eval();
   torch::Tensor predictions = model->forward(xtrain);
   predictions = predictions.to(torch::kCPU).contiguous();
   std::memcpy(img.ytrain.data(), predictions.data_ptr<type_t>(), predictions.numel() * sizeof(type_t));

   cudaFree(datax);
   cudaFree(datay);
   return duration.count();
}
