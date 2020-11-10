#include "src/blob.h"
#include "src/mnist.h"

#include <iostream>

int main(int argc, char *argv[]) {
  // cudl::Blob<double> *blob_test = new cudl::Blob<double>(1, 1, 1, 1);
  // std::cout << "blob_test buf_size is " << blob_test->buf_size() <<
  // std::endl;
  // delete blob_test;

  int batch_size_train = 10;
  cudl::MNIST train_dataloader = cudl::MNIST("./dataset");
  train_dataloader.train(batch_size_train);
  cudl::Blob<float> *train_data = train_dataloader.get_data();
  cudl::Blob<float> *train_target = train_dataloader.get_target();
  train_dataloader.get_batch();
  std::cout << train_data->n() << " " << train_data->c() << " "
            << train_data->h() << " " << train_data->w() << std::endl;
  return 0;
}
