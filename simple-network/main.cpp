#include "src/blob.h"
#include "src/mnist.h"
#include "src/network.h"

#include <iostream>

int main(int argc, char *argv[]) {
  // cudl::Blob<double> *blob_test = new cudl::Blob<double>(1, 1, 1, 1);
  // std::cout << "blob_test buf_size is " << blob_test->buf_size() <<
  // std::endl;
  // delete blob_test;

  int batch_size_train = 256;
  int num_steps_train = 1600;
  int monitoring_step = 200;

  double learning_rate = 0.02f;
  double lr_decay = 0.00005f;

  cudl::MNIST train_dataloader = cudl::MNIST("./dataset");
  train_dataloader.train(batch_size_train, true);

  cudl::Network model;
  model.add_layer(new cudl::Dense("dense", 10));
  model.add_layer(new cudl::Softmax("softmax"));
  model.cuda();
  
  model.train();

  int step = 0;
  cudl::Blob<float> *train_data = train_dataloader.get_data();
  cudl::Blob<float> *train_target = train_dataloader.get_target();
  train_dataloader.get_batch();
  // int tp_count = 0;
  while (step < num_steps_train) {
    train_data->to(cudl::cuda);
    train_target->to(cudl::cuda);
    
    model.forward(train_data);
    // tp_count += model.get_accuracy(train_target);

    model.backward(train_target);

    learning_rate *= 1.f / (1.f + lr_decay * step);
    model.update(learning_rate);

    step = train_dataloader.next();
  }

  // std::cout << train_data->n() << " " << train_data->c() << " "
  //           << train_data->h() << " " << train_data->w() << std::endl;
  return 0;
}
