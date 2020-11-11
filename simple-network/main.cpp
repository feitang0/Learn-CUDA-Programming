#include "src/blob.h"
#include "src/mnist.h"
#include "src/network.h"

#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{
  // cudl::Blob<double> *blob_test = new cudl::Blob<double>(1, 1, 1, 1);
  // std::cout << "blob_test buf_size is " << blob_test->buf_size() <<
  // std::endl;
  // delete blob_test;

  int batch_size_train = 256;
  int num_steps_train = 1600;
  int monitoring_step = 200;

  double learning_rate = 0.02f;
  double lr_decay = 0.00005f;

  int batch_size_test = 10;
  int num_steps_test = 1000;

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
  int tp_count = 0;
  while (step < num_steps_train)
  {
    train_data->to(cudl::cuda);
    train_target->to(cudl::cuda);

    model.forward(train_data);
    tp_count += model.get_accuracy(train_target);

    model.backward(train_target);

    learning_rate *= 1.f / (1.f + lr_decay * step);
    model.update(learning_rate);

    step = train_dataloader.next();

    if (step % monitoring_step == 0)
    {
      float loss = model.loss(train_target);
      float accuracy = 100.f * tp_count / monitoring_step / batch_size_train;

      std::cout << "step: " << std::right << std::setw(4) << step << ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << ", accuracy: " << accuracy << "%" << std::endl;
    }
  }

  cudl::MNIST test_data_loader = cudl::MNIST("./dataset");
  test_data_loader.test(batch_size_test);

  model.test();

  cudl::Blob<float> *test_data = test_data_loader.get_data();
  cudl::Blob<float> *test_target = test_data_loader.get_target();
  test_data_loader.get_target();
  step = 0;
  while (step < num_steps_test)
  {
    test_data->to(cudl::cuda);
    test_target->to(cudl::cuda);

    model.forward(test_data);
    step = test_data_loader.next();
  }

  // std::cout << train_data->n() << " " << train_data->c() << " "
  //           << train_data->h() << " " << train_data->w() << std::endl;
  return 0;
}
