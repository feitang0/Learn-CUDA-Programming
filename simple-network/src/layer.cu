#include "layer.h"

#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace cudl;

Layer::Layer() {}

Layer::~Layer() {
  if (output_ != nullptr) {
    delete output_;
    output_ = nullptr;
  }
  if (grad_input_ != nullptr) {
    delete grad_input_;
    grad_input_ = nullptr;
  }

  if (weights_ != nullptr) {
    delete weights_;
    weights_ = nullptr;
  }
  if (biases_ != nullptr) {
    delete biases_;
    biases_ = nullptr;
  }
  if (grad_weights_ != nullptr) {
    delete grad_weights_;
    grad_weights_ = nullptr;
  }
  if (grad_biases_ != nullptr) {
    delete grad_biases_;
    grad_biases_ = nullptr;
  }
}

void Layer::init_weight_bias(unsigned int seed) {
  checkCudaErrors(cudaDeviceSynchronize());

  if (weights_ == nullptr || biases_ == nullptr) {
    return;
  }

  std::random_device rd;
  std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

  float range = sqrt(6.f / input_->size());
  std::uniform_real_distribution<> dis(-range, range);

  for (int i = 0; i < weights_->len(); i++) {
    weights_->ptr()[i] = static_cast<float>(dis(gen));
  }
  for (int i = 0; i < biases_->len(); i++) {
    biases_->ptr()[i] = 0.f;
  }

  weights_->to(DeviceType::cuda);
  biases_->to(DeviceType::cuda);

  std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}

void Layer::update_weights_biases(float learning_rate) {
  float eps = -1.f * learning_rate;
  if (weights_ != nullptr && grad_weights_ != nullptr) {
    checkCublasErrors(cublasSaxpy(cuda_->cublas(), weights_->len(), &eps,
                                  grad_weights_->cuda(), 1, weights_->cuda(),
                                  1));
  }

  if (biases_ != nullptr && grad_weights_ != nullptr) {
    checkCublasErrors(cublasSaxpy(cuda_->cublas(), biases_->len(), &eps,
                                  grad_biases_->cuda(), 1, biases_->cuda(), 1));
  }
}

float Layer::get_loss(Blob<float> *target) {
  assert("No Loss layer has no loss." && false);
  return EXIT_FAILURE;
}

int Layer::get_accuracy(Blob<float> *target) {
  assert("No Loss layer cannot estimate accuracy." && false);
  return EXIT_FAILURE;
}

Dense::Dense(std::string name, int output_size) {
  name_ = name;
  output_size_ = output_size;
}

Dense::~Dense() {
  if (d_one_vec != nullptr) {
    cudaFree(d_one_vec);
    d_one_vec = nullptr;
  }
}

__global__ void init_one_vec(float *d_one_vec, size_t length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= length)
    return;

  d_one_vec[i] = 1.f;
}

void Dense::fwd_initialize(Blob<float> *input) {
  if (weights_ == nullptr) {
    input_size_ = input->c() * input->h() * input->w();

    weights_ = new Blob<float>(1, 1, input_size_, output_size_);
    biases_ = new Blob<float>(1, 1, output_size_);
  }

  if (input_ == nullptr || batch_size_ != input->n()) {
    input_ = input;
    batch_size_ = input->n();

    if (output_ == nullptr) {
      output_ = new Blob<float>(batch_size_, output_size_);
    } else {
      output_->reset(batch_size_, output_size_);
    }

    output_->tensor();

    if (d_one_vec != nullptr) {
      cudaFree(d_one_vec);
    }
    checkCudaErrors(
        cudaMalloc((void **)&d_one_vec, sizeof(float) * batch_size_));
    init_one_vec<<<(batch_size_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D,
                   BLOCK_DIM_1D>>>(d_one_vec, batch_size_);

    // if (load_pretrain_ && !freeze_) {
    //   if (load_parameter()) {
    //     std::cout << "error occurred.." << std::endl;
    //     exit(-1);
    //   }
    // } else if (!freeze_) {
    //   init_weight_bias();
    // } else {
    // }
    if (!freeze_) {
      init_weight_bias();
    }
  }
}

Blob<float> *Dense::forward(Blob<float> *input) {
  checkCublasErrors(cublasSgemm(
      cuda_->cublas(), CUBLAS_OP_T, CUBLAS_OP_N, output_size_, batch_size_,
      input_size_, &cuda_->one, weights_->cuda(), input_size_, input_->cuda(),
      input_size_, &cuda_->zero, output_->cuda(), output_size_));
  checkCublasErrors(cublasSgemm(cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
                                output_size_, batch_size_, 1, &cuda_->one,
                                biases_->cuda(), output_size_, d_one_vec, 1,
                                &cuda_->one, output_->cuda(), output_size_));
  return output_;
}

void Dense::bwd_initialize(Blob<float> *grad_output) {
  if (grad_weights_ == nullptr) {
    grad_weights_ = new Blob<float>(weights_->shape());
    grad_biases_ = new Blob<float>(biases_->shape());
  }

  if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
    grad_output_ = grad_output;

    if (grad_input_ == nullptr) {
      grad_input_ = new Blob<float>(input_->shape());
    } else {
      grad_input_->reset(input_->shape());
    }
  }
}

Blob<float> *Dense::backward(Blob<float> *grad_output) {
  cublasSgemv(cuda_->cublas(), CUBLAS_OP_N, output_size_, batch_size_,
              &cuda_->one, grad_output_->cuda(), output_size_, d_one_vec, 1,
              &cuda_->zero, grad_biases_->cuda(), 1);
  cublasSgemm(cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_T, input_size_,
              output_size_, batch_size_, &cuda_->one, input_->cuda(),
              input_size_, grad_output_->cuda(), output_size_, &cuda_->zero,
              grad_weights_->cuda(), input_size_);
  if (!gradient_stop_) {
    cublasSgemm(cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_N, input_size_,
                batch_size_, output_size_, &cuda_->one, weights_->cuda(),
                input_size_, grad_output_->cuda(), output_size_, &cuda_->zero,
                grad_input_->cuda(), input_size_);
  }
  return grad_input_;
}

Softmax::Softmax(std::string name) { name_ = name; }

Softmax::~Softmax() {}

void Softmax::fwd_initialize(Blob<float> *input) {
  if (input_ == nullptr || batch_size_ != input->n()) {
    input_ = input;
    input_desc_ = input->tensor();
    batch_size_ = input->n();

    if (output_ == nullptr) {
      output_ = new Blob<float>(input->shape());
    } else {
      output_->reset(input->shape());
    }

    output_desc_ = output_->tensor();
  }
}

Blob<float> *Softmax::forward(Blob<float> *input) {
  checkCudnnErrors(cudnnSoftmaxForward(cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE,
                                       CUDNN_SOFTMAX_MODE_CHANNEL, &cuda_->one,
                                       input_desc_, input->cuda(), &cuda_->zero,
                                       output_desc_, output_->cuda()));

  return output_;
}

void Softmax::bwd_initialize(Blob<float> *target) {
  if (grad_input_ == nullptr || batch_size_ != target->n()) {
    if (grad_input_ == nullptr) {
      grad_input_ = new Blob<float>(input_->shape());
    } else {
      grad_input_->reset(input_->shape());
    }
  }
}

Blob<float> *Softmax::backward(Blob<float> *target) {
  checkCudaErrors(cudaMemcpyAsync(grad_input_->cuda(), output_->cuda(),
                                  output_->buf_size(),
                                  cudaMemcpyDeviceToDevice));
  checkCublasErrors(cublasSaxpy(cuda_->cublas(), target->len(),
                                &cuda_->minus_one, target->cuda(), 1,
                                grad_input_->cuda(), 1));

  int grad_output_size = target->n() * target->n() * target->h() * target->w();
  float scale = 1.f / static_cast<float>(target->n());
  checkCublasErrors(cublasSscal(cuda_->cublas(), grad_output_size, &scale,
                                grad_input_->cuda(), 1));

  return grad_input_;
}

float Softmax::get_loss(Blob<float> *target) {
  return loss_.loss(output_, target);
}

int Softmax::get_accuracy(Blob<float> *target) {
  int batch_size = output_->n();
  int output_size = output_->size();

  assert(batch_size == target->n());
  assert(output_size == target->size());

  float *h_output, *h_target;
  int idx_output, idx_target;
  int hit_count = 0;

  h_output = output_->to(host);
  h_target = target->to(host);

  for (int b = 0; b < batch_size; b++) {
    idx_output = 0;
    idx_target = 0;

    for (int i = 0; i < 10; i++) {
      if (h_output[b * output_size + i] >
          h_output[b * output_size + idx_output]) {
        idx_output = i;
      }
      if (h_target[b * output_size + i] >
          h_target[b * output_size + idx_target]) {
        idx_target = i;
      }

      if (idx_output == idx_target) {
        hit_count++;
      }
    }
  }

  return hit_count;
}
