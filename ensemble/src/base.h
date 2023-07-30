#pragma once
#include<torch/torch.h>
#include"types.h"

namespace wx{

namespace regression{
class base_algorithm{
public:
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels) final;
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power) = 0;
    virtual int test(const torch::Tensor& sample, float& result) = 0;
    virtual  int set_error(const float error) final;
    virtual int get_error(float& error) final;
    virtual ~base_algorithm();
private:
    float error_;
};
class base_factory{
public:
    virtual base_algorithm* create_alg() = 0;
};

}


namespace classification{
class base_algorithm{
public:
    virtual int train() = 0;
    virtual int test(const torch::Tensor& sample, int& result) = 0;
    virtual int set_error(const float error);
private:

};

class base_factory{
public:
    virtual base_algorithm* create_alg() = 0;
};
}

}
