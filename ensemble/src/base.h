#pragma once
#include<torch/torch.h>
#include"types.h"

namespace wx{

namespace regression{
class base_algorithm{
public:
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels);
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power) = 0;
    virtual int test(const torch::Tensor& sample, float& result) = 0;
    virtual int set_error(const float error);
    virtual int get_error(float& error);
private:
    float error_;
};
virtual int base_algorithm::train(const torch::Tensor& samples, const torch::Tensor& labels){
    float n = samples.sizes()[0];
    torch::Tensor samples_power = torch::zeros(n);
    for(int i=0;i<n;++i){
        samples_power[i] = 1.0/n;
    }
    return train(samples,labels,samples_power);
}
virtual int base_algorithm::set_error(const float error){
    error_ = error;
    return WX_SUCCESS;
}
virtual int base_algorithm::get_error(float& error){
    error = error_;
    return WX_SUCCESS;
}

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
