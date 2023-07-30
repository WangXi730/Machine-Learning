#include"base.h"

int wx::regression::base_algorithm::train(const torch::Tensor& samples, const torch::Tensor& labels){
    float n = samples.sizes()[0];
    torch::Tensor samples_power = torch::zeros(n);
    for(int i=0;i<n;++i){
        samples_power[i] = 1.0/n;
    }
    return train(samples,labels,samples_power);
}
int wx::regression::base_algorithm::set_error(const float error){
    error_ = error;
    return WX_SUCCESS;
}
int wx::regression::base_algorithm::get_error(float& error){
    error = error_;
    return WX_SUCCESS;
}
wx::regression::base_algorithm::~base_algorithm(){}

