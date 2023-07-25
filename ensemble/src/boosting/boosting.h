#pragma once
#include"../base.h"
#include<vector>
#include<algorithm>
#include<cmath>
#include<utility>

namespace wx{
namespace regression{

class Adaboost : public base_algorithm{
public:
    Adaboost(base_factory* factory);
    ~Adaboost();
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power);
    virtual int test(const torch::Tensor& sample, float& result);
    virtual int set_error(const float error);
private:
    std::vector<std::pair<base_algorithm*,float>> weak_;
    base_factory* factory_ = nullptr;
};

class Adaboost_factory : public base_factory{
public:
    Adaboost_factory(base_factory* factory = nullptr);
    virtual Adaboost* create_alg();
private:
    base_factory* factory_;
};
}
}

