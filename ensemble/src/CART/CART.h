#pragma once
#include"../base.h"
#include<vector>

namespace wx{
namespace regression{
class DecisionTree : public base_algorithm{
public:
    DecisionTree();
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power)override;
    virtual int test(const torch::Tensor& sample, float& result)override;
    struct TreeNode{
        TreeNode(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power, std::vector<int>& exist_point, int attr_size, float error);
        float lim_point = 0;
        int lim_dim = 0;
        TreeNode* left_child = nullptr;
        TreeNode* right_child = nullptr;
    }; 
private:
    TreeNode* root = nullptr;
};
class DecisionTree_factory : public base_factory{
public:
    DecisionTree* create_alg();
};
}

extern regression::DecisionTree_factory rtf;//regression tree factory
}

