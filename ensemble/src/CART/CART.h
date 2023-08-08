#pragma once
#include"../base.h"
#include<vector>

namespace wx{
namespace regression{
class DecisionTree : public base_algorithm{
public:
    DecisionTree();
    ~DecisionTree();
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power)override;
    virtual int test(const torch::Tensor& sample, float& result)override;
    virtual int cut_Alpha(float& Alpha);
    struct TreeNode{
        TreeNode(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power, std::vector<int>& exist_point, int attr_size, float error);
        ~TreeNode();
        //计算节点属性：回归值、不纯度，以及可以计算样本数量（参数sample_count < 0时计算，否则视为传入参数），范围：[left,right)
        static void get_node_val(const torch::Tensor& sample_power, std::vector<int>&exist_point, int left, int right, float& regression_value, float& impurity, float& sample_count);
        //分裂点
        float lim_point_ = 0;
        //分裂特征
        int lim_dim_ = 0;
        //子节点
        TreeNode* left_child_ = nullptr;
        TreeNode* right_child_ = nullptr;
        //样本数量(受到样本权重影响)
        float sample_count_;
        //回归值
        float regression_value_;
        //不纯度（方差）
        float impurity_;
        //深度
        int deep_;
    }; 
private:
    TreeNode* root_ = nullptr;
};
class DecisionTree_factory : public base_factory{
public:
    virtual DecisionTree* create_alg() override;
};
}
namespace classification{
class DecisionTree : public base_algorithm{
public:
    DecisionTree();
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power)override;
    virtual int test(const torch::Tensor& sample, int& result)override;
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
    virtual DecisionTree* create_alg() override;
};
}



extern regression::DecisionTree_factory rtf;//regression tree factory
}

