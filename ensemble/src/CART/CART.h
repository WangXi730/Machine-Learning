#include"../base.h"
#include<vector>

namespace wx{
namespace regression{
class DecisionTree{
public:
    DecisionTree();
    virtual int train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power);
    virtual int test(const torch::Tensor& sample, float& result);
    virtual int set_error(const float error);
    struct TreeNode{
        TreeNode(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power, std::vector<int>& exist_point);
        float lim_point;
        int lim_dim;
        TreeNode* left_child = nullptr;
        TreeNode* right_child = nullptr;
    }; 
private:
    TreeNode* root;
};
}


}

