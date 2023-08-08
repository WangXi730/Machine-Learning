#include"CART.h"
namespace wx{
namespace regression{
    DecisionTree::DecisionTree(){
        
    }
    DecisionTree::~DecisionTree(){
        delete root;
    }
    int DecisionTree::train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power){
        //检验标准值的数量是否与样本相同
        auto shape = samples.sizes();
        if(shape[0] != labels.sizes()[0]){
            return EXCEPTION_NULL;
        }
        else if(shape.size() != 2){
            return EXCEPTION_SAMPLES_SHAPE;
        }
        else if(labels.sizes().size() != 1){
            return EXCEPTION_LABELS_SHAPE;
        }
        //构造树，根结点具有所有数据，所以构造一个全部下标都在的向量
        std::vector<int> exist_point(shape[0]);
        for(int i=0;i<shape[0];++i){
            exist_point[i] = i;
        }
        if(root != nullptr){
            delete root;
            root = nullptr;
        }
        float error = 0.0;
        get_error(error);
        root = new TreeNode(samples,labels,samples_power,exist_point, shape[1], error);
        
        return WX_SUCCESS;
    }
    int DecisionTree::test(const torch::Tensor& sample, float& result){
        if(root == nullptr){
            return EXCEPTION_MODE_NULL;
        }
        TreeNode* node = root;
        while(node->left_child != nullptr){
            float val = sample.index({node->lim_dim}).item<float>();
            if(val >= node->lim_point){
                node = node->right_child;
            }
            else{
                node = node->left_child;
            }
        }
        result = node->regression_value;
        return WX_SUCCESS;
    }
    DecisionTree::TreeNode::~TreeNode(){
        if(left_child != nullptr){
            delete left_child;
            delete right_child;
        }
    }
    static void DecisionTree::TreeNode::get_node_val(const torch::Tensor& samples_power, std::vector<int>&exist_point, int left, int right, float& regression_value, float& impurity, float& sample_count){
        //计算样本数量
        if(sample_count < 0){
            sample_count = 0.0;
            for(int i=left;i<right;++i){
                point = exist_point[i];
                sample_count += samples_power.index({point}).item<float>();
            }
        }
        //计算样本数量以及这个点的回归值
        regression_value = 0.0;
        for(int i=left;i<right;++i){
            point = exist_point[i];
            regression_value += labels.index({point}).item<float>() * samples_power.index({point}).item<float>();
        }
        regression_value /= sample_count;
        //计算不纯度(方差)
        impurity = 0.0;
        for(int i=left;i<right;++i){
            int point = exist_point[i];
            impurity += std::pow(labels.index({point}).item<float>() - regression_value, 2) * samples_power.index({point}).item<float>();
        }
        impurity /= sample_count;
    }
        
    DecisionTree::TreeNode::TreeNode(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power, std::vector<int>& exist_point, int attr_size, float error, int deep){
        //递归结束条件
        if(error < 0){
            return;
        }

        //遍历所有属性，并计算一些属性
        float error_left = -1;
        float error_right = -1;
        float regression_value_left = -1;
        float regression_value_right = -1;
        int lim_dim = -1;
        for(int attribute=0;attribute<attr_size;++attribute){
            //按照这个属性的值，对exit_point进行排序
            std::sort(exist_point.begin(),exist_point.end(),
                [=](int a, int b){
                    return samples[a][attribute] < samples[b][attribute];
                });
            float sample_count_left = -1;
            float sample_count_right = -1;
            //遍历这个属性可能存在的所有切分点
            for(int i = 1;i < exist_point.size() - 1; ++i){
                int lim = exist_point[i];
                //计算方差和回归值
                float left_regression_value_tmp;
                float right_regression_value_tmp;
                float left_error_tmp;
                float right_error_tmp;
                get_node_val(samples_power,exist_point,0,lim,left_regression_value_tmp,left_error_tmp,sample_count_left);
                get_node_val(samples_power,exist_point,lim,exist_point.size(),right_regression_value_tmp,right_error_tmp,sample_count_right);
                //迭代sample_count_left和sample_count_right
                sample_count_left += samples_power.index({lim}).item<float>();
                sample_count_right -= samples_power.index({lim}).item<float>();
            }
        }
        //分配子节点
        std::vector<int> exist_left;
        std::vector<int> exist_right;
        for(int& point:exist_point){
            if(samples.index({point,lim_dim}).item<float>() >= lim_point){
                //划分为r
                exist_right.emplace_back(point);
            }
            else{
                //划分为l
                exist_left.emplace_back(point);
            }
        }
        left_child = new TreeNode(samples,labels,samples_power,exist_left,attr_size,error);
        right_child = new TreeNode(samples,labels,samples_power,exist_right,attr_size,error);
    }
    DecisionTree* DecisionTree_factory::create_alg() {
        return new DecisionTree();
    }
}

regression::DecisionTree_factory rtf;
}




