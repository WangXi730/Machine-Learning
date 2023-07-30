#include"CART.h"
namespace wx{
namespace regression{
    DecisionTree::DecisionTree(){
        
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
        result = node->lim_point;
        return WX_SUCCESS;
    }
    DecisionTree::TreeNode::TreeNode(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power, std::vector<int>& exist_point, int attr_size, float error){
        //递归结束条件
        if(error < 0){
            //计算所有的结点的平均数，作为这个点的值
            float power_account = 0.0;
            for(int& point:exist_point){
                float label = labels.index({point}).item<float>();
                float sample_power = samples_power.index({point}).item<float>();
                lim_point += label * sample_power;
                power_account += sample_power;
            }
            lim_point /= power_account;
            return;
        }

        //遍历所有属性
        float node_error = -1;
        for(int attribute=0;attribute<attr_size;++attribute){
            //遍历这个属性可能存在的所有切分点
            for(int& lim:exist_point){
                //计算方差
                float tmp_error = 0.0;
                float meanr = 0.0;
                float meanl = 0.0;
                float sizer = 0;
                float sizel = 0;
                for(int& point : exist_point){
                    float val = samples.index({point,attribute}).item<float>();
                    float lim_val = samples.index({lim,attribute}).item<float>();
                    if(val >= lim_val){
                        //划分为r
                        sizer += samples_power.index({point}).item<float>();
                        meanr += labels.index({point}).item<float>() * samples_power.index({point}).item<float>();
                    }
                    else{
                        //划分为l
                        sizel += samples_power.index({point}).item<float>();
                        meanl += labels.index({point}).item<float>() * samples_power.index({point}).item<float>();
                    }
                }
                meanl /= sizel;
                meanr /= sizer;
                float power_account = 0.0;
                for(int& point : exist_point){
                    float val = samples.index({point,attribute}).item<float>();
                    float lim_val = samples.index({lim,attribute}).item<float>();
                    float sample_power = samples_power.index({point}).item<float>();
                    float label = labels.index({point}).item<float>();
                    if(val >= lim_val){
                        //划分为r
                        tmp_error += std::pow(label - meanr, 2) * sample_power;
                    }
                    else{
                        //划分为l
                        tmp_error += std::pow(label - meanl, 2) * sample_power;
                    }
                    power_account += sample_power;
                }
                tmp_error /= power_account;
                if(node_error == -1 || node_error < tmp_error){
                    node_error = tmp_error;
                    lim_dim = attribute;
                    lim_point = samples.index({lim,attribute}).item<float>();
                }
            }
        }

        //目前是已经把误差最小的切分点和切分维度找出来了，接下来开始判断是否符合误差吧
        if(error >= node_error){
            error = -1;
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




