#include"CART.h"
namespace wx{
namespace regression{
    DecisionTree::DecisionTree(){
        
    }
    
    virtual int DecisionTree::train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power){
        //检验标准值的数量是否与样本相同
        auto shape = samples.sizes();
        if(shape[0] != labels.sizes()[0]){
            return EXCEPTION_NULL;
        }
        //构造树，根结点具有所有数据，所以构造一个全部下标都在的向量
        std::vector<int> exist_point(shape[0]);
        for(int i=0;i<shape[0];++i){
            exist_point[i] = i;
        }
        
    }
    virtual int DecisionTree::test(const torch::Tensor& sample, float& result){
        
    }
    virtual int DecisionTree::set_error(const float error){

    }

}


}




