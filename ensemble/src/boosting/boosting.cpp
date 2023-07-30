#include"boosting.h"

namespace wx{
namespace regression{
    Adaboost::Adaboost(base_factory* factory){
        factory_ = factory;
    }
    Adaboost::~Adaboost(){
        for(int i=0;i<weak_.size();++i){
            delete weak_[i].first;
        }
    }
 
    int Adaboost::train(const torch::Tensor& samples, const torch::Tensor& labels, const torch::Tensor& samples_power){
        auto shape = samples.sizes();
        if(samples_power.sizes()[0] != shape[0]){
            return EXCEPTION_MEMORY_OVERRUN;
        }
        while(1){
            //step
            //1 创建弱分类器，并且进行训练
            base_algorithm* weaki = factory_->create_alg();
            weaki->train(samples,labels,samples_power);
            //2 计算弱模型的误差，从而计算出这个弱模型的权重
            // 计算 eti[i] 的相对误差
            float Et = 0.0;
            std::vector<float> eti(shape[0],0.0);
            for(int i=0;i<shape[0];++i){
                float ht = 0.0;
                weaki->test(samples[i],ht);
                eti[i] = (labels[i] - ht) ^ (labels[i] - ht);
                Et = max(Et,eti[i]);
            }
            for(int i=0;i<shape[0];++i){
                eti[i] /= Et;
            }
            //计算弱模型的误差
            float et = 0.0;
            for(int i=0;i<shape[0];++i){
                et += samples_power[i] * eti[i];
            }
            //计算权重
            float weaki_power = et / (1-et);
            weak_.emplace_back(std::make_pair(weaki,weaki_power));
            //3 更新样本的权重
            // 计算 normalized factor 记为 Zt
            float Zt = 0.0;
            for(int i=0;i<shape[0];++i){
                Zt += samples_power[i] * std::pow(weaki_power,1-eti[i]);
            }
            //4 测试现有的回归模型
            float error = 0.0;
            for(int i=0;i<weak_.size();++i){
                float res  = 0.0;
                test(samples[i],res);
                error += std::fabs(res - labels[i]);
            }
            float allow_error;
            get_error(allow_error);
            if(error < allow_error){
                break;
            }
        }


        return WX_SUCCESS;
    }
    int Adaboost::test(const torch::Tensor& sample, float& result){
        if(sample.sizes()[0] != weak_.size()){
            return EXCEPTION_MEMORY_OVERRUN;
        }
        for(int i=0;i<weak_.size();++i){
            float result_tmp = 0.0;
            weak_[i].first->test(sample,&result_tmp);
            result += result_tmp * weak_[i].second;
        }
        return WX_SUCCESS;
    }
    Adaboost_factory::Adaboost_factory(base_factory* factory){
        factory_ = factory;
        if(factory_ == nullptr){
            factory_ = rtf;
        }
    }
    Adaboost* Adaboost_factory::create_alg(){
        return new Adaboost(factory_);
    }
}
}

