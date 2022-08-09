#pragma once
#include<iostream>
#include"Decision_Tree.hpp"
namespace wx {
	//模板参数：weak_type弱分类器，N代表是N分类模型
	template<size_t N, class weak_type = Decision_Tree<N,int,int>>
	class AdaBoost {
	public:
		//样本集合sample，标签lable，每种属性的种类数量，以及相关参数k（弱分类器的迭代次数或精度，换言之：弱分类器究竟有多弱？）,e强分类器最终的误差允许
		AdaBoost(std::vector<std::vector<int>>& sample, std::vector<int>& lable, std::vector<int>& attr_size, int k, double e)
			:_sample(sample), _lable(lable), _attr_size(attr_size), _k(k), _e(e)
		{
			m = attr_size.size();
			n = lable.size();
			_k = _k > m ? (m+1) : _k;
		}
		//深拷贝
		AdaBoost(const AdaBoost<N,weak_type>& cp)
			:_sample(cp._sample), _lable(cp._lable), _attr_size(cp._attr_size), _k(cp._k), _e(cp._e)
		{
			m = _attr_size.size();
			n = _lable.size();
			_k = _k > m ? (m + 1) : _k;
		}
		//接下来开始训练
		void Train() {
			//1、初始化权值
			std::vector<double> _power(n, 1.0 / (double)n);
			while (1) {
				//创建this_time_error
				std::unordered_set<int> this_time_error;
				//2
				//(a)创造一个弱分类器，然后进行训练
				weak_type Gm((std::vector<std::vector<int>>&)_sample, _lable, _attr_size, _k, _power);
				Gm.Train();
				//(b)计算Gm的分类误差率
				double em = 0.0;
				for (auto& e : Gm.Get_error()) {
					em += _power[e];
				}
				//(c)计算Gm的系数
				double am = 0.5 * log2((1 - em) / em);
				//(d)更新训练数据集的权值分布
				//先求出泛化因子
				double Zm = 0.0;
				//空间换时间，提升效率
				std::vector<double> Gmxi(n, 0);
				for (int i = 0; i < n; ++i) {
					Gmxi[i] = Gm.Test(_sample[i]);
					double yi = 0;//定义符号函数yi
					if (_lable[i] == Gmxi[i]) {
						yi = 1.0;//分类正确
					}
					else {
						yi = -1.0;
					}
					Zm += _power[i] * exp(-am * yi);
				}
				std::vector<double> D(n, 0.0);
				for (int i = 0; i < n; ++i) {
					double yi = 0;//定义符号函数yi
					if (_lable[i] == Gmxi[i]) {
						yi = 1.0;
					}
					else {
						yi = -1.0;
					}
					D[i] = _power[i] / Zm * exp(-am * yi);
				}
				//更新权值
				_power = D;
				_weak_power.push_back(am);
				//3，把Gm加入强分类器中，试试效果，并得到误分类率，判断循环是否退出
				_strong_algorithm.push_back(Gm);
				for (int i = 0; i < n; ++i) {
					int t = Test(_sample[i]);   ///
					if (t == _lable[i]) {
						//分类正确，可以走了
					}
					else {
						//加入错误列表
						this_time_error.insert(i);
					}
				}
				double e = ((double)this_time_error.size() / (double)n);
				if (e <= _e) {
					//创建error_list并退出
					error_list = this_time_error;
					break;
				}
			}
		}
		//对给定数据进行分类
		int Test(std::vector<int> t) {
			if (t.size() != m) {
				throw m;
			}
			//每一个弱分类器都进行分类并投票
			std::vector<double> results(m, 0.0);
			for (int i = 0; i < _strong_algorithm.size(); ++i) {
				results[_strong_algorithm[i].Test(t)] += _weak_power[i];
			}
			//最终选择投票最多的选项
			int result = 0;
			double mm = results[0];
			for (int i = 0; i < m; ++i) {
				if (mm < results[i]) {
					result = i;
					mm = results[i];
				}
			}
			return result;
		}
	private:
		std::vector<weak_type> _strong_algorithm;//最终的强分类器
		std::vector<double> _weak_power;//强分类器中，各个弱分类器的权重
		std::vector<std::vector<int>> _sample;//样本矩阵
		std::vector<int> _lable;//样本标签向量
		std::vector<int> _attr_size;//属性种类数量
		int _k = 0;//k次截止
		double _e = 0;//误差小于e截止
		int m; //属性数量
		int n; //样本数量
		std::unordered_set<int> error_list;	//错误列表
	};
}
