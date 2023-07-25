#pragma once
#include<iostream>
#include"Decision_Tree.hpp"
namespace wx {
	//ģ�������weak_type����������N������N����ģ��
	template<size_t N, class weak_type = Decision_Tree<N,int,int>>
	class AdaBoost {
	public:
		//��������sample����ǩlable��ÿ�����Ե������������Լ���ز���k�����������ĵ��������򾫶ȣ�����֮���������������ж�������,eǿ���������յ��������
		AdaBoost(std::vector<std::vector<int>>& sample, std::vector<int>& lable, std::vector<int>& attr_size, int k, double e)
			:_sample(sample), _lable(lable), _attr_size(attr_size), _k(k), _e(e)
		{
			m = attr_size.size();
			n = lable.size();
			_k = _k > m ? (m+1) : _k;
		}
		//���
		AdaBoost(const AdaBoost<N,weak_type>& cp)
			:_sample(cp._sample), _lable(cp._lable), _attr_size(cp._attr_size), _k(cp._k), _e(cp._e)
		{
			m = _attr_size.size();
			n = _lable.size();
			_k = _k > m ? (m + 1) : _k;
		}
		//��������ʼѵ��
		void Train() {
			//1����ʼ��Ȩֵ
			std::vector<double> _power(n, 1.0 / (double)n);
			while (1) {
				//����this_time_error
				std::unordered_set<int> this_time_error;
				//2
				//(a)����һ������������Ȼ�����ѵ��
				weak_type Gm((std::vector<std::vector<int>>&)_sample, _lable, _attr_size, _k, _power);
				Gm.Train();
				//(b)����Gm�ķ��������
				double em = 0.0;
				for (auto& e : Gm.Get_error()) {
					em += _power[e];
				}
				//(c)����Gm��ϵ��
				double am = 0.5 * log2((1 - em) / em);
				//(d)����ѵ�����ݼ���Ȩֵ�ֲ�
				//�������������
				double Zm = 0.0;
				//�ռ任ʱ�䣬����Ч��
				std::vector<double> Gmxi(n, 0);
				for (int i = 0; i < n; ++i) {
					Gmxi[i] = Gm.Test(_sample[i]);
					double yi = 0;//������ź���yi
					if (_lable[i] == Gmxi[i]) {
						yi = 1.0;//������ȷ
					}
					else {
						yi = -1.0;
					}
					Zm += _power[i] * exp(-am * yi);
				}
				std::vector<double> D(n, 0.0);
				for (int i = 0; i < n; ++i) {
					double yi = 0;//������ź���yi
					if (_lable[i] == Gmxi[i]) {
						yi = 1.0;
					}
					else {
						yi = -1.0;
					}
					D[i] = _power[i] / Zm * exp(-am * yi);
				}
				//����Ȩֵ
				_power = D;
				_weak_power.push_back(am);
				//3����Gm����ǿ�������У�����Ч�������õ�������ʣ��ж�ѭ���Ƿ��˳�
				_strong_algorithm.push_back(Gm);
				for (int i = 0; i < n; ++i) {
					int t = Test(_sample[i]);   ///
					if (t == _lable[i]) {
						//������ȷ����������
					}
					else {
						//��������б�
						this_time_error.insert(i);
					}
				}
				double e = ((double)this_time_error.size() / (double)n);
				if (e <= _e) {
					//����error_list���˳�
					error_list = this_time_error;
					break;
				}
			}
		}
		//�Ը������ݽ��з���
		int Test(std::vector<int> t) {
			if (t.size() != m) {
				throw m;
			}
			//ÿһ���������������з��ಢͶƱ
			std::vector<double> results(m, 0.0);
			for (int i = 0; i < _strong_algorithm.size(); ++i) {
				results[_strong_algorithm[i].Test(t)] += _weak_power[i];
			}
			//����ѡ��ͶƱ����ѡ��
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
		std::vector<weak_type> _strong_algorithm;//���յ�ǿ������
		std::vector<double> _weak_power;//ǿ�������У���������������Ȩ��
		std::vector<std::vector<int>> _sample;//��������
		std::vector<int> _lable;//������ǩ����
		std::vector<int> _attr_size;//������������
		int _k = 0;//k�ν�ֹ
		double _e = 0;//���С��e��ֹ
		int m; //��������
		int n; //��������
		std::unordered_set<int> error_list;	//�����б�
	};
}
