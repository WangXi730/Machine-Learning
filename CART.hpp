#pragma once
//CART�㷨�Լ�����㷨
#include"Decision_Tree.hpp"

namespace wx {
	//�ع������
	template<class value_type>
	struct Regression_Tree_node : public Decision_Tree_node <value_type> {
		Regression_Tree_node() 
		:splitting_variable(this->this_attr) ,splitting_variable_pointval(this->attr) {}
		//���˾��������е������⣬����Ҫ������������
		std::unordered_map<int, double> Rm_cm;//���������ڵ�Ԫ���������ֵ��Rm_cm[i]��ʾ��i������
		std::unordered_map<int, value_type>& splitting_variable_pointval;//��������֮ǰ�Ѿ����ֵ��зֱ���(����)��Ӧ��ȡֵ
		std::pair<double,double> cm;//������ȷ�����з�����ֵ�����ڻع����Ƕ�����������ֱ����pair
		int& splitting_variable;//������ȷ�����зֱ���
		double splitting_point;//�������зֱ�����Ӧ���зֵ�
	};
	//�ع�����Ĭ��Ϊ��С���˻ع���
	template<size_t type_size = 0, class value_type = double, class lable_type = double, class strategy_t = square_error_strategy, class algorithm_t = least_squares_regression, class Tree_node = Regression_Tree_node<value_type>>
	class Regression_Tree : public  Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node> {
	public:
		Regression_Tree(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<double> power, int k)
			:Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>(sample, lable, std::move(std::vector<int>(sample[0].size(), 2)), k, power)
		{
			this->_k = k;
		}
		Regression_Tree(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<double> power, double e)
			:Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>(sample, lable, std::move(std::vector<int>(sample[0].size(), 2)), e, power)
		{}
		//�������캯��
		Regression_Tree(const Regression_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>& cp)
		{
			this->error_list = cp.error_list;
			this->func = cp.func;
			this->m = cp.m;
			this->n = cp.n;
			this->_attr_size = cp._attr_size;
			this->_kore = cp._kore;
			this->_lable = cp._lable;
			this->_power = cp._power;
			this->_sample = cp._sample;
		}
		
	protected:
		virtual void _train(Tree_node& node, Tree_node* pr, std::vector<int>& node_data, std::unordered_map<int, value_type> attr, int k, std::vector<int> used) {
			//�Խ���ʼ��
			node.parent = pr;//����㸸�����null
			node.data = node_data;//�ý����е�����
			node.attr = attr;//��ǰ�����е�����
			if (pr) {
				node.Rm_cm = pr->Rm_cm;//��ǰ���Ӧ���̳и������з�����
			//ͬʱ����������з�����Ӧ���ڸ����Ļ����ϣ����Ӹ����������Ӧ���Ե��з�
				if (this->_sample[node.data[0]][pr->splitting_variable] <= this->_sample[pr->splitting_point][pr->splitting_variable]) {
					//С���зֵ��ֵ����Ϊ����	
					node.Rm_cm[pr->splitting_variable] = pr->cm.first;
				}
				else {
					node.Rm_cm[pr->splitting_variable] = pr->cm.second;
				}
			}
			//�ݹ��������
			if (this->_kore) {
				//������e
				if (pr != nullptr && pr->entropy < this->_e) {
					std::unordered_map<double,double> tmp;//tmp���±��Ǳ�ǩ�������������ǩ��������ȡ������Ϊ����
					for (auto& i : node_data) {
						tmp[this->_lable[i]] += this->_power[i];
					}
					int t = 0;
					double mm = tmp[0];
					for (int i = 1; i < type_size; ++i) {
						if (mm < tmp[i]) {
							mm = tmp[i];
							t = i;
						}
					}
					node.this_attr = t;//ȷ����������
					node.entropy = 0.0;//����Ҷ�ӽ�㲻���з��࣬��������Ϊ0��
					//����error_list
					for (auto& e : node_data) {
						if (this->_lable[e] != node.this_attr) {
							node.error_list.insert(e);
							this->error_list.insert(e);
						}
					}
					this->leaf_node.insert(&node);
					return;
				}
			}
			//������k(ʣ��Ŀ�ʹ�����Ը���)
			if (k == 0) {
				std::unordered_map<double,double> tmp;
				for (auto& i : node_data) {
					tmp[this->_lable[i]] += this->_power[i];
				}
				int t = 0;
				double mm = tmp[0];
				for (int i = 1; i < type_size; ++i) {
					if (mm < tmp[i]) {
						mm = tmp[i];
						t = i;
					}
				}
				node.this_attr = t;
				//����error_list
				for (auto& e : node_data) {
					if (this->_lable[e] != node.this_attr) {
						node.error_list.insert(e);
						this->error_list.insert(e);
					}
				}
				this->leaf_node.insert(&node);
				return;
			}
			//�ߵ�����˵��û�н��������ü�����
			//����ÿһ��������أ�ѡ������
			double mm = -1;//�����mm��ʾ��С�������
			int sub = -1;//��С�ߵ��±�
			double sp = 0;//�зֵ�
			std::pair<double, double> cm;//�зֵ������Ĺ̶����ֵ
			for (int i = 0; i < this->m; ++i) {
				////����Ҫ��ȷ��һ�㣬������֮ǰ�ù�������
				//if (used[i] == 1)
				//	continue;
				double tmp_sp = 0;//��һ�ε��зֵ�
				std::pair<double, double> tmp_cm;//��һ�ε��������ֵ
				double tmp = this->func(this->_sample, this->_lable, type_size, i, node_data, this->_attr_size[i], this->_power, tmp_cm, tmp_sp);
				if (tmp < mm||mm == -1) {
					sub = i;
					mm = tmp;
					sp = tmp_sp;
					cm = tmp_cm;
				}
			}
			//ȷ����ǰ���ľ�������
			node.entropy = mm;
			node.this_attr = sub;
			node.splitting_point = sp;
			node.cm = cm;
			//�Ѿ�ʹ�ù�sub�����ˣ����������ӽ��Ͳ�������
			used[sub] = 1;

			//��ǰ�����ӽ��ָ��
			Tree_node* child = new Tree_node;
			//ȷ����һ�������е�����
			std::unordered_map<int, double> child_attr = attr;//�����������ϣ��ӽ�����Ÿ��������ͬ������
			child_attr[sub] = cm.first;//�ڵ�ǰ���ȷ���������ϣ��ӽ�������Ӧ��ÿ�����Զ���
			//��һ�������е�����
			std::vector<int> child_data;
			for (auto& e : node_data) {
				if (this->_sample[e][sub] <= this->_sample[sp][sub]) {
					child_data.push_back(e);
				}
			}
			//���뵽��ǰ������ָ���б�
			node.child.push_back(child);
			_train(*child, &node, child_data, child_attr, k - 1, used);

			child = new Tree_node;
			//ȷ����һ�������е�����
			child_attr[sub] = cm.second;//�ڵ�ǰ���ȷ���������ϣ��ӽ�������Ӧ��ÿ�����Զ���
			//��һ�������е�����
			child_data.clear();
			for (auto& e : node_data) {
				if (this->_sample[e][sub] > this->_sample[sp][sub]) {
					child_data.push_back(e);
				}
			}
			//���뵽��ǰ������ָ���б�
			node.child.push_back(child);
			_train(*child, &node, child_data, child_attr, k - 1, used);
		}
	};
	//��С����Ѱ�������з��㷨
	class least_squares_regression {
	public:
		double operator()(std::vector<std::vector<double>>& sample, std::vector<double>& lable, int type_size, int i, std::vector<int>& node_data, int attr_size, std::vector<double> power, std::pair<double,double>& cm, double& sp) {
			//����������зֱ���������ͨ������������̶����ֵ���зֵ�
			double result = -1;//������С��0�����Կ�����-1�����ֵ
			int ptr = -1;//��С���±����Ҹ�-1
			for (int k = 0; k < node_data.size(); ++k) {
				//ÿһ���зֵ㶼��һ�飬�ҵ���С�ģ�kΪ�зֵ���±꣬ƽ�����㷨
				double tmp_result = 0;
				double n1 = 0;//�� 1 ���� n1 ��Ԫ��
				double n2 = 0;//�� 2 ���� n2 ��Ԫ��
				double tmp_x1 = 0;//��һ���ܺ�
				double tmp_x2 = 0;//�ڶ����ܺ�
				for (int j = 0; j < node_data.size(); ++j) {
					if (sample[node_data[j]][i] <= sample[node_data[k]][i]) {
						//��һ���
						tmp_x1 += lable[node_data[j]] * power[node_data[j]];
						n1 += power[node_data[j]];
					}
					else {
						//�ڶ����
						tmp_x2 += lable[node_data[j]] * power[node_data[j]];
						n2 += power[node_data[j]];
					}
				}
				//���ֵ
				if (n1)
					tmp_x1 = tmp_x1 / n1;
				if (n2)
					tmp_x2 = tmp_x2 / n2;
				//����
				for (int j = 0; j < node_data.size(); ++j) {
					if (sample[node_data[j]][i] <= sample[node_data[k]][i]) {
						//��һ��
						tmp_result += (lable[node_data[j]] - tmp_x1) * (lable[node_data[j]] - tmp_x1) * power[node_data[j]];
					}
					else {
						//�ڶ���
						tmp_result += (lable[node_data[j]] - tmp_x2) * (lable[node_data[j]] - tmp_x2) * power[node_data[j]];
					}
				}
				if (tmp_result < result || result == -1) {
					result = tmp_result;
					ptr = k;
					cm.first = tmp_x1;
					cm.second = tmp_x2;
				}
			}
			//��ֵsp
			sp = node_data[ptr];
			//����result
			return result;
		}
	};
	//ƽ��������
	class square_error_strategy {
	public:
		//���������µ�����Ҷ�ӽ������ƽ����
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double operator()(Regression_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree, std::vector<Decision_Tree_node<lable_type>*>* leaf = nullptr) {
			//Ҫ��ÿ���ӽ������
			//�����Ҷ�ӣ�ֱ�Ӿ����ˣ������ݹ�
			if (node->is_leaf()) 
				return node_strategy<value_type,lable_type,type_size,strategy_t,algorithm_t,Tree_node>(node, tree);
			//������ǣ���ݹ�����ӽ��
			double result = 0;
			for (int i = 0; i < node->child.size(); ++i) {
				result += operator()(node, tree);
			}
			return result;
		}
		//�ṩ�ӿڣ����㵱ǰ�������ƽ����
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double node_strategy(Regression_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree) {
			//ͨ���������attr���ԣ�֪���������Ѿ����ֵ����ԣ�������
			//��ʵҲ����������
			//�������e
			double e = 0;
			//���������ϵ�ÿ�����ݶ�����ƽ����
			for (int i = 0; i < node->data.size(); ++i) {
				//i�������������
				if (tree->Get_sample()[node->data[i]][node->splitting_variable] <= node->splitting_point)
					e += (tree->Get_lable()[node->data[i]] - node->cm.first) * (tree->Get_lable()[node->data[i]] - node->cm.first) * tree->Get_power()[node->data[i]];
				else
					e += (tree->Get_lable()[node->data[i]] - node->cm.second) * (tree->Get_lable()[node->data[i]] - node->cm.second) * tree->Get_power()[node->data[i]];
			}
			//������������ɵ�Ԥ�����
			return e;
		}
	};
}