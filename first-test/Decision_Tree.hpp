#pragma once
#include<iostream>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<math.h>
#include<memory>
#include<algorithm>
namespace wx {
	class Information_gain;
	class Information_gain_ratio;
	class Empirical_entropy_strategy;
	//�����
	template<class value_type>
	struct Decision_Tree_node {
		std::unordered_map<int, value_type> attr;//������ĵ�i���������������
		int this_attr;//������ȷ����������ԣ������Ҷ�ӽ�㣬��ȷ����������
		std::vector<int> data;//������ʣ����������
		std::vector<Decision_Tree_node<value_type>*> child;//�ӽ��ָ�룬�±꼴��������
		Decision_Tree_node<value_type>* parent;//�����ָ��
		double entropy = 0;//�ý����Ϣ����(��)
		std::unordered_set<int> error_list;//�����б�
		//���캯������
		Decision_Tree_node() {}
		//��һ��ȷ��Ҷ�ӽ��ķ���
		bool is_leaf() {
			return child.size() == 0;
		}
	};

	//ģ��������type_size����ǩ����������strategy_t������ѡ��,�������Ĳ��������ʾΪC(t) = Ca(t) - a * |t|��|t|��ʾ�������Ҷ�ӽ���������algorithm_t���㷨ѡ�񣬿�ѡ��Ϣ�������Ϣ����ȣ�Tree_node�����Ľ�����ͣ�һ�㽨��̳�Decision_Tree_node����
	template<size_t type_size, class value_type = int, class lable_type = int, class strategy_t = Empirical_entropy_strategy, class algorithm_t = Information_gain_ratio, class Tree_node = Decision_Tree_node<value_type>>
	class Decision_Tree {
	public:
		//���ɾ�������������
		// ��������m��n�У�m�����ԣ�n��������ÿһ����������һ��������vector<vector<algorithm_t::value_type>> sample��
		// ��ǩ������vector<algorithm_t::lable_type> lable��lable.size() == n��ÿ��������һ����
		// �������Ե���������������vector<int> attr_size, attr_size.size() == m��ÿ��������һ����
		// ���ȣ�double e/�������߶ȣ�int k
		// ����Ȩ��������ȱʡΪ 1 ��������vector<double> power��power.size() == n
		Decision_Tree(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<int>& attr_size, int k, std::vector<double> power)
			:_sample(sample), _lable(lable), _attr_size(attr_size), _k(k), _kore(0), root(nullptr)
		{
			m = attr_size.size();
			n = lable.size();
			_power = power;
			_k = _k > m+1 ? m+1 : _k;
		}
		Decision_Tree(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<int>& attr_size, double e, std::vector<double> power)
			:_sample(sample), _lable(lable), _attr_size(attr_size), _e(e), _kore(1), root(nullptr)
		{
			m = attr_size.size();
			n = lable.size();
			_power = power;
			_k = m+1;
		}
		Decision_Tree(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<int>&& attr_size, int k, std::vector<double> power)
			:_sample(sample), _lable(lable), _attr_size(std::move(attr_size)), _k(k), _kore(0), root(nullptr)
		{
			m = _attr_size.size();
			n = lable.size();
			_power = power;
			_k = _k > m+1 ? m+1 : _k;
		}
		Decision_Tree(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<int>&& attr_size, double e, std::vector<double> power)
			:_sample(sample), _lable(lable), _attr_size(std::move(attr_size)), _e(e), _kore(1), root(nullptr)
		{
			m = _attr_size.size();
			n = lable.size();
			_power = power;
			_k = m+1;
		}
		//�������캯��
		Decision_Tree(const Decision_Tree<type_size,value_type, lable_type, strategy_t,algorithm_t,Tree_node>& cp)
		{
			this->error_list = cp.error_list;
			this->func = cp.func;
			this->m = cp.m;
			this->n = cp.n;
			this->_attr_size = cp._attr_size;
			this->_e = cp._e;
			this->_k = cp._k;
			this->_kore = cp._kore;
			this->_lable = cp._lable;
			this->_power = cp._power;
			this->_sample = cp._sample;
		}
		//���������ͷſռ�
		virtual ~Decision_Tree()
		{}
		//�޸�k
		bool Re_k(int k) {
			_k = k;		//�޸�k��ֵ
			_kore = 0;	//�޸�Ϊk����
			return true;
		}
		//�޸�e
		bool Re_e(double e) {
			_e = e;
			_kore = 1;//ͬ��
			return true;
		}
		//�޸�Ȩ������
		bool Re_power(std::vector<double>& power) {
			if (power.size() == n) {
				_power = power;
				return true;
			}
			return false;
		}
		//�޸���������
		bool Re_samset(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<int>& attr_size) {
			_sample = sample;
			_lable = lable;
			_attr_size = attr_size;
			//˳��ѵ��һ��
			Train();
			return true;
		}
		//ѵ����
		void Train() {
			//������ԭ�������������ͻ
			_clear();
			//��ȡÿһ�����Ե���Ϣ����ѡ��������Ϊ�����
			root.reset(new Tree_node);
			std::vector<int> root_data(n);//��������rootӵ����������
			for (int i = 0; i < n; ++i) {
				root_data[i] = i;
			}
			std::unordered_map<int, value_type> mp;//����㲻�߱��κ�����
			std::vector<int> used(m, 0);//�Ѿ�ʹ�ù�������
			_train(*root, nullptr, root_data, mp, _k - 1, used);

		}
		//�Ը������ݽ��з���
		lable_type Test(std::vector<value_type> t) {
			if (root.get() == nullptr) {
				Train();
			}
			//Ҫ����һ��ά����ͬ������
			if (t.size() != m) {
				throw m;
			}
			Tree_node* ptr = root.get();
			while (!(ptr->is_leaf())) {
				ptr = ptr->child[t[ptr->this_attr]];
			}
			return ptr->this_attr;
		}
		//��ȡ�������ļ���
		std::unordered_set<int> Get_error() {
			return error_list;
		}
		//��ȡ�ຯ��
		std::vector<std::vector<value_type>> Get_sample() {
			return _sample;
		}
		std::vector<double> Get_power() {
			return _power;
		}
		std::vector<lable_type> Get_lable() {
			return _lable;
		}
		//��ȡ��ʧ����loss function��ֵ
		virtual double Loss_function(double a) {
			//��ʧ���� Ca(t) = C(t) + a * |t|
			return C.operator()(root.get(),this) + a * (double)leaf_node.size();
		}
		//��������֦
		void Pruning(double a) {
			int e = 1;
			while (e) {
				e = 0;
				//��ÿһ��Ҷ�ӵĸ������б�����ֱ�������ҳ��Ż�Ϊֹ
				for (int i = 0; i < leaf_node.size(); ++i) {
					//��¼�����
					Tree_node* pr = leaf_node[i]->parent;
					//�������µ�Ҷ�ӽ�������
					std::vector<Tree_node*> leafs;
					//���㵱ǰ����ʧ
					double now_lose = C(pr, this, &leafs) + a * (double)leaf_node.size();
					//�����֦֮�����ʧ
					double cut_lose = C.node_strategy(pr, this) + a * ((double)leaf_node.size() - (double)leafs.size() + 1);
					//�Ƚ�
					if (now_lose > cut_lose) {
						e = 1;
						//ȥ��Ҷ�ӽ�㼯���е��ӽ��
						for (int j = 0; j < leafs.size(); ++j) {
							auto tmp = leaf_node.find(leafs[j]);
							if (tmp != leaf_node.end()) {
								leaf_node.erase(tmp);
							}
						}
						//�����µ�Ҷ�ӽ��
						leaf_node.insert(pr);
						//�Ӵ洢�ϼ�֦
						pr.clear();
						break;
					}
				}
			}
		}
	protected:
		//������
		void _clear() {
			leaf_node.clear();
			std::unordered_set<int>(std::move(error_list));
			//��վ����������ͷſռ䣬����ʹ����vector����������ֻnew��һ���ڴ棬��root���ͷż���
			root.reset(nullptr);
		}
		virtual void _train(Tree_node& node, Tree_node* pr, std::vector<int>& node_data, std::unordered_map<int, value_type> attr, int k, std::vector<int> used) {
			//�Խ���ʼ��
			node.parent = pr;//����㸸�����null
			node.data = node_data;//�ý����е�����
			node.attr = attr;//��ǰ�����е�����
			//�ݹ��������
			if (_kore) {
				//������e
				if (pr!=nullptr && pr->entropy < _e) {
					std::vector<double> tmp(type_size, 0);//tmp���±��Ǳ�ǩ�������������ǩ��������ȡ������Ϊ����
					for (auto& i : node_data) {
						tmp[_lable[i]] += _power[i];
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
						if (_lable[e] != node.this_attr) {
							node.error_list.insert(e);
							error_list.insert(e);
						}
					}
					leaf_node.insert(&node);
					return;
				}
			}
			//������k(ʣ��Ŀ�ʹ�����Ը���)
			if (k == 0) {
				std::vector<double> tmp(type_size, 0);
				for (auto& i : node_data) {
					tmp[_lable[i]] += _power[i];
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
					if (_lable[e] != node.this_attr) {
						node.error_list.insert(e);
						error_list.insert(e);
					}
				}
				leaf_node.insert(&node);
				return;
			}
			//����֮�⣬�������Ҳ������Ϊ�˳��ݹ������
			int E = node_data[0];
			for (auto& i : node_data) {
				if (_lable[i] != _lable[E]) {
					break;
				}
				if (i == node_data[node_data.size() - 1]) {
					//֤�����еı�ǩ�����
					node.this_attr = _lable[E];
					node.entropy = 0;
					leaf_node.insert(&node);
					return;
				}
			}

			//�ߵ�����˵��û�н��������ü�����
			//����ÿһ��������أ�ѡ������
			double mm = 0.0;//������Σ���Ϣ���治����С��0��
			int sub = -1;//����ߵ��±�
			for (int i = 0; i < m; ++i) {
				//����Ҫ��ȷ��һ�㣬������֮ǰ�ù�������
				if (used[i] == 1)
					continue;
				std::pair<double, double> tmp_pair;
				double tmp_double = 0;
				double tmp = func(_sample, _lable, type_size, i, node_data, _attr_size[i], _power,tmp_pair, tmp_double);
				if (tmp > mm) {
					sub = i;
					mm = tmp;
				}
			}
			//ȷ����ǰ���ľ�������
			node.entropy = mm;
			node.this_attr = sub;
			//�Ѿ�ʹ�ù�sub�����ˣ����������ӽ��Ͳ�������
			used[sub] = 1;
			//��ǰ�����ӽ��ָ�룬��ɵݹ�
			for (int i = 0; i < _attr_size[sub]; ++i) {
				Tree_node* child = new Tree_node;
				//ȷ����һ�������е�����
				std::unordered_map<int, lable_type> child_attr = attr;//�����������ϣ��ӽ�����Ÿ��������ͬ������
				child_attr[sub] = i;//�ڵ�ǰ���ȷ���������ϣ��ӽ�������Ӧ��ÿ�����Զ���
				//��һ�������е�����
				std::vector<int> child_data;
				for (auto& e : node_data) {
					if (_sample[e][sub] == i) {
						child_data.push_back(e);
					}
				}
				//���뵽��ǰ������ָ���б�
				node.child.push_back(child);
				_train(*child, &node, child_data, child_attr, k - 1, used);
			}
		}
		std::vector<std::vector<value_type>> _sample;//��������
		std::vector<lable_type> _lable;//������ǩ����
		std::vector<int> _attr_size;//������������
		std::vector<double> _power;//����Ȩ������
		int _k = 0;//k���ֹ
		double _e = 0;//func(����i)С��e��ֹ
		int _kore = 0;//0������k��1������e
		typename strategy_t C;//���ԣ����ڼ���Ԥ�����
		typename algorithm_t func;//�㷨���Ƚ��صķº���
		int m; //��������
		int n; //��������
		std::unordered_set<int> error_list;	//�����б�
		std::auto_ptr<Tree_node> root;//�����
		std::unordered_set<Tree_node*> leaf_node;//Ϊ�˷��㣬��Ҷ�ӽ��洢����
	};


	//��Ϣ�����㷨
	class Information_gain {
	public:
		Information_gain(){}
		//���ݼ��У���i�����Ե���Ϣ���棬���У�H(D)����һ��ľ����أ�node_data��ʾ�����ʣ�����±꣬�����attr_size����ʾ��i�����Ե�������Ŀ
		double operator()(std::vector<std::vector<int>>& sample, std::vector<int>& lable, int type_size, int i, std::vector<int>& node_data, int attr_size, std::vector<double> power, std::pair<double, double> cm = std::pair<double, double>(), double sp = double()) {
			//����HD
			double HD = 0;
			std::vector<double> tmp(type_size);//��ǩ�����Ŀռ�
			for (auto& e : node_data) {
				tmp[lable[e]] += power[e];
			}
			//����D�Ĵ�С
			double D_count = 0.0;
			for (auto& e : power) {
				D_count += e;
			}
			for (auto& e : tmp) {
				if(e != 0)
					HD += -(e / D_count * log(e / D_count));
			}
			//����HDA
			std::vector<std::vector<int>> count(attr_size);
			for (auto& e : node_data) {
				count[sample[e][i]].push_back(e);//��e�������ĵ�i�����Ե���𣬴������������±�
			}
			//�������� i ��Ӧ���� node_data �ľ��������أ�Ȼ�����
			double HDA = 0;
			for (auto& Di : count) {
				double sum = 0;
				if (Di.size() == 0) {
					continue;//���������û�����ݣ���������
				}
				//���ÿ����ǩ����������
				std::unordered_map<int, double> mp;
				double Di_count = 0.0;
				for (auto& Dik : Di) {
					mp[lable[Dik]] += power[Dik];
					Di_count += power[Dik];
				}
				//���
				for (auto& k : mp) {
					if (k.second != 0) 
						sum += (k.second / Di_count * log(k.second / Di_count));
				}
				//���ܵ� HDA ����
				HDA += -(Di_count / D_count * sum);
			}
			return HD - HDA;//������Ϣ����
		}
	};
	//��Ϣ������㷨
	class Information_gain_ratio {
	public:	 
		Information_gain_ratio(){}
		double operator()(std::vector<std::vector<int>>& sample, std::vector<int>& lable, int type_size, int i, std::vector<int>& node_data, int attr_size, std::vector<double> power,std::pair<double, double> cm = std::pair<double,double>(), double sp = double()) {
			Information_gain f;
			//������Ϣ����gR(DA)
			double gRDA = f(sample, lable, type_size, i, node_data, attr_size, power);
			//����sample��������i����HA(D)
			double HAD = 0.0;
			std::vector<double> Di_count(attr_size, 0.0);
			double D_count = 0.0;
			for (auto& e : node_data) {
				Di_count[sample[e][i]] += power[e];
				D_count += power[e];
			}
			for (auto& e : Di_count) {
				if (e != 0)
					HAD += -(e / D_count * log(e / D_count));
			}
			//������Ϣ������ 
			return gRDA / HAD;
		}
	};

	//��Ϣ�ؼ���
	class Empirical_entropy_strategy {
	public:
		Empirical_entropy_strategy(){}
		//�������������е�Ҷ�ӽ�����
		//����һ���������ĸ����㣬����ʹ�ö�̬��Ӧ�Ա���
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double operator()(Decision_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree, std::vector<Decision_Tree_node<lable_type>*>* leaf = nullptr) {
			double result = 0;
			_func(node, result, tree, leaf);
			return result;
		}
		//�ṩ�ӿڣ����㵱ǰ������
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double node_strategy (Decision_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree) {
			double Nt = 0;
			double result = 0;
			std::vector<double> power = tree->Get_power();
			for (int i = 0; i < node->data.size(); ++i) {
				Nt += power[node->data[i]];
			}
			for (int i = 0; i < node->data.size(); ++i) {
				result -= power[node->data[i]] * log(power[node->data[i] / Nt]);
			}
			return result;
		}
	private:
		//�ڲ�����������Ϣ�أ���������㣬Ҷ�ӽ�㣬��
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		void _func(Decision_Tree_node<lable_type>* node, double& entropy, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree, std::vector<Decision_Tree_node<lable_type>*>* count) {
			//�ݹ��������
			if (node->is_leaf()) {
				entropy += node_strategy(node,tree);
				if (count != nullptr) {
					count->push_back(node);
				}
				return;
			}
			//�������Ҷ�ӽ�㣬�ͼ���
			for (int i = 0; i < node->child.size(); ++i) {
				_func(node->child[i], entropy, tree, count);
			}
		}
	};
}