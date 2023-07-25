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
	//结点类
	template<class value_type>
	struct Decision_Tree_node {
		std::unordered_map<int, value_type> attr;//这个结点的第i个属性所属的类别
		int this_attr;//这个结点确定分类的属性，如果是叶子结点，则确定最后的类型
		std::vector<int> data;//这个结点剩余的样本序号
		std::vector<Decision_Tree_node<value_type>*> child;//子结点指针，下标即代表种类
		Decision_Tree_node<value_type>* parent;//父结点指针
		double entropy = 0;//该结点信息增益(比)
		std::unordered_set<int> error_list;//错误列表
		//构造函数给上
		Decision_Tree_node() {}
		//给一个确定叶子结点的方法
		bool is_leaf() {
			return child.size() == 0;
		}
	};

	//模板声明：type_size：标签种类数量，strategy_t：策略选择,决策树的策略这里表示为C(t) = Ca(t) - a * |t|，|t|表示这棵树的叶子结点的数量，algorithm_t：算法选择，可选信息增益或信息增益比，Tree_node：树的结点类型，一般建议继承Decision_Tree_node类型
	template<size_t type_size, class value_type = int, class lable_type = int, class strategy_t = Empirical_entropy_strategy, class algorithm_t = Information_gain_ratio, class Tree_node = Decision_Tree_node<value_type>>
	class Decision_Tree {
	public:
		//生成决策树，参数：
		// 样本矩阵（m行n列，m个属性，n个样本，每一个样本算作一个向量，vector<vector<algorithm_t::value_type>> sample）
		// 标签向量（vector<algorithm_t::lable_type> lable，lable.size() == n，每个样本有一个）
		// 各个属性的种类数量向量（vector<int> attr_size, attr_size.size() == m，每个属性有一个）
		// 精度：double e/决策树高度：int k
		// 样本权重向量（缺省为 1 向量）：vector<double> power，power.size() == n
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
		//拷贝构造函数
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
		//析构方法释放空间
		virtual ~Decision_Tree()
		{}
		//修改k
		bool Re_k(int k) {
			_k = k;		//修改k的值
			_kore = 0;	//修改为k作用
			return true;
		}
		//修改e
		bool Re_e(double e) {
			_e = e;
			_kore = 1;//同理
			return true;
		}
		//修改权重向量
		bool Re_power(std::vector<double>& power) {
			if (power.size() == n) {
				_power = power;
				return true;
			}
			return false;
		}
		//修改样本集合
		bool Re_samset(std::vector<std::vector<value_type>>& sample, std::vector<lable_type>& lable, std::vector<int>& attr_size) {
			_sample = sample;
			_lable = lable;
			_attr_size = attr_size;
			//顺带训练一次
			Train();
			return true;
		}
		//训练树
		void Train() {
			//先清理原本的树，避免冲突
			_clear();
			//获取每一个属性的信息量，选择最大的作为根结点
			root.reset(new Tree_node);
			std::vector<int> root_data(n);//毫无疑问root拥有所有数据
			for (int i = 0; i < n; ++i) {
				root_data[i] = i;
			}
			std::unordered_map<int, value_type> mp;//根结点不具备任何属性
			std::vector<int> used(m, 0);//已经使用过的属性
			_train(*root, nullptr, root_data, mp, _k - 1, used);

		}
		//对给定数据进行分类
		lable_type Test(std::vector<value_type> t) {
			if (root.get() == nullptr) {
				Train();
			}
			//要求是一个维度相同的样本
			if (t.size() != m) {
				throw m;
			}
			Tree_node* ptr = root.get();
			while (!(ptr->is_leaf())) {
				ptr = ptr->child[t[ptr->this_attr]];
			}
			return ptr->this_attr;
		}
		//获取错误分类的集合
		std::unordered_set<int> Get_error() {
			return error_list;
		}
		//获取类函数
		std::vector<std::vector<value_type>> Get_sample() {
			return _sample;
		}
		std::vector<double> Get_power() {
			return _power;
		}
		std::vector<lable_type> Get_lable() {
			return _lable;
		}
		//获取损失函数loss function的值
		virtual double Loss_function(double a) {
			//损失函数 Ca(t) = C(t) + a * |t|
			return C.operator()(root.get(),this) + a * (double)leaf_node.size();
		}
		//决策树剪枝
		void Pruning(double a) {
			int e = 1;
			while (e) {
				e = 0;
				//对每一个叶子的父结点进行遍历，直到不能找出优化为止
				for (int i = 0; i < leaf_node.size(); ++i) {
					//记录父结点
					Tree_node* pr = leaf_node[i]->parent;
					//这个结点下的叶子结点的数量
					std::vector<Tree_node*> leafs;
					//计算当前的损失
					double now_lose = C(pr, this, &leafs) + a * (double)leaf_node.size();
					//计算剪枝之后的损失
					double cut_lose = C.node_strategy(pr, this) + a * ((double)leaf_node.size() - (double)leafs.size() + 1);
					//比较
					if (now_lose > cut_lose) {
						e = 1;
						//去除叶子结点集合中的子结点
						for (int j = 0; j < leafs.size(); ++j) {
							auto tmp = leaf_node.find(leafs[j]);
							if (tmp != leaf_node.end()) {
								leaf_node.erase(tmp);
							}
						}
						//加入新的叶子结点
						leaf_node.insert(pr);
						//从存储上剪枝
						pr.clear();
						break;
					}
				}
			}
		}
	protected:
		//清理树
		void _clear() {
			leaf_node.clear();
			std::unordered_set<int>(std::move(error_list));
			//清空决策树，并释放空间，由于使用了vector，导致我们只new了一个内存，即root，释放即可
			root.reset(nullptr);
		}
		virtual void _train(Tree_node& node, Tree_node* pr, std::vector<int>& node_data, std::unordered_map<int, value_type> attr, int k, std::vector<int> used) {
			//对结点初始化
			node.parent = pr;//根结点父结点是null
			node.data = node_data;//该结点具有的数据
			node.attr = attr;//当前结点具有的属性
			//递归结束条件
			if (_kore) {
				//这里是e
				if (pr!=nullptr && pr->entropy < _e) {
					std::vector<double> tmp(type_size, 0);//tmp的下标是标签，数据是这个标签的数量，取最多的作为类型
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
					node.this_attr = t;//确定最后的类型
					node.entropy = 0.0;//由于叶子结点不进行分类，所以收益为0；
					//构造error_list
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
			//这里是k(剩余的可使用属性个数)
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
				//构造error_list
				for (auto& e : node_data) {
					if (_lable[e] != node.this_attr) {
						node.error_list.insert(e);
						error_list.insert(e);
					}
				}
				leaf_node.insert(&node);
				return;
			}
			//除此之外，分类完成也可以作为退出递归的条件
			int E = node_data[0];
			for (auto& i : node_data) {
				if (_lable[i] != _lable[E]) {
					break;
				}
				if (i == node_data[node_data.size() - 1]) {
					//证明所有的标签都相等
					node.this_attr = _lable[E];
					node.entropy = 0;
					leaf_node.insert(&node);
					return;
				}
			}

			//走到这里说明没有结束，还得继续分
			//计算每一个种类的熵，选择最大的
			double mm = 0.0;//无论如何，信息增益不可能小于0吧
			int sub = -1;//最大者的下标
			for (int i = 0; i < m; ++i) {
				//首先要明确的一点，不能用之前用过的属性
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
			//确定当前结点的决策属性
			node.entropy = mm;
			node.this_attr = sub;
			//已经使用过sub属性了，所以它的子结点就不能用了
			used[sub] = 1;
			//当前结点的子结点指针，完成递归
			for (int i = 0; i < _attr_size[sub]; ++i) {
				Tree_node* child = new Tree_node;
				//确定下一代结点具有的属性
				std::unordered_map<int, lable_type> child_attr = attr;//在其他属性上，子结点有着跟父结点相同的属性
				child_attr[sub] = i;//在当前结点确定的属性上，子结点的属性应当每个属性都有
				//下一代结点具有的样本
				std::vector<int> child_data;
				for (auto& e : node_data) {
					if (_sample[e][sub] == i) {
						child_data.push_back(e);
					}
				}
				//加入到当前结点的子指针列表
				node.child.push_back(child);
				_train(*child, &node, child_data, child_attr, k - 1, used);
			}
		}
		std::vector<std::vector<value_type>> _sample;//样本矩阵
		std::vector<lable_type> _lable;//样本标签向量
		std::vector<int> _attr_size;//属性种类数量
		std::vector<double> _power;//样本权重向量
		int _k = 0;//k层截止
		double _e = 0;//func(特征i)小于e截止
		int _kore = 0;//0代表用k，1代表用e
		typename strategy_t C;//策略：用于计算预测误差
		typename algorithm_t func;//算法：比较熵的仿函数
		int m; //属性数量
		int n; //样本数量
		std::unordered_set<int> error_list;	//错误列表
		std::auto_ptr<Tree_node> root;//根结点
		std::unordered_set<Tree_node*> leaf_node;//为了方便，将叶子结点存储起来
	};


	//信息增益算法
	class Information_gain {
	public:
		Information_gain(){}
		//数据集中，第i个属性的信息增益，其中，H(D)是上一层的经验熵，node_data表示本层的剩余结点下标，这里的attr_size仅表示第i个属性的种类数目
		double operator()(std::vector<std::vector<int>>& sample, std::vector<int>& lable, int type_size, int i, std::vector<int>& node_data, int attr_size, std::vector<double> power, std::pair<double, double> cm = std::pair<double, double>(), double sp = double()) {
			//计算HD
			double HD = 0;
			std::vector<double> tmp(type_size);//标签数量的空间
			for (auto& e : node_data) {
				tmp[lable[e]] += power[e];
			}
			//计算D的大小
			double D_count = 0.0;
			for (auto& e : power) {
				D_count += e;
			}
			for (auto& e : tmp) {
				if(e != 0)
					HD += -(e / D_count * log(e / D_count));
			}
			//计算HDA
			std::vector<std::vector<int>> count(attr_size);
			for (auto& e : node_data) {
				count[sample[e][i]].push_back(e);//第e个变量的第i个属性的类别，存放这个变量的下标
			}
			//计算特征 i 对应集合 node_data 的经验条件熵，然后求和
			double HDA = 0;
			for (auto& Di : count) {
				double sum = 0;
				if (Di.size() == 0) {
					continue;//如果该种类没有数据，跳过即可
				}
				//求出每个标签的样本数量
				std::unordered_map<int, double> mp;
				double Di_count = 0.0;
				for (auto& Dik : Di) {
					mp[lable[Dik]] += power[Dik];
					Di_count += power[Dik];
				}
				//求和
				for (auto& k : mp) {
					if (k.second != 0) 
						sum += (k.second / Di_count * log(k.second / Di_count));
				}
				//汇总到 HDA 里面
				HDA += -(Di_count / D_count * sum);
			}
			return HD - HDA;//返回信息增益
		}
	};
	//信息增益比算法
	class Information_gain_ratio {
	public:	 
		Information_gain_ratio(){}
		double operator()(std::vector<std::vector<int>>& sample, std::vector<int>& lable, int type_size, int i, std::vector<int>& node_data, int attr_size, std::vector<double> power,std::pair<double, double> cm = std::pair<double,double>(), double sp = double()) {
			Information_gain f;
			//定义信息增益gR(DA)
			double gRDA = f(sample, lable, type_size, i, node_data, attr_size, power);
			//定义sample关于属性i的熵HA(D)
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
			//返回信息增益率 
			return gRDA / HAD;
		}
	};

	//信息熵计算
	class Empirical_entropy_strategy {
	public:
		Empirical_entropy_strategy(){}
		//求这个结点下所有的叶子结点的熵
		//接收一个决策树的父类结点，方便使用多态来应对变数
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double operator()(Decision_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree, std::vector<Decision_Tree_node<lable_type>*>* leaf = nullptr) {
			double result = 0;
			_func(node, result, tree, leaf);
			return result;
		}
		//提供接口，计算当前结点的熵
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
		//内部方法：求信息熵，参数：结点，叶子结点，熵
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		void _func(Decision_Tree_node<lable_type>* node, double& entropy, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree, std::vector<Decision_Tree_node<lable_type>*>* count) {
			//递归结束条件
			if (node->is_leaf()) {
				entropy += node_strategy(node,tree);
				if (count != nullptr) {
					count->push_back(node);
				}
				return;
			}
			//如果不是叶子结点，就继续
			for (int i = 0; i < node->child.size(); ++i) {
				_func(node->child[i], entropy, tree, count);
			}
		}
	};
}