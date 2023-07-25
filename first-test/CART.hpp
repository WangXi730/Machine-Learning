#pragma once
//CART算法以及相关算法
#include"Decision_Tree.hpp"

namespace wx {
	//回归树结点
	template<class value_type>
	struct Regression_Tree_node : public Decision_Tree_node <value_type> {
		Regression_Tree_node() 
		:splitting_variable(this->this_attr) ,splitting_variable_pointval(this->attr) {}
		//除了决策树共有的属性外，还需要加入以下属性
		std::unordered_map<int, double> Rm_cm;//这个结点所在单元的最优输出值，Rm_cm[i]表示第i个属性
		std::unordered_map<int, value_type>& splitting_variable_pointval;//在这个结点之前已经划分的切分变量(属性)相应的取值
		std::pair<double,double> cm;//这个结点确定的切分最优值，由于回归树是二叉树，这里直接用pair
		int& splitting_variable;//这个结点确定的切分变量
		double splitting_point;//这个结点切分变量相应的切分点
	};
	//回归树，默认为最小二乘回归树
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
		//拷贝构造函数
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
			//对结点初始化
			node.parent = pr;//根结点父结点是null
			node.data = node_data;//该结点具有的数据
			node.attr = attr;//当前结点具有的属性
			if (pr) {
				node.Rm_cm = pr->Rm_cm;//当前结点应当继承父结点的切分属性
			//同时，这个结点的切分属性应当在父结点的基础上，增加父结点对于其对应属性的切分
				if (this->_sample[node.data[0]][pr->splitting_variable] <= this->_sample[pr->splitting_point][pr->splitting_variable]) {
					//小于切分点的值，分为左类	
					node.Rm_cm[pr->splitting_variable] = pr->cm.first;
				}
				else {
					node.Rm_cm[pr->splitting_variable] = pr->cm.second;
				}
			}
			//递归结束条件
			if (this->_kore) {
				//这里是e
				if (pr != nullptr && pr->entropy < this->_e) {
					std::unordered_map<double,double> tmp;//tmp的下标是标签，数据是这个标签的数量，取最多的作为类型
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
					node.this_attr = t;//确定最后的类型
					node.entropy = 0.0;//由于叶子结点不进行分类，所以收益为0；
					//构造error_list
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
			//这里是k(剩余的可使用属性个数)
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
				//构造error_list
				for (auto& e : node_data) {
					if (this->_lable[e] != node.this_attr) {
						node.error_list.insert(e);
						this->error_list.insert(e);
					}
				}
				this->leaf_node.insert(&node);
				return;
			}
			//走到这里说明没有结束，还得继续分
			//计算每一个种类的熵，选择最大的
			double mm = -1;//这里的mm表示最小分类误差
			int sub = -1;//最小者的下标
			double sp = 0;//切分点
			std::pair<double, double> cm;//切分点的两侧的固定输出值
			for (int i = 0; i < this->m; ++i) {
				////首先要明确的一点，不能用之前用过的属性
				//if (used[i] == 1)
				//	continue;
				double tmp_sp = 0;//这一次的切分点
				std::pair<double, double> tmp_cm;//这一次的两侧输出值
				double tmp = this->func(this->_sample, this->_lable, type_size, i, node_data, this->_attr_size[i], this->_power, tmp_cm, tmp_sp);
				if (tmp < mm||mm == -1) {
					sub = i;
					mm = tmp;
					sp = tmp_sp;
					cm = tmp_cm;
				}
			}
			//确定当前结点的决策属性
			node.entropy = mm;
			node.this_attr = sub;
			node.splitting_point = sp;
			node.cm = cm;
			//已经使用过sub属性了，所以它的子结点就不能用了
			used[sub] = 1;

			//当前结点的子结点指针
			Tree_node* child = new Tree_node;
			//确定下一代结点具有的属性
			std::unordered_map<int, double> child_attr = attr;//在其他属性上，子结点有着跟父结点相同的属性
			child_attr[sub] = cm.first;//在当前结点确定的属性上，子结点的属性应当每个属性都有
			//下一代结点具有的样本
			std::vector<int> child_data;
			for (auto& e : node_data) {
				if (this->_sample[e][sub] <= this->_sample[sp][sub]) {
					child_data.push_back(e);
				}
			}
			//加入到当前结点的子指针列表
			node.child.push_back(child);
			_train(*child, &node, child_data, child_attr, k - 1, used);

			child = new Tree_node;
			//确定下一代结点具有的属性
			child_attr[sub] = cm.second;//在当前结点确定的属性上，子结点的属性应当每个属性都有
			//下一代结点具有的样本
			child_data.clear();
			for (auto& e : node_data) {
				if (this->_sample[e][sub] > this->_sample[sp][sub]) {
					child_data.push_back(e);
				}
			}
			//加入到当前结点的子指针列表
			node.child.push_back(child);
			_train(*child, &node, child_data, child_attr, k - 1, used);
		}
	};
	//最小二乘寻找最优切分算法
	class least_squares_regression {
	public:
		double operator()(std::vector<std::vector<double>>& sample, std::vector<double>& lable, int type_size, int i, std::vector<int>& node_data, int attr_size, std::vector<double> power, std::pair<double,double>& cm, double& sp) {
			//返回输入的切分变量的误差，并通过参数返回其固定输出值、切分点
			double result = -1;//误差不可能小于0，所以可以用-1代表初值
			int ptr = -1;//最小的下标暂且给-1
			for (int k = 0; k < node_data.size(); ++k) {
				//每一个切分点都试一遍，找到最小的，k为切分点的下标，平方阶算法
				double tmp_result = 0;
				double n1 = 0;//第 1 类有 n1 个元素
				double n2 = 0;//第 2 类有 n2 个元素
				double tmp_x1 = 0;//第一类总和
				double tmp_x2 = 0;//第二类总和
				for (int j = 0; j < node_data.size(); ++j) {
					if (sample[node_data[j]][i] <= sample[node_data[k]][i]) {
						//第一类的
						tmp_x1 += lable[node_data[j]] * power[node_data[j]];
						n1 += power[node_data[j]];
					}
					else {
						//第二类的
						tmp_x2 += lable[node_data[j]] * power[node_data[j]];
						n2 += power[node_data[j]];
					}
				}
				//求均值
				if (n1)
					tmp_x1 = tmp_x1 / n1;
				if (n2)
					tmp_x2 = tmp_x2 / n2;
				//计算
				for (int j = 0; j < node_data.size(); ++j) {
					if (sample[node_data[j]][i] <= sample[node_data[k]][i]) {
						//第一类
						tmp_result += (lable[node_data[j]] - tmp_x1) * (lable[node_data[j]] - tmp_x1) * power[node_data[j]];
					}
					else {
						//第二类
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
			//赋值sp
			sp = node_data[ptr];
			//返回result
			return result;
		}
	};
	//平方误差策略
	class square_error_strategy {
	public:
		//求这个结点下的所有叶子结点的误差平方和
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double operator()(Regression_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree, std::vector<Decision_Tree_node<lable_type>*>* leaf = nullptr) {
			//要求每个子结点的误差
			//如果是叶子，直接就求了，结束递归
			if (node->is_leaf()) 
				return node_strategy<value_type,lable_type,type_size,strategy_t,algorithm_t,Tree_node>(node, tree);
			//如果不是，则递归访问子结点
			double result = 0;
			for (int i = 0; i < node->child.size(); ++i) {
				result += operator()(node, tree);
			}
			return result;
		}
		//提供接口，计算当前结点的误差平方和
		template<class value_type, class lable_type, size_t type_size, class strategy_t, class algorithm_t, class Tree_node>
		double node_strategy(Regression_Tree_node<lable_type>* node, Decision_Tree<type_size, value_type, lable_type, strategy_t, algorithm_t, Tree_node>* tree) {
			//通过这个结点的attr属性，知道这个结点已经划分的属性，求出误差
			//其实也就两种类型
			//定义误差e
			double e = 0;
			//对这个结点上的每个数据都进行平方和
			for (int i = 0; i < node->data.size(); ++i) {
				//i仅代表迭代次数
				if (tree->Get_sample()[node->data[i]][node->splitting_variable] <= node->splitting_point)
					e += (tree->Get_lable()[node->data[i]] - node->cm.first) * (tree->Get_lable()[node->data[i]] - node->cm.first) * tree->Get_power()[node->data[i]];
				else
					e += (tree->Get_lable()[node->data[i]] - node->cm.second) * (tree->Get_lable()[node->data[i]] - node->cm.second) * tree->Get_power()[node->data[i]];
			}
			//返回这个结点造成的预测误差
			return e;
		}
	};
}