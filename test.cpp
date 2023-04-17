#include<iostream>
#include<vector>
#include<cmath>
using namespace std;

//封装枚举类
template<class t,size_t l>
class attri {
public:
	attri(t data) :val(data) {}
	t get_val() {
		return val;
	}
	//获取枚举的数量
	size_t size() {
		return l;
	}
	//
private:
	t val;
};
//仿函数ID3：已完成
template<class t>//参数t为样本类
class ID3 {
public:
	//返回信息增益
	double operator()(vector<t> Set, int i, int HD) {//标本集合，以及第i个属性，HD是上一个结点的熵
		vector<vector<double>> account(Set[0].attr_size(i));//这个特征每一种的个数
		//初始化account
		for (int j = 0; j < Set.size(); ++j) {
			account[j].resize(Set[0].attr_size(Set[0].size()));
		}
		for (int j = 0; j < Set.size(); ++j) {
			account[Set[j].attr(i)][Set[j].attr(Set[j].size())] += 1;
		}
		//计算这一个结点的熵也就是H(D|A)
		double HDA = 0;
		for (int j = 0; j < account.size();++j) {
			double tmp;
			for (int k = 0; k < account.size(); ++k) 
				tmp += account[j][k];
			for (int k = 0; k < account[j].size();++k) 
				HDA += -(tmp/Set.size())*(account[j][k] / tmp * log2(account[j][k] / tmp));//熵就是信息量的期望
		}
		return HD - HDA;
	}
};
//仿函数C4.5：已完成
template<class Type>
class C4_5 {
public:
	//返回信息增益率
	double operator()(vector<Type> Set,int i,int HD) {
		vector<vector<double>> account(Set[0].attr_size(i));//这个特征每一种的个数
		//初始化account
		for (int j = 0; j < Set.size(); ++j) {
			account[j].resize(Set[0].attr_size(Set[0].size()));
		}
		for (int j = 0; j < Set.size(); ++j) {
			account[Set[j].attr(i)][Set[j].attr(Set[j].size())] += 1;
		}
		//计算这一个结点的熵也就是H(D|A)
		double HDA = 0;
		//以及这个属性的熵也就是HA(D)
		double HAD = 0;
		for (int j = 0; j < account.size(); ++j) {
			double tmp = 0;
			for (int k = 0; k < account.size(); ++k)
				tmp += account[j][k];
			for (int k = 0; k < account[j].size(); ++k)
				HDA += -(tmp / Set.size()) * (account[j][k] / tmp * log2(account[j][k] / tmp));//熵就是信息量的期望
			HAD += -tmp / Set.size();
		}
		return (HD - HDA) / HAD;
	}
};
//决策树结点类
template<class Type>//Type类型表示实例类型
struct Tree_node {
	//决策树的结点一般需要包含以下信息
	vector<Tree_node<Type>> _ptr;	//1、子结点指针
	int i;							//2、记录本结点保存的是第i种特征
	vector<Type> set;				//3、保存所有该结点上的实例
};
////决策树
//template<class Type,class Node = Tree_node<Type>,class dicision = C4_5<Type>>//Type类型表示实例类型，node表示结点类，dicision表示算法
//class Dicision_Tree{
//public:
//	//构建决策树
//	Dicision_Tree(vector<Type> Set, size_t ε = 0) {
//		test(Set, ε, root, Set[0].size());
//	}
//	//训练
//	void test(vector<Type> Set, size_t ε,Node* node, vector<>  ) {
//		//递归结束条件
//		if (n == 0)
//			return;   
//		//决断出哪一种最优，即信息增益（率）最大
//		double max = 0;
//		int i = 0;
//		for (int j = 0; j < Type.size(), ++j) {
//			double tmp = _dic(Set, j);//计算第 j 种特征的信息增益
//			if (tmp > max) {
//				i = j;
//				max = tmp;
//			}
//		}
//		//接下来创造结点
//		node = new Node;
//		node.i = i;
//		node.
//	}
//	//测试
//	
//	//剪枝
//
//private:
//	dicision _dic;//使用的算法
//	Node* _root;
//};

enum age{teenage,midlife,elderly};
enum job{have1,without1};
enum house{have2,without2};
enum credi{best,good,ordinary};

enum y{yes,no};

//样本类
class sample {
public:
	sample(attri<age, 3> Age, attri<job, 2> Job, attri<house, 2> Hou, attri<credi, 3> Cre, attri<y,2> t)
		:s1(Age), s2(Job), s3(Hou), s4(Cre), s(t)
	{}
	//获取特征数量
	size_t size() {
		return 4;
	}
	//获取每一种特征
	int attr(int iter) {
		if (iter == 0)
			return s1.get_val();
		else if (iter == 1)
			return s2.get_val();
		else if (iter == 2)
			return s3.get_val();
		else if (iter == 3)
			return s4.get_val();
		else if (iter == 4)
			return s.get_val();
		return -1;
	}
	//获取分类标签
	int Type() {
		return attr(4);
	}
	int attr_size(int iter) {
		if (iter == 0)
			return s1.size();
		else if (iter == 1)
			return s2.size();
		else if (iter == 2)
			return s3.size();
		else if (iter == 3)
			return s4.size();
		else if (iter == 4)
			return s.size();
		return -1;
	}
private:
	attri<age, 3> s1;
	attri<job, 2> s2;
	attri<house, 2> s3;
	attri<credi, 3> s4;

	attri<y, 2> s;
};



int main() {
	C4_5<sample> c;	
	vector<sample> set;
	set.push_back(sample(teenage, without1, without2, ordinary, no));//1
	set.push_back(sample(teenage, without1, without2, good, no));//2
	set.push_back(sample(teenage, have1, without2, good, yes));//3
	set.push_back(sample(teenage, have1, have2, ordinary, yes));//4
	set.push_back(sample(teenage, without1, without2, ordinary, no));//5
	set.push_back(sample(midlife, without1, without2, ordinary, no));//6
	set.push_back(sample(midlife, without1, without2, good, no));//7
	set.push_back(sample(midlife, have1, have2, good, yes));//8
	set.push_back(sample(midlife, without1, have2, best, yes));//9
	set.push_back(sample(midlife, without1, have2, best, yes));//10
	set.push_back(sample(elderly, without1, have2, best, yes));//11
	set.push_back(sample(elderly, without1, have2, good, yes));//12
	set.push_back(sample(elderly, have1, without2, good, yes));//13
	set.push_back(sample(elderly, have1, without2, best, yes));//14
	set.push_back(sample(elderly, without1, without2, ordinary, no));//15
	int j = 0;
	double max = c(set, 0 , 0);
	for (int i = 0; i < 15; ++i) {

	}
	return 0;
}
