#include<iostream>
#include<vector>
#include<cmath>
using namespace std;

//��װö����
template<class t,size_t l>
class attri {
public:
	attri(t data) :val(data) {}
	t get_val() {
		return val;
	}
	//��ȡö�ٵ�����
	size_t size() {
		return l;
	}
	//
private:
	t val;
};
//�º���ID3�������
template<class t>//����tΪ������
class ID3 {
public:
	//������Ϣ����
	double operator()(vector<t> Set, int i, int HD) {//�걾���ϣ��Լ���i�����ԣ�HD����һ��������
		vector<vector<double>> account(Set[0].attr_size(i));//�������ÿһ�ֵĸ���
		//��ʼ��account
		for (int j = 0; j < Set.size(); ++j) {
			account[j].resize(Set[0].attr_size(Set[0].size()));
		}
		for (int j = 0; j < Set.size(); ++j) {
			account[Set[j].attr(i)][Set[j].attr(Set[j].size())] += 1;
		}
		//������һ��������Ҳ����H(D|A)
		double HDA = 0;
		for (int j = 0; j < account.size();++j) {
			double tmp;
			for (int k = 0; k < account.size(); ++k) 
				tmp += account[j][k];
			for (int k = 0; k < account[j].size();++k) 
				HDA += -(tmp/Set.size())*(account[j][k] / tmp * log2(account[j][k] / tmp));//�ؾ�����Ϣ��������
		}
		return HD - HDA;
	}
};
//�º���C4.5�������
template<class Type>
class C4_5 {
public:
	//������Ϣ������
	double operator()(vector<Type> Set,int i,int HD) {
		vector<vector<double>> account(Set[0].attr_size(i));//�������ÿһ�ֵĸ���
		//��ʼ��account
		for (int j = 0; j < Set.size(); ++j) {
			account[j].resize(Set[0].attr_size(Set[0].size()));
		}
		for (int j = 0; j < Set.size(); ++j) {
			account[Set[j].attr(i)][Set[j].attr(Set[j].size())] += 1;
		}
		//������һ��������Ҳ����H(D|A)
		double HDA = 0;
		//�Լ�������Ե���Ҳ����HA(D)
		double HAD = 0;
		for (int j = 0; j < account.size(); ++j) {
			double tmp = 0;
			for (int k = 0; k < account.size(); ++k)
				tmp += account[j][k];
			for (int k = 0; k < account[j].size(); ++k)
				HDA += -(tmp / Set.size()) * (account[j][k] / tmp * log2(account[j][k] / tmp));//�ؾ�����Ϣ��������
			HAD += -tmp / Set.size();
		}
		return (HD - HDA) / HAD;
	}
};
//�����������
template<class Type>//Type���ͱ�ʾʵ������
struct Tree_node {
	//�������Ľ��һ����Ҫ����������Ϣ
	vector<Tree_node<Type>> _ptr;	//1���ӽ��ָ��
	int i;							//2����¼����㱣����ǵ�i������
	vector<Type> set;				//3���������иý���ϵ�ʵ��
};
////������
//template<class Type,class Node = Tree_node<Type>,class dicision = C4_5<Type>>//Type���ͱ�ʾʵ�����ͣ�node��ʾ����࣬dicision��ʾ�㷨
//class Dicision_Tree{
//public:
//	//����������
//	Dicision_Tree(vector<Type> Set, size_t �� = 0) {
//		test(Set, ��, root, Set[0].size());
//	}
//	//ѵ��
//	void test(vector<Type> Set, size_t ��,Node* node, vector<>  ) {
//		//�ݹ��������
//		if (n == 0)
//			return;   
//		//���ϳ���һ�����ţ�����Ϣ���棨�ʣ����
//		double max = 0;
//		int i = 0;
//		for (int j = 0; j < Type.size(), ++j) {
//			double tmp = _dic(Set, j);//����� j ����������Ϣ����
//			if (tmp > max) {
//				i = j;
//				max = tmp;
//			}
//		}
//		//������������
//		node = new Node;
//		node.i = i;
//		node.
//	}
//	//����
//	
//	//��֦
//
//private:
//	dicision _dic;//ʹ�õ��㷨
//	Node* _root;
//};

enum age{teenage,midlife,elderly};
enum job{have1,without1};
enum house{have2,without2};
enum credi{best,good,ordinary};

enum y{yes,no};

//������
class sample {
public:
	sample(attri<age, 3> Age, attri<job, 2> Job, attri<house, 2> Hou, attri<credi, 3> Cre, attri<y,2> t)
		:s1(Age), s2(Job), s3(Hou), s4(Cre), s(t)
	{}
	//��ȡ��������
	size_t size() {
		return 4;
	}
	//��ȡÿһ������
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
	//��ȡ�����ǩ
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
