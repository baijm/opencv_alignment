#include <math.h>
#include <nlopt.hpp>
#include <iostream>

using namespace std;

/*
nloptʾ��
min (x \in R^2) sqrt(x2)
����:
	x2 >= 0
	x2 >= (a1x1 + b1)^3
	x2 >= (a2x1 + b2)^3
	����a1 = 2, b1 = 0, a2 = -1, b2 = 1
*/

// ��������
int iters = 0;

// ����Ŀ�꺯��
// vector��������������ı���ά����ͬ
// my_func_data��������myfunc���ݸ��ӵ�����
double myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
	++iters;

	// Ŀ�꺯������ڱ�����ƫ����, ֻ�л����ݶȵ��㷨��Ҫ, �����ݶȵ��Ż��㷨��grad==NULL
	if (!grad.empty()) {
		grad[0] = 0.0;
		grad[1] = 0.5 / sqrt(x[1]);
	}

	// ����Ŀ�꺯����ֵ
	return sqrt(x[1]);
}

// ��ΪԼ����a��b������, ��������DS���洢��Щ��Ϣ
typedef struct {
	double a, b;
} my_constraint_data;

// Լ������
// vector��������������ı���ά����ͬ
// nloptʹ�õ�Լ����myconstraint(x) <= 0
double myvconstraint(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
	my_constraint_data *d = reinterpret_cast<my_constraint_data*>(data);
	double a = d->a, b = d->b;

	// Ŀ�꺯������ڱ�����ƫ����, ֻ�л����ݶȵ��㷨��Ҫ, �����ݶȵ��Ż��㷨��grad==NULL
	if (!grad.empty()) {
		grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
		grad[1] = -1.0;
	}

	// ����Լ��������ֵ
	return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

int main()
{
	// �����㷨��ά��
	// �㷨������L��ʾ�ֲ��Ż�, D��ʾ���ڵ���/�ݶ�, N��ʾ���õ���
	nlopt::opt opt(nlopt::LD_MMA, 2);

	// ���ñ����½�
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL; lb[1] = 0;
	opt.set_lower_bounds(lb);
	// �������nlopt_set_upper_bounds, ���Ͻ�Ĭ��Ϊ������

	// ����Ŀ�꺯��
	opt.set_min_objective(myvfunc, NULL);

	// ���Լ��
	my_constraint_data data[2] = { { 2, 0 }, { -1, 1 } };
	// 1e-8ΪԼ����tolerance: ��������ʱ���the constraint is violated (is positive) by that tolerance����Ϊ�����
	opt.add_inequality_constraint(myvconstraint, &data[0], 1e-8);
	opt.add_inequality_constraint(myvconstraint, &data[1], 1e-8);

	// ֹͣ����, >=1��
	// set_xtol_rel : ����x�����tolerance, δָ���Ĳ���ȡĬ��ֵ
	opt.set_xtol_rel(1e-4); 
	// set_stopval : ��ĳһ���е��Ŀ�꺯��ֵС��stopvalʱֹͣ
	// opt.set_stopval(opt, sqrt(8. / 27.) + 1e-3);

	// �����Ż�
	std::vector<double> x(2);
	x[0] = 1.234; x[1] = 5.678; // ��ʼֵ
	double minf; // Ŀ�꺯����Сֵ
	
	// �����Ż��Ƿ�ɹ�
	nlopt::result result = opt.optimize(x, minf); // �����������쳣
	
	/*
	// nlopt�������ܶຯ�����ش�����; �������NLOPT_ROUNDOFF_LIMITED, ���ʾ������������ж�, �������������
	if (nlopt_optimize(opt, x, &minf) < 0) {
	// ���ʧ��(���紫��Ƿ�����, �ù��ڴ�)��nlopt_optimize���ظ�ֵ
	cout << "nlopt failed!" << endl;
	}
	else {
	// ����ɹ�, �����СĿ�꺯��ֵ�Ͳ���ֵ
	cout << "found minimum at f(" << x[0] << "," << x[1] << ") = " << minf << endl;
	// �����������
	cout << "found minimum after " << iters << " evaluations" << endl;
	}
	*/

	return 0;
}
