#include <math.h>
#include <nlopt.hpp>
#include <iostream>

using namespace std;

/*
nlopt示例
min (x \in R^2) sqrt(x2)
条件:
	x2 >= 0
	x2 >= (a1x1 + b1)^3
	x2 >= (a2x1 + b2)^3
	其中a1 = 2, b1 = 0, a2 = -1, b2 = 1
*/

// 迭代次数
int iters = 0;

// 定义目标函数
// vector变量长度与问题的变量维数相同
// my_func_data可用于向myfunc传递附加的数据
double myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
	++iters;

	// 目标函数相对于变量的偏导数, 只有基于梯度的算法需要, 不用梯度的优化算法中grad==NULL
	if (!grad.empty()) {
		grad[0] = 0.0;
		grad[1] = 0.5 / sqrt(x[1]);
	}

	// 返回目标函数的值
	return sqrt(x[1]);
}

// 因为约束用a和b参数化, 所以声明DS来存储这些信息
typedef struct {
	double a, b;
} my_constraint_data;

// 约束函数
// vector变量长度与问题的变量维数相同
// nlopt使用的约束是myconstraint(x) <= 0
double myvconstraint(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
	my_constraint_data *d = reinterpret_cast<my_constraint_data*>(data);
	double a = d->a, b = d->b;

	// 目标函数相对于变量的偏导数, 只有基于梯度的算法需要, 不用梯度的优化算法中grad==NULL
	if (!grad.empty()) {
		grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
		grad[1] = -1.0;
	}

	// 返回约束函数的值
	return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

int main()
{
	// 设置算法和维数
	// 算法种类中L表示局部优化, D表示基于导数/梯度, N表示不用导数
	nlopt::opt opt(nlopt::LD_MMA, 2);

	// 设置变量下界
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL; lb[1] = 0;
	opt.set_lower_bounds(lb);
	// 如果不用nlopt_set_upper_bounds, 则上界默认为正无穷

	// 设置目标函数
	opt.set_min_objective(myvfunc, NULL);

	// 添加约束
	my_constraint_data data[2] = { { 2, 0 }, { -1, 1 } };
	// 1e-8为约束的tolerance: 测试收敛时如果the constraint is violated (is positive) by that tolerance则认为点可行
	opt.add_inequality_constraint(myvconstraint, &data[0], 1e-8);
	opt.add_inequality_constraint(myvconstraint, &data[1], 1e-8);

	// 停止条件, >=1个
	// set_xtol_rel : 参数x的相对tolerance, 未指定的参数取默认值
	opt.set_xtol_rel(1e-4); 
	// set_stopval : 当某一可行点的目标函数值小于stopval时停止
	// opt.set_stopval(opt, sqrt(8. / 27.) + 1e-3);

	// 进行优化
	std::vector<double> x(2);
	x[0] = 1.234; x[1] = 5.678; // 初始值
	double minf; // 目标函数最小值
	
	// 测试优化是否成功
	nlopt::result result = opt.optimize(x, minf); // 如果出错会抛异常
	
	/*
	// nlopt的其他很多函数返回错误码; 如果返回NLOPT_ROUNDOFF_LIMITED, 则表示由于舍入误差中断, 结果可能仍能用
	if (nlopt_optimize(opt, x, &minf) < 0) {
	// 如果失败(比如传入非法参数, 用光内存)则nlopt_optimize返回负值
	cout << "nlopt failed!" << endl;
	}
	else {
	// 如果成功, 输出最小目标函数值和参数值
	cout << "found minimum at f(" << x[0] << "," << x[1] << ") = " << minf << endl;
	// 输出迭代次数
	cout << "found minimum after " << iters << " evaluations" << endl;
	}
	*/

	return 0;
}
