
#define BOOST_TEST_MODULE MCL_TEST
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <mcl/mcl.hpp>

double inflat(double val)
{
	return std::pow(val, 1.5);
}

BOOST_AUTO_TEST_CASE(my_test)
{
	Eigen::MatrixXd in(10.0*Eigen::MatrixXd::Random(5, 5).cwiseAbs());
	mcl_cpp::mcl mclob(in);
	mclob.cluster_mcl(4, 1.5, 60, 2);

	//Eigen::MatrixXd in(3, 3);
	//in << (Eigen::Matrix3d() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
	/*,
		Eigen::MatrixXd::Zero(3, 5 - 3),
		Eigen::MatrixXd::Zero(5 - 3, 3),
		Eigen::MatrixXd::Identity(5 - 3, 5 - 3);
		*/
	std::cout << in << std::endl;
	auto one_over_col_sum = in.colwise().sum().cwiseInverse();
	std::cout << "one_over_col_sum" << std::endl << one_over_col_sum << std::endl;

	Eigen::MatrixXd M_normalized = in;// *one_over_col_sum.asDiagonal();
	std::cout << "M_normalized" << std::endl << M_normalized << std::endl;

	Eigen::MatrixPower<Eigen::MatrixXd> Apow(M_normalized);
	auto mp = Apow(3.0);
	auto inf = M_normalized.pow(1.5);
	double inf_pow = 1.5;
	auto lam = [&](double x) -> double { return std::pow(x, inf_pow); };
	auto inf2 = M_normalized.unaryExpr(lam);
	std::cout << std::endl << std::endl << inf2 << std::endl << std::endl
		<< mp << std::endl << std::endl
		<< inf << std::endl << std::endl;

}