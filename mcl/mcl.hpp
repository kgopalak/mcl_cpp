#include <mcl/license.hpp>

#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <map>
#include <vector>

using std::vector;
using std::map;


namespace mcl_cpp
{
	class mcl
	{
	public:
		//! @brief MCL ctor
		mcl(const Eigen::MatrixXd& Min) : M(Min) { }

		/*! @brief Apply Markov clustering algorithm with specified parameters and return clusters
			For each cluster, returns the list of node-ids that belong to that cluster
			*/
		map<size_t, vector<size_t> > cluster_mcl(double expand_factor = 2, double inflate_factor = 2, double max_loop = 10, double mult_factor = 1)
		{
			Eigen::MatrixXd M_selfloop = M + mult_factor * Eigen::MatrixXd::Identity(M.cols(), M.rows());
			Eigen::MatrixXd M_normalized = normalize(M_selfloop);

			for (int i = 0; i < max_loop; i++)
			{
				inflate(M_normalized, inflate_factor);
				expand(M_normalized, expand_factor);
				if (stop(M_normalized))
					break;
			}
			return map<size_t, vector<size_t> >();
		}

	private:
		bool stop(const Eigen::MatrixXd& in)
		{
			auto diff = (in*in - in);
			return (diff.maxCoeff() - diff.minCoeff() < 1e-15);
		}

		Eigen::MatrixXd normalize(Eigen::MatrixXd& in)
		{
			auto one_over_col_sum = in.colwise().sum().cwiseInverse();
			Eigen::MatrixXd M_normalized = in * one_over_col_sum.asDiagonal();
			return std::move(M_normalized);
		}

		void expand(Eigen::MatrixXd& in, double expand_factor)
		{
			Eigen::MatrixPower<Eigen::MatrixXd> Apow(in);
			in = Apow(expand_factor);
		}

		void inflate(Eigen::MatrixXd& in, double inflate_factor)
		{
			auto lam = [inflate_factor](double x) -> double { return std::pow(x, inflate_factor); };
			in = in.unaryExpr(lam);
		}

		//std::function<void(const mcl&)> IterEndvisitor;
		const Eigen::MatrixXd & M;
	};






}