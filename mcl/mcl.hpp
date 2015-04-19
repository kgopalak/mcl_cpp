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
		//! @brief MCL ctor. Register a callback function to return the cluster results
		mcl(const Eigen::MatrixXd& Min, std::function< void(size_t cluster_j, size_t member_i) > f) : M(Min), ClusterResultCallback(f) {}

		/*! @brief Apply Markov clustering algorithm with specified parameters and return clusters
			For each cluster, returns the list of node-ids that belong to that cluster
			*/
		void cluster_mcl(double expand_factor = 2, double inflate_factor = 2, double max_loop = 10, double mult_factor = 1)
		{
			Eigen::MatrixXd M_selfloop = M + mult_factor * Eigen::MatrixXd::Identity(M.cols(), M.rows());
			Eigen::MatrixXd M_normalized = normalize(M_selfloop);

			for (int i = 0; i < max_loop && !stop(M_normalized); i++)
			{
				inflate(M_normalized, inflate_factor);
				expand(M_normalized, expand_factor);
			}

			for (auto row = 0; row<M_normalized.rows(); row++)
			{
				if (M_normalized(row, row) > 0)
				{
					for (auto col = 0; col < M_normalized.cols(); col++)
					{
						if (M_normalized(row, col) > 0)
							ClusterResultCallback(row, col);
					}
				}
			}
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
		std::function< void(size_t cls_i, size_t mem_j) > ClusterResultCallback;
	};






}