#include "mf_quantum_graph.h"
#include <Eigen/Core>
#include <memory>

#pragma once

template <typename Scalar_>
class NeumannNeumannPreconditioner {
	typedef Scalar_ Scalar;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
public:
	typedef typename Vector::StorageIndex StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic
	};

	NeumannNeumannPreconditioner() : isInitialized(false) {}

	explicit NeumannNeumannPreconditioner(const MFQuantumGraph& mfqg) : vertex_weights(mfqg.vertex_weights.size()) {
		compute(mfqg);
	}

	EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return vertex_weights.size(); }
	EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return vertex_weights.size(); }

	NeumannNeumannPreconditioner& analyzePattern(const MFQuantumGraph&) {
		return *this;
	}

	NeumannNeumannPreconditioner& factorize(const MFQuantumGraph& mfqg) {
		vertices = mfqg.vertices;
		N = mfqg.N;
		edges = mfqg.edges;
		const int sizeI = N-2;
		const int nnzII = 3*(N-4)+2*2;
		const int nnzIG = 2;
		const int nnzGG = 2;

		Eigen::SparseMatrix<double> local_A(sizeI+2, sizeI+2);
		std::vector<Eigen::Triplet<double>> local_A_coefficients;
		local_A.reserve(nnzII+2*nnzIG+nnzGG);

		Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 1);
		double h = x(1)-x(0);
		Eigen::VectorXd	cx;
		Eigen::VectorXd	vx;

		int out, in;
		double lcoeff, rcoeff;
		for (QGEdge edge: mfqg.edges) {
			out = edge.out;
			in = edge.in;
			cx = x.unaryExpr(edge.c);
			vx = x.unaryExpr(edge.v);
			lcoeff = -(cx(0)+cx(1))/(2*h);
			rcoeff = -(cx(N-2)+cx(N-1))/(2*h);

			local_A_coefficients.clear();
			local_A_coefficients.reserve(nnzII+2*nnzIG+nnzGG);
			local_A_coefficients.emplace_back(0, 0, (cx(0)+2*cx(1)+cx(2))/(2*h)+h*vx(1));
			local_A_coefficients.emplace_back(0, 1, -(cx(1)+cx(2))/(2*h));
			for (int i=1; i<N-3; ++i) {
				local_A_coefficients.emplace_back(i, i-1, -(cx(i)+cx(i+1))/(2*h));
				local_A_coefficients.emplace_back(i, i, (cx(i)+2*cx(i+1)+cx(i+2))/(2*h)+h*vx(i+1));
				local_A_coefficients.emplace_back(i, i+1, -(cx(i+1)+cx(i+2))/(2*h));
			}
			local_A_coefficients.emplace_back(N-3, N-4, -(cx(N-3)+cx(N-2))/(2*h));
			local_A_coefficients.emplace_back(N-3, N-3, (cx(N-3)+2*cx(N-2)+cx(N-1))/(2*h)+h*vx(N-2));
			local_A.setFromTriplets(local_A_coefficients.begin(), local_A_coefficients.end());
			local_A.insert(0, sizeI) = lcoeff;
			local_A.insert(sizeI, 0) = lcoeff;
			local_A.insert(N-3, sizeI+1) = rcoeff;
			local_A.insert(sizeI+1, N-3) = rcoeff;
			local_A.insert(sizeI, sizeI) = -lcoeff+vx(0)*h/2;
			local_A.insert(sizeI+1, sizeI+1) = -rcoeff+vx(N-1)*h/2;

			neumann_solvers.push_back(std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>());
			neumann_solvers.back()->compute(local_A);
		}

		vertex_weights.resize(mfqg.vertex_weights.size());
		for (int i = 0; i < mfqg.vertex_weights.size(); ++i) {
			if (mfqg.vertex_weights[i] != 0) {
				vertex_weights(i) = Scalar(1)/mfqg.vertex_weights[i];
			} else {
				vertex_weights(i) = Scalar(1);
			}
		}

		isInitialized = true;
		return *this;
	}

	NeumannNeumannPreconditioner& compute(const MFQuantumGraph& mfqg) {
		return factorize(mfqg);
	}

	template<typename Rhs, typename Dest>
	void _solve_impl(const Rhs& b, Dest& x) const {
		x = Rhs::Zero(vertices);
		Rhs tmp = Rhs::Zero(vertices);
		Eigen::VectorXd neumann_fx = Eigen::VectorXd::Zero(N);
		int out, in;
		for (int i = 0; i < neumann_solvers.size(); ++i) {
			out = edges[i].out;
			in = edges[i].in;
			neumann_fx(N-2) = b(out)*vertex_weights(out);
			neumann_fx(N-1) = b(in)*vertex_weights(in);
			tmp = neumann_solvers[i]->solve(neumann_fx);
			x(out) += tmp(N-2)*vertex_weights(out);
			x(in) += tmp(N-1)*vertex_weights(in);
		}
	}

	template<typename Rhs> inline const Eigen::Solve<NeumannNeumannPreconditioner, Rhs>
	solve(const Eigen::MatrixBase<Rhs>& b) const {
		return Eigen::Solve<NeumannNeumannPreconditioner, Rhs>(*this, b.derived());
	}

	Eigen::ComputationInfo info() { return Eigen::Success; }

protected:
	std::vector<QGEdge> edges;
	std::vector<std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>> neumann_solvers;
	int vertices;
	int N;
	Vector vertex_weights;
	bool isInitialized;
};
