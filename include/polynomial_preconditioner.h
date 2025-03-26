#include "mf_quantum_graph.h"
#include <Eigen/Core>

#pragma once

template <typename Scalar_>
class PolynomialPreconditioner {
	typedef Scalar_ Scalar;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
public:
	typedef typename Vector::StorageIndex StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic
	};

	PolynomialPreconditioner() : isInitialized(false) {}

	explicit PolynomialPreconditioner(const MFQuantumGraph& mfqg) : vertex_weights(mfqg.vertex_weights.size()) {
		compute(mfqg);
	}

	EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return vertex_weights.size(); }
	EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return vertex_weights.size(); }

	PolynomialPreconditioner& analyzePattern(const MFQuantumGraph&) {
		return *this;
	}

	PolynomialPreconditioner& factorize(const MFQuantumGraph& mfqg) {
		AIG = mfqg.AIG;
		AGG = mfqg.AGG;
		mysolver.compute(mfqg.AII);
		vertex_weights.resize(mfqg.vertex_weights.size());
		Eigen::VectorXd rhs;
		double dii;
		for (int i = 0; i < mfqg.vertex_weights.size(); ++i) {
			rhs = Eigen::VectorXd::Zero(mfqg.vertex_weights.size());
			rhs(i) = 1.0;
			dii = (AGG*rhs-AIG.transpose()*mysolver.solve(AIG*rhs))(i);
			if (dii != 0) {
				vertex_weights(i) = Scalar(1)/dii;
			} else {
				vertex_weights(i) = Scalar(1);
			}
		}

		isInitialized = true;
		return *this;
	}

	PolynomialPreconditioner& compute(const MFQuantumGraph& mfqg) {
		return factorize(mfqg);
	}

	template<typename Rhs, typename Dest>
	void _solve_impl(const Rhs& b, Dest& x) const {
		x = vertex_weights.cwiseProduct(2*b-(AGG*vertex_weights.cwiseProduct(b)-AIG.transpose()*mysolver.solve(AIG*vertex_weights.cwiseProduct(b))));
	}

	template<typename Rhs> inline const Eigen::Solve<PolynomialPreconditioner, Rhs>
	solve(const Eigen::MatrixBase<Rhs>& b) const {
		return Eigen::Solve<PolynomialPreconditioner, Rhs>(*this, b.derived());
	}

	Eigen::ComputationInfo info() { return Eigen::Success; }

protected:
  Eigen::SparseMatrix<double> AII, AIG, AGG;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mysolver;
	Vector vertex_weights;
	bool isInitialized;
};
