#include "mf_quantum_graph.h"
#include <Eigen/Core>

#pragma once

template <typename Scalar_>
class DegreePreconditioner {
	typedef Scalar_ Scalar;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
public:
	typedef typename Vector::StorageIndex StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic
	};

	DegreePreconditioner() : isInitialized(false) {}

	explicit DegreePreconditioner(const MFQuantumGraph& mfqg) : vertex_weights(mfqg.vertex_weights.size()) {
		compute(mfqg);
	}

	EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return vertex_weights.size(); }
	EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return vertex_weights.size(); }

	DegreePreconditioner& analyzePattern(const MFQuantumGraph&) {
		return *this;
	}

	DegreePreconditioner& factorize(const MFQuantumGraph& mfqg) {
		vertex_weights.resize(mfqg.vertex_weights.size());
		for(int i = 0; i < mfqg.vertex_weights.size(); ++i) {
			if (mfqg.vertex_weights[i] != 0) {
				vertex_weights(i) = Scalar(1)/mfqg.vertex_weights[i];
			} else {
				vertex_weights(i) = Scalar(1);
			}
		}

		isInitialized = true;
		return *this;
	}

	DegreePreconditioner& compute(const MFQuantumGraph& mfqg) {
		return factorize(mfqg);
	}

	template<typename Rhs, typename Dest>
	void _solve_impl(const Rhs& b, Dest& x) const {
		x = vertex_weights.cwiseProduct(b);
	}

	template<typename Rhs> inline const Eigen::Solve<DegreePreconditioner, Rhs>
	solve(const Eigen::MatrixBase<Rhs>& b) const {
		return Eigen::Solve<DegreePreconditioner, Rhs>(*this, b.derived());
	}

	Eigen::ComputationInfo info() { return Eigen::Success; }

protected:
	Vector vertex_weights;
	bool isInitialized;
};
