#include "mf_quantum_graph.h"
#include <Eigen/Core>

#pragma once

template <typename Scalar_>
class DiagonalPreconditioner {
	typedef Scalar_ Scalar;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
public:
	typedef typename Vector::StorageIndex StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic
	};

	DiagonalPreconditioner() : m_isInitialized(false) {}

	explicit DiagonalPreconditioner(const MFQuantumGraph& mfqg) : m_invdiag(mfqg.vertex_weights.size()) {
		compute(mfqg);
	}

	EIGEN_CONSTEXPR Eigen::Index rows() const EIGEN_NOEXCEPT { return m_invdiag.size(); }
	EIGEN_CONSTEXPR Eigen::Index cols() const EIGEN_NOEXCEPT { return m_invdiag.size(); }

	DiagonalPreconditioner& analyzePattern(const MFQuantumGraph&) {
		return *this;
	}

	DiagonalPreconditioner& factorize(const MFQuantumGraph& mfqg) {
		m_invdiag.resize(mfqg.vertex_weights.size());
		Eigen::VectorXd rhs;
		double dii;
		for (int i = 0; i < mfqg.vertex_weights.size(); ++i) {
			rhs = Eigen::VectorXd::Zero(mfqg.vertex_weights.size());
			rhs(i) = 1.0;
			dii = (mfqg.AGG*rhs-mfqg.AIG.transpose()*mfqg.mysolver.solve(mfqg.AIG*rhs))(i);
			if (dii != 0) {
				m_invdiag(i) = Scalar(1)/dii;
			} else {
				m_invdiag(i) = Scalar(1);
			}
		}

		m_isInitialized = true;
		return *this;
	}

	DiagonalPreconditioner& compute(const MFQuantumGraph& mfqg) {
		return factorize(mfqg);
	}

	template<typename Rhs, typename Dest>
	void _solve_impl(const Rhs& b, Dest& x) const {
		x = m_invdiag.array() * b.array() ;
	}

	template<typename Rhs> inline const Eigen::Solve<DiagonalPreconditioner, Rhs>
	solve(const Eigen::MatrixBase<Rhs>& b) const {
		return Eigen::Solve<DiagonalPreconditioner, Rhs>(*this, b.derived());
	}

	Eigen::ComputationInfo info() { return Eigen::Success; }

protected:
	Vector m_invdiag;
	bool m_isInitialized;
};
