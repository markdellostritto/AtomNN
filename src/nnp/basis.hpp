#pragma once
#ifndef BASIS_HPP
#define BASIS_HPP

// eigen
#include <Eigen/Dense>
//nnp
#include "src/nnp/cutoff.hpp"
//mem
#include "src/mem/serialize.hpp"

#ifndef BASIS_PRINT_FUNC
#define BASIS_PRINT_FUNC 0
#endif

struct Basis{
protected:
	cutoff::Name cutname_;//cutoff name
	cutoff::Norm cutnorm_;//cutoff normalization
	int size_;//number of functions
	double rc_;//cutoff radius
	double rci_;//cutoff radius inverse
	double pirci_;//PI*rci_
	double norm_;//normalization factor
	Eigen::VectorXd symm_;//symmetry function
public:
	//==== constructors/destructors ====
	Basis();
	Basis(double rc, cutoff::Name cutname, cutoff::Norm cutnorm, int nf);
	virtual ~Basis();
	
	//==== member access ====
	const cutoff::Name& cutname()const{return cutname_;}
	const cutoff::Norm& cutnorm()const{return cutnorm_;}
	const double& rc()const{return rc_;}
	const int& size()const{return size_;}
	const double norm()const{return norm_;}
	Eigen::VectorXd& symm(){return symm_;}
	const Eigen::VectorXd& symm()const{return symm_;}
	
	//==== member functions ====
	void clear();
	void resize(int nf);
	double cut_func(double dr)const;
	double cut_grad(double dr)const;
	void cut_comp(double dr, double& v, double& g)const;
	
	//==== static functions ====
	static double norm(cutoff::Norm cutnorm, double rc);
};

#endif