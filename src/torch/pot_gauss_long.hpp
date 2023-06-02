#pragma once
#ifndef POT_GAUSS_LONG_HPP
#define POT_GAUSS_LONG_HPP

//mem
#include "src/mem/serialize.hpp"
// torch
#include "src/torch/pot.hpp"
#include "src/torch/kspace_coul.hpp"

#ifndef PGL_PRINT_FUNC
#define PGL_PRINT_FUNC 0
#endif

#ifndef PGL_PRINT_DATA
#define PGL_PRINT_DATA 0
#endif

namespace ptnl{
	
class PotGaussLong: public Pot{
private:
	//parameters - global
	double eps_;
	double prec_;
	//parameters - atomic
	Eigen::VectorXi f_;
	Eigen::VectorXd radius_;
	Eigen::MatrixXd rij_;
	//kspace
	KSpace::Coul coul_;
public:
	//==== constructors/destructors ====
	PotGaussLong():Pot(Pot::Name::GAUSS_LONG),eps_(1.0){}
	~PotGaussLong(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotGaussLong& pot);
	
	//==== access ====
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	KSpace::Coul& coul(){return coul_;}
	const KSpace::Coul& coul()const{return coul_;}
	Eigen::VectorXi& f(){return f_;}
	const Eigen::VectorXi& f()const{return f_;}
	int& f(int i){return f_[i];}
	const int& f(int i)const{return f_[i];}
	Eigen::VectorXd& radius(){return radius_;}
	const Eigen::VectorXd& radius()const{return radius_;}
	double& radius(int i){return radius_[i];}
	const double& radius(int i)const{return radius_[i];}
	const Eigen::MatrixXd& rij()const{return rij_;}
	const double& rij(int i, int j)const{return rij_(i,j);}
	
	//==== member functions ====
	void read(Token& token);
	void coeff(Token& token);
	void resize(int);
	void init();
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
	Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J);
	double cQ(Structure& struc);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotGaussLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotGaussLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotGaussLong& obj, const char* arr);
	
}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotGaussLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotGaussLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotGaussLong& obj, const char* arr);
	
}

#endif