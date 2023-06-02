#pragma once
#ifndef BASIS_ANGULAR_HPP
#define BASIS_ANGULAR_HPP

// c++ libraries
#include <ostream>
//eigen
#include <Eigen/Dense>
// symmetry functions
#include "src/nnp/cutoff.hpp"
#include "src/nnp/basis.hpp"
// serialize
#include "src/mem/serialize.hpp"

#ifndef BASIS_ANGULAR_PRINT_FUNC
#define BASIS_ANGULAR_PRINT_FUNC 0
#endif

//*****************************************
// PhiAN - angular function names
//*****************************************

class PhiAN{
public:
	enum Type{
		UNKNOWN,
		GAUSS,//gaussian
		IPOWP,//inverse power - product
		IPOWS,//inverse power - sum
		SECHP,//sech - product
		SECHS//sech - sum
	};
	//constructor
	PhiAN():t_(Type::UNKNOWN){}
	PhiAN(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static PhiAN read(const char* str);
	static const char* name(const PhiAN& phiAN);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const PhiAN& t);

//*****************************************
// BasisA - angular basis
//*****************************************

class BasisA: public Basis{
private:
	PhiAN phiAN_;//type of angular functions
	int alpha_;//power
	std::vector<double> eta_;//radial width
	std::vector<double> zeta_;//angular width
	std::vector<double> ietap_;//eta^-p
	std::vector<int> lambda_;//sign of cosine window
	std::vector<double> phif_;
	std::vector<std::vector<double> > etaf_;
public:
	//==== constructors/destructors ====
	BasisA():Basis(),phiAN_(PhiAN::UNKNOWN){}
	BasisA(double rc, cutoff::Name cutname, cutoff::Norm cutnorm, int nf, PhiAN phiAN);
	~BasisA();
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const BasisA& basisA);
	
	//==== reading/writing ====
	static void write(FILE* writer, const BasisA& basis);
	static void read(FILE* writer, BasisA& basis);
	
	//==== member access ====
	PhiAN& phiAN(){return phiAN_;}
	const PhiAN& phiAN()const{return phiAN_;}
	int& alpha(){return alpha_;}
	const int& alpha()const{return alpha_;}
	double& eta(int i){return eta_[i];}
	const double& eta(int i)const{return eta_[i];}
	double& zeta(int i){return zeta_[i];}
	const double& zeta(int i)const{return zeta_[i];}
	int& lambda(int i){return lambda_[i];}
	const int& lambda(int i)const{return lambda_[i];}
	std::vector<double>& eta(){return eta_;}
	const std::vector<double>& eta()const{return eta_;}
	std::vector<double>& zeta(){return zeta_;}
	const std::vector<double>& zeta()const{return zeta_;}
	std::vector<int>& lambda(){return lambda_;}
	const std::vector<int>& lambda()const{return lambda_;}
	std::vector<double>& phif(){return phif_;}
	const std::vector<double>& phif()const{return phif_;}
	std::vector<std::vector<double> >& etaf(){return etaf_;}
	const std::vector<std::vector<double> >& etaf()const{return etaf_;}
	
	Eigen::VectorXd& symm(){return symm_;}
	const Eigen::VectorXd& symm()const{return symm_;}
	
	//==== member functions ====
	void clear();
	void resize(int size);
	void init();
	double symmf(double cos, const double dr[3], double eta, double zeta, int lambda, int alpha)const;
	void symmd(double& fphi, double* feta, double cos, const double dr[3], double eta, double zeta, int lambda, int alpha)const;
	void symm(double cos, const double d[3]);
	void force(double& phi, double* eta, double cos, const double d[3], const double* dEdG)const;
	void compute(double cos, const double d[3], double* symm, double* phi, double* eta0, double* eta1, double* eta2)const;
	void forcep(double cos, const double dr[3]);
};

bool operator==(const BasisA& basis1, const BasisA& basis2);
inline bool operator!=(const BasisA& basis1, const BasisA& basis2){return !(basis1==basis2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisA& obj, const char* arr);
	
}

#endif