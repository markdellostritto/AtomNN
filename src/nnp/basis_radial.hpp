#pragma once
#ifndef BASIS_RADIAL_HPP
#define BASIS_RADIAL_HPP

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// symmetry functions
#include "src/nnp/cutoff.hpp"
#include "src/nnp/basis.hpp"
// ann - serialization
#include "src/mem/serialize.hpp"

#ifndef BASIS_RADIAL_PRINT_FUNC
#define BASIS_RADIAL_PRINT_FUNC 0
#endif

//*****************************************
// PhiRN - radial function names
//*****************************************

class PhiRN{
public:
	enum Type{
		UNKNOWN,
		GAUSSIAN,//Behler G2
		TANH,//tanh
		SOFTPLUS,//softplus
		LOGCOSH,//log-cosh
		SWISH,
		MISH
	};
	//constructor
	PhiRN():t_(Type::UNKNOWN){}
	PhiRN(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static PhiRN read(const char* str);
	static const char* name(const PhiRN& phiRN);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const PhiRN& phiRN);

//*****************************************
// BasisR - radial basis
//*****************************************

class BasisR: public Basis{
private:
	PhiRN phiRN_;//type of radial functions
	std::vector<double> rs_;//center
	std::vector<double> eta_;//width
public:
	//==== constructors/destructors ====
	BasisR():Basis(),phiRN_(PhiRN::UNKNOWN){}
	BasisR(double rc, cutoff::Name cutname, cutoff::Norm cutnorm, int nf, PhiRN phiRN);
	~BasisR();
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const BasisR& basisR);
	
	//==== reading/writing ====
	static void write(FILE* writer, const BasisR& basis);
	static void read(FILE* writer, BasisR& basis);
	
	//==== member access ====
	PhiRN& phiRN(){return phiRN_;}
	const PhiRN& phiRN()const{return phiRN_;}
	double& rs(int i){return rs_[i];}
	const double& rs(int i)const{return rs_[i];}
	double& eta(int i){return eta_[i];}
	const double& eta(int i)const{return eta_[i];}
	std::vector<double>& rs(){return rs_;}
	const std::vector<double>& rs()const{return rs_;}
	std::vector<double>& eta(){return eta_;}
	const std::vector<double>& eta()const{return eta_;}
	
	Eigen::VectorXd& symm(){return symm_;}
	const Eigen::VectorXd& symm()const{return symm_;}
	
	//==== member functions ====
	void clear();
	void resize(int size);
	double symmf(double dr, double eta, double rs)const;
	double symmd(double dr, double eta, double rs)const;
	void symm(double dr);
	double force(double dr, const double* dEdG)const;
	void compute(double dr, double* symm, double* amp)const;
};
std::ostream& operator<<(std::ostream& out, const BasisR& basisR);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisR& obj, const char* arr);
	
}

#endif