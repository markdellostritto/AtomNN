#pragma once
#ifndef BASIS_ANGULAR_HPP
#define BASIS_ANGULAR_HPP

// c++ libraries
#include <ostream>
//eigen
#include <Eigen/Dense>
// symmetry functions
#include "src/nnp/cutoff.hpp"
#include "src/nnp/symm_angular.hpp"
// serialize
#include "src/mem/serialize.hpp"

#ifndef BASIS_ANGULAR_PRINT_FUNC
#define BASIS_ANGULAR_PRINT_FUNC 0
#endif

struct BasisA{
private:
	double rc_;//cutoff radius
	double norm_;//normalization factor
	PhiAN phiAN_;//type of angular functions
	cutoff::Norm normT_;//normalization scheme
	int nfA_;//number of angular functions
	cutoff::Func* cutoff_;//cutoff function
	PhiA** fA_;//angular functions
	Eigen::VectorXd symm_;//symmetry function
public:
	//==== constructors/destructors ====
	BasisA():normT_(cutoff::Norm::UNKNOWN),phiAN_(PhiAN::UNKNOWN),nfA_(0),fA_(NULL),cutoff_(NULL){}
	BasisA(const BasisA& basisA);
	~BasisA();
	
	//==== operators ====
	BasisA& operator=(const BasisA& basisA);
	friend std::ostream& operator<<(std::ostream& out, const BasisA& basisA);
	
	//==== initialization ====
	void init_G3(int nA, cutoff::Norm norm, cutoff::Name tcut, double rcut);
	void init_G4(int nA, cutoff::Norm norm, cutoff::Name tcut, double rcut);
	
	//==== reading/writing ====
	static void write(const char* file,const BasisA& basis);
	static void read(const char* file, BasisA& basis);
	static void write(FILE* writer,const BasisA& basis);
	static void read(FILE* writer, BasisA& basis);
	
	//==== member access ====
	cutoff::Func* cutoff(){return cutoff_;}
	const cutoff::Func* cutoff()const{return cutoff_;}
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	const int& nfA()const{return nfA_;}
	PhiAN& phiAN(){return phiAN_;}
	const PhiAN& phiAN()const{return phiAN_;}
	cutoff::Norm& normT(){return normT_;}
	const cutoff::Norm& normT()const{return normT_;}
	PhiA& fA(int i){return *fA_[i];}
	const PhiA& fA(int i)const{return *fA_[i];}
	Eigen::VectorXd& symm(){return symm_;}
	const Eigen::VectorXd& symm()const{return symm_;}
	
	//==== member functions ====
	void clear();
	void symm(double cos, const double d[3]);
	void force(double& phi, double* eta, double cos, const double d[3], const double* dEdG)const;
	
	//==== static functions ====
	static double norm(cutoff::Norm normT, double rc);
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