#pragma once
#ifndef BASIS_RADIAL_HPP
#define BASIS_RADIAL_HPP

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// symmetry functions
#include "ann_cutoff.h"
#include "ann_symm_radial.h"
// ann - serialization
#include "ann_serialize.h"

#ifndef BASIS_RADIAL_PRINT_FUNC
#define BASIS_RADIAL_PRINT_FUNC 0
#endif

struct BasisR{
private:
	PhiRN phiRN_;//type of radial functions
	cutoff::Norm normT_;//normalization scheme
	int nfR_;//number of radial functions
	double rc_;//cutoff radius
	double norm_;//normalization factor
	cutoff::Func* cutoff_;//cutoff function
	PhiR** fR_;//radial functions
	Eigen::VectorXd symm_;//symmetry function
public:
	//==== constructors/destructors ====
	BasisR():norm_(cutoff::Norm::UNKNOWN),phiRN_(PhiRN::UNKNOWN),nfR_(0),fR_(NULL),cutoff_(NULL){}
	BasisR(const BasisR& basisR);
	~BasisR();
	
	//==== operators ====
	BasisR& operator=(const BasisR& basis);
	friend std::ostream& operator<<(std::ostream& out, const BasisR& basisR);
	
	//==== initialization ====
	void init_G1(int nR, cutoff::Norm norm, cutoff::Name tcut, double rcut);
	void init_G2(int nR, cutoff::Norm norm, cutoff::Name tcut, double rcut);
	void init_T1(int nR, cutoff::Norm norm, cutoff::Name tcut, double rcut);
	
	//==== reading/writing ====
	static void write(const char* filename, const BasisR& basis);
	static void read(const char* filename, BasisR& basis);
	static void write(FILE* writer, const BasisR& basis);
	static void read(FILE* reader, BasisR& basis);
	
	//==== member access ====
	cutoff::Func* cutoff(){return cutoff_;}
	const cutoff::Func* cutoff()const{return cutoff_;}
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	const int& nfR()const{return nfR_;}
	PhiRN& phiRN(){return phiRN_;}
	const PhiRN& phiRN()const{return phiRN_;}
	cutoff::Norm& normT(){return normT_;}
	const cutoff::Norm& normT()const{return normT_;}
	PhiR& fR(int i){return *fR_[i];}
	const PhiR& fR(int i)const{return *fR_[i];}
	Eigen::VectorXd& symm(){return symm_;}
	const Eigen::VectorXd& symm()const{return symm_;}
	
	//==== member functions ====
	void clear();
	void symm(double dr);
	double force(double dr, const double* dEdG)const;
	
	//==== static functions ====
	static double norm(cutoff::Norm normT, double rc);
};
std::ostream& operator<<(std::ostream& out, const BasisR& basisR);

bool operator==(const BasisR& basis1, const BasisR& basis2);
inline bool operator!=(const BasisR& basis1, const BasisR& basis2){return !(basis1==basis2);}

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