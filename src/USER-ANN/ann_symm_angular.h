#pragma once
#ifndef SYMM_ANGULAR_HPP
#define SYMM_ANGULAR_HPP

// c++ libraries
#include <iosfwd>
// ann - serialization
#include "ann_serialize.h"

//*****************************************
// PhiAN - angular function names
//*****************************************

class PhiAN{
public:
	enum Type{
		UNKNOWN=0,
		G3=1,//Behler G3
		G4=2//Behler G4
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
// PhiA - angular function base class
//*****************************************

struct PhiA{
	//==== constructors/destructors ====
	virtual ~PhiA(){}
	//==== member functions - evaluation ====
	virtual double val(double cos, const double r[3], const double c[3])const=0;
	virtual double dist(const double r[3], const double c[3])const=0;
	virtual double angle(double cos)const=0;
	virtual double grad_angle(double cos)const=0;
	virtual double grad_dist_0(const double r[3], const double c[3], double gij)const=0;
	virtual double grad_dist_1(const double r[3], const double c[3], double gik)const=0;
	virtual double grad_dist_2(const double r[3], const double c[3], double gjk)const=0;
	virtual void compute_angle(double cos, double& val, double& grad)const=0;
	virtual void compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const=0;
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiA& f);
bool operator==(const PhiA& phia1, const PhiA& phia2);
inline bool operator!=(const PhiA& phia1, const PhiA& phia2){return !(phia1==phia2);};

//*****************************************
// PhiA - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiA& obj, const char* arr);
	
}

#endif