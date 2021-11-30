#pragma once
#ifndef SYMM_RADIAL_HPP
#define SYMM_RADIAL_HPP

// c++ libraries
#include <iosfwd>
// ann - serialization
#include "src/mem/serialize.hpp"

//*****************************************
// PhiRN - radial function names
//*****************************************

class PhiRN{
public:
	enum Type{
		UNKNOWN=0,
		G1=1,//Behler G1
		G2=2,//Behler G2
		T1=3//tanh
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
// PhiR - radial function base class
//*****************************************

struct PhiR{
	//==== constructors/destructors ====
	virtual ~PhiR(){}
	//==== member functions - evaluation ====
	virtual double val(double r, double cut)const noexcept=0;
	virtual double grad(double r, double cut, double gcut)const noexcept=0;
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiR& f);
bool operator==(const PhiR& phir1, const PhiR& phir2);
inline bool operator!=(const PhiR& phir1, const PhiR& phir2){return !(phir1==phir2);}

//*****************************************
// PhiR - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiR& obj, const char* arr);
	
}

#endif