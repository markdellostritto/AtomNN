#pragma once
#ifndef SYMM_RADIAL_TANH_HPP
#define SYMM_RADIAL_TANH_HPP

// c++ libaries
#include <iosfwd>
// ann - symm - radial
#include "ann_symm_radial.h"
// ann - serialization
#include "ann_serialize.h"

//*****************************************
// PHIR - T1 - DelloStritto
//*****************************************

struct PhiR_T1 final: public PhiR{
	//==== function parameters ====
	double eta;//radial exponential width 
	double rs;//center of radial window
	//==== constructors/destructors ====
	PhiR_T1():PhiR(),eta(0.0),rs(0.0){}
	PhiR_T1(double rs_, double eta_):PhiR(),eta(eta_),rs(rs_){}
	//==== member functions - evaluation ====
	double val(double r, double cut)const;
	double grad(double r, double cut, double gcut)const;
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiR_T1& f);
bool operator==(const PhiR_T1& phir1, const PhiR_T1& phir2);
inline bool operator!=(const PhiR_T1& phir1, const PhiR_T1& phir2){return !(phir1==phir2);}

//*****************************************
// PHIR - T1 - DelloStritto - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiR_T1& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiR_T1& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiR_T1& obj, const char* arr);
	
}

#endif